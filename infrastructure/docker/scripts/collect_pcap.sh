#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
if (( $# > 1 )); then
    echo "Usage: $0 [output_root]" >&2
    exit 2
fi
OUTPUT_ROOT="${1:-$PROJECT_ROOT/baseline}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
fi
PCAP_DIR="$OUTPUT_ROOT/network"
NETWORK_NAME="autoshield_net"
CAPTURE_SECONDS=12
INCIDENT_MODE=0
(( $# == 0 )) || INCIDENT_MODE=1
if (( INCIDENT_MODE == 0 )); then
    CURL_BIN="$(command -v curl || true)"
    if [[ -z "$CURL_BIN" ]]; then
        for candidate in /usr/bin/curl /usr/local/bin/curl; do
            if [[ -x "$candidate" ]]; then
                CURL_BIN="$candidate"
                break
            fi
        done
    fi
    if [[ -z "$CURL_BIN" ]]; then
        echo "curl is required to generate controlled normal baseline requests." >&2
        exit 1
    fi
fi

mkdir -p "$PCAP_DIR"

if (( INCIDENT_MODE )) && [[ "$(uname -s)" == "Darwin" ]]; then
    CAPTURE_TARGET="--network container:autoshield-nginx --cap-add NET_RAW --cap-add NET_ADMIN"
    CAPTURE_INTERFACE="eth0"
elif (( INCIDENT_MODE )); then
    CAPTURE_TARGET="--network host --privileged"
    CAPTURE_INTERFACE="br-$(docker network inspect "$NETWORK_NAME" --format '{{.Id}}' | cut -c1-12)"
else
    CAPTURE_TARGET="--network host --privileged"
    CAPTURE_INTERFACE="br-$(docker network inspect "$NETWORK_NAME" --format '{{.Id}}' | cut -c1-12)"
fi
CAPTURE_LOG="$(mktemp)"
REQUEST_RESULTS="$(mktemp)"
CAPTURE_PID=""
cleanup() {
    if [[ -n "$CAPTURE_PID" ]] && kill -0 "$CAPTURE_PID" 2>/dev/null; then
        kill "$CAPTURE_PID" 2>/dev/null || true
        wait "$CAPTURE_PID" 2>/dev/null || true
    fi
    rm -f "$CAPTURE_LOG" "$REQUEST_RESULTS"
}
trap cleanup EXIT

echo "Network: $NETWORK_NAME"
echo "Capture interface: $CAPTURE_INTERFACE"
echo "Output: $PCAP_DIR/traffic.pcap"
if (( INCIDENT_MODE )); then
    CAPTURE_MODE="passive_incident_collection_window"
    echo "Starting a passive $CAPTURE_SECONDS-second incident capture; no requests will be generated."
else
    CAPTURE_MODE="baseline_collector_generated_normal_requests"
    echo "Starting a $CAPTURE_SECONDS-second capture for controlled normal requests."
fi

docker run --rm \
    $CAPTURE_TARGET \
    -v "$PCAP_DIR:/captures" \
    alpine:3.20 \
    sh -c "
        apk add --no-cache tcpdump >/dev/null 2>&1 &&
        timeout $CAPTURE_SECONDS tcpdump \
            -i $CAPTURE_INTERFACE \
            -nn \
            -s 0 \
            -w /captures/traffic.pcap
    " >"$CAPTURE_LOG" 2>&1 &
CAPTURE_PID=$!

capture_ready=0
for _ in {1..90}; do
    if grep -q 'listening on' "$CAPTURE_LOG"; then
        capture_ready=1
        break
    fi
    if ! kill -0 "$CAPTURE_PID" 2>/dev/null; then
        cat "$CAPTURE_LOG" >&2
        echo "Packet capture exited before becoming ready." >&2
        exit 1
    fi
    sleep 1
done
if (( capture_ready == 0 )); then
    cat "$CAPTURE_LOG" >&2
    echo "Timed out waiting for tcpdump to become ready." >&2
    exit 1
fi

record_request() {
    local path="$1"
    local code
    code="$("$CURL_BIN" --max-time 10 --silent --show-error \
        --output /dev/null --write-out '%{http_code}' "http://localhost:8080$path")"
    printf '%s\t%s\n' "$path" "$code" >> "$REQUEST_RESULTS"
    [[ "$code" == "200" ]] || {
        echo "Normal request $path returned HTTP $code." >&2
        return 1
    }
}

if (( INCIDENT_MODE == 0 )); then
    echo "Generating ordinary GET requests through Nginx..."
    record_request "/"
    record_request "/api/health"
    record_request "/api/data"
    record_request "/api/data"
    record_request "/"
fi

capture_status=0
wait "$CAPTURE_PID" || capture_status=$?
CAPTURE_PID=""
if (( capture_status != 0 && capture_status != 124 )); then
    cat "$CAPTURE_LOG" >&2
    echo "Packet capture failed with exit status $capture_status." >&2
    exit "$capture_status"
fi

if [[ ! -s "$PCAP_DIR/traffic.pcap" ]]; then
    echo "Packet capture did not produce a non-empty PCAP." >&2
    exit 1
fi

SUMMARY_FILENAME="normal_traffic_summary.json"
(( INCIDENT_MODE == 0 )) || SUMMARY_FILENAME="traffic_summary.json"
python3 "$SCRIPT_DIR/summarize_pcap.py" \
    "$PCAP_DIR/traffic.pcap" \
    "$PCAP_DIR/$SUMMARY_FILENAME" \
    "$OUTPUT_ROOT/network/autoshield_net.json" \
    "$REQUEST_RESULTS" \
    "$CAPTURE_MODE"

echo "PCAP and evidence-derived normal traffic summary collected."