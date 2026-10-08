#!/bin/bash

set -e

BASELINE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/baseline"
PCAP_DIR="$BASELINE_DIR/network"

mkdir -p "$PCAP_DIR"

NETWORK_NAME="autoshield_net"
CAPTURE_INTERFACE="br-$(docker network inspect "$NETWORK_NAME" --format '{{.Id}}' | cut -c1-12)"

echo "Network: $NETWORK_NAME"
echo "Capture interface: $CAPTURE_INTERFACE"
echo "Output: $PCAP_DIR/traffic.pcap"
echo ""

docker run --rm --privileged \
    --network host \
    -v "$PCAP_DIR:/captures" \
    alpine:3.20 \
    sh -c "
        apk add --no-cache tcpdump >/dev/null 2>&1 &&
        timeout 30 tcpdump \
            -i $CAPTURE_INTERFACE \
            -nn \
            -s 0 \
            -w /captures/traffic.pcap
    "

echo ""
echo "PCAP collection complete."
echo "Output: $PCAP_DIR/traffic.pcap"