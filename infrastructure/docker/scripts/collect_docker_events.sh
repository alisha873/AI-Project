#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
if (( $# > 1 )); then
    echo "Usage: $0 [output_root]" >&2
    exit 2
fi
OUTPUT_ROOT="${1:-$PROJECT_ROOT/baseline}"
if [[ "$OUTPUT_ROOT" != /* ]]; then
    OUTPUT_ROOT="$PROJECT_ROOT/$OUTPUT_ROOT"
fi
if (( $# == 0 )); then
    EVENT_DIR="$OUTPUT_ROOT/events"
else
    EVENT_DIR="$OUTPUT_ROOT/logs"
fi

mkdir -p "$EVENT_DIR"

echo "Collecting Docker lifecycle and network events..."

docker events \
    --since 10m \
    --until 0s \
    --filter type=container \
    --filter type=network \
    --filter event=start \
    --filter event=stop \
    --filter event=die \
    --filter event=kill \
    --filter event=restart \
    --filter event=oom \
    --filter event=connect \
    --filter event=disconnect \
    > "$EVENT_DIR/docker_events.log"

echo ""
echo "Docker event collection complete."
echo "Output: $EVENT_DIR/docker_events.log"