#!/bin/bash

set -e

BASELINE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/baseline"

mkdir -p "$BASELINE_DIR/events"

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
    > "$BASELINE_DIR/events/docker_events.log"

echo ""
echo "Docker event collection complete."
echo "Output: $BASELINE_DIR/events/docker_events.log"