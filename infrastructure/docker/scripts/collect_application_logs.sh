#!/bin/bash

set -e

BASELINE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/baseline"

mkdir -p "$BASELINE_DIR/logs/juice_shop"

echo "[1/3] Collecting Juice Shop container logs..."

docker logs autoshield-juice-shop \
    > "$BASELINE_DIR/logs/juice_shop/container.log" 2>&1


echo "[2/3] Collecting Juice Shop access logs..."

docker cp \
    autoshield-juice-shop:/juice-shop/logs/access.log.2026-10-08 \
    "$BASELINE_DIR/logs/juice_shop/access.log"


echo "[3/3] Collecting Juice Shop audit manifest..."

docker cp \
    autoshield-juice-shop:/juice-shop/logs/audit.json \
    "$BASELINE_DIR/logs/juice_shop/audit.json"


echo ""
echo "Application log collection complete."
echo "Output: $BASELINE_DIR/logs/juice_shop"