#!/bin/bash

set -e

BASELINE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/baseline"

mkdir -p "$BASELINE_DIR/logs/juice_shop"
mkdir -p "$BASELINE_DIR/logs/autoshield_api"
mkdir -p "$BASELINE_DIR/logs/linux_admin"

echo "[1/5] Collecting Juice Shop container logs..."

docker logs autoshield-juice-shop \
    > "$BASELINE_DIR/logs/juice_shop/container.log" 2>&1


echo "[2/5] Collecting Juice Shop access logs..."

docker cp \
    autoshield-juice-shop:/juice-shop/logs/access.log.2026-10-08 \
    "$BASELINE_DIR/logs/juice_shop/access.log"


echo "[3/5] Collecting Juice Shop audit manifest..."

docker cp \
    autoshield-juice-shop:/juice-shop/logs/audit.json \
    "$BASELINE_DIR/logs/juice_shop/audit.json"


echo "[4/5] Collecting AutoShield API logs..."

docker logs autoshield-api \
    > "$BASELINE_DIR/logs/autoshield_api/container.log" 2>&1


echo "[5/5] Collecting Linux authentication logs..."

docker cp \
    autoshield-linux-admin:/var/log/auth.log \
    "$BASELINE_DIR/logs/linux_admin/auth.log"


echo ""
echo "Application and authentication log collection complete."
echo "Output: $BASELINE_DIR/logs"