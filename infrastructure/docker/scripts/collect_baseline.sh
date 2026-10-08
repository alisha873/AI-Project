#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

BASELINE_DIR="$PROJECT_ROOT/baseline"

echo "========================================"
echo " AutoShield Baseline Collection"
echo "========================================"
echo ""

echo "[1/5] Collecting system state..."
"$SCRIPT_DIR/collect_system_state.sh"

echo ""
echo "[2/5] Collecting application and authentication logs..."
"$SCRIPT_DIR/collect_application_logs.sh"

echo ""
echo "[3/5] Collecting Docker events..."
"$SCRIPT_DIR/collect_docker_events.sh"

echo ""
echo "[4/5] Collecting packet capture..."
"$SCRIPT_DIR/collect_pcap.sh"

echo ""
echo "[5/5] Collecting core Docker/network configuration..."

mkdir -p "$BASELINE_DIR/configs"
mkdir -p "$BASELINE_DIR/network"
mkdir -p "$BASELINE_DIR/system"

echo "  - Nginx inspection..."
docker inspect autoshield-nginx \
    > "$BASELINE_DIR/system/nginx_inspect.json"

echo "  - Juice Shop inspection..."
docker inspect autoshield-juice-shop \
    > "$BASELINE_DIR/system/juice_shop_inspect.json"

echo "  - AutoShield API inspection..."
docker inspect autoshield-api \
    > "$BASELINE_DIR/system/autoshield_api_inspect.json"

echo "  - PostgreSQL inspection..."
docker inspect autoshield-postgres \
    > "$BASELINE_DIR/system/postgres_inspect.json"

echo "  - Redis inspection..."
docker inspect autoshield-redis \
    > "$BASELINE_DIR/system/redis_inspect.json"

echo "  - Linux Admin inspection..."
docker inspect autoshield-linux-admin \
    > "$BASELINE_DIR/system/linux_admin_inspect.json"

echo "  - Docker network inspection..."
docker network inspect autoshield_net \
    > "$BASELINE_DIR/network/autoshield_net.json"

echo "  - Resolved Compose configuration..."
docker compose \
    -f infrastructure/docker/docker-compose.yml \
    config \
    > "$BASELINE_DIR/configs/docker-compose.resolved.yml"

echo ""
echo "========================================"
echo " Baseline collection complete"
echo "========================================"
echo ""
echo "Output:"
echo "  $BASELINE_DIR"
echo ""