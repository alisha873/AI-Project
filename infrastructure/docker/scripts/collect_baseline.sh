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
echo "[2/5] Collecting core Docker/network configuration..."

mkdir -p "$BASELINE_DIR/configs" "$BASELINE_DIR/network" "$BASELINE_DIR/system"

for spec in \
    "autoshield-nginx:nginx_inspect.json" \
    "autoshield-juice-shop:juice_shop_inspect.json" \
    "autoshield-api:autoshield_api_inspect.json" \
    "autoshield-postgres:postgres_inspect.json" \
    "autoshield-redis:redis_inspect.json" \
    "autoshield-linux-admin:linux_admin_inspect.json"; do
    container="${spec%%:*}"
    filename="${spec#*:}"
    echo "  - Inspecting $container..."
    docker inspect "$container" > "$BASELINE_DIR/system/$filename"
done

docker network inspect autoshield_net > "$BASELINE_DIR/network/autoshield_net.json"
docker compose \
    -f infrastructure/docker/docker-compose.yml \
    config \
    > "$BASELINE_DIR/configs/docker-compose.resolved.yml"

echo ""
echo "[3/5] Collecting Docker events..."
"$SCRIPT_DIR/collect_docker_events.sh"

echo ""
echo "[4/5] Capturing controlled normal traffic..."
"$SCRIPT_DIR/collect_pcap.sh"

echo ""
echo "[5/5] Collecting application and service logs..."
"$SCRIPT_DIR/collect_application_logs.sh"

echo ""
echo "========================================"
echo " Baseline collection complete"
echo "========================================"
echo ""
echo "Output:"
echo "  $BASELINE_DIR"
echo ""