#!/bin/bash

set -euo pipefail

BASELINE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/baseline"
LOG_DIR="$BASELINE_DIR/logs"
JUICE_SHOP_EXPORT="$(mktemp -d)"
trap 'rm -rf "$JUICE_SHOP_EXPORT"' EXIT

mkdir -p \
    "$LOG_DIR/juice_shop" \
    "$LOG_DIR/autoshield_api" \
    "$LOG_DIR/postgres" \
    "$LOG_DIR/redis" \
    "$LOG_DIR/linux_admin"

echo "Collecting container logs for all six AutoShield components..."
docker logs --timestamps autoshield-nginx \
    > "$LOG_DIR/nginx.log" 2>&1
docker logs --timestamps autoshield-juice-shop \
    > "$LOG_DIR/juice_shop/container.log" 2>&1
docker logs --timestamps autoshield-api \
    > "$LOG_DIR/autoshield_api/container.log" 2>&1
docker logs --timestamps autoshield-postgres \
    > "$LOG_DIR/postgres/container.log" 2>&1
docker logs --timestamps autoshield-redis \
    > "$LOG_DIR/redis/container.log" 2>&1
docker logs --timestamps autoshield-linux-admin \
    > "$LOG_DIR/linux_admin/container.log" 2>&1

echo "Discovering Juice Shop access logs dynamically..."
mkdir -p "$JUICE_SHOP_EXPORT/logs"
docker cp autoshield-juice-shop:/juice-shop/logs/. "$JUICE_SHOP_EXPORT/logs/"

shopt -s nullglob
access_logs=("$JUICE_SHOP_EXPORT/logs"/access.log*)
if (( ${#access_logs[@]} == 0 )); then
    echo "No Juice Shop access log found under /juice-shop/logs." >&2
    exit 1
fi

latest_access_log="${access_logs[0]}"
for candidate in "${access_logs[@]:1}"; do
    if [[ "$candidate" -nt "$latest_access_log" ]]; then
        latest_access_log="$candidate"
    fi
done
cp "$latest_access_log" "$LOG_DIR/juice_shop/access.log"

if [[ ! -s "$JUICE_SHOP_EXPORT/logs/audit.json" ]]; then
    echo "Juice Shop audit manifest is missing or empty." >&2
    exit 1
fi
cp "$JUICE_SHOP_EXPORT/logs/audit.json" "$LOG_DIR/juice_shop/audit.json"

echo "Collecting Linux Admin authentication evidence..."
docker cp autoshield-linux-admin:/var/log/auth.log "$LOG_DIR/linux_admin/auth.log"

echo "Application, database, cache, proxy, and authentication logs collected."
echo "Output: $LOG_DIR"