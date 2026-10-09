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
LOG_DIR="$OUTPUT_ROOT/logs"
JUICE_SHOP_EXPORT="$(mktemp -d)"
trap 'rm -rf "$JUICE_SHOP_EXPORT"' EXIT

mkdir -p \
    "$LOG_DIR/juice_shop" \
    "$LOG_DIR/autoshield_api" \
    "$LOG_DIR/postgres" \
    "$LOG_DIR/redis" \
    "$LOG_DIR/linux_admin"

echo "Collecting container logs for all six AutoShield components..."
collect_container_log() {
    docker logs "$1" 2>&1 | python3 -c \
        'import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().replace(b"\r\n", b"\n"))' \
        > "$2"
}
collect_container_log autoshield-nginx "$LOG_DIR/nginx.log"
collect_container_log autoshield-juice-shop "$LOG_DIR/juice_shop/container.log"
collect_container_log autoshield-api "$LOG_DIR/autoshield_api/container.log"
collect_container_log autoshield-postgres "$LOG_DIR/postgres/container.log"
collect_container_log autoshield-redis "$LOG_DIR/redis/container.log"
collect_container_log autoshield-linux-admin "$LOG_DIR/linux_admin/container.log"

echo "Discovering Juice Shop access logs dynamically..."
mkdir -p "$JUICE_SHOP_EXPORT/logs"
JUICE_SHOP_CURRENT_DIR="$LOG_DIR/juice_shop/current"
mkdir -p "$JUICE_SHOP_CURRENT_DIR"
JUICE_SHOP_COPY_STATUS="copied"
if docker cp autoshield-juice-shop:/juice-shop/logs/. "$JUICE_SHOP_EXPORT/logs/"; then
    cp -R "$JUICE_SHOP_EXPORT/logs/." "$JUICE_SHOP_CURRENT_DIR/"
else
    JUICE_SHOP_COPY_STATUS="copy_failed"
    echo "Warning: could not copy current /juice-shop/logs from autoshield-juice-shop." >&2
fi

python3 - "$JUICE_SHOP_CURRENT_DIR" "$JUICE_SHOP_COPY_STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

current_dir = Path(sys.argv[1])
copy_status = sys.argv[2]
files = []
for path in sorted(current_dir.rglob("*")):
    if not path.is_file() or path.name == "source_manifest.json":
        continue
    relative_path = path.relative_to(current_dir).as_posix()
    if path.name.startswith("access.log"):
        category = "access_log"
    elif path.name == "audit.json":
        category = "audit_manifest"
    else:
        category = "other_current_log"
    files.append({
        "path": relative_path,
        "source": f"autoshield-juice-shop:/juice-shop/logs/{relative_path}",
        "category": category,
        "size_bytes": path.stat().st_size,
    })

manifest = {
    "source_container": "autoshield-juice-shop",
    "source_directory": "/juice-shop/logs",
    "collection_status": copy_status,
    "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "files": files,
}
(current_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
if copy_status != "copied":
    print("Warning: current Juice Shop log collection failed.", file=sys.stderr)
if not any(item["category"] == "access_log" for item in files):
    print("Warning: no current Juice Shop access log was available.", file=sys.stderr)
elif not any(item["category"] == "access_log" and item["size_bytes"] > 0 for item in files):
    print("Warning: current Juice Shop access log files exist but are empty.", file=sys.stderr)
if not any(item["category"] == "audit_manifest" for item in files):
    print("Warning: no current Juice Shop audit.json was available.", file=sys.stderr)
elif not any(item["category"] == "audit_manifest" and item["size_bytes"] > 0 for item in files):
    print("Warning: current Juice Shop audit.json exists but is empty.", file=sys.stderr)
PY

echo "Collecting Linux Admin authentication evidence..."
if docker exec autoshield-linux-admin test -r /var/log/auth.log; then
    docker cp autoshield-linux-admin:/var/log/auth.log "$LOG_DIR/linux_admin/auth.log"
else
    echo "Warning: Linux Admin has no /var/log/auth.log yet; no authentication events have been logged." >&2
fi

echo "Application, database, cache, proxy, and authentication logs collected."
echo "Output: $LOG_DIR"