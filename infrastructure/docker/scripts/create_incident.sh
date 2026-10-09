#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_ROOT"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <incident_id>"
    echo "Example: $0 incident_002"
    exit 1
fi

INCIDENT_ID="$1"
INCIDENT_DIR="$PROJECT_ROOT/$INCIDENT_ID"

if [[ ! "$INCIDENT_ID" =~ ^incident_[0-9]{3}$ ]]; then
    echo "Error: incident ID must match incident_NNN"
    exit 1
fi

if [[ -e "$INCIDENT_DIR" ]]; then
    echo "Error: incident directory already exists: $INCIDENT_DIR"
    exit 1
fi

echo "========================================"
echo " AutoShield Incident Package Builder"
echo "========================================"
echo ""
echo "Incident ID: $INCIDENT_ID"
echo "Output:      $INCIDENT_DIR"
echo ""

echo "[1/5] Creating incident directory..."

mkdir -p \
    "$INCIDENT_DIR/logs" \
    "$INCIDENT_DIR/network" \
    "$INCIDENT_DIR/configs" \
    "$INCIDENT_DIR/system"

echo "  Created:"
echo "    logs/"
echo "    network/"
echo "    configs/"
echo "    system/"

echo ""
echo "[2/5] Collecting live system state..."
"$SCRIPT_DIR/collect_system_state.sh" "$INCIDENT_DIR"

echo "[3/5] Collecting Docker and topology configuration..."
mkdir -p "$INCIDENT_DIR/configs/containers"

docker compose \
    -f infrastructure/docker/docker-compose.yml \
    config \
    > "$INCIDENT_DIR/configs/docker-compose.resolved.yml"
cp infrastructure/docker/nginx/nginx.conf "$INCIDENT_DIR/configs/nginx.conf"

for container in \
    autoshield-nginx \
    autoshield-juice-shop \
    autoshield-api \
    autoshield-postgres \
    autoshield-redis \
    autoshield-linux-admin; do
    docker inspect "$container" > "$INCIDENT_DIR/configs/containers/$container.json"
done
docker network inspect autoshield_net > "$INCIDENT_DIR/network/autoshield_net.json"

echo "[4/5] Collecting Docker events, traffic, and service logs..."
"$SCRIPT_DIR/collect_docker_events.sh" "$INCIDENT_DIR"
"$SCRIPT_DIR/collect_pcap.sh" "$INCIDENT_DIR"
"$SCRIPT_DIR/collect_application_logs.sh" "$INCIDENT_DIR"

echo "[5/5] Generating public metadata and validating package..."
python3 - "$INCIDENT_DIR" "$INCIDENT_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

incident_dir = Path(sys.argv[1]).resolve()
incident_id = sys.argv[2]
required_directories = ("configs", "logs", "network", "system")
required_files = (
    "configs/docker-compose.resolved.yml",
    "configs/nginx.conf",
    "configs/containers/autoshield-nginx.json",
    "configs/containers/autoshield-juice-shop.json",
    "configs/containers/autoshield-api.json",
    "configs/containers/autoshield-postgres.json",
    "configs/containers/autoshield-redis.json",
    "configs/containers/autoshield-linux-admin.json",
    "logs/nginx.log",
    "logs/juice_shop/container.log",
    "logs/juice_shop/current/source_manifest.json",
    "logs/autoshield_api/container.log",
    "logs/postgres/container.log",
    "logs/redis/container.log",
    "logs/linux_admin/container.log",
    "network/autoshield_net.json",
    "network/traffic.pcap",
    "network/traffic_summary.json",
    "system/system_state.json",
)
for directory in required_directories:
    path = incident_dir / directory
    if not path.is_dir():
        raise SystemExit(f"Required package directory is missing: {path}")
for relative_path in required_files:
    path = incident_dir / relative_path
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Required evidence file is missing or empty: {relative_path}")
if not (incident_dir / "logs/docker_events.log").is_file():
    raise SystemExit("Required current Docker event collection output is missing.")

for path in sorted(incident_dir.rglob("*.json")):
    if path.name == "metadata.json":
        continue
    try:
        with path.open(encoding="utf-8") as source:
            json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Invalid JSON evidence {path.relative_to(incident_dir)}: {error}")

state_path = incident_dir / "system/system_state.json"
with state_path.open(encoding="utf-8") as source:
    system_state = json.load(source)
juice_manifest_path = incident_dir / "logs/juice_shop/current/source_manifest.json"
with juice_manifest_path.open(encoding="utf-8") as source:
    juice_manifest = json.load(source)
if juice_manifest.get("source_container") != "autoshield-juice-shop":
    raise SystemExit("Juice Shop current-log provenance is invalid.")
if juice_manifest.get("collection_status") != "copied":
    collection_warnings = ["Juice Shop current log directory collection failed or was incomplete."]
else:
    collection_warnings = []
for artifact in juice_manifest.get("files", []):
    if not artifact.get("source", "").startswith("autoshield-juice-shop:/juice-shop/logs/"):
        raise SystemExit(f"Unexpected Juice Shop evidence source: {artifact.get('source')}")
expected = {
    "autoshield-nginx",
    "autoshield-juice-shop",
    "autoshield-api",
    "autoshield-postgres",
    "autoshield-redis",
    "autoshield-linux-admin",
}
containers = {item["name"]: item for item in system_state.get("running_containers", [])}
missing = expected - containers.keys()
if missing:
    raise SystemExit(f"System state is missing expected containers: {sorted(missing)}")
for name in expected:
    if "autoshield_net" not in containers[name].get("network_membership", {}):
        raise SystemExit(f"Container is not represented on autoshield_net: {name}")

evidence_files = sorted(
    path.relative_to(incident_dir).as_posix()
    for path in incident_dir.rglob("*")
    if path.is_file() and path.name != "metadata.json"
)
if (incident_dir / "logs/docker_events.log").stat().st_size == 0:
    collection_warnings.append("No Docker events were returned for the collector's current event window.")
if not (incident_dir / "logs/linux_admin/auth.log").is_file():
    collection_warnings.append(
        "Linux Admin had not generated /var/log/auth.log at collection time; "
        "no authentication events were available. The Linux Admin container log is included."
    )
access_logs = [item for item in juice_manifest["files"] if item["category"] == "access_log"]
if not access_logs:
    collection_warnings.append("No current Juice Shop access log was available in the container.")
elif not any(item["size_bytes"] > 0 for item in access_logs):
    collection_warnings.append("Current Juice Shop access log file(s) existed but were empty.")
audit_logs = [item for item in juice_manifest["files"] if item["category"] == "audit_manifest"]
if not audit_logs:
    collection_warnings.append("No current Juice Shop audit.json was available in the container.")
elif not any(item["size_bytes"] > 0 for item in audit_logs):
    collection_warnings.append("Current Juice Shop audit.json existed but was empty.")
traffic_summary_path = incident_dir / "network/traffic_summary.json"
with traffic_summary_path.open(encoding="utf-8") as source:
    traffic_summary = json.load(source)
if traffic_summary.get("capture_mode") != "passive_incident_collection_window":
    raise SystemExit("Incident PCAP was not collected in passive incident mode.")
if traffic_summary.get("packet_count", 0) == 0:
    collection_warnings.append("No packets were observed during the passive PCAP collection window.")
metadata = {
    "incident_id": incident_id,
    "collection_version": "1.0",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": "docker",
    "network_capture_mode": traffic_summary["capture_mode"],
    "evidence_files": evidence_files,
    "collection_warnings": collection_warnings,
}
(incident_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(f"Validated {len(evidence_files)} evidence files; all six expected containers are on autoshield_net.")
PY

python3 -m json.tool "$INCIDENT_DIR/metadata.json" > /dev/null

test -d "$INCIDENT_DIR/logs"
test -d "$INCIDENT_DIR/network"
test -d "$INCIDENT_DIR/configs"
test -d "$INCIDENT_DIR/system"
test -f "$INCIDENT_DIR/metadata.json"

python3 -m json.tool "$INCIDENT_DIR/metadata.json" > /dev/null

echo ""
echo "========================================"
echo " Incident package created"
echo "========================================"
echo ""
echo "Output:"
echo "  $INCIDENT_DIR"
echo ""