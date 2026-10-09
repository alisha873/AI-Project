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
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"
SYSTEM_DIR="$OUTPUT_ROOT/system"

mkdir -p "$SYSTEM_DIR"

echo "[1/5] Collecting all running containers..."
docker ps \
    --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' \
    > "$SYSTEM_DIR/containers.txt"

echo "[2/5] Collecting container IP addresses..."
{
    echo "container|network|ip_address"

    while IFS= read -r container; do
        docker inspect --format \
            '{{.Name}}{{range $network, $settings := .NetworkSettings.Networks}}{{printf "|%s|%s\n" $network $settings.IPAddress}}{{end}}' \
            "$container" | sed 's#^/##'
    done < <(docker ps --format '{{.Names}}')
} > "$SYSTEM_DIR/container_ips.txt"

echo "[3/5] Collecting Docker networks..."
docker network ls > "$SYSTEM_DIR/docker_networks.txt"

echo "[4/5] Collecting processes for all running containers..."
{
    while IFS= read -r container; do
        [[ -n "$container" ]] || continue
        printf '===== %s =====\n' "$container"
        docker top "$container"
        printf '\n'
    done < <(docker ps --format '{{.Names}}')
} > "$SYSTEM_DIR/container_processes.txt"

echo "[5/5] Collecting published ports..."
docker ps \
    --format '{{.Names}}|{{.Ports}}' \
    > "$SYSTEM_DIR/container_ports.txt"

echo "Collecting consolidated Docker and runtime state..."
python3 - "$OUTPUT_ROOT" "$([[ $# -eq 0 ]] && echo baseline || echo incident)" <<'PY'
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output_root = Path(sys.argv[1])
output_mode = sys.argv[2]
docker = os.environ.get("DOCKER_BIN") or shutil.which("docker") or "/usr/local/bin/docker"
expected_components = [
    "autoshield-nginx",
    "autoshield-juice-shop",
    "autoshield-api",
    "autoshield-postgres",
    "autoshield-redis",
    "autoshield-linux-admin",
]


def command(*args):
    return subprocess.check_output([docker, *args], text=True).strip()


container_ids = command("ps", "-q", "--no-trunc").splitlines()
containers = json.loads(command("inspect", *container_ids)) if container_ids else []
network_ids = command("network", "ls", "-q", "--no-trunc").splitlines()
networks = json.loads(command("network", "inspect", *network_ids)) if network_ids else []

running = []
for container in containers:
    state = container.get("State", {})
    network_membership = {}
    for name, details in container.get("NetworkSettings", {}).get("Networks", {}).items():
        network_membership[name] = {
            "network_id": details.get("NetworkID"),
            "ip_address": details.get("IPAddress"),
            "global_ipv6_address": details.get("GlobalIPv6Address"),
            "gateway": details.get("Gateway"),
            "mac_address": details.get("MacAddress"),
        }

    published_ports = []
    for container_port, bindings in container.get("NetworkSettings", {}).get("Ports", {}).items():
        for binding in bindings or []:
            published_ports.append({
                "container_port": container_port,
                "host_ip": binding.get("HostIp"),
                "host_port": binding.get("HostPort"),
            })

    running.append({
        "container_id": container.get("Id"),
        "name": container.get("Name", "").lstrip("/"),
        "image": container.get("Config", {}).get("Image"),
        "state": state.get("Status"),
        "running": state.get("Running", False),
        "started_at": state.get("StartedAt"),
        "health": state.get("Health", {}).get("Status"),
        "network_membership": network_membership,
        "published_ports": published_ports,
        "exposed_ports": sorted(container.get("Config", {}).get("ExposedPorts", {}).keys()),
    })

present = {container["name"] for container in running}
state = {
    "schema_version": 1,
    "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {
        "hostname": platform.node(),
        "operating_system": platform.platform(),
        "python_version": platform.python_version(),
        "docker_server_version": command("version", "--format", "{{.Server.Version}}"),
        "docker_compose_version": command("compose", "version", "--short"),
    },
    "expected_components": expected_components,
    "missing_expected_components": sorted(set(expected_components) - present),
    "running_containers": running,
    "docker_networks": [
        {
            "network_id": network.get("Id"),
            "name": network.get("Name"),
            "driver": network.get("Driver"),
            "scope": network.get("Scope"),
            "internal": network.get("Internal"),
            "attachable": network.get("Attachable"),
            "ipam": network.get("IPAM", {}).get("Config", []),
            "containers": {
                entry.get("Name"): {
                    "endpoint_id": entry.get("EndpointID"),
                    "ipv4_address": entry.get("IPv4Address"),
                    "ipv6_address": entry.get("IPv6Address"),
                }
                for entry in network.get("Containers", {}).values()
            },
        }
        for network in networks
    ],
}

output = json.dumps(state, indent=2) + "\n"
(output_root / ("system_state.json" if output_mode == "baseline" else "system/system_state.json")).write_text(
    output, encoding="utf-8"
)
print(f"Wrote system_state.json for {len(running)} running container(s).")
PY

echo "System state collection complete."