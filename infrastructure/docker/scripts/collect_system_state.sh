#!/bin/bash

set -e

BASELINE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)/baseline"

mkdir -p "$BASELINE_DIR/system"

echo "[1/5] Collecting running containers..."

docker ps \
    --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.Ports}}' \
    > "$BASELINE_DIR/system/containers.txt"


echo "[2/5] Collecting container IP addresses..."

{
    echo "container|ip_address"

    for container in $(docker ps --format '{{.Names}}'); do
        ip=$(docker inspect -f \
            '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' \
            "$container")

        echo "$container|$ip"
    done

} > "$BASELINE_DIR/system/container_ips.txt"


echo "[3/5] Collecting Docker networks..."

docker network ls \
    > "$BASELINE_DIR/system/docker_networks.txt"


echo "[4/5] Collecting container processes..."

{
    echo "===== autoshield-nginx ====="
    docker top autoshield-nginx

    echo ""
    echo "===== autoshield-juice-shop ====="
    docker top autoshield-juice-shop

} > "$BASELINE_DIR/system/container_processes.txt"


echo "[5/5] Collecting published ports..."

docker ps \
    --format '{{.Names}}|{{.Ports}}' \
    > "$BASELINE_DIR/system/container_ports.txt"


echo ""
echo "System state collection complete."