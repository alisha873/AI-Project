from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidence.schema import AttackGraph, GraphEdge, GraphNode


class DeterministicGraphBuilder:
    """Build a deterministic graph from incident evidence."""

    def build(self, incident_dir: Path) -> AttackGraph:
        incident_dir = Path(incident_dir)

        containers_path = (
            incident_dir / "docker" / "containers_after_attack.json"
        )
        network_path = (
            incident_dir / "docker" / "network_after_attack.json"
        )
        events_path = incident_dir / "normalized" / "events.jsonl"

        for path in (containers_path, network_path, events_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required graph evidence not found: {path}"
                )

        containers = json.loads(containers_path.read_text())
        # Validate and retain the network snapshot as a required input.
        json.loads(network_path.read_text())

        events = [
            json.loads(line)
            for line in events_path.read_text().splitlines()
            if line.strip()
        ]

        nodes: dict[str, GraphNode] = {}
        edges: dict[tuple[str, str, str], GraphEdge] = {}

        def add_node(node: GraphNode) -> None:
            existing = nodes.get(node.node_id)

            if existing is None:
                nodes[node.node_id] = node
                return

            if (
                existing.node_type != node.node_type
                or existing.name != node.name
            ):
                raise ValueError(
                    f"Conflicting graph node identity: {node.node_id}"
                )

            merged_attributes = dict(existing.attributes)
            for key, value in node.attributes.items():
                if (
                    key not in merged_attributes
                    or merged_attributes[key] is None
                ):
                    merged_attributes[key] = value

            nodes[node.node_id] = existing.model_copy(
                update={
                    "attributes": merged_attributes,
                    "evidence_ids": sorted(
                        set(existing.evidence_ids + node.evidence_ids)
                    ),
                    "source_artifacts": sorted(
                        set(
                            existing.source_artifacts
                            + node.source_artifacts
                        )
                    ),
                    "basis": (
                        "observed"
                        if "observed"
                        in {existing.basis, node.basis}
                        else "configured"
                    ),
                }
            )

        def add_edge(edge: GraphEdge) -> None:
            key = (edge.source, edge.target, edge.edge_type)
            existing = edges.get(key)

            if existing is None:
                edges[key] = edge
                return

            edges[key] = existing.model_copy(
                update={
                    "evidence_ids": sorted(
                        set(existing.evidence_ids + edge.evidence_ids)
                    ),
                    "source_artifacts": sorted(
                        set(
                            existing.source_artifacts
                            + edge.source_artifacts
                        )
                    ),
                }
            )

        # 1. Container nodes and configured startup processes.
        for container in containers:
            raw_name = container.get("Name", "").lstrip("/")
            docker_id = container.get("Id")

            if not raw_name or not docker_id:
                continue

            container_id = f"container:{raw_name}"

            add_node(
                GraphNode(
                    node_id=container_id,
                    node_type="container",
                    name=raw_name,
                    basis="configured",
                    source_artifacts=[
                        "docker/containers_after_attack.json"
                    ],
                    attributes={
                        "docker_id": docker_id,
                        "image": container.get("Config", {}).get("Image"),
                        "status": container.get("State", {}).get("Status"),
                    },
                )
            )

            executable = container.get("Path")
            args = container.get("Args") or []

            if not executable:
                continue

            process_name = Path(executable).name
            process_id = f"process:{raw_name}:{process_name}"

            add_node(
                GraphNode(
                    node_id=process_id,
                    node_type="process",
                    name=process_name,
                    basis="configured",
                    source_artifacts=[
                        "docker/containers_after_attack.json"
                    ],
                    attributes={
                        "executable": executable,
                        "args": args,
                        "container": raw_name,
                    },
                )
            )

            add_edge(
                GraphEdge(
                    source=container_id,
                    target=process_id,
                    edge_type="executes",
                    basis="configured",
                    source_artifacts=[
                        "docker/containers_after_attack.json"
                    ],
                    attributes={
                        "relationship_basis": "configured_startup"
                    },
                )
            )

        # 2. Configured container-to-IP edges from network membership.
        for event in events:
            if event.get("event_type") != "network_membership":
                continue

            attrs = event.get("attributes") or {}
            container_id = attrs.get("source_canonical_entity", "")
            ip_address = attrs.get("ip_address")

            if (
                not container_id.startswith("container:")
                or container_id not in nodes
                or not ip_address
            ):
                continue

            ip_id = f"ip:{ip_address}"

            add_node(
                GraphNode(
                    node_id=ip_id,
                    node_type="ip",
                    name=ip_address,
                    basis="configured",
                    source_artifacts=[
                        "docker/network_after_attack.json"
                    ],
                    attributes={"address": ip_address},
                )
            )

            evidence_ids = {
                item
                for item in (
                    [event.get("event_id", "")]
                    + attrs.get("source_event_ids", [])
                )
                if item
            }

            add_edge(
                GraphEdge(
                    source=container_id,
                    target=ip_id,
                    edge_type="connects_to",
                    basis="configured",
                    evidence_ids=sorted(evidence_ids),
                    source_artifacts=[
                        "docker/network_after_attack.json",
                        "normalized/events.jsonl",
                    ],
                    attributes={
                        "network_id": attrs.get("network_id"),
                        "network_name": event.get("destination_entity"),
                        "relationship_basis": "docker_network_membership",
                    },
                )
            )

        # 3. IP and port nodes from observed packet and HTTP evidence.
        for event in events:
            attrs: dict[str, Any] = event.get("attributes") or {}
            event_type = event.get("event_type")
            event_id = event.get("event_id", "")

            if event_type == "network_packet":
                endpoints = [
                    (attrs.get("source_ip"), attrs.get("source_port")),
                    (
                        attrs.get("destination_ip"),
                        attrs.get("destination_port"),
                    ),
                ]

            elif event_type == "http_request":
                upstream = attrs.get("upstream", "")
                _, separator, upstream_port = upstream.rpartition(":")

                port_number = (
                    int(upstream_port)
                    if separator and upstream_port.isdigit()
                    else None
                )

                endpoints = [
                    (event.get("source_entity"), None),
                    (event.get("destination_entity"), port_number),
                ]
            else:
                continue

            for ip_address, port_number in endpoints:
                if not ip_address:
                    continue

                add_node(
                    GraphNode(
                        node_id=f"ip:{ip_address}",
                        node_type="ip",
                        name=ip_address,
                        basis="observed",
                        evidence_ids=[event_id] if event_id else [],
                        source_artifacts=["normalized/events.jsonl"],
                        attributes={"address": ip_address},
                    )
                )

                if port_number is None:
                    continue

                add_node(
                    GraphNode(
                        node_id=f"port:{ip_address}:{port_number}",
                        node_type="port",
                        name=str(port_number),
                        basis="observed",
                        evidence_ids=[event_id] if event_id else [],
                        source_artifacts=["normalized/events.jsonl"],
                        attributes={
                            "ip": ip_address,
                            "port": port_number,
                        },
                    )
                )

        return AttackGraph(
            nodes=sorted(
                nodes.values(),
                key=lambda node: node.node_id,
            ),
            edges=sorted(
                edges.values(),
                key=lambda edge: (
                    edge.source,
                    edge.target,
                    edge.edge_type,
                ),
            ),
            attack_path=[],
        )
