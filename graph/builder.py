from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evidence.schema import AttackGraph, GraphEdge, GraphNode


class DeterministicGraphBuilder:
    """Build a deterministic graph from incident evidence."""

    def build(self, incident_dir: Path) -> AttackGraph:
        incident_dir = Path(incident_dir)


        legacy_containers_path = (
            incident_dir / "docker" / "containers_after_attack.json"
        )
        legacy_network_path = (
            incident_dir / "docker" / "network_after_attack.json"
        )
        current_containers_dir = incident_dir / "configs" / "containers"
        current_network_path = incident_dir / "network" / "autoshield_net.json"
        events_path = incident_dir / "normalized" / "events.jsonl"

        if not events_path.is_file():
            raise FileNotFoundError(
                f"Required graph evidence not found: {events_path}"
            )

        if legacy_containers_path.is_file() and legacy_network_path.is_file():
            containers = json.loads(
                legacy_containers_path.read_text(encoding="utf-8")
            )
            json.loads(legacy_network_path.read_text(encoding="utf-8"))
        elif current_containers_dir.is_dir() and current_network_path.is_file():
            containers = []
            for path in sorted(current_containers_dir.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    containers.extend(data)
                elif isinstance(data, dict):
                    containers.append(data)

            if not containers:
                raise ValueError(
                    f"No container snapshots found in {current_containers_dir}"
                )

            json.loads(current_network_path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(
                "Required Docker evidence not found in either supported layout: "
                f"{legacy_containers_path} / {legacy_network_path}, or "
                f"{current_containers_dir} / {current_network_path}"
            )


        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
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

        def add_packet_communication(event: dict[str, Any]) -> None:
            """Add a communication edge only for observed, non-loopback IP pairs."""
            attrs = event.get("attributes") or {}
            source_ip = attrs.get("source_ip")
            destination_ip = attrs.get("destination_ip")
            event_id = event.get("event_id", "")

            if not source_ip or not destination_ip:
                return

            # This baseline only models communication between distinct IPs.
            # In particular, don't treat host loopback captures as container traffic.
            if source_ip == destination_ip:
                return

            source_id = f"ip:{source_ip}"
            target_id = f"ip:{destination_ip}"
            if source_id not in nodes or target_id not in nodes:
                return

            add_edge(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    edge_type="communicates_with",
                    basis="observed",
                    evidence_ids=[event_id] if event_id else [],
                    source_artifacts=["normalized/events.jsonl"],
                    attributes={
                        "protocol": attrs.get("protocol"),
                        "source_port": attrs.get("source_port"),
                        "destination_port": attrs.get("destination_port"),
                        "relationship_basis": "observed_packet",
                    },
                )
            )


        # Resolve configured service dependencies from captured container
        # environment variables. These are configuration relationships,
        # not proof of observed network communication.
        configured_service_hosts: list[tuple[str, str, str]] = []

        for container in containers:
            raw_name = container.get("Name", "").lstrip("/")
            config = container.get("Config", {})
            service_name = raw_name.removeprefix("autoshield-")

            for item in config.get("Env", []) or []:
                key, separator, value = item.partition("=")
                if (
                    separator
                    and key.endswith("_HOST")
                    and value.strip()
                ):
                    configured_service_hosts.append(
                        (service_name, value.strip(), key)
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

            # Explicit service node hosted by this container.
            config = container.get("Config", {})
            service_name = raw_name.removeprefix("autoshield-")
            service_id = f"service:{service_name}"
            add_node(
                GraphNode(
                    node_id=service_id,
                    node_type="service",
                    name=service_name,
                    basis="configured",
                    source_artifacts=[
                        "docker/containers_after_attack.json"
                    ],
                    attributes={
                        "container": raw_name,
                        "image": config.get("Image"),
                        "exposed_ports": sorted(
                            (config.get("ExposedPorts") or {}).keys()
                        ),
                    },
                )
            )
            add_edge(
                GraphEdge(
                    source=container_id,
                    target=service_id,
                    edge_type="hosts",
                    basis="configured",
                    source_artifacts=[
                        "docker/containers_after_attack.json"
                    ],
                    attributes={
                        "relationship_basis": "container_service_configuration"
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


        # 2b. Configured service dependencies (not observed traffic).
        services_by_name = {
            node.name: node
            for node in nodes.values()
            if node.node_type == "service"
        }

        for source_service, target_host, configuration_key in sorted(
            configured_service_hosts
        ):
            source_node = services_by_name.get(source_service)
            target_node = services_by_name.get(target_host)

            if source_node is None or target_node is None:
                continue

            add_edge(
                GraphEdge(
                    source=source_node.node_id,
                    target=target_node.node_id,
                    edge_type="depends_on",
                    basis="configured",
                    source_artifacts=[
                        "docker/containers_after_attack.json"
                    ],
                    attributes={
                        "configuration_key": configuration_key,
                        "configured_host": target_host,
                        "relationship_basis": "container_environment",
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
                source_ip = event.get("source_entity")
                destination_ip = event.get("destination_entity")
                upstream = attrs.get("upstream", "")
                _, separator, upstream_port = upstream.rpartition(":")

                port_number = (
                    int(upstream_port)
                    if separator and upstream_port.isdigit()
                    else None
                )

                # Model the observed Nginx upstream relationship.
                if upstream and separator and upstream_port.isdigit():
                    upstream_ip = upstream.rsplit(":", 1)[0]
                    upstream_port_number = int(upstream_port)
                    target_service = next(
                        (
                            node
                            for node in nodes.values()
                            if node.node_type == "service"
                            and upstream_ip
                            in [
                                ip_node.name
                                for ip_node in nodes.values()
                                if ip_node.node_type == "ip"
                                and any(
                                    edge.source
                                    == f"container:{node.attributes.get('container')}"
                                    and edge.target == ip_node.node_id
                                    and edge.edge_type == "connects_to"
                                    for edge in edges.values()
                                )
                            ]
                        ),
                        None,
                    )
                    if target_service:
                        add_edge(
                            GraphEdge(
                                source="service:nginx",
                                target=target_service.node_id,
                                edge_type="forwards",
                                basis="observed",
                                evidence_ids=[event_id] if event_id else [],
                                source_artifacts=[
                                    "normalized/events.jsonl"
                                ],
                                attributes={
                                    "upstream": upstream,
                                    "upstream_port": upstream_port_number,
                                    "query": attrs.get("query"),
                                    "method": attrs.get("method"),
                                    "path": attrs.get("path"),
                                    "relationship_basis": "observed_http_upstream",
                                },
                            )
                        )

                endpoints = [
                    (source_ip, None),
                    (destination_ip, port_number),
                ]

                if (
                    source_ip
                    and destination_ip
                    and source_ip != destination_ip
                ):
                    source_id = f"ip:{source_ip}"
                    target_id = f"ip:{destination_ip}"

                    for ip_address in (source_ip, destination_ip):
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

                    if source_id in nodes and target_id in nodes:
                        add_edge(
                            GraphEdge(
                                source=source_id,
                                target=target_id,
                                edge_type="accesses",
                                basis="observed",
                                evidence_ids=(
                                    [event_id] if event_id else []
                                ),
                                source_artifacts=[
                                    "normalized/events.jsonl"
                                ],
                                attributes={
                                    "method": attrs.get("method"),
                                    "path": attrs.get("path"),
                                    "query": attrs.get("query"),
                                    "status": attrs.get("status"),
                                    "upstream": upstream,
                                    "relationship_basis": "observed_http_request",
                                },
                            )
                        )
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

        # 4. Build communication edges after all endpoint IP nodes exist.
        for event in events:
            if event.get("event_type") == "network_packet":
                add_packet_communication(event)

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
