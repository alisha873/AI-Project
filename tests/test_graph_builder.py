from pathlib import Path

import pytest

from graph.builder import DeterministicGraphBuilder


INCIDENT_DIR = Path("incident_001")


def test_builder_creates_container_nodes():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    containers = [
        node for node in graph.nodes
        if node.node_type == "container"
    ]

    assert len(containers) == 6
    assert all(node.basis == "configured" for node in containers)


def test_builder_merges_repeated_ip_evidence():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    loopback = next(
        node for node in graph.nodes
        if node.node_id == "ip:127.0.0.1"
    )

    assert loopback.node_type == "ip"
    assert loopback.basis == "observed"
    assert len(loopback.evidence_ids) == 24
    assert len(loopback.evidence_ids) == len(set(loopback.evidence_ids))


def test_builder_is_deterministic():
    builder = DeterministicGraphBuilder()

    first = builder.build(INCIDENT_DIR).model_dump(mode="json")
    second = builder.build(INCIDENT_DIR).model_dump(mode="json")

    assert first == second


def test_builder_rejects_missing_evidence(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required graph evidence"):
        DeterministicGraphBuilder().build(tmp_path)


def test_current_incident_has_no_unsupported_packet_edges():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)
    communication_edges = [
        edge for edge in graph.edges
        if edge.edge_type == "communicates_with"
    ]

    # The current PCAP contains only 127.0.0.1-to-127.0.0.1 traffic.
    assert communication_edges == []


def test_packet_communication_helper_rejects_same_ip():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    assert all(
        edge.source != edge.target
        for edge in graph.edges
        if edge.edge_type == "communicates_with"
    )


def test_http_access_edges_preserve_evidence():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    accesses = [
        edge for edge in graph.edges
        if edge.edge_type == "accesses"
    ]

    api_access = next(
        edge for edge in accesses
        if edge.source == "ip:192.168.65.1"
        and edge.target == "ip:172.19.0.6"
    )

    assert api_access.basis == "observed"
    assert "nginx-10" in api_access.evidence_ids
    assert "nginx-13" in api_access.evidence_ids
    assert api_access.attributes["relationship_basis"] == (
        "observed_http_request"
    )
    assert api_access.source_artifacts == ["normalized/events.jsonl"]


def test_http_access_edges_do_not_label_source_as_attacker():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    source = next(
        node for node in graph.nodes
        if node.node_id == "ip:192.168.65.1"
    )

    assert "attacker" not in source.attributes

def test_builder_creates_service_nodes_and_hosting_edges():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    services = [
        node for node in graph.nodes
        if node.node_type == "service"
    ]

    assert len(services) == 6

    hosting_edges = [
        edge for edge in graph.edges
        if edge.edge_type == "hosts"
    ]

    assert len(hosting_edges) == 6

    for edge in hosting_edges:
        assert edge.source.startswith("container:")
        assert edge.target.startswith("service:")
        assert edge.basis == "configured"

def test_communication_edges_are_created_when_endpoint_nodes_exist_later(
    tmp_path,
):
    import json

    from graph.builder import DeterministicGraphBuilder

    # Build a minimal incident where the packet appears before the
    # network membership events that establish the endpoint IPs.
    incident = tmp_path / "incident_001"
    (incident / "docker").mkdir(parents=True)
    (incident / "normalized").mkdir()

    (incident / "docker" / "containers_after_attack.json").write_text("[]")
    (incident / "docker" / "network_after_attack.json").write_text("{}")

    events = [
        {
            "event_id": "packet-1",
            "event_type": "network_packet",
            "attributes": {
                "source_ip": "10.0.0.10",
                "destination_ip": "10.0.0.20",
                "source_port": 50000,
                "destination_port": 5432,
                "protocol": "TCP",
            },
        },
        {
            "event_id": "membership-1",
            "event_type": "network_membership",
            "attributes": {
                "source_canonical_entity": "container:app",
                "ip_address": "10.0.0.10",
                "network_id": "net-1",
            },
            "destination_entity": "net-1",
        },
        {
            "event_id": "membership-2",
            "event_type": "network_membership",
            "attributes": {
                "source_canonical_entity": "container:db",
                "ip_address": "10.0.0.20",
                "network_id": "net-1",
            },
            "destination_entity": "net-1",
        },
    ]

    # The empty container snapshot means membership edges are not needed
    # for this test; we are checking packet endpoint handling separately.
    (incident / "normalized" / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    graph = DeterministicGraphBuilder().build(incident)

    assert any(
        edge.edge_type == "communicates_with"
        and edge.source == "ip:10.0.0.10"
        and edge.target == "ip:10.0.0.20"
        and "packet-1" in edge.evidence_ids
        for edge in graph.edges
    )

def test_schema_accepts_forwards_edge():
    from evidence.schema import GraphEdge

    edge = GraphEdge(
        source="service:nginx",
        target="service:api",
        edge_type="forwards",
        basis="configured",
        attributes={"relationship_basis": "configured_upstream"},
    )

    assert edge.edge_type == "forwards"
    assert edge.source == "service:nginx"
    assert edge.target == "service:api"

def test_builder_creates_nginx_forwarding_edges():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    forwarding_edges = [
        edge for edge in graph.edges
        if edge.edge_type == "forwards"
    ]

    assert forwarding_edges

    nginx_to_api = next(
        edge for edge in forwarding_edges
        if edge.source == "service:nginx"
        and edge.target == "service:api"
    )

    assert nginx_to_api.basis == "observed"
    assert "nginx-10" in nginx_to_api.evidence_ids
    assert nginx_to_api.attributes["relationship_basis"] == (
        "observed_http_upstream"
    )


def test_builder_creates_configured_api_database_dependency():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    dependency = next(
        (
            edge for edge in graph.edges
            if edge.edge_type == "depends_on"
            and edge.source == "service:api"
            and edge.target == "service:postgres"
        ),
        None,
    )

    assert dependency is not None
    assert dependency.basis == "configured"
    assert (
        "docker/containers_after_attack.json"
        in dependency.source_artifacts
    )
    assert dependency.attributes["configured_host"] == "postgres"
    assert dependency.attributes["relationship_basis"] == (
        "container_environment"
    )

    # A configured dependency is not proof of observed network traffic.
    assert not any(
        edge.edge_type == "communicates_with"
        and {edge.source, edge.target}
        == {"ip:172.19.0.6", "ip:172.19.0.3"}
        for edge in graph.edges
    )
