
import pytest
from pydantic import ValidationError

from evidence.schema import AttackGraph, GraphEdge, GraphNode


@pytest.mark.parametrize(
    "node_type",
    ["container", "service", "ip", "port", "process", "user"],
)
def test_graph_node_accepts_valid_types(node_type):
    node = GraphNode(
        node_id="node-1",
        node_type=node_type,
        name="test-node",
    )
    assert node.node_type == node_type


def test_graph_node_rejects_invalid_type():
    with pytest.raises(ValidationError):
        GraphNode(
            node_id="node-1",
            node_type="database",
            name="test-node",
        )


@pytest.mark.parametrize(
    "edge_type",
    [
        "communicates_with",
        "connects_to",
        "accesses",
        "executes",
        "authenticates_to",
    ],
)
def test_graph_edge_accepts_valid_types(edge_type):
    edge = GraphEdge(
        source="node-1",
        target="node-2",
        edge_type=edge_type,
    )
    assert edge.edge_type == edge_type


def test_graph_edge_rejects_invalid_type():
    with pytest.raises(ValidationError):
        GraphEdge(
            source="node-1",
            target="node-2",
            edge_type="owns",
        )


def test_graph_provenance_defaults():
    node = GraphNode(
        node_id="node-1",
        node_type="container",
        name="api",
    )
    edge = GraphEdge(
        source="node-1",
        target="node-2",
        edge_type="connects_to",
    )

    assert node.evidence_ids == []
    assert node.source_artifacts == []
    assert node.basis == "observed"
    assert edge.evidence_ids == []
    assert edge.source_artifacts == []
    assert edge.basis == "observed"


def test_graph_preserves_attributes_and_attack_path():
    node = GraphNode(
        node_id="node-1",
        node_type="container",
        name="api",
        attributes={"image": "api:latest"},
    )
    graph = AttackGraph(
        nodes=[node],
        attack_path=["node-1"],
    )

    assert graph.nodes[0].attributes["image"] == "api:latest"
    assert graph.attack_path == ["node-1"]


def test_graph_supports_configured_provenance():
    node = GraphNode(
        node_id="process:api",
        node_type="process",
        name="uvicorn",
        basis="configured",
        evidence_ids=["docker-container-1"],
        source_artifacts=["docker/containers_after_attack.json"],
    )

    assert node.basis == "configured"
    assert node.evidence_ids == ["docker-container-1"]
    assert node.source_artifacts == [
        "docker/containers_after_attack.json"
    ]
