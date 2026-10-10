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
