from evidence.schema import AttackGraph, GraphEdge, GraphNode
from evaluation.gnn_path_metrics import map_ground_truth_path
from graph.path_reasoning import (
    find_candidate_paths,
    score_candidate_path_with_gcn,
)
from graph.gnn import AttackGraphGCN
from evaluation.gnn_path_metrics import measure_path_coverage

def make_graph():
    nodes = [
        GraphNode(node_id="a", node_type="service", name="a"),
        GraphNode(node_id="b", node_type="service", name="b"),
        GraphNode(node_id="c", node_type="service", name="c"),
        GraphNode(node_id="d", node_type="service", name="d"),
    ]
    edges = [
        GraphEdge(
            source="a",
            target="b",
            edge_type="forwards",
            basis="observed",
            evidence_ids=["event-1"],
        ),
        GraphEdge(
            source="b",
            target="c",
            edge_type="depends_on",
            basis="configured",
        ),
        GraphEdge(
            source="a",
            target="d",
            edge_type="connects_to",
            basis="configured",
        ),
        GraphEdge(
            source="d",
            target="c",
            edge_type="forwards",
            basis="observed",
            evidence_ids=["event-2"],
        ),
    ]
    return AttackGraph(nodes=nodes, edges=edges)


def test_finds_paths_and_preserves_provenance():
    paths = find_candidate_paths(make_graph(), "a", "c")

    assert len(paths) == 2
    best = paths[0]
    assert best.nodes == ("a", "b", "c")
    assert best.hops[0].basis == "observed"
    assert best.hops[0].evidence_ids == ("event-1",)
    assert best.hops[1].basis == "configured"


def test_observed_only_excludes_configured_edges():
    paths = find_candidate_paths(
        make_graph(), "a", "c", observed_only=True
    )

    assert paths == []


def test_missing_nodes_return_no_paths():
    assert find_candidate_paths(make_graph(), "missing", "c") == []


def test_hop_limit_is_respected():
    paths = find_candidate_paths(
        make_graph(), "a", "c", max_hops=1
    )
    assert paths == []


def test_invalid_limits_raise():
    import pytest

    with pytest.raises(ValueError):
        find_candidate_paths(make_graph(), "a", "c", max_hops=0)


def test_gcn_path_score_is_finite_and_repeatable_with_same_model():
    import torch

    graph = make_graph()
    path = find_candidate_paths(graph, "a", "c")[0]

    torch.manual_seed(7)
    model = AttackGraphGCN()
    first = score_candidate_path_with_gcn(graph, path, model=model)
    second = score_candidate_path_with_gcn(graph, path, model=model)

    assert -1.0 <= first <= 1.0
    assert first == second



def test_path_coverage_counts_nodes_and_direct_transitions():
    graph = AttackGraph(
        nodes=[
            GraphNode(node_id="a", node_type="service", name="a"),
            GraphNode(node_id="b", node_type="service", name="b"),
            GraphNode(node_id="c", node_type="service", name="c"),
        ],
        edges=[
            GraphEdge(
                source="a",
                target="b",
                edge_type="forwards",
                basis="observed",
            )
        ],
    )
    mapping = map_ground_truth_path(graph, ["a", "b", "c"])
    coverage = measure_path_coverage(
        graph,
        ["a", "b", "c"],
        mapping,
    )

    assert coverage.node_coverage == 1.0
    assert coverage.transition_coverage == 0.5
    assert coverage.represented_transitions == 1
    assert coverage.expected_transitions == 2
