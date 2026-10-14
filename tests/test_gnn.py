
from pathlib import Path

import torch

from graph.builder import DeterministicGraphBuilder
from graph.gnn import AttackGraphGCN, EDGE_TYPES, NODE_TYPES, encode_graph
from graph.http_features import HTTP_FEATURE_DIM


INCIDENT_DIR = Path("incident_001")


def test_encode_graph_preserves_nodes_and_edge_types():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)
    tensors = encode_graph(graph)

    assert len(tensors.node_ids) == len(graph.nodes)
    assert tensors.node_features.shape == (
        len(graph.nodes),
        len(NODE_TYPES) + 2 + HTTP_FEATURE_DIM,
    )
    assert tensors.edge_index.shape == (2, len(graph.edges))
    assert tensors.edge_type_ids.shape == (len(graph.edges),)
    assert all(
        0 <= int(edge_type_id) < len(EDGE_TYPES)
        for edge_type_id in tensors.edge_type_ids
    )
    assert tensors.edge_basis_ids.shape == (len(graph.edges),)
    assert set(tensors.edge_basis_ids.tolist()) <= {0, 1}


def test_gcn_returns_one_embedding_per_node():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)
    tensors = encode_graph(graph)

    model = AttackGraphGCN()
    embeddings = model(tensors.node_features, tensors.edge_index,tensors.edge_type_ids)

    assert embeddings.shape == (len(graph.nodes), 16)
    assert torch.isfinite(embeddings).all()


def test_gcn_handles_graph_without_edges():
    features = torch.zeros((3, len(NODE_TYPES) + 2 + HTTP_FEATURE_DIM))
    edge_index = torch.empty((2, 0), dtype=torch.long)

    model = AttackGraphGCN()
    embeddings = model(features, edge_index)

    assert embeddings.shape == (3, 16)
    assert torch.isfinite(embeddings).all()


def test_graph_encoding_is_deterministic():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)

    first = encode_graph(graph)
    second = encode_graph(graph)

    assert first.node_ids == second.node_ids
    assert torch.equal(first.node_features, second.node_features)
    assert torch.equal(first.edge_index, second.edge_index)
    assert torch.equal(first.edge_type_ids, second.edge_type_ids)
    assert torch.equal(first.edge_type_ids, second.edge_type_ids)


def test_gcn_uses_edge_type_ids():
    torch.manual_seed(7)
    model = AttackGraphGCN()
    model.eval()

    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    features = torch.nn.functional.pad(features, (0, HTTP_FEATURE_DIM))
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    first_relation = torch.tensor(
        [EDGE_TYPES.index("forwards")], dtype=torch.long
    )
    second_relation = torch.tensor(
        [EDGE_TYPES.index("depends_on")], dtype=torch.long
    )

    with torch.no_grad():
        first = model(features, edge_index, first_relation)
        second = model(features, edge_index, second_relation)

    assert not torch.allclose(first, second)

def test_edge_scorer_returns_one_score_per_candidate():
    from graph.gnn import EdgeScorer

    scorer = EdgeScorer(embedding_dim=16)

    source_embeddings = torch.randn(3, 16)
    target_embeddings = torch.randn(3, 16)
    edge_type_ids = torch.tensor(
        [
            EDGE_TYPES.index("forwards"),
            EDGE_TYPES.index("depends_on"),
            EDGE_TYPES.index("accesses"),
        ],
        dtype=torch.long,
    )

    scores = scorer(
        source_embeddings,
        target_embeddings,
        edge_type_ids,
    )

    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()

def test_gcn_uses_edge_provenance():
    torch.manual_seed(7)
    model = AttackGraphGCN()
    model.eval()

    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    features = torch.nn.functional.pad(features, (0, HTTP_FEATURE_DIM))
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    edge_type_ids = torch.tensor(
        [EDGE_TYPES.index("forwards")], dtype=torch.long
    )

    configured = torch.tensor([0], dtype=torch.long)
    observed = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        configured_output = model(
            features, edge_index, edge_type_ids, configured
        )
        observed_output = model(
            features, edge_index, edge_type_ids, observed
        )

    assert not torch.allclose(configured_output, observed_output)
