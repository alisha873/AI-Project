import pytest

from explainability.gnn_edge_explainer import explain_candidate_edge


def test_rejects_unknown_edge_type():
    with pytest.raises(ValueError, match="Unsupported edge type"):
        explain_candidate_edge(
            graph=None,
            gcn=None,
            scorer=None,
            source_id="source",
            target_id="target",
            edge_type="not_a_real_edge_type",
        )


def test_rejects_zero_background_samples():
    with pytest.raises(ValueError, match="background_samples"):
        explain_candidate_edge(
            graph=None,
            gcn=None,
            scorer=None,
            source_id="source",
            target_id="target",
            edge_type="forwards",
            background_samples=0,
        )
