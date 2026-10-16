import numpy as np
import pytest

from explainability.gnn_shap import explain_edge_features


FEATURES = ["http_status", "query_indicator", "observed_edge"]


def test_explains_candidate_edge_score_and_additivity():
    def scorer(x):
        return 0.4 * x[:, 0] + 0.3 * x[:, 1] - 0.2 * x[:, 2]

    background = np.array([
        [200, 0, 1],
        [200, 1, 1],
        [404, 0, 0],
        [500, 1, 0],
    ], dtype=float)
    samples = np.array([[200, 1, 1]], dtype=float)

    result = explain_edge_features(
        scorer, background, samples, FEATURES,
        edge_description="service:nginx -> service:api (forwards)",
    )
    row = result["results"][0]

    assert row["output_type"] == "raw_gcn_edge_score"
    assert {item["feature"] for item in row["contributions"]} == set(FEATURES)
    assert row["additivity_error"] < 1e-5
    assert "does not establish" in row["interpretation"]


def test_rejects_feature_count_mismatch():
    with pytest.raises(ValueError, match="match feature_names"):
        explain_edge_features(
            lambda x: x[:, 0],
            np.ones((2, 3)),
            np.ones((1, 3)),
            ["only_one"],
            edge_description="test edge",
        )


def test_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="finite"):
        explain_edge_features(
            lambda x: x[:, 0],
            np.array([[1.0, np.nan]]),
            np.array([[1.0, 0.0]]),
            ["a", "b"],
            edge_description="test edge",
        )


def test_rejects_vector_output_per_row():
    with pytest.raises(ValueError, match="one scalar"):
        explain_edge_features(
            lambda x: np.column_stack([x[:, 0], x[:, 1]]),
            np.ones((3, 2)),
            np.ones((1, 2)),
            ["a", "b"],
            edge_description="test edge",
        )
