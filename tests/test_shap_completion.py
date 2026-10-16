import numpy as np
import pytest

from risk.shap_explainer import explain_predictions


FEATURES = [
    "attack_blocked",
    "attack_path_closed",
    "remaining_exposed_ports",
    "remaining_vulnerabilities",
    "patch_applied",
    "configuration_change",
    "validation_evidence_count",
]


def model(x):
    return (
        0.30 * x[:, 0]
        + 0.20 * x[:, 1]
        + 0.10 * x[:, 2]
        + 0.15 * x[:, 3]
        + 0.10 * x[:, 4]
        + 0.05 * x[:, 5]
        + 0.02 * x[:, 6]
    )


@pytest.fixture
def background():
    return np.array([
        [0, 0, 4, 5, 0, 0, 0],
        [1, 1, 0, 1, 1, 1, 3],
        [0, 1, 2, 3, 0, 1, 1],
        [1, 0, 1, 2, 1, 0, 2],
    ], dtype=float)


def test_returns_one_explanation_per_sample(background):
    samples = np.array([
        [1, 1, 0, 0, 1, 1, 4],
        [0, 0, 3, 4, 0, 0, 0],
    ], dtype=float)

    results = explain_predictions(
        model, background, samples, FEATURES, decision="TEST_ONLY"
    )

    assert len(results) == 2
    for result in results:
        assert result.decision == "TEST_ONLY"
        assert len(result.features) == len(FEATURES)
        assert result.explanation


def test_feature_names_and_contributions_are_finite(background):
    sample = np.array([[1, 0, 2, 1, 1, 0, 3]], dtype=float)

    result = explain_predictions(
        model, background, sample, FEATURES, decision="TEST_ONLY"
    )[0]

    names = {item.feature for item in result.features}
    assert names == set(FEATURES)

    for item in result.features:
        assert np.isfinite(item.contribution)
        assert item.direction in {
            "increases_model_output",
            "decreases_model_output",
            "no_change",
        }


@pytest.mark.parametrize("bad_output", [
    lambda x: np.full(len(x), np.nan),
    lambda x: np.full(len(x), np.inf),
    lambda x: np.zeros((len(x), 2)),
])
def test_rejects_invalid_model_outputs(background, bad_output):
    sample = np.array([[1, 0, 2, 1, 1, 0, 3]], dtype=float)

    with pytest.raises((ValueError, TypeError)):
        explain_predictions(
            bad_output, background, sample, FEATURES, decision="TEST_ONLY"
        )


def test_rejects_wrong_feature_count(background):
    sample = np.array([[1, 0, 2]], dtype=float)

    with pytest.raises((ValueError, TypeError, IndexError)):
        explain_predictions(
            model, background, sample, FEATURES, decision="TEST_ONLY"
        )


def test_rejects_nonfinite_input(background):
    sample = np.array([[1, 0, np.nan, 1, 1, 0, 3]], dtype=float)

    with pytest.raises((ValueError, TypeError)):
        explain_predictions(
            model, background, sample, FEATURES, decision="TEST_ONLY"
        )
