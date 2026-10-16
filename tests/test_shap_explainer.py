import numpy as np
import pytest

pytest.importorskip("shap")

from risk.shap_explainer import explain_predictions


def test_explainer_returns_existing_autoshield_schema():
    def demo_model(x):
        # Synthetic score for integration testing only.
        return 0.7 * x[:, 0] - 0.2 * x[:, 1]

    background = np.array([
        [0.0, 0.0],
        [0.2, 0.4],
        [0.4, 0.2],
        [0.6, 0.8],
    ])
    samples = np.array([[1.0, 0.0]])
    names = ["attack_blocked", "remaining_vulnerabilities"]

    results = explain_predictions(
        model_fn=demo_model,
        background_data=background,
        samples=samples,
        feature_names=names,
        decision="DEMO_ONLY",
    )

    assert len(results) == 1
    result = results[0]
    assert result.decision == "DEMO_ONLY"
    assert {item.feature for item in result.features} == set(names)
    assert all(np.isfinite(item.contribution) for item in result.features)
    assert result.explanation.startswith("Model output:")
    assert result.features[0].contribution > 0


def test_shap_baseline_plus_contributions_reconstructs_prediction():
    import re
    import math

    def demo_model(x):
        return 0.7 * x[:, 0] - 0.2 * x[:, 1]

    background = np.array([
        [0.0, 0.0],
        [0.2, 0.4],
        [0.4, 0.2],
        [0.6, 0.8],
    ])
    sample = np.array([[1.0, 0.0]])
    result = explain_predictions(
        model_fn=demo_model,
        background_data=background,
        samples=sample,
        feature_names=["feature_a", "feature_b"],
        decision="DEMO_ONLY",
    )[0]

    prediction_match = re.search(
        r"Model output: ([^;]+);", result.explanation
    )
    baseline_match = re.search(
        r"SHAP baseline: ([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
        result.explanation,
    )
    assert prediction_match and baseline_match

    prediction = float(prediction_match.group(1))
    baseline = float(baseline_match.group(1))
    reconstructed = baseline + sum(
        item.contribution for item in result.features
    )
    assert math.isclose(
        reconstructed, prediction, rel_tol=1e-6, abs_tol=1e-6
    )
