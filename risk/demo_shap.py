"""Run a clearly synthetic SHAP integration demo."""

import json

import numpy as np

from risk.features import RiskFeatures
from risk.shap_explainer import explain_predictions


def synthetic_demo_score(x: np.ndarray) -> np.ndarray:
    """Demo score only; not trained and not a security-risk estimate."""
    return (
        -0.40 * x[:, 0]
        -0.50 * x[:, 1]
        + 0.10 * x[:, 2]
        + 0.20 * x[:, 3]
        - 0.30 * x[:, 4]
        + 0.05 * x[:, 5]
        - 0.05 * x[:, 6]
    )


def main() -> None:
    # Synthetic examples only, used to exercise the integration.
    background_records = [
        RiskFeatures(
            attack_blocked=False,
            attack_path_closed=False,
            remaining_exposed_ports=4,
            remaining_vulnerabilities=6,
            patch_applied=False,
            configuration_change=False,
            validation_evidence_count=0,
        ),
        RiskFeatures(
            attack_blocked=True,
            attack_path_closed=False,
            remaining_exposed_ports=2,
            remaining_vulnerabilities=3,
            patch_applied=True,
            configuration_change=True,
            validation_evidence_count=2,
        ),
        RiskFeatures(
            attack_blocked=True,
            attack_path_closed=True,
            remaining_exposed_ports=0,
            remaining_vulnerabilities=0,
            patch_applied=True,
            configuration_change=False,
            validation_evidence_count=5,
        ),
    ]
    sample = RiskFeatures(
        attack_blocked=True,
        attack_path_closed=True,
        remaining_exposed_ports=1,
        remaining_vulnerabilities=1,
        patch_applied=True,
        configuration_change=True,
        validation_evidence_count=3,
    )

    background = np.vstack([item.to_model_array() for item in background_records])
    sample_array = sample.to_model_array().reshape(1, -1)

    results = explain_predictions(
        model_fn=synthetic_demo_score,
        background_data=background,
        samples=sample_array,
        feature_names=RiskFeatures.feature_names(),
        decision="DEMO_ONLY",
    )

    print("DEMO ONLY — NOT A REAL AUTOSHIELD SECURITY ASSESSMENT")
    print(json.dumps(results[0].model_dump(), indent=2))


if __name__ == "__main__":
    main()
