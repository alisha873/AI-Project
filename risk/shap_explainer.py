"""Reusable SHAP explanations for AutoShield scalar-output models."""

from collections.abc import Callable, Sequence

import numpy as np

from evidence.schema import ExplainabilityResult, FeatureImportance


ModelFunction = Callable[[np.ndarray], np.ndarray]


def explain_predictions(
    model_fn: ModelFunction,
    background_data: np.ndarray,
    samples: np.ndarray,
    feature_names: Sequence[str],
    decision: str,
) -> list[ExplainabilityResult]:
    """Explain model outputs for each sample using SHAP.

    Positive contributions increase the model output relative to the
    SHAP baseline; negative contributions decrease it. Interpret them
    as risk contributions only when the model output represents risk.

    Args:
        model_fn: Callable accepting an (n_samples, n_features) array
            and returning one scalar output per sample.
        background_data: Representative baseline data, shape (n, features).
        samples: Samples to explain, shape (n, features).
        feature_names: Names matching the columns in both arrays.
        decision: Decision label to include in the result schema.

    Returns:
        One ExplainabilityResult per sample.

    Raises:
        ValueError: If input shapes, names, or model outputs are invalid.
        ImportError: If the optional SHAP dependency is not installed.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is not installed. Activate your project venv and run: "
            "python -m pip install shap"
        ) from exc

    background = np.asarray(background_data, dtype=float)
    inputs = np.asarray(samples, dtype=float)
    names = list(feature_names)

    if background.ndim != 2 or inputs.ndim != 2:
        raise ValueError("background_data and samples must be 2D arrays.")
    if background.shape[0] == 0 or inputs.shape[0] == 0:
        raise ValueError("background_data and samples cannot be empty.")
    if background.shape[1] != inputs.shape[1]:
        raise ValueError("Background and sample feature counts must match.")
    if len(names) != inputs.shape[1]:
        raise ValueError("feature_names length must match the feature count.")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique.")
    if not np.isfinite(background).all() or not np.isfinite(inputs).all():
        raise ValueError("Input arrays must contain only finite values.")

    def checked_model_fn(data: np.ndarray) -> np.ndarray:
        predictions = np.asarray(model_fn(np.asarray(data, dtype=float)))
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        if predictions.ndim != 1 or len(predictions) != len(data):
            raise ValueError(
                "model_fn must return one scalar output per input row."
            )
        if not np.isfinite(predictions).all():
            raise ValueError("model_fn returned non-finite predictions.")
        return predictions.astype(float)

    # SHAP uses the supplied background rows as the reference distribution.
    explainer = shap.Explainer(checked_model_fn, background)
    explanation = explainer(inputs)

    values = np.asarray(explanation.values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != inputs.shape:
        raise ValueError(
            "Expected one SHAP value per sample and feature; "
            f"received shape {values.shape} for input shape {inputs.shape}."
        )

    baseline_values = np.asarray(explanation.base_values)
    results = []

    for row_index, row_values in enumerate(values):
        base = baseline_values[row_index] if baseline_values.ndim else baseline_values
        base = float(np.asarray(base).reshape(-1)[0])

        features = []
        for name, contribution in zip(names, row_values, strict=True):
            contribution = float(contribution)
            direction = (
                "increases_model_output"
                if contribution > 0
                else "decreases_model_output"
                if contribution < 0
                else "no_change"
            )
            features.append(
                FeatureImportance(
                    feature=name,
                    contribution=contribution,
                    direction=direction,
                )
            )

        features.sort(key=lambda item: abs(item.contribution), reverse=True)

        prediction = float(checked_model_fn(inputs[row_index : row_index + 1])[0])
        summary = (
            f"Model output: {prediction:.12g}; "
            f"SHAP baseline: {base:.12g}. "
            "Contributions describe changes relative to the baseline; "
            "they are not causal effects."
        )

        results.append(
            ExplainabilityResult(
                decision=decision,
                features=features,
                explanation=summary,
            )
        )

    return results
