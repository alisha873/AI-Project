"""SHAP explanations for AutoShield's GCN edge scorer.

This module deliberately accepts the existing graph, trained GCN and edge
scorer as arguments. It does not instantiate an untrained model or claim
that an edge score is a calibrated probability.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import shap


def explain_edge_features(
    score_features: Callable[[np.ndarray], np.ndarray],
    background: np.ndarray,
    samples: np.ndarray,
    feature_names: Sequence[str],
    *,
    edge_description: str,
    max_background: int = 50,
) -> dict:
    """Explain a scalar edge-score function over an explicit feature vector.

    score_features must run the existing GCN + EdgeScorer while holding the
    selected candidate edge and graph context fixed, and perturbing only
    the documented feature vector.
    """
    background = np.asarray(background, dtype=float)
    samples = np.asarray(samples, dtype=float)
    names = list(feature_names)

    if background.ndim != 2 or samples.ndim != 2:
        raise ValueError("background and samples must be 2D matrices")
    if background.shape[1] != len(names) or samples.shape[1] != len(names):
        raise ValueError("Feature columns must match feature_names")
    if not len(background) or not len(samples):
        raise ValueError("background and samples must not be empty")
    if not np.isfinite(background).all() or not np.isfinite(samples).all():
        raise ValueError("Feature matrices must contain only finite values")

    background = background[:max_background]

    def checked_score(batch):
        values = np.asarray(score_features(np.asarray(batch, dtype=float)))
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) != len(batch):
            raise ValueError("Edge scorer must return one scalar per row")
        if not np.isfinite(values).all():
            raise ValueError("Edge scorer returned non-finite scores")
        return values

    explainer = shap.Explainer(
        checked_score,
        background,
        feature_names=names,
    )
    explanation = explainer(samples)
    predictions = checked_score(samples)

    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values).reshape(-1)

    results = []
    for index, prediction in enumerate(predictions):
        base = float(base_values[index] if len(base_values) > 1 else base_values[0])
        contributions = values[index].reshape(-1)
        reconstructed = base + float(contributions.sum())

        features = []
        for name, contribution in zip(names, contributions):
            contribution = float(contribution)
            features.append({
                "feature": name,
                "contribution": contribution,
                "direction": (
                    "increases_edge_score" if contribution > 1e-10
                    else "decreases_edge_score" if contribution < -1e-10
                    else "no_change"
                ),
            })
        features.sort(key=lambda item: abs(item["contribution"]), reverse=True)

        results.append({
            "edge": edge_description,
            "output_type": "raw_gcn_edge_score",
            "score": float(prediction),
            "baseline": base,
            "contributions": features,
            "reconstructed_score": reconstructed,
            "additivity_error": float(abs(prediction - reconstructed)),
            "interpretation": (
                "Explains this candidate edge score only. It does not establish "
                "a confirmed attack transition or successful exploitation."
            ),
        })

    return {
        "explainer": "SHAP",
        "model": "existing AutoShield GCN + edge scorer",
        "results": results,
    }
