from __future__ import annotations

from collections.abc import Iterable


def precision_recall_f1(
    predicted: Iterable[str],
    expected: Iterable[str],
) -> dict[str, float | int]:
    predicted_set = set(predicted)
    expected_set = set(expected)

    true_positives = len(predicted_set & expected_set)
    false_positives = len(predicted_set - expected_set)
    false_negatives = len(expected_set - predicted_set)

    precision = (
        true_positives / len(predicted_set)
        if predicted_set
        else 0.0
    )
    recall = (
        true_positives / len(expected_set)
        if expected_set
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
