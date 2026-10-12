import pytest

from evaluation.gnn_dataset import (
    CandidateEdgeLabel,
    PreparedEdgeExample,
)
from evaluation.train_gnn import (
    _calculate_metrics,
    _require_both_classes,
)


def make_example(label: int, scenario_id: str = "scenario_001"):
    return PreparedEdgeExample(
        scenario_id=scenario_id,
        split="train",
        edge=CandidateEdgeLabel(
            source="service:nginx",
            target="service:api",
            edge_type="forwards",
            label=label,
        ),
    )


def test_training_split_requires_positive_and_negative_examples():
    examples = [make_example(1), make_example(1)]

    with pytest.raises(ValueError, match="both positive and negative"):
        _require_both_classes(examples, "Training")


def test_training_split_accepts_both_classes():
    _require_both_classes([make_example(0), make_example(1)], "Training")


def test_validation_metrics_are_calculated_correctly():
    metrics = _calculate_metrics(
        predictions=[1, 0, 1, 0],
        labels=[1, 0, 0, 1],
    )

    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_perfect_predictions_produce_perfect_metrics():
    metrics = _calculate_metrics(
        predictions=[1, 0, 1, 0],
        labels=[1, 0, 1, 0],
    )

    assert all(value == pytest.approx(1.0) for value in metrics.values())
