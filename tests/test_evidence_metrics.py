from evaluation.evidence_metrics import precision_recall_f1


def test_perfect_evidence_selection():
    result = precision_recall_f1(
        predicted=["event-1", "event-2"],
        expected=["event-1", "event-2"],
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_over_selection_reduces_precision():
    result = precision_recall_f1(
        predicted=["event-1", "event-2", "event-3"],
        expected=["event-1"],
    )

    assert result["precision"] == 1 / 3
    assert result["recall"] == 1.0
    assert result["f1"] == 0.5


def test_under_selection_reduces_recall():
    result = precision_recall_f1(
        predicted=["event-1"],
        expected=["event-1", "event-2"],
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert abs(result["f1"] - 2 / 3) < 1e-9


def test_empty_selection_and_empty_expected_set():
    result = precision_recall_f1([], [])

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
