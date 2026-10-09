import sys

import pytest

from evaluation.evaluate_evidence_extraction import main


def test_extraction_argument_is_required(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate_evidence_extraction"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    error = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "--extraction" in error
    assert "required" in error


def test_missing_extraction_file_reports_resolved_path(monkeypatch, capsys):
    missing_path = "evaluation/results/missing_extraction.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_evidence_extraction", "--extraction", missing_path],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    error = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "/evaluation/results/missing_extraction.json" in error
    assert "does not exist" in error