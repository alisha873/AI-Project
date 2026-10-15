import sys
from pathlib import Path

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
    assert "does not exist" in error

    # The application resolves relative paths against the project root and
    # prints them with the operating system's own separator. Build the
    # expected path from components instead of hardcoding "/" or "\\".
    project_root = Path(__file__).resolve().parents[1]
    expected_path = (
        project_root / "evaluation" / "results" / "missing_extraction.json"
    )
    assert str(expected_path) in error