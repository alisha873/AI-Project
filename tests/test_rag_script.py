"""
Tests for scripts/run_grounded_answer.py (Phase 11D CLI). Mocked model, synthetic store; plus one real
context-only run when the pinned archives are present. No network, no real language model.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scripts import run_grounded_answer as cli
from tests.rag_helpers import FakeLLM, as_json, build_context, good_answer, reconstruction_doc, root_cause_doc, synthetic_store
from tests.retrieval_helpers import INCIDENT_001, archives_present

SCRIPT = Path(cli.__file__)


@pytest.fixture()
def incident(tmp_path):
    path = tmp_path / "incident_001"
    (path / "analysis").mkdir(parents=True)
    (path / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc()), encoding="utf-8")
    (path / "analysis" / "incident_reconstruction.json").write_text(json.dumps(reconstruction_doc()), encoding="utf-8")
    return path


@pytest.fixture()
def synthetic(monkeypatch):
    monkeypatch.setattr(cli, "_open_store", lambda args: synthetic_store())


def run(incident, *extra):
    return cli.main(["incident_001", "--incident-dir", str(incident), *extra])


def test_default_run_builds_context_only_and_names_fts5_v1(incident, synthetic, capsys):
    assert run(incident) == 0
    out = capsys.readouterr().out
    assert "fts5-bm25-v1" in out and "[production baseline]" in out
    assert "No language model was called" in out and "application-4" in out and "cwe:CWE-1#0000" in out


def test_output_report_is_written_only_on_request(incident, synthetic, tmp_path, capsys):
    target = tmp_path / "out" / "report.json"
    assert run(incident, "--output", str(target)) == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["retrieval_methods_used"] == ["fts5-bm25-v1"] and data["answers"] == []
    assert data["ground_truth_used"] is False and data["contexts"][0]["references"]
    assert not any(p.name.endswith(".json") for p in (incident / "analysis").iterdir() if p.name not in
                   {"root_cause_analysis.json", "incident_reconstruction.json"})  # nothing written into the incident


def test_with_a_mocked_valid_answer(incident, synthetic, monkeypatch, capsys):
    context = build_context()
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(as_json(good_answer(context))))
    assert run(incident, "--provider", "groq") == 0
    assert "answered" in capsys.readouterr().out


@pytest.mark.parametrize("response, expected_status", [
    ("not json", "rejected_invalid_response"),
    (RuntimeError("provider down"), "llm_error"),
])
def test_a_failed_or_invalid_answer_gives_exit_status_3(incident, synthetic, monkeypatch, capsys, response, expected_status):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(response))
    assert run(incident, "--provider", "groq") == 3
    assert expected_status in capsys.readouterr().out


def test_rejected_citations_exit_3_and_list_issues(incident, synthetic, monkeypatch, capsys):
    answer = good_answer(build_context())
    answer["observed_evidence"][0]["evidence_ids"] = ["nginx-404"]
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(as_json(answer)))
    assert run(incident, "--provider", "ollama") == 3
    out = capsys.readouterr().out
    assert "rejected_citations" in out and "[unknown_evidence_id]" in out


def test_failures_are_reported_not_raised(incident, tmp_path, capsys):
    assert cli.main(["incident_1"]) == 1
    assert cli.main(["incident_002", "--incident-dir", str(incident)]) == 1  # name mismatch
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--database", str(tmp_path / "nope.sqlite")]) == 1
    assert not (tmp_path / "nope.sqlite").exists()  # a missing store is never created
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--max-documents", "0"]) == 1
    assert "FAILED" in capsys.readouterr().err


def test_a_provider_that_cannot_be_created_is_a_clean_failure(incident, synthetic, monkeypatch, capsys):
    def boom(provider, model):
        raise RuntimeError("GROQ_API_KEY is not set")

    monkeypatch.setattr(cli, "make_llm_client", boom)
    assert run(incident, "--provider", "groq") == 1
    assert "cannot create the 'groq' client" in capsys.readouterr().err


def test_unknown_provider_is_command_line_misuse():
    with pytest.raises(SystemExit) as info:
        cli.main(["incident_001", "--provider", "openai"])
    assert info.value.code == 2


def test_the_default_provider_needs_no_llm_module():
    assert cli.make_llm_client("none", None) is None


def test_there_is_no_way_to_select_another_retriever():
    options = {opt for action in cli._parser()._actions for opt in action.option_strings}
    assert not any(word in opt for opt in options for word in ("retriever", "semantic", "rrf", "embed", "fusion"))
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not {m for m in imported if m.startswith(("retrieval.rrf", "retrieval.semantic", "retrieval.embed",
                                                      "retrieval.vector", "retrieval.fts5_v2"))}


@pytest.mark.skipif(not archives_present() or not (INCIDENT_001 / "analysis").is_dir(),
                    reason="pinned archives or incident_001 analysis missing; the real CLI path is NOT tested")
def test_real_context_only_run_on_incident_001(capsys):
    assert cli.main(["incident_001"]) == 0
    out = capsys.readouterr().out
    assert "fts5-bm25-v1" in out and "No language model was called" in out and "cwe:CWE-89#" in out
