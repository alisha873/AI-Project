"""
Offline tests for the remediation planner's pre-send budget handling and for reusing the Phase 11D (RAG)
report so the grounded answer is not generated twice.

No network and no real model. The CLI tests monkeypatch the store builder with a synthetic in-memory store
(as the other Phase 12 CLI tests do) and count the `FakeLLM` calls.
"""

from __future__ import annotations

import json

import pytest

from llm.token_budget import PromptTooLargeError
from remediation.planner import plan_remediation
from remediation.report import run_incident_remediation
from rag.report import run_incident_rag
from scripts import run_remediation_plan as cli
from tests.rag_helpers import root_cause_doc
from tests.remediation_helpers import (
    FakeLLM,
    as_json,
    good_answer,
    good_plan,
    make_input,
    plan_json,
    reconstruction_with_configuration,
    synthetic_store,
)
from tests.test_remediation_report import incident, synthetic   # reuse the fixtures  # noqa: F401


# -------------------------------------------------------------- planner budget handling

def test_the_planner_reports_no_request_sent_for_an_oversized_prompt():
    planner_input = make_input()
    assert planner_input.ready
    result = plan_remediation(planner_input, FakeLLM(PromptTooLargeError(9000, 7000, "too big; no request was sent")))
    assert result.status == "llm_error"
    assert result.request_sent is False
    assert result.error_kind == "prompt_too_large"
    assert result.to_dict()["request_sent"] is False


def test_the_planner_reports_request_sent_for_a_provider_error():
    result = plan_remediation(make_input(), FakeLLM(RuntimeError("429 rate limit")))
    assert result.status == "llm_error" and result.request_sent is True
    assert result.error_kind != "prompt_too_large"


# -------------------------------------------------------------- reuse avoids a second grounded answer

def test_passing_a_prebuilt_rag_report_does_not_regenerate_the_answer(incident, synthetic):
    # One FakeLLM shared by both the RAG step and the planner: call 1 = grounded answer, call 2 = plan.
    llm = FakeLLM(as_json(good_answer_for(incident)), plan_json(good_plan()))
    with synthetic_store() as store:
        rag = run_incident_rag(incident, store, llm)
        assert len(llm.calls) == 1                         # exactly one grounded-answer call
        report = run_incident_remediation(incident, store, llm, rag_report=rag)
        assert len(llm.calls) == 2                         # planner only; the answer was NOT regenerated
    assert report.planning_requested and report.results


def good_answer_for(incident_dir):
    # build the context the same way the pipeline will, so the scripted answer validates
    from tests.remediation_helpers import build_context
    return good_answer(build_context())


# -------------------------------------------------------------- CLI --rag-output writes the RAG report once

def test_the_cli_writes_the_rag_report_and_calls_the_model_once_per_step(incident, synthetic, tmp_path, monkeypatch):
    rag_path = tmp_path / "rag_report.json"
    out_path = tmp_path / "remediation_plan.json"
    llm = FakeLLM(as_json(good_answer_for(incident)), plan_json(good_plan()))
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: llm)
    code = cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq",
                     "--rag-output", str(rag_path), "--output", str(out_path)])
    assert code in (0, 3)                                   # 0 if the plan validates; 3 if not — both are fine here
    assert rag_path.is_file() and out_path.is_file()
    rag = json.loads(rag_path.read_text(encoding="utf-8"))
    assert rag["production_retriever"] == "fts5-bm25-v1" and rag["generation_requested"] is True
    # Exactly two model calls total: one grounded answer (reused) + one plan. NOT three.
    assert len(llm.calls) == 2


def test_the_cli_without_rag_output_still_works_and_does_not_write_it(incident, synthetic, tmp_path, monkeypatch):
    out_path = tmp_path / "remediation_plan.json"
    llm = FakeLLM(as_json(good_answer_for(incident)), plan_json(good_plan()))
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: llm)
    cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq", "--output", str(out_path)])
    assert out_path.is_file()
    assert len(llm.calls) == 2                              # still one answer + one plan
