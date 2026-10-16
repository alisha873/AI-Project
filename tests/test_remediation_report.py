"""
Phase 12 tests: the incident-level report and the CLI, with a MOCKED model.

The synthetic tests build an incident directory in a temp folder. One test uses the real `incident_001`
analysis files and the real pinned archives when they are present (skipped, with a reason, otherwise). No
network and no real language model is used anywhere.
"""

from __future__ import annotations

import argparse
import json

import pytest

from llm.base import LLMClient
from remediation.report import run_incident_remediation
from scripts import run_remediation_plan as cli
from tests.rag_helpers import root_cause_doc
from tests.remediation_helpers import (
    RELEVANT_A,
    FakeLLM,
    as_json,
    build_context,
    good_answer,
    good_plan,
    good_proposal,
    make_input,
    plan_json,
    reconstruction_with_configuration,
    synthetic_store,
)
from tests.retrieval_helpers import INCIDENT_001, archives_present


@pytest.fixture()
def incident(tmp_path):
    path = tmp_path / "incident_001"
    (path / "analysis").mkdir(parents=True)
    (path / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc()), encoding="utf-8")
    (path / "analysis" / "incident_reconstruction.json").write_text(
        json.dumps(reconstruction_with_configuration()), encoding="utf-8")
    return path


@pytest.fixture()
def synthetic(monkeypatch):
    monkeypatch.setattr(cli, "_open_store", lambda args: synthetic_store())


def scripted_llm(*, answer=None, plan=None) -> FakeLLM:
    """First call: the Phase 11D grounded answer; second call: the remediation plan."""
    answer = answer if answer is not None else as_json(good_answer(build_context()))
    plan = plan if plan is not None else plan_json(good_plan())
    return FakeLLM(answer, plan)


def run(incident, *extra):
    return cli.main(["incident_001", "--incident-dir", str(incident), *extra])


# ------------------------------------------------------------
# deterministic end-to-end pipeline with a mocked LLMClient
# ------------------------------------------------------------

def test_end_to_end_pipeline_with_a_mocked_llm_client(incident):
    llm = scripted_llm()
    assert isinstance(llm, LLMClient)
    with synthetic_store() as store:
        report = run_incident_remediation(incident, store, llm)
    assert len(llm.calls) == 2                                  # one grounded answer, one plan
    assert "evidence_ids" in llm.calls[0]["user"] or "observed_evidence" in llm.calls[0]["user"]
    assert report.planning_requested and report.uses_only_production_retriever and not report.any_failure
    assert report.grounded_answer_statuses == ((0, "answered"),)
    (result,) = report.results
    assert result.status == "planned" and len(result.plan.proposals) == 3
    assert [p.priority_rank for p in result.plan.proposals] == [1, 2, 3]
    summary = result.plan.input_summary
    assert summary.configuration_status == "available" and summary.configuration_fact_ids == ["CFG-1", "CFG-2"]
    assert summary.relevance_source == "validated_phase_11d_answer"


def test_the_report_is_deterministic_advisory_and_secret_free(incident):
    dumps = []
    for _ in range(2):
        with synthetic_store() as store:
            dumps.append(json.dumps(run_incident_remediation(incident, store, scripted_llm()).to_dict(), sort_keys=True))
    assert dumps[0] == dumps[1]
    data = json.loads(dumps[0])
    assert (data["advisory_only"], data["execution_performed"], data["ground_truth_used"]) == (True, False, False)
    assert data["retrieval_methods_used"] == ["fts5-bm25-v1"] and data["uses_only_production_retriever"]
    assert data["caveats"] and data["plans"][0]["plan"]["caveats"]
    assert "gsk_" not in dumps[0]


def test_without_a_model_nothing_is_planned(incident):
    with synthetic_store() as store:
        report = run_incident_remediation(incident, store, None)
    assert not report.planning_requested and report.results == () and not report.any_failure


def test_a_rejected_phase_11d_answer_leaves_the_planner_without_relevance_labels(incident):
    llm = scripted_llm(answer="not json")
    with synthetic_store() as store:
        report = run_incident_remediation(incident, store, llm)
    assert len(llm.calls) == 1                                   # the planner never called the model
    assert report.grounded_answer_statuses == ((0, "rejected_invalid_response"),)
    (result,) = report.results
    assert result.status == "not_planned_insufficient_context"
    assert [r.code for r in result.plan.insufficient_reasons] == ["answer_not_validated"]
    assert report.any_failure                                    # a rejected answer is a failure of the run


def test_an_insufficient_context_is_not_a_failure_of_the_run(incident, tmp_path):
    # no incident evidence at all: remove the reconstruction
    (incident / "analysis" / "incident_reconstruction.json").unlink()
    llm = FakeLLM("never used")
    with synthetic_store() as store:
        report = run_incident_remediation(incident, store, llm)
    assert llm.calls == []
    (result,) = report.results
    assert result.status == "not_planned_insufficient_context" and not report.any_failure
    assert "no_observed_evidence" in [r.code for r in result.plan.insufficient_reasons]


def test_a_model_failure_during_planning_is_reported_not_raised(incident):
    llm = FakeLLM(as_json(good_answer(build_context())), TimeoutError("timed out"))
    with synthetic_store() as store:
        report = run_incident_remediation(incident, store, llm)
    (result,) = report.results
    assert result.status == "llm_error" and result.error_kind == "timeout" and result.plan is None
    assert report.any_failure


def test_a_precomputed_rag_report_avoids_repeating_the_first_half(incident):
    from rag.report import run_incident_rag

    first = FakeLLM(as_json(good_answer(build_context())))
    with synthetic_store() as store:
        rag_report = run_incident_rag(incident, store, first)
        second = FakeLLM(plan_json(good_plan()))
        report = run_incident_remediation(incident, store, second, rag_report=rag_report)
    assert len(first.calls) == 1 and len(second.calls) == 1
    assert report.results[0].status == "planned"


def test_a_rag_report_for_another_incident_is_refused(incident, tmp_path):
    from rag.report import run_incident_rag

    other = tmp_path / "other" / "incident_002"
    (other / "analysis").mkdir(parents=True)
    (other / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc(incident_id="incident_002")), encoding="utf-8")
    with synthetic_store() as store:
        foreign = run_incident_rag(other, store, None)
        with pytest.raises(ValueError):
            run_incident_remediation(incident, store, FakeLLM("{}"), rag_report=foreign)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def test_default_cli_run_calls_no_model_and_plans_nothing(incident, synthetic, capsys):
    assert run(incident) == 0
    out = capsys.readouterr().out
    assert "fts5-bm25-v1" in out and "No language model was called" in out and "ADVISORY ONLY" in out


def test_cli_with_a_mocked_model_prints_ranked_proposals_and_writes_only_on_request(incident, synthetic, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: scripted_llm())
    target = tmp_path / "out" / "plan.json"
    assert run(incident, "--provider", "groq", "--output", str(target)) == 0
    out = capsys.readouterr().out
    assert "plan planned" in out and "#1 REM-incident_001-F0-" in out and "human review required" in out
    assert "model's own judgement" in out
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["plans"][0]["status"] == "planned" and data["execution_performed"] is False
    assert not any(p.name not in {"root_cause_analysis.json", "incident_reconstruction.json"}
                   for p in (incident / "analysis").iterdir())


def test_cli_without_output_writes_nothing(incident, synthetic, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: scripted_llm())
    before = sorted(p.name for p in tmp_path.rglob("*"))
    assert run(incident, "--provider", "groq") == 0
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


@pytest.mark.parametrize("llm, expected", [
    (lambda: FakeLLM(as_json(good_answer(build_context())), "not json"), "rejected_invalid_response"),
    (lambda: FakeLLM(as_json(good_answer(build_context())), RuntimeError("provider down")), "llm_error"),
    (lambda: FakeLLM(as_json(good_answer(build_context())), plan_json(good_plan(None, good_proposal(reference_ids=[])))), "rejected_grounding"),
])
def test_a_failed_or_rejected_plan_gives_exit_status_3(incident, synthetic, monkeypatch, capsys, llm, expected):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: llm())
    assert run(incident, "--provider", "groq") == 3
    assert expected in capsys.readouterr().out


def test_a_grounding_rejection_lists_the_issues(incident, synthetic, monkeypatch, capsys):
    bad = good_plan(None, good_proposal(observed_evidence_ids=["nginx-404"]))
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: scripted_llm(plan=plan_json(bad)))
    assert run(incident, "--provider", "ollama") == 3
    assert "[unknown_evidence_id]" in capsys.readouterr().out


def test_an_insufficient_evidence_plan_exits_zero(incident, synthetic, monkeypatch, capsys):
    plan = good_plan()
    plan.update(outcome="insufficient_evidence", proposals=[], insufficient_reason="The passages do not fit this evidence.")
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: scripted_llm(plan=plan_json(plan)))
    assert run(incident, "--provider", "groq") == 0
    assert "model_declared_insufficient" in capsys.readouterr().out


def test_failures_are_reported_not_raised(incident, tmp_path, capsys):
    assert cli.main(["incident_1"]) == 1
    assert cli.main(["incident_002", "--incident-dir", str(incident)]) == 1
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--database", str(tmp_path / "nope.sqlite")]) == 1
    assert not (tmp_path / "nope.sqlite").exists()
    assert "FAILED" in capsys.readouterr().err


def test_a_provider_that_cannot_be_created_is_a_clean_failure(incident, synthetic, monkeypatch, capsys):
    def boom(provider, model):
        raise RuntimeError("GROQ_API_KEY is not set")

    monkeypatch.setattr(cli, "make_llm_client", boom)
    assert run(incident, "--provider", "groq") == 1
    assert "cannot create the 'groq' client" in capsys.readouterr().err


def test_malformed_configuration_is_a_clean_failure(incident, synthetic, monkeypatch, capsys):
    path = incident / "analysis" / "incident_reconstruction.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["graph_relationships"]["configured_only"] = "not a list"
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: scripted_llm())
    assert run(incident, "--provider", "groq") == 1
    assert "ConfigurationError" in capsys.readouterr().err


def test_unknown_provider_is_command_line_misuse():
    with pytest.raises(SystemExit) as info:
        cli.main(["incident_001", "--provider", "openai"])
    assert info.value.code == 2


def test_there_is_no_way_to_select_another_retriever():
    options = {opt for action in cli._parser()._actions for opt in action.option_strings}
    assert not any(word in opt for opt in options for word in ("retriever", "semantic", "rrf", "embed", "fusion"))


# ------------------------------------------------------------
# real incident_001 files and real pinned archives, mocked model
# ------------------------------------------------------------

@pytest.mark.skipif(not archives_present() or not (INCIDENT_001 / "analysis").is_dir(),
                    reason="pinned archives or incident_001 analysis missing; the real-data path is NOT tested")
def test_real_incident_001_with_a_mocked_model():
    args = argparse.Namespace(database=None, manifest=cli.DEFAULT_MANIFEST, cwe_archive=cli.DEFAULT_CWE,
                              capec_archive=cli.DEFAULT_CAPEC)
    from rag.pipeline import build_contexts_for_incident

    with cli._open_store(args) as store:
        context = build_contexts_for_incident(INCIDENT_001, store).contexts[0]
        refs = context.reference_ids
        plan = good_plan(make_input(), good_proposal(reference_ids=[refs[0]]))
        llm = FakeLLM(as_json(good_answer(context)), plan_json(plan))
        report = run_incident_remediation(INCIDENT_001, store, llm)
    assert len(llm.calls) == 2 and report.uses_only_production_retriever
    (result,) = report.results
    assert result.status == "planned", result.validation and result.validation.issues
    summary = result.plan.input_summary
    assert summary.configuration_status == "available" and len(summary.configuration_fact_ids) >= 1
    assert summary.observed_evidence_ids == ["application-4", "nginx-19"]
    assert [h.status for h in summary.hypotheses] == ["hypothesis_not_confirmed"]
    first = result.plan.proposals[0]
    assert first.references[0].reference_id == refs[0] and first.references[0].provenance["adapter"]
    assert RELEVANT_A not in json.dumps(result.plan.model_dump(mode="json"))  # synthetic ids never leak in
