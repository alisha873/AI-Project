"""
Phase 12 tests: a failure UPSTREAM of the planner (a Phase 11D answer that was not accepted) must stay visible,
sanitized, in the Phase 12 report and in the CLI output.

All with a mocked model; no network, no real key. The "provider errors" below are shaped like real Groq 400
responses but are built locally.
"""

from __future__ import annotations

import json

import pytest

from remediation.planner import scrub_message
from remediation.report import run_incident_remediation
from scripts import run_remediation_plan as cli
from tests.rag_helpers import root_cause_doc
from tests.remediation_helpers import (
    FakeLLM,
    as_json,
    build_context,
    good_answer,
    good_plan,
    plan_json,
    reconstruction_with_configuration,
    synthetic_store,
)

SECRETS = ("gsk_abcdef1234567890", "hunter2hunter2", "super-secret-value-12345", "Bearer zzz.yyy-xxx")
INCIDENT_TEXT = "SELECT password FROM incident_users WHERE nginx-19"


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


@pytest.fixture()
def key_in_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value-12345")


def leaky_error() -> Exception:
    """A provider error in the shape of a Groq 400, deliberately carrying secrets and model output."""
    return RuntimeError(
        "Error code: 400 - Authorization: Bearer zzz.yyy-xxx api_key=hunter2hunter2 key gsk_abcdef1234567890 "
        "super-secret-value-12345 {'error': {'message': 'Failed to validate JSON.', 'type': 'invalid_request_error', "
        f"'code': 'json_validate_failed', 'failed_generation': '{INCIDENT_TEXT}'}}}}"
    )


def run(incident, first, second=None):
    llm = FakeLLM(first) if second is None else FakeLLM(first, second)
    with synthetic_store() as store:
        return run_incident_remediation(incident, store, llm), llm


def assert_clean(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text, secret
    assert INCIDENT_TEXT not in text


# ------------------------------------------------------------
# the scrubber
# ------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "token gsk_abcdef1234567890 leaked",
    "key sk-abcdefgh12345678 leaked",
    "Authorization: Bearer zzz.yyy-xxx",
    "{'Authorization': 'Basic dXNlcjpwYXNz'}",
    '{"authorization": "Bearer zzz.yyy-xxx"}',
    "x-api-key: hunter2hunter2",
    "api_key=hunter2hunter2",
    "Bearer zzz.yyy-xxx",
])
def test_scrub_message_removes_secret_shapes(raw):
    cleaned = scrub_message(raw)
    for secret in ("gsk_abcdef1234567890", "sk-abcdefgh12345678", "zzz.yyy-xxx", "dXNlcjpwYXNz", "hunter2hunter2"):
        assert secret not in cleaned
    assert "[redacted]" in cleaned


def test_scrub_message_removes_the_configured_key_value(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "another-secret-value-98765")
    assert "another-secret-value-98765" not in scrub_message("failed with another-secret-value-98765 here")


@pytest.mark.parametrize("raw", [
    "x {'failed_generation': 'SELECT password FROM users', 'other': 1}",
    'x {"failed_generation": "SELECT password FROM users"}',
    "x failed_generation: SELECT password FROM users",
])
def test_scrub_message_drops_the_providers_failed_generation_body(raw):
    cleaned = scrub_message(raw)
    assert "SELECT" not in cleaned and "password" not in cleaned
    assert "failed_generation: [omitted]" in cleaned and cleaned.startswith("x ")


def test_scrub_message_truncates_and_accepts_any_object():
    assert len(scrub_message("x" * 5000)) == 300
    assert len(scrub_message("x" * 5000, limit=40)) == 40
    assert scrub_message(ValueError("plain problem")) == "plain problem"
    assert scrub_message("Failed to validate JSON, no secrets here") == "Failed to validate JSON, no secrets here"


# ------------------------------------------------------------
# an upstream failure is reported, sanitized, and the planner is not claimed to have run
# ------------------------------------------------------------

def test_a_failed_phase_11d_call_is_preserved_scrubbed_in_the_report(incident, key_in_environment):
    report, llm = run(incident, leaky_error())
    assert len(llm.calls) == 1                                    # only the Phase 11D call; the planner never called a model
    (diag,) = report.grounded_answer_diagnostics
    assert (diag.finding_index, diag.status, diag.planner_model_called) == (0, "llm_error", False)
    assert "RuntimeError" in diag.message and "Failed to validate JSON" in diag.message   # actionable
    assert diag.llm == "FakeLLM(fake-model)" and diag.issue_codes == ()
    assert_clean(diag.message)
    assert len(diag.message) <= 300
    (result,) = report.results
    assert result.status == "not_planned_insufficient_context" and result.plan is not None and not result.model_called
    assert [r.code for r in result.plan.insufficient_reasons] == ["answer_not_validated"]
    assert report.any_failure                                     # the existing failure semantics are unchanged


def test_the_serialized_report_carries_the_diagnostic_and_no_secret(incident, key_in_environment):
    report, _ = run(incident, leaky_error())
    data = report.to_dict()
    dumped = json.dumps(data, sort_keys=True)
    assert_clean(dumped)
    (entry,) = data["grounded_answer_diagnostics"]
    assert entry["upstream_phase"] == "11D" and entry["status"] == "llm_error"
    assert entry["planner_model_called"] is False and data["plans"][0]["planner_model_called"] is False
    assert data["plans"][0]["llm"] is None                        # no planner model is claimed
    assert (data["advisory_only"], data["execution_performed"], data["ground_truth_used"]) == (True, False, False)
    assert data["nothing_planned_reason"] is None
    assert json.loads(dumped) == data                              # serializes and round-trips


def test_the_report_is_deterministic_with_diagnostics(incident):
    dumps = [json.dumps(run(incident, leaky_error())[0].to_dict(), sort_keys=True) for _ in range(2)]
    assert dumps[0] == dumps[1]


def test_an_invalid_phase_11d_response_keeps_its_text_only_in_the_rejected_answer_field(incident):
    report, llm = run(incident, "not json at all")
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "rejected_invalid_response" and diag.planner_model_called is False and len(llm.calls) == 1
    entry = report.to_dict()["grounded_answer_diagnostics"][0]
    assert entry["rejected_answer"] == "not json at all"          # the returned text, kept for diagnosis
    assert "not json at all" not in entry["message"]               # the short message still carries no model text


def test_rejected_citations_keep_their_issue_codes_and_the_rejected_answer(incident):
    answer = good_answer(build_context())
    answer["hypotheses"][0]["evidence_ids"] = ["nginx-404"]       # fabricated evidence id
    report, llm = run(incident, as_json(answer))
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "rejected_citations" and "unknown_evidence_id" in diag.issue_codes
    assert diag.rejected_answer == as_json(answer) and "nginx-404" in diag.rejected_answer
    assert diag.planner_model_called is False and len(llm.calls) == 1
    assert report.any_failure and report.results[0].status == "not_planned_insufficient_context"


def test_a_planner_failure_after_an_accepted_answer_has_no_upstream_diagnostic(incident):
    report, llm = run(incident, as_json(good_answer(build_context())), TimeoutError("timed out"))
    assert report.grounded_answer_diagnostics == () and len(llm.calls) == 2
    (result,) = report.results
    assert result.status == "llm_error" and result.model_called
    assert report.to_dict()["plans"][0]["planner_model_called"] is True and report.any_failure


def test_a_successful_run_has_no_diagnostics(incident):
    report, _ = run(incident, as_json(good_answer(build_context())), plan_json(good_plan()))
    data = report.to_dict()
    assert data["grounded_answer_diagnostics"] == [] and data["nothing_planned_reason"] is None
    assert data["plans"][0]["status"] == "planned" and data["plans"][0]["planner_model_called"] is True
    assert not report.any_failure


def test_an_insufficient_context_is_reported_upstream_without_being_a_failure(incident):
    (incident / "analysis" / "incident_reconstruction.json").unlink()   # no observed evidence at all
    report, llm = run(incident, "never used")
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "not_generated_no_incident_evidence" and diag.planner_model_called is False
    assert llm.calls == [] and not report.any_failure              # the existing semantics: insufficient is not a failure


def test_without_a_model_there_are_no_diagnostics(incident):
    with synthetic_store() as store:
        report = run_incident_remediation(incident, store, None)
    assert report.grounded_answer_diagnostics == () and report.nothing_planned_reason is None


# ------------------------------------------------------------
# all findings skipped: explicit, not a failure
# ------------------------------------------------------------

@pytest.fixture()
def skipped_incident(incident):
    empty = {"status": "unspecified", "vulnerability_type": "unknown", "root_cause": "unknown",
             "affected_component": "unknown", "supporting_evidence_ids": [], "rationale": "x",
             "evidence_needed_to_confirm": []}
    (incident / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc(empty)), encoding="utf-8")
    return incident


def test_all_findings_skipped_is_explicit_and_not_a_failure(skipped_incident):
    report, llm = run(skipped_incident, "never used")
    assert llm.calls == [] and report.results == () and report.skipped_findings
    assert report.nothing_planned_reason and "all 1 finding was skipped" in report.nothing_planned_reason
    assert not report.any_failure
    data = report.to_dict()
    assert data["nothing_planned_reason"] == report.nothing_planned_reason and data["plans"] == []


def test_the_cli_says_nothing_was_planned_and_still_exits_zero(skipped_incident, synthetic, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM("never used"))
    assert cli.main(["incident_001", "--incident-dir", str(skipped_incident), "--provider", "groq"]) == 0
    out = capsys.readouterr().out
    assert "NOTHING WAS PLANNED" in out and "finding 0 skipped: no_informative_subject" in out


def test_a_normal_run_does_not_print_the_nothing_planned_line(incident, synthetic, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_llm_client",
                        lambda provider, model: FakeLLM(as_json(good_answer(build_context())), plan_json(good_plan())))
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq"]) == 0
    out = capsys.readouterr().out
    assert "NOTHING WAS PLANNED" not in out and "UPSTREAM" not in out


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def test_the_cli_prints_the_upstream_failure_and_exits_3(incident, synthetic, monkeypatch, capsys, key_in_environment):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(leaky_error()))
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq"]) == 3
    captured = capsys.readouterr()
    out = captured.out
    assert "UPSTREAM (Phase 11D) answer not accepted [llm_error] from FakeLLM(fake-model)" in out
    assert "Failed to validate JSON" in out
    assert "The planner's model was NOT called for this finding." in out
    assert "plan not_planned_insufficient_context" in out and "answer_not_validated" in out
    assert "ADVISORY ONLY" in out
    assert_clean(out + captured.err)


def test_the_cli_output_file_carries_the_diagnostic_without_secrets(incident, synthetic, monkeypatch, tmp_path, capsys, key_in_environment):
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(leaky_error()))
    target = tmp_path / "report.json"
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq", "--output", str(target)]) == 3
    text = target.read_text(encoding="utf-8")
    assert_clean(text + capsys.readouterr().out)
    data = json.loads(text)
    assert data["grounded_answer_diagnostics"][0]["status"] == "llm_error"
    assert data["plans"][0]["status"] == "not_planned_insufficient_context" and data["execution_performed"] is False


def test_the_cli_prints_validation_codes_for_rejected_citations(incident, synthetic, monkeypatch, capsys):
    answer = good_answer(build_context())
    answer["hypotheses"][0]["evidence_ids"] = ["nginx-404"]
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(as_json(answer)))
    assert cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq"]) == 3
    out = capsys.readouterr().out
    assert "[rejected_citations]" in out and "validation issue codes:" in out and "unknown_evidence_id" in out
