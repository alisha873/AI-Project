"""
Tests for the diagnostic field `rejected_answer` in the Phase 12 report.

Background: a live Phase 11D answer was rejected (`rejected_citations`) and the Phase 12 report kept only the
issue CODES, so the model's ids could not be inspected. These tests use MOCKED model responses only (no network,
no real key) and show that the returned text is kept, credential-redacted, serializable and clearly labelled,
that it exists only when a response was actually returned and refused, and that validation is unchanged.
"""

from __future__ import annotations

import json

import pytest

from rag.citations import validate_answer
from remediation.planner import redact_credentials
from remediation.report import REJECTED_ANSWER_WARNING, run_incident_remediation
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

KEY = "gsk_abcdef1234567890"


@pytest.fixture()
def incident(tmp_path):
    path = tmp_path / "incident_001"
    (path / "analysis").mkdir(parents=True)
    (path / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc()), encoding="utf-8")
    (path / "analysis" / "incident_reconstruction.json").write_text(
        json.dumps(reconstruction_with_configuration()), encoding="utf-8")
    return path


def run(incident, *responses):
    llm = FakeLLM(*responses)
    with synthetic_store() as store:
        return run_incident_remediation(incident, store, llm), llm


def wrong_id_answer(transform) -> dict:
    """A copy of a valid answer whose reference ids were passed through `transform` (a model that mangled them)."""
    answer = good_answer(build_context())
    for entry in answer["reference_relevance"]:
        entry["reference_id"] = transform(entry["reference_id"])
    for hypothesis in answer["hypotheses"]:
        hypothesis["reference_ids"] = [transform(r) for r in hypothesis["reference_ids"]]
    for item in answer["reference_guidance"]:
        item["reference_ids"] = [transform(r) for r in item["reference_ids"]]
    return answer


# ------------------------------------------------------------
# redaction: credentials only, nothing else
# ------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    f"leaked {KEY} here",
    "key sk-abcdefgh12345678 here",
    "Authorization: Bearer zzz.yyy-xxx",
    '{"authorization": "Bearer zzz.yyy-xxx"}',
    "x-api-key: hunter2hunter2",
    "api_key=hunter2hunter2",
])
def test_redact_credentials_removes_secret_shapes(raw):
    cleaned = redact_credentials(raw)
    for secret in (KEY, "sk-abcdefgh12345678", "zzz.yyy-xxx", "hunter2hunter2"):
        assert secret not in cleaned
    assert "[redacted]" in cleaned


def test_redact_credentials_removes_the_configured_key_value(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value-12345")
    assert "super-secret-value-12345" not in redact_credentials("oops super-secret-value-12345 oops")


def test_redact_credentials_leaves_ordinary_text_and_ids_exactly_as_they_were():
    text = (
        '{"reference_id": "cwe:CWE-89#0001", "statement": "GET /search?message=%27%20OR%201=1 with status 200 '
        'was recorded in nginx-19 and application-4; this is not confirmed (see CAPEC-66).", '
        '"note": "SELECT name FROM users WHERE id = 1"}'
    )
    assert redact_credentials(text) == text


def test_redact_credentials_does_not_truncate_and_does_not_apply_the_failed_generation_rule():
    long_text = "word " * 5000
    assert redact_credentials(long_text) == long_text
    assert "failed_generation" in redact_credentials('the model wrote "failed_generation" in a sentence')
    assert redact_credentials(ValueError("plain")) == "plain"


# ------------------------------------------------------------
# the report field
# ------------------------------------------------------------

def test_rejected_citations_keep_the_exact_answer_so_the_offending_ids_can_be_read(incident):
    answer = wrong_id_answer(lambda ref: ref.split("#")[0])        # ids without the "#NNNN" suffix
    report, llm = run(incident, as_json(answer))
    assert len(llm.calls) == 1                                       # no retry, no repair
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "rejected_citations" and diag.planner_model_called is False
    assert diag.rejected_answer == as_json(answer)
    kept = json.loads(diag.rejected_answer)
    assert [e["reference_id"] for e in kept["reference_relevance"]] == [
        r["reference_id"] for r in answer["reference_relevance"]]
    assert all("#" not in e["reference_id"] for e in kept["reference_relevance"])


def test_the_kept_answer_alone_is_enough_to_reproduce_the_issue_codes_offline(incident):
    answer = wrong_id_answer(lambda ref: ref.split("#")[0])
    report, _ = run(incident, as_json(answer))
    (diag,) = report.grounded_answer_diagnostics
    context = build_context()
    again = validate_answer(json.loads(diag.rejected_answer), context)   # deterministic, no model involved
    assert not again.passed and tuple(again.codes) == diag.issue_codes
    assert set(diag.issue_codes) == {"unknown_reference_id", "missing_assessment"}


def test_an_unparsable_response_is_kept_too_with_its_status_unchanged(incident):
    report, llm = run(incident, "this is not json")
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "rejected_invalid_response" and diag.rejected_answer == "this is not json"
    assert diag.issue_codes == () and len(llm.calls) == 1


def test_the_serialized_field_is_labelled_json_safe_and_round_trips(incident):
    answer = wrong_id_answer(lambda ref: ref.split("#")[0])
    report, _ = run(incident, as_json(answer))
    data = report.to_dict()
    (entry,) = data["grounded_answer_diagnostics"]
    assert entry["rejected_answer"] == as_json(answer)
    assert entry["rejected_answer_warning"] == REJECTED_ANSWER_WARNING
    assert "incident details" in REJECTED_ANSWER_WARNING and "NOT accepted" in REJECTED_ANSWER_WARNING
    assert json.loads(json.dumps(data)) == data
    for key in ("finding_index", "upstream_phase", "status", "message", "issue_codes", "llm", "planner_model_called"):
        assert key in entry                                          # the earlier keys are all still there
    assert entry["message"] == "The answer was rejected: %d citation or grounding issue(s)." % len(entry["issue_codes"])


def test_credentials_in_a_rejected_answer_are_redacted_but_citation_ids_survive(incident):
    answer = wrong_id_answer(lambda ref: ref.split("#")[0])
    answer["limitations"] = [f"Authorization: Bearer zzz.yyy-xxx and {KEY} were pasted by the model."]
    report, _ = run(incident, as_json(answer))
    (diag,) = report.grounded_answer_diagnostics
    assert KEY not in diag.rejected_answer and "zzz.yyy-xxx" not in diag.rejected_answer
    assert "[redacted]" in diag.rejected_answer
    kept = json.loads(diag.rejected_answer)                          # still valid JSON
    assert kept["reference_relevance"] == answer["reference_relevance"]
    assert kept["hypotheses"] == answer["hypotheses"]
    assert KEY not in json.dumps(report.to_dict())


def test_the_configured_key_value_never_reaches_the_report(incident, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value-12345")
    answer = wrong_id_answer(lambda ref: ref.split("#")[0])
    answer["limitations"] = ["leaked super-secret-value-12345"]
    report, _ = run(incident, as_json(answer))
    assert "super-secret-value-12345" not in json.dumps(report.to_dict())


def test_a_provider_error_is_never_stored_as_a_rejected_answer(incident):
    error = RuntimeError("Error code: 400 - {'failed_generation': 'SELECT password FROM incident_users'}")
    report, _ = run(incident, error)
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "llm_error" and diag.rejected_answer is None
    dumped = json.dumps(report.to_dict())
    assert "SELECT password" not in dumped and "rejected_answer_warning" not in dumped


def test_an_accepted_answer_has_no_diagnostic_and_keeps_the_report_shape(incident):
    report, llm = run(incident, as_json(good_answer(build_context())), plan_json(good_plan()))
    data = report.to_dict()
    assert data["grounded_answer_diagnostics"] == [] and "rejected_answer" not in json.dumps(data)
    assert data["plans"][0]["status"] == "planned" and len(llm.calls) == 2 and not report.any_failure


def test_nothing_generated_keeps_no_rejected_answer(incident):
    (incident / "analysis" / "incident_reconstruction.json").unlink()
    report, llm = run(incident, "never used")
    (diag,) = report.grounded_answer_diagnostics
    assert diag.status == "not_generated_no_incident_evidence" and diag.rejected_answer is None and llm.calls == []


# ------------------------------------------------------------
# the CLI: kept in the file, not printed
# ------------------------------------------------------------

def test_the_cli_file_has_the_rejected_answer_but_the_terminal_does_not(incident, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_open_store", lambda args: synthetic_store())
    answer = wrong_id_answer(lambda ref: ref.split("#")[0])
    answer["limitations"] = ["a distinctive sentence that should stay out of the terminal"]
    monkeypatch.setattr(cli, "make_llm_client", lambda provider, model: FakeLLM(as_json(answer)))
    target = tmp_path / "report.json"
    code = cli.main(["incident_001", "--incident-dir", str(incident), "--provider", "groq", "--output", str(target)])
    out = capsys.readouterr().out
    assert code == 3                                                 # the exit-code semantics are unchanged
    assert "distinctive sentence" not in out and "rejected_answer" not in out
    saved = json.loads(target.read_text(encoding="utf-8"))
    kept = json.loads(saved["grounded_answer_diagnostics"][0]["rejected_answer"])
    assert kept["limitations"] == ["a distinctive sentence that should stay out of the terminal"]
    assert saved["plans"][0]["planner_model_called"] is False
