"""
Phase 12 closure: the planner's own rejected model output (`raw_response`) must be redacted and labelled.

Defect fixed here: for `rejected_invalid_response` and `rejected_grounding` the plan result stored the model's
text as `raw_response` unredacted and with no label in the JSON, unlike the upstream Phase 11D diagnostic
(`rejected_answer`), which is redacted and carries a warning. Everything below uses MOCKED model responses
(no network, no real key, a fake key-shaped string only).
"""

from __future__ import annotations

import json

from remediation.planner import plan_remediation
from tests.remediation_helpers import FakeLLM, good_plan, make_input, plan_json

KEY = "gsk_abcdef1234567890"


def run(response):
    llm = FakeLLM(response)
    return plan_remediation(make_input(), llm), llm


def rejected_by_grounding_with_key():
    plan = good_plan()
    plan["proposals"][1]["observed_evidence_ids"] = ["nginx-404"]  # fabricated id -> rejected_grounding
    plan["limitations"] = [f"debug text {KEY}"]
    return run(plan_json(plan))


def test_rejected_grounding_raw_response_is_redacted_and_labelled():
    result, _ = rejected_by_grounding_with_key()
    assert result.status == "rejected_grounding"
    assert KEY not in (result.raw_response or "")
    assert "[redacted]" in result.raw_response
    data = result.to_dict()
    assert KEY not in json.dumps(data)
    warning = data["raw_response_warning"].lower()
    assert "unvalidated" in warning and "sensitive" in warning and "not accepted" in warning


def test_invalid_response_raw_response_is_redacted_and_labelled():
    result, _ = run(f'{{"outcome": "proposals", "note": "{KEY}"')  # truncated JSON
    assert result.status == "rejected_invalid_response"
    assert KEY not in json.dumps(result.to_dict())
    assert "raw_response_warning" in result.to_dict()


def test_rejected_raw_response_is_otherwise_kept_unchanged_for_diagnosis():
    plan = good_plan()
    plan["proposals"][1]["observed_evidence_ids"] = ["nginx-404"]
    result, _ = run(plan_json(plan))
    assert result.raw_response == plan_json(plan)  # no credential, so nothing is altered or truncated


def test_no_warning_when_there_is_no_rejected_output():
    accepted, _ = run(plan_json(good_plan()))
    assert accepted.status == "planned" and "raw_response_warning" not in accepted.to_dict()
    insufficient = plan_remediation(make_input(with_answer=False), FakeLLM("never used"))
    assert insufficient.status == "not_planned_insufficient_context"
    assert "raw_response_warning" not in insufficient.to_dict()
    errored = plan_remediation(make_input(), FakeLLM(RuntimeError("boom")))
    assert errored.status == "llm_error" and "raw_response_warning" not in errored.to_dict()


def test_the_warning_never_reaches_the_prompt_or_changes_validation():
    result, llm = rejected_by_grounding_with_key()
    assert len(llm.calls) == 1 and result.plan is None and not result.accepted
