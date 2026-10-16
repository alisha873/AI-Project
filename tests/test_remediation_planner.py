"""
Phase 12 tests: `plan_remediation` end to end with a MOCKED `LLMClient`.

No live API, no key, no network. Scripted responses stand in for the model; the real Groq client class
is exercised only with a stubbed SDK object (no HTTP request is made).
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from llm.base import LLMClient
from remediation.contract import RemediationPlan
from remediation.inputs import build_planner_input
from remediation.planner import plan_remediation, safe_error_message
from remediation.prompt import SYSTEM_PROMPT, build_user_prompt, plan_json_schema
from tests.remediation_helpers import (
    OFF_TOPIC,
    RELEVANT_A,
    RELEVANT_B,
    FakeLLM,
    configuration,
    good_plan,
    good_proposal,
    make_input,
    plan_json,
    three_reference_context,
)
from remediation.configuration import UNAVAILABLE


def run(response, planner_input=None):
    planner_input = planner_input or make_input()
    llm = FakeLLM(response)
    return plan_remediation(planner_input, llm), llm


# ------------------------------------------------------------
# a valid context produces several supported proposals
# ------------------------------------------------------------

def test_a_valid_context_produces_multiple_supported_proposals():
    planner_input = make_input()
    result, llm = run(plan_json(good_plan(planner_input)), planner_input)
    assert result.status == "planned" and result.accepted and result.validation.passed
    plan = result.plan
    assert plan.outcome == "proposals" and len(plan.proposals) == 3 and not plan.insufficient_reasons
    assert len(llm.calls) == 1  # one model call, no retry


def test_proposals_are_ranked_by_the_documented_rule_not_the_models_order():
    planner_input = make_input()
    result, _ = run(plan_json(good_plan(planner_input)), planner_input)  # model order: general, high, medium
    proposals = result.plan.proposals
    assert [p.priority_rank for p in proposals] == [1, 2, 3]
    assert [p.priority_basis.evidence_linkage for p in proposals] == ["incident_linked", "incident_linked", "general_hardening"]
    assert [p.priority_basis.severity for p in proposals][:2] == ["high", "medium"]
    assert "not a risk score" in " ".join(result.plan.priority_criteria)
    assert "uncalibrated" in " ".join(result.plan.priority_criteria)


def test_every_proposal_is_marked_proposed_never_executed_and_needs_human_review():
    result, _ = run(plan_json(good_plan()))
    for proposal in result.plan.proposals:
        assert (proposal.status, proposal.executed, proposal.human_review_required) == ("proposed", False, True)
    assert (result.plan.advisory_only, result.plan.execution_performed, result.plan.ground_truth_used) == (True, False, False)


def test_proposal_ids_are_stable_unique_and_content_derived():
    planner_input = make_input()
    first, _ = run(plan_json(good_plan(planner_input)), planner_input)
    second, _ = run(plan_json(good_plan(planner_input)), planner_input)
    ids = [p.proposal_id for p in first.plan.proposals]
    assert ids == [p.proposal_id for p in second.plan.proposals] and len(set(ids)) == 3
    assert all(i.startswith("REM-incident_001-F0-") for i in ids)
    changed = good_plan(planner_input)
    changed["proposals"][1]["description"] = "A different description."
    third, _ = run(plan_json(changed), planner_input)
    assert set(p.proposal_id for p in third.plan.proposals) != set(ids)


# ------------------------------------------------------------
# evidence categories stay separate in the output
# ------------------------------------------------------------

def test_a_hypothesis_is_preserved_as_a_hypothesis_in_the_output():
    result, _ = run(plan_json(good_plan()))
    for proposal in (p for p in result.plan.proposals if p.hypothesis_ids):
        assert [(h.hypothesis_id, h.status, h.is_confirmed, h.basis) for h in proposal.linked_hypotheses] == [
            ("H1", "hypothesis_not_confirmed", False, "inferred_hypothesis")
        ]
    summary = result.plan.input_summary
    assert not any(h.is_confirmed for h in summary.hypotheses)


def test_observed_evidence_ids_and_reference_provenance_stay_distinct():
    planner_input = make_input()
    result, _ = run(plan_json(good_plan(planner_input)), planner_input)
    context_refs = {r.reference_id: r for r in planner_input.usable_references}
    for proposal in result.plan.proposals:
        assert set(proposal.observed_evidence_ids).isdisjoint(proposal.reference_ids)
        assert [c.reference_id for c in proposal.references] == proposal.reference_ids
        for citation in proposal.references:
            source = context_refs[citation.reference_id]
            assert (citation.doc_id, citation.chunk_id, citation.chunk_sha256) == (source.doc_id, source.chunk_id, source.chunk_sha256)
            assert (citation.source, citation.source_id, citation.source_version) == (source.source, source.source_id, source.source_version)
            assert citation.document_content_sha256 == source.document_content_sha256
            assert citation.provenance == source.provenance and citation.provenance["origin_kind"] == "bundled_fixture"
            assert citation.relevance_label == "relevant" and citation.basis == "general_reference_knowledge"
    assert OFF_TOPIC not in json.dumps(result.plan.model_dump(mode="json")["proposals"])


def test_the_input_summary_records_what_was_excluded_and_why():
    result, _ = run(plan_json(good_plan()))
    summary = result.plan.input_summary
    assert [(e.reference_id, e.label) for e in summary.excluded_references] == [(OFF_TOPIC, "not_relevant")]
    assert set(summary.usable_reference_ids) == {RELEVANT_A, RELEVANT_B}
    assert summary.relevance_source == "validated_phase_11d_answer" and summary.configuration_status == "available"
    assert summary.retriever == "fts5-bm25-v1" and summary.is_production_retriever


# ------------------------------------------------------------
# insufficient context: the model is not called and nothing is invented
# ------------------------------------------------------------

@pytest.mark.parametrize("scenario", ["no_answer", "no_evidence", "no_references", "none_relevant"])
def test_insufficient_context_never_reaches_the_model(scenario):
    from rag.answer import generate_grounded_answer
    from tests.rag_helpers import build_context, good_answer
    from tests.retrieval_helpers import make_doc, make_store
    from tests.remediation_helpers import as_json

    if scenario == "no_answer":
        planner_input = make_input(with_answer=False)
    elif scenario == "no_evidence":
        planner_input = build_planner_input(build_context(reconstruction=None), None, configuration())
    elif scenario == "no_references":
        store = make_store(make_doc("CWE-9", "Unrelated", "gardening vocabulary only"))
        try:
            planner_input = build_planner_input(build_context(store), None, configuration())
        finally:
            store.close()
    else:
        context = three_reference_context()
        answer = good_answer(context)
        answer["reference_relevance"] = [{"reference_id": r, "assessment": "not_relevant"} for r in context.reference_ids]
        answer["hypotheses"][0]["reference_ids"] = []
        answer["reference_guidance"] = []
        planner_input = build_planner_input(context, generate_grounded_answer(context, FakeLLM(as_json(answer))), configuration())

    assert not planner_input.ready
    llm = FakeLLM(plan_json(good_plan()))
    result = plan_remediation(planner_input, llm)
    assert llm.calls == []                                   # the model was never called
    assert result.status == "not_planned_insufficient_context" and not result.accepted
    plan = result.plan
    assert plan.outcome == "insufficient_evidence" and plan.proposals == [] and plan.insufficient_reasons
    assert [r.code for r in plan.insufficient_reasons] == [r.code for r in planner_input.insufficiency]
    assert "model was not called" in result.message


def test_the_model_may_validly_declare_the_material_insufficient():
    planner_input = make_input()
    plan = good_plan(planner_input)
    plan.update(outcome="insufficient_evidence", proposals=[],
                insufficient_reason="The passages do not describe a remediation that fits this evidence.",
                unresolved_questions=["Which component handles the search query?"])
    result, _ = run(plan_json(plan), planner_input)
    assert result.status == "model_declared_insufficient" and not result.accepted
    assert result.plan.outcome == "insufficient_evidence" and result.plan.proposals == []
    assert result.plan.insufficient_reasons[0].code == "model_declared_insufficient"
    assert result.plan.unresolved_questions == ["Which component handles the search query?"]


# ------------------------------------------------------------
# invalid or incomplete model output, and rejected content
# ------------------------------------------------------------

@pytest.mark.parametrize("response", [None, "", "not json", "[]", '{"schema_version": "1.0"}', "```json\n{}\n```"])
def test_invalid_model_output_gives_no_plan(response):
    result, llm = run(response)
    assert result.status == "rejected_invalid_response" and result.plan is None and not result.accepted
    assert len(llm.calls) == 1                                # no retry and no repair attempt


def test_incomplete_output_is_not_completed_or_defaulted():
    plan = good_plan()
    del plan["proposals"][0]["validation"]
    result, _ = run(plan_json(plan))
    assert result.status == "rejected_invalid_response" and result.plan is None
    assert "validation" in result.message


def test_output_with_a_model_supplied_priority_is_rejected():
    plan = good_plan()
    plan["proposals"][0]["priority_rank"] = 1
    result, _ = run(plan_json(plan))
    assert result.status == "rejected_invalid_response" and "priority_rank" in result.message


def test_grounding_failures_are_rejected_with_every_issue_and_no_partial_plan():
    plan = good_plan()
    plan["proposals"][1]["observed_evidence_ids"] = ["nginx-404"]       # fabricated
    plan["proposals"][2]["reference_ids"] = [OFF_TOPIC]                 # not_relevant
    plan["proposals"][0]["expected_security_effect"] = "This guarantees the attack is stopped."
    result, llm = run(plan_json(plan))
    assert result.status == "rejected_grounding" and result.plan is None and not result.accepted
    assert {"unknown_evidence_id", "excluded_reference_cited", "remediation_success_claim"} <= set(result.validation.codes)
    assert result.raw_response == plan_json(plan) and len(llm.calls) == 1


def test_a_rejected_plan_never_validates_a_good_proposal_by_association():
    plan = good_plan()
    plan["proposals"][0]["reference_ids"] = []
    result, _ = run(plan_json(plan))
    assert result.plan is None  # two valid proposals do not rescue the plan


def test_a_plan_for_the_wrong_incident_is_rejected():
    plan = good_plan()
    plan["incident_id"] = "incident_002"
    result, _ = run(plan_json(plan))
    assert result.status == "rejected_grounding" and "wrong_incident" in result.validation.codes


# ------------------------------------------------------------
# provider errors, timeouts and secrets
# ------------------------------------------------------------

def test_a_provider_error_gives_llm_error_without_a_plan():
    result, llm = run(RuntimeError("503 service unavailable"))
    assert result.status == "llm_error" and result.plan is None and result.error_kind == "provider_error"
    assert "RuntimeError" in result.message and len(llm.calls) == 1   # not retried
    assert result.llm == "FakeLLM(fake-model)"


@pytest.mark.parametrize("exc", [TimeoutError("timed out"), type("APITimeoutError", (Exception,), {})("Request timed out.")])
def test_a_timeout_is_reported_as_a_timeout(exc):
    result, _ = run(exc)
    assert result.status == "llm_error" and result.error_kind == "timeout" and result.plan is None


def test_real_groq_sdk_exceptions_are_handled():
    groq = pytest.importorskip("groq")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    timeout = groq.APITimeoutError(request=request)
    connection = groq.APIConnectionError(request=request)
    rate = groq.RateLimitError("rate limited", response=httpx.Response(429, request=request), body=None)
    kinds = {}
    for exc in (timeout, connection, rate):
        result, _ = run(exc)
        assert result.status == "llm_error" and result.plan is None
        kinds[type(exc).__name__] = result.error_kind
    assert kinds == {"APITimeoutError": "timeout", "APIConnectionError": "provider_error", "RateLimitError": "provider_error"}


def test_secrets_in_provider_errors_are_not_exposed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value-12345")
    leaked = "auth failed for key gsk_AbCdEf123456 Bearer abc.def-ghi api_key=hunter2hunter2 and super-secret-value-12345"
    result, _ = run(RuntimeError(leaked))
    text = json.dumps(result.to_dict()) + result.message
    for secret in ("gsk_AbCdEf123456", "abc.def-ghi", "hunter2hunter2", "super-secret-value-12345"):
        assert secret not in text
    assert "[redacted]" in result.message


def test_error_messages_are_truncated():
    assert len(safe_error_message(RuntimeError("x" * 5000))) < 400


def test_the_prompt_and_report_contain_no_key_and_no_ground_truth():
    planner_input = make_input()
    prompt = SYSTEM_PROMPT + build_user_prompt(planner_input)
    assert "gsk_" not in prompt and "ground_truth/" not in prompt and "GROQ_API_KEY" not in prompt
    result, _ = run(plan_json(good_plan(planner_input)), planner_input)
    assert "gsk_" not in json.dumps(result.to_dict())


# ------------------------------------------------------------
# the existing LLMClient abstraction is used, unchanged
# ------------------------------------------------------------

def test_the_model_is_called_through_llmclient_generate_with_the_inline_schema():
    planner_input = make_input()
    llm = FakeLLM(plan_json(good_plan(planner_input)))
    assert isinstance(llm, LLMClient)
    plan_remediation(planner_input, llm)
    call = llm.calls[0]
    assert call["system"] == SYSTEM_PROMPT and call["user"] == build_user_prompt(planner_input)
    assert call["response_format"] == plan_json_schema()


def test_the_schema_is_unchanged_by_the_groq_clients_strict_conversion():
    pytest.importorskip("groq")
    from llm.groq import GroqLLMClient

    schema = plan_json_schema()
    assert GroqLLMClient._make_strict_schema(schema) == schema
    assert "$ref" not in json.dumps(schema) and "$defs" not in json.dumps(schema)


def test_the_real_groq_client_class_works_with_a_stubbed_sdk_and_no_network():
    pytest.importorskip("groq")
    from llm.groq import GroqLLMClient

    planner_input = make_input()
    captured = {}

    class StubCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=plan_json(good_plan(planner_input))))])

    client = GroqLLMClient(api_key="test-key-not-real")        # constructing the SDK object makes no request
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))
    result = plan_remediation(planner_input, client)
    assert result.status == "planned" and result.llm == f"GroqLLMClient({client.model})"
    assert captured["response_format"]["type"] == "json_schema" and captured["response_format"]["json_schema"]["strict"] is True
    assert captured["messages"][0]["role"] == "system" and "test-key-not-real" not in json.dumps(result.to_dict())


def test_the_real_groq_client_surfaces_sdk_timeouts_as_llm_error():
    groq = pytest.importorskip("groq")
    httpx = pytest.importorskip("httpx")
    from llm.groq import GroqLLMClient

    class StubCompletions:
        def create(self, **kwargs):
            raise groq.APITimeoutError(request=httpx.Request("POST", "https://api.groq.com/x"))

    client = GroqLLMClient(api_key="test-key-not-real")
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=StubCompletions()))
    result = plan_remediation(make_input(), client)
    assert result.status == "llm_error" and result.error_kind == "timeout" and result.plan is None


def test_wrong_argument_types_are_refused():
    with pytest.raises(TypeError):
        plan_remediation("input", FakeLLM("{}"))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        plan_remediation(make_input(), object())  # type: ignore[arg-type]


# ------------------------------------------------------------
# stable validation and serialization
# ------------------------------------------------------------

def test_the_plan_serializes_deterministically_and_round_trips():
    planner_input = make_input()
    first, _ = run(plan_json(good_plan(planner_input)), planner_input)
    second, _ = run(plan_json(good_plan(planner_input)), planner_input)
    a = json.dumps(first.plan.model_dump(mode="json"), sort_keys=True)
    b = json.dumps(second.plan.model_dump(mode="json"), sort_keys=True)
    assert a == b
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(second.to_dict(), sort_keys=True)
    restored = RemediationPlan.model_validate(json.loads(a))
    assert restored == first.plan and restored.schema_version == "1.0"


def test_the_output_contract_is_immutable_and_rejects_tampering():
    plan = run(plan_json(good_plan()))[0].plan
    with pytest.raises(Exception):
        plan.advisory_only = False  # type: ignore[misc]
    data = plan.model_dump(mode="json")
    for mutate in (
        lambda d: d.update(execution_performed=True),
        lambda d: d["proposals"][0].update(executed=True),
        lambda d: d["proposals"][0].update(human_review_required=False),
        lambda d: d["proposals"][0].update(status="applied"),
        lambda d: d.update(unexpected="field"),
        lambda d: d.update(schema_version="9.9"),
    ):
        broken = copy.deepcopy(data)
        mutate(broken)
        with pytest.raises(Exception):
            RemediationPlan.model_validate(broken)


def test_planning_does_not_modify_the_context():
    planner_input = make_input()
    before = copy.deepcopy(planner_input.context.to_dict())
    run(plan_json(good_plan(planner_input)), planner_input)
    assert planner_input.context.to_dict() == before


def test_unavailable_configuration_shows_up_in_the_plan():
    planner_input = make_input(config=UNAVAILABLE)
    plan = good_plan(planner_input, good_proposal(configuration_dependent=True))
    result, _ = run(plan_json(plan), planner_input)
    assert result.status == "planned"
    assert result.plan.input_summary.configuration_status == "unavailable" and result.plan.input_summary.configuration_fact_ids == []
    assert all(p.configuration_dependent for p in result.plan.proposals)
