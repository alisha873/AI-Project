"""
Denied compromise wording is shortened; an affirmative claim is not.

The grounding screen is not changed. These tests use a mocked model response and do not call a provider.
"""

from __future__ import annotations

from remediation.grounding import PlanIssue, PlanValidation, parse_plan, validate_plan
from remediation.planner import DENIED_COMPROMISE_NOT_USABLE, plan_remediation
from remediation.qualify import CANONICAL_DENIAL, qualify_denied_compromise
from remediation.text_checks import compromise_claims, confirmation_claims
from tests.remediation_helpers import FakeLLM, good_plan, good_proposal, make_input, plan_json

# The incident_001 rerun used this denial shape. Its CWE-89 and CAPEC-7 were allowed there; this
# fixture cites CWE-1, so the sentence keeps the same list denial without an id this input does not state.
REFERENCE_DENIAL = (
    "The reference passages describe a weakness class, but they do not "
    "confirm that an attack succeeded or that the target system was compromised in this incident; the "
    "finding is hypothesis_not_confirmed."
)
LIMITATION_DENIAL = (
    "No application source code inspection or database execution logs were available to confirm "
    "successful exploitation."
)
RATIONALE_DENIAL = (
    "The recorded requests show injection-like syntax, but they do not confirm that an attack succeeded "
    "or that the target system was compromised."
)

QUALIFIED_RATIONALE = (
    "The events contain SQL-injection-like request syntax. The finding status is "
    "hypothesis_not_confirmed, and the events do not establish successful exploitation."
)
QUALIFIED_LIMITS = (
    "The passage describes a weakness class. The finding status is hypothesis_not_confirmed, "
    "and the passage does not establish that the system was compromised."
)
QUALIFIED_LIMITATION = (
    "There is no evidence of successful exploitation. The finding status is hypothesis_not_confirmed."
)

# Exact sentences from incident_001/analysis/e2e_remediation_rerun2/remediation_plan.json.
LIVE_REFERENCE = (
    "The reference passages describe CWE-89 and blind SQL injection patterns (CAPEC-7), but they do not "
    "confirm that an attack succeeded or that the target system was compromised in this incident; the "
    "finding is hypothesis_not_confirmed."
)
LIVE_LIMITATION = (
    "No application source code inspection or database execution logs were available to confirm "
    "successful exploitation."
)


def _plan(**proposal_overrides) -> dict:
    limitations = proposal_overrides.pop("limitations", None)
    plan = good_plan(make_input(), good_proposal(**proposal_overrides))
    if limitations is not None:
        plan["limitations"] = limitations
    return plan


def _run(plan: dict):
    planner_input = make_input()
    llm = FakeLLM(plan_json(plan))
    return plan_remediation(planner_input, llm), llm, planner_input


def test_the_incident_001_denial_sentences_shorten_without_asserting_compromise():
    plan = _plan(
        reference_support_limits=LIVE_REFERENCE,
        limitations=[
            "Evidence is limited to observed HTTP request patterns with SQL-injection-like syntax.",
            LIVE_LIMITATION,
        ],
    )
    draft = parse_plan(plan_json(plan))
    validation = PlanValidation(False, (
        PlanIssue(
            "unsupported_compromise_claim", "proposals[0].reference_support_limits",
            "asserts 'system was compromised'; no event in this input states that outcome",
        ),
        PlanIssue(
            "unsupported_compromise_claim", "proposals[0].reference_support_limits",
            "asserts 'system was compromised'; no event in this input states that outcome",
        ),
        PlanIssue(
            "unsupported_compromise_claim", "limitations[1]",
            "asserts 'successful exploitation'; no event in this input states that outcome",
        ),
    ))
    qualified = qualify_denied_compromise(draft, validation)
    assert qualified is not None
    reference = qualified.proposals[0].reference_support_limits
    limitation = qualified.limitations[1]
    assert reference == (
        "The finding status is hypothesis_not_confirmed, and the supplied material does not establish "
        "that outcome. The finding is hypothesis_not_confirmed."
    )
    assert limitation == CANONICAL_DENIAL
    assert compromise_claims(reference) == [] and compromise_claims(limitation) == []
    assert confirmation_claims(reference) == [] and confirmation_claims(limitation) == []


def test_the_grounding_screen_still_flags_the_long_denials():
    plan = _plan(
        rationale=RATIONALE_DENIAL,
        reference_support_limits=REFERENCE_DENIAL,
        limitations=["Evidence is limited to the recorded requests.", LIMITATION_DENIAL],
    )
    result = validate_plan(parse_plan(plan_json(plan)), make_input())
    flagged = {
        (issue.location, issue.message)
        for issue in result.issues
        if issue.code == "unsupported_compromise_claim"
    }
    assert any(location == "proposals[0].rationale" and "system was compromised" in message
               for location, message in flagged)
    assert any(location == "proposals[0].reference_support_limits" and "system was compromised" in message
               for location, message in flagged)
    assert any(location == "limitations[1]" and "successful exploitation" in message
               for location, message in flagged)


def test_unsupported_compromise_claims_in_all_three_fields_are_rejected():
    plan = _plan(
        rationale="The system was compromised by the search request.",
        reference_support_limits="The attempt succeeded against the recorded search.",
        limitations=[
            "Evidence is limited to the recorded requests.",
            "Successful exploitation occurred during the requests.",
        ],
    )
    result, llm, _ = _run(plan)
    assert result.status == "rejected_grounding" and result.plan is None
    locations = {issue.location for issue in result.validation.issues}
    assert "proposals[0].rationale" in locations
    assert "proposals[0].reference_support_limits" in locations
    assert "limitations[1]" in locations
    assert set(result.validation.codes) == {"unsupported_compromise_claim"}
    assert len(llm.calls) == 1


def test_a_claim_followed_by_a_hedge_is_not_rewritten_into_a_denial():
    plan = _plan(
        rationale="The system was compromised, which may be unconfirmed.",
        reference_support_limits=QUALIFIED_LIMITS,
        limitations=[QUALIFIED_LIMITATION],
    )
    result, llm, _ = _run(plan)
    assert result.status == "rejected_grounding" and result.plan is None
    assert len(llm.calls) == 1


def test_a_denial_followed_by_a_reassertion_is_not_partially_rewritten():
    plan = _plan(
        rationale="They do not confirm that the system was compromised, but the system was compromised.",
        reference_support_limits=QUALIFIED_LIMITS,
        limitations=[QUALIFIED_LIMITATION],
    )
    draft = parse_plan(plan_json(plan))
    validation = validate_plan(draft, make_input())
    assert qualify_denied_compromise(draft, validation) is None
    result, _llm, _ = _run(plan)
    assert result.status == "rejected_grounding" and result.plan is None


def test_a_correctly_qualified_plan_passes_without_being_rewritten():
    plan = _plan(
        rationale=QUALIFIED_RATIONALE,
        reference_support_limits=QUALIFIED_LIMITS,
        limitations=[QUALIFIED_LIMITATION],
    )
    result, llm, _ = _run(plan)
    proposal = result.plan.proposals[0]
    assert result.status == "planned" and result.validation.passed
    assert proposal.rationale == QUALIFIED_RATIONALE
    assert proposal.reference_support_limits == QUALIFIED_LIMITS
    assert result.plan.limitations == [QUALIFIED_LIMITATION]
    assert proposal.linked_hypotheses[0].status == "hypothesis_not_confirmed"
    assert proposal.linked_hypotheses[0].is_confirmed is False
    assert len(llm.calls) == 1


def test_long_denials_in_all_three_fields_keep_the_finding_unconfirmed():
    plan = _plan(
        rationale=RATIONALE_DENIAL,
        reference_support_limits=REFERENCE_DENIAL,
        limitations=["Evidence is limited to the recorded requests.", LIMITATION_DENIAL],
    )
    result, llm, _ = _run(plan)
    assert result.status == "planned" and result.validation.passed and result.accepted
    proposal = result.plan.proposals[0]
    stored = [proposal.rationale, proposal.reference_support_limits, *result.plan.limitations]
    for text in stored:
        assert compromise_claims(text) == []
        assert confirmation_claims(text) == []
    assert "system was compromised" not in proposal.reference_support_limits
    assert "successful exploitation" not in " ".join(result.plan.limitations)
    assert CANONICAL_DENIAL in proposal.rationale
    assert CANONICAL_DENIAL in proposal.reference_support_limits
    assert result.plan.limitations[0] == "Evidence is limited to the recorded requests."
    assert result.plan.limitations[1] == CANONICAL_DENIAL
    assert proposal.linked_hypotheses[0].status == "hypothesis_not_confirmed"
    assert proposal.linked_hypotheses[0].is_confirmed is False
    assert result.raw_response == plan_json(plan)
    assert len(llm.calls) == 1


def test_a_shortened_denial_that_still_fails_grounding_drops_the_proposals(monkeypatch):
    real = validate_plan
    seen = {"n": 0}

    def wrapped(draft, planner_input):
        seen["n"] += 1
        if seen["n"] == 1:
            return real(draft, planner_input)
        return PlanValidation(False, (PlanIssue(
            "unsupported_compromise_claim",
            "proposals[0].rationale",
            "asserts 'system was compromised'; leftover",
        ),))

    monkeypatch.setattr("remediation.planner.validate_plan", wrapped)
    plan = _plan(
        rationale=RATIONALE_DENIAL,
        reference_support_limits=REFERENCE_DENIAL,
        limitations=[LIMITATION_DENIAL],
    )
    result, llm, _ = _run(plan)
    assert result.status == "model_declared_insufficient" and result.plan is not None
    assert result.plan.outcome == "insufficient_evidence" and result.plan.proposals == []
    assert result.plan.insufficient_reasons[0].code == DENIED_COMPROMISE_NOT_USABLE
    assert "did not itself declare" in result.message
    assert "not inverted" in result.message
    assert len(llm.calls) == 1
