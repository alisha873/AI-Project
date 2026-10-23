"""
Phase 12 tests: the wording screens in `remediation.text_checks`.

These are HEURISTIC safeguards. The tests below show that the listed unsafe claims are refused and that a
list of cautious statements (negated, hypothetical, conditional) is not refused. They do not show that the
screens understand meaning, and no false-positive or false-negative rate is claimed: paraphrases that are
not listed here may still pass. Known gaps are asserted in `test_known_limits_are_documented_not_hidden`.

Synthetic data only; no language model; no network.
"""

from __future__ import annotations

import json

import pytest

from remediation import text_checks as t
from remediation.grounding import parse_plan, validate_plan
from remediation.planner import plan_remediation
from tests.remediation_helpers import FakeLLM, good_plan, good_proposal, make_input, plan_json

# ------------------------------------------------------------
# claims of compromise or successful exploitation
# ------------------------------------------------------------

UNSAFE_COMPROMISE = [
    "The database was breached.",
    "Records were exfiltrated.",
    "The table was dumped.",
    "The attack succeeded.",
    "The attacker dumped the users table.",
    "The attacker gained access to the server.",
    "An intruder obtained the credentials.",
    "The injection attempts were successful.",
    "The exploit worked against the search endpoint.",
    "A breach occurred.",
    "Exfiltration took place overnight.",
    "Sensitive data has been stolen.",
    "The customer database was successfully breached via the search request.",
    "Credentials were leaked.",
    "The server was hacked.",
    "SQL injection was used to dump the database.",
    "The attackers then exfiltrated the records.",
    # claims the Phase 11D list already covered must still be refused
    "The system was compromised.",
    "The vulnerability was successfully exploited.",
]

CAUTIOUS_COMPROMISE = [
    "The database may have been breached.",
    "It is possible that records were exfiltrated.",
    "If records were exfiltrated, rotate the credentials.",
    "If the database was breached, review the audit logs.",
    "Whether the database was breached is unknown.",
    "There is no evidence that the table was dumped.",
    "No evidence shows that the attack succeeded.",
    "The events do not show that records were exfiltrated.",
    "We suspect the attack succeeded, but this is unconfirmed.",
    "The hypothesis is that records were exfiltrated, which the events do not show.",
    "An attacker could dump the table if the query is unsafe.",
    "It is not known whether the injection attempts were successful.",
    "Check whether the attack succeeded.",
    "Replay the requests and confirm that the injection attempts no longer succeed.",
    "Injection weaknesses can expose data if present; presence here is unconfirmed.",
    "The attack surface is reduced.",
    "The hypothesis has not been confirmed and no breach is established.",
    # A negating verb over a list. The high-impact phrase is not the first item, so a two-word window misses it.
    "The citations do not confirm that the code is vulnerable or that exploitation succeeded.",
    "This supports an injection-attempt hypothesis but does not establish vulnerable server-side "
    "query construction, successful exploitation, or database access.",
]


@pytest.mark.parametrize("text", UNSAFE_COMPROMISE)
def test_unsupported_claims_of_compromise_are_flagged(text):
    assert t.compromise_claims(text), text


@pytest.mark.parametrize("text", CAUTIOUS_COMPROMISE)
def test_cautious_statements_about_compromise_are_allowed(text):
    assert t.compromise_claims(text) == [], text


@pytest.mark.parametrize("text", [
    "The database was breached, which may be unconfirmed.",      # the hedge comes AFTER the claim
    "It may be late, but the database was breached.",            # a hedge in an earlier clause does not excuse it
])
def test_a_hedge_must_frame_the_claim_not_merely_appear_nearby(text):
    assert t.compromise_claims(text), text


# ------------------------------------------------------------
# claims that a proposal works
# ------------------------------------------------------------

UNSAFE_SUCCESS = [
    "The application is now safe.",
    "This will prevent all SQL injection.",
    "The endpoint is no longer exploitable.",
    "This makes the application safe.",
    "This keeps the search endpoint secure.",
    "This will prevent SQL injection attacks.",
    "The endpoint is not exploitable after this change.",
    "The search endpoint cannot be exploited.",
    "This change prevents all injection.",
    "This could prevent all attacks.",                           # hedged overclaims are still overclaims
    "This will completely eliminate the vulnerability.",
    "The vulnerability has been fixed.",
    "This guarantees the attack is stopped.",
    "The application will be no longer vulnerable.",
    "This will block attacks.",
]

CAUTIOUS_SUCCESS = [
    "This change is intended to reduce the chance of SQL injection.",
    "This may help reduce risk and does not make the application safe.",
    "This does not guarantee that the endpoint cannot be exploited.",
    "This does not mean the application is now safe.",
    "No claim is made that this will prevent all SQL injection.",
    "This has not been shown to prevent all attacks.",
    "We cannot say that the endpoint is no longer exploitable.",
    "This makes logging more secure.",
    "The fix is not guaranteed to prevent all injection.",
    "If implemented correctly this is expected to lower, not remove, the exposure.",
    "Replay the original requests and compare the responses.",
]


@pytest.mark.parametrize("text", UNSAFE_SUCCESS)
def test_claims_that_a_fix_works_are_flagged(text):
    assert t.success_claims(text), text


@pytest.mark.parametrize("text", CAUTIOUS_SUCCESS)
def test_cautious_statements_about_effect_are_allowed(text):
    assert t.success_claims(text) == [], text


# ------------------------------------------------------------
# confirmation wording
# ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "It was established that the application is injectable.",
    "We have demonstrated that the search handler is exploitable.",
    "The application is definitely vulnerable.",
    "The SQL injection is confirmed.",
])
def test_presenting_a_hypothesis_as_confirmed_is_flagged(text):
    assert t.confirmation_claims(text), text


@pytest.mark.parametrize("text", [
    "It has not been established that the application is injectable.",
    "If it were established that the app is vulnerable, escalate.",
    "Whether the search handler is exploitable has not been determined.",
    "The hypothesis remains unconfirmed.",
])
def test_cautious_statements_about_confirmation_are_allowed(text):
    assert t.confirmation_claims(text) == [], text


# ------------------------------------------------------------
# the screens are wired into plan validation, in every free-text field
# ------------------------------------------------------------

def _codes(**overrides) -> list[str]:
    plan = good_plan(make_input(), good_proposal(**overrides))
    return list(validate_plan(parse_plan(json.dumps(plan)), make_input()).codes)


@pytest.mark.parametrize("field", ["title", "description", "problem_addressed", "rationale",
                                   "reference_support_limits", "expected_security_effect"])
@pytest.mark.parametrize("text, code", [
    ("The database was breached.", "unsupported_compromise_claim"),
    ("Records were exfiltrated.", "unsupported_compromise_claim"),
    ("The table was dumped.", "unsupported_compromise_claim"),
    ("The attack succeeded.", "unsupported_compromise_claim"),
    ("The application is now safe.", "remediation_success_claim"),
    ("This will prevent all SQL injection.", "remediation_success_claim"),
    ("The endpoint is no longer exploitable.", "remediation_success_claim"),
])
def test_each_unsafe_claim_rejects_the_plan_in_each_proposal_field(field, text, code):
    assert code in _codes(**{field: text})


def test_unsafe_claims_in_nested_and_plan_level_text_are_rejected_too():
    assert "unsupported_compromise_claim" in _codes(
        validation={"procedure": "Replay requests in the lab.", "expected_result": "Records were exfiltrated before the change."})
    assert "remediation_success_claim" in _codes(
        implementation_steps=[{"text": "Apply the change; the endpoint is no longer exploitable.", "origin": "general_practice"}])
    assert "remediation_success_claim" in _codes(operational_side_effects=["None, because this will prevent all SQL injection."])
    plan = good_plan(make_input(), good_proposal())
    plan["limitations"] = ["The database was breached."]
    assert "unsupported_compromise_claim" in validate_plan(parse_plan(json.dumps(plan)), make_input()).codes


def test_a_negated_list_does_not_reject_a_plan_and_an_affirmative_claim_still_does():
    cautious = (
        "Citations describe a weakness, but do not confirm that the code is vulnerable "
        "or that exploitation succeeded."
    )
    assert _codes(reference_support_limits=cautious) == []
    assert "unsupported_compromise_claim" in _codes(reference_support_limits="Exploitation succeeded.")


def test_cautious_wording_does_not_reject_an_otherwise_valid_plan():
    codes = _codes(
        rationale="If records were exfiltrated, that would matter, but the events do not show that the attack succeeded.",
        expected_security_effect="This is intended to reduce the chance of SQL injection and does not make the application safe.",
        reference_support_limits="The passage does not establish that the endpoint is no longer exploitable.",
    )
    assert codes == []


def test_long_coordinated_negation_is_still_flagged_which_is_the_conservative_direction():
    # The negation window is short on purpose: widening it would also excuse real claims. A cautious sentence
    # that negates two things in one clause ("... or that ...") can therefore be flagged and must be reworded.
    text = "The passage does not establish that records were exfiltrated or that the endpoint is no longer exploitable."
    assert t.success_claims(text)                      # known false positive of the heuristic
    assert t.compromise_claims(text) == []


def test_an_unsafe_claim_gives_no_plan_through_the_whole_planner():
    planner_input = make_input()
    plan = good_plan(planner_input, good_proposal(description="The database was breached and records were exfiltrated."))
    llm = FakeLLM(plan_json(plan))
    result = plan_remediation(planner_input, llm)
    assert result.status == "rejected_grounding" and result.plan is None and not result.accepted
    assert "unsupported_compromise_claim" in result.validation.codes
    assert len(llm.calls) == 1                                   # no retry and no repair


def test_known_limits_are_documented_not_hidden():
    # Paraphrases outside the pattern lists still pass. This test pins that fact so nobody mistakes the
    # screens for understanding; if a pattern is added for one of these, move the sentence to the unsafe lists.
    assert "heuristic" in (t.__doc__ or "").lower()
    assert "not a test of whether a statement is entailed" in " ".join((t.__doc__ or "").split())
    for text in ("Someone in the team made off with the user table.",
                 "The adversary walked away with the customer list."):
        assert t.compromise_claims(text) == [] and t.success_claims(text) == []
