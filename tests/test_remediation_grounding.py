"""
Phase 12 tests: parsing and grounding validation. These prove that the planner FAILS SAFELY: each test
takes a valid plan, breaks exactly one thing and checks that the specific problem is reported.

Synthetic data only; no language model; no network.
"""

from __future__ import annotations

import copy
import json

import pytest

from remediation.configuration import UNAVAILABLE, ConfigurationFactInput, build_configuration_context
from remediation.contract import PlanDraft
from remediation.grounding import InvalidPlanError, parse_plan, validate_plan
from tests.remediation_helpers import (
    OFF_TOPIC,
    RELEVANT_A,
    RELEVANT_B,
    good_plan,
    good_proposal,
    make_input,
)


def check(plan: dict, planner_input=None):
    planner_input = planner_input or make_input()
    return validate_plan(parse_plan(json.dumps(plan)), planner_input)


def single(**overrides) -> dict:
    """A one-proposal plan whose proposal has `overrides` applied."""
    return good_plan(make_input(), good_proposal(**overrides))


def codes_for(planner_input=None, **overrides) -> list[str]:
    return list(check(single(**overrides), planner_input).codes)


# ------------------------------------------------------------
# the baseline really is valid (otherwise every test below proves nothing)
# ------------------------------------------------------------

def test_the_baseline_plans_are_valid():
    assert check(single()).passed
    assert check(good_plan()).passed  # the three-proposal plan


# ------------------------------------------------------------
# invalid or incomplete model output (structure)
# ------------------------------------------------------------

@pytest.mark.parametrize("response", [None, "", "   ", "not json", "[1, 2]", '"text"', "```json\n{}\n```", "{"])
def test_a_non_json_object_is_refused(response):
    with pytest.raises(InvalidPlanError):
        parse_plan(response)


def test_a_non_string_response_is_refused():
    with pytest.raises(InvalidPlanError):
        parse_plan(12345)


def test_missing_keys_are_refused():
    plan = single()
    del plan["limitations"]
    with pytest.raises(InvalidPlanError, match="limitations"):
        parse_plan(json.dumps(plan))
    plan = single()
    del plan["proposals"][0]["validation"]
    with pytest.raises(InvalidPlanError, match="validation"):
        parse_plan(json.dumps(plan))


@pytest.mark.parametrize("path, value", [
    (("schema_version",), "2.0"),
    (("outcome",), "maybe"),
    (("proposals", 0, "severity", "level"), "critical"),
    (("proposals", 0, "disruption"), "none"),
    (("proposals", 0, "reversibility"), "easily"),
    (("proposals", 0, "implementation_steps", 0, "origin"), "internet"),
    (("proposals", 0, "affected_area", "state"), "somewhere"),
    (("proposals", 0, "configuration_dependent"), "yes"),
    (("proposals", 0, "title"), ""),
    (("proposals", 0, "title"), "   "),
    (("proposals", 0, "title"), "x" * 5000),
    (("proposals", 0, "reference_ids"), "cwe:CWE-1#0000"),
    (("proposals", 0, "operational_side_effects"), [1, 2]),
    (("proposals", 0, "observed_evidence_ids"), None),
])
def test_wrong_values_and_types_are_refused(path, value):
    plan = single()
    target = plan
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(InvalidPlanError):
        parse_plan(json.dumps(plan))


@pytest.mark.parametrize("field", ["priority_rank", "proposal_id", "status", "executed", "human_review_required",
                                   "priority_basis", "references", "linked_hypotheses"])
def test_the_model_may_not_supply_derived_fields(field):
    plan = single()
    plan["proposals"][0][field] = 1
    with pytest.raises(InvalidPlanError, match=field):
        parse_plan(json.dumps(plan))


def test_unknown_top_level_and_nested_keys_are_refused():
    plan = single()
    plan["confidence"] = 0.9
    with pytest.raises(InvalidPlanError):
        parse_plan(json.dumps(plan))
    plan = single()
    plan["proposals"][0]["severity"]["score"] = 9.1
    with pytest.raises(InvalidPlanError):
        parse_plan(json.dumps(plan))


def test_structure_errors_do_not_echo_the_model_text():
    plan = single(title="x" * 5000)
    with pytest.raises(InvalidPlanError) as raised:
        parse_plan(json.dumps(plan))
    assert "xxxxxxxx" not in str(raised.value)


# ------------------------------------------------------------
# fabricated or unknown evidence, document, chunk, hypothesis and configuration ids
# ------------------------------------------------------------

@pytest.mark.parametrize("field, bad, code", [
    ("observed_evidence_ids", ["nginx-404"], "unknown_evidence_id"),
    ("observed_evidence_ids", ["application-4", "application-99"], "unknown_evidence_id"),
    ("reference_ids", ["cwe:CWE-999#0000"], "unknown_reference_id"),
    ("reference_ids", ["cwe:CWE-1"], "unknown_reference_id"),            # a DOCUMENT id is not a chunk id
    ("reference_ids", ["cwe:CWE-1#0042"], "unknown_reference_id"),      # real document, fabricated chunk
    ("reference_ids", ["made up"], "unknown_reference_id"),
    ("hypothesis_ids", ["H9"], "unknown_hypothesis_id"),
    ("configuration_fact_ids", ["CFG-99"], "unknown_configuration_id"),
])
def test_fabricated_ids_are_rejected(field, bad, code):
    assert code in codes_for(**{field: bad})


def test_an_evidence_id_that_phase_9_cites_but_phase_8_does_not_describe_is_not_citable():
    from tests.rag_helpers import HYPOTHESIS, build_context, reconstruction_doc, root_cause_doc

    reconstruction = reconstruction_doc()
    reconstruction["timeline"] = [e for e in reconstruction["timeline"] if e["event_id"] != "nginx-19"]
    reconstruction["claims"] = [c for c in reconstruction["claims"] if c["evidence_ids"] != ["nginx-19"]]
    context = build_context(reconstruction=reconstruction, root_cause=root_cause_doc(HYPOTHESIS))
    assert "nginx-19" in context.unresolved_evidence_ids
    from rag.answer import generate_grounded_answer
    from remediation.inputs import build_planner_input
    from tests.remediation_helpers import FakeLLM, as_json, configuration, good_answer

    answer = good_answer(context)
    answer["hypotheses"][0]["evidence_ids"] = ["application-4"]  # the 11D answer may only cite resolved events
    result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
    assert result.accepted, result.validation
    planner_input = build_planner_input(context, result, configuration())
    plan = good_plan(planner_input, good_proposal())  # cites application-4 and the unresolved nginx-19
    assert "unknown_evidence_id" in check(plan, planner_input).codes


def test_duplicate_citations_are_reported():
    assert "duplicate_citation" in codes_for(observed_evidence_ids=["application-4", "application-4"])


# ------------------------------------------------------------
# references presented as evidence, and the reverse
# ------------------------------------------------------------

def test_a_reference_passage_cannot_be_cited_as_observed_incident_evidence():
    codes = codes_for(observed_evidence_ids=["application-4", RELEVANT_B])
    assert "category_mixing" in codes


def test_incident_evidence_cannot_be_cited_as_a_reference():
    assert "category_mixing" in codes_for(reference_ids=[RELEVANT_A, "nginx-19"])


def test_a_hypothesis_or_configuration_id_in_the_wrong_field_is_category_mixing():
    assert "category_mixing" in codes_for(observed_evidence_ids=["H1"])
    assert "category_mixing" in codes_for(reference_ids=["CFG-1"])
    assert "category_mixing" in codes_for(configuration_fact_ids=["application-4"])


def test_prose_that_treats_a_passage_as_an_incident_event_is_flagged():
    text = f"The passage {RELEVANT_B} shows the attack happened."
    assert "prose_citation_mismatch" in codes_for(rationale=text)  # mentioned but not cited by this proposal


def test_prose_naming_an_unknown_or_uncited_event_is_flagged():
    assert "unknown_evidence_id" in codes_for(rationale="Event nginx-404 shows more requests.")
    assert "prose_citation_mismatch" in codes_for(observed_evidence_ids=["application-4"], rationale="See nginx-19 too.")


# ------------------------------------------------------------
# passages labelled not_relevant (or unclear, or unassessed)
# ------------------------------------------------------------

def test_a_not_relevant_passage_cannot_be_cited():
    assert "excluded_reference_cited" in codes_for(reference_ids=[RELEVANT_A, OFF_TOPIC])


def test_a_not_relevant_passage_cannot_be_smuggled_in_through_prose():
    assert "excluded_reference_cited" in codes_for(rationale=f"Also see {OFF_TOPIC} for context.")


def test_an_unclear_passage_cannot_be_cited_either():
    planner_input = make_input(labels={OFF_TOPIC: "unclear"})
    assert "excluded_reference_cited" in codes_for(planner_input, reference_ids=[OFF_TOPIC])


def test_a_not_relevant_passage_in_a_non_reference_field_is_category_mixing():
    assert "category_mixing" in codes_for(observed_evidence_ids=[OFF_TOPIC])


# ------------------------------------------------------------
# hypotheses stay hypotheses; no unsupported claims of compromise
# ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The SQL injection is confirmed.",
    "This is a confirmed vulnerability in the search handler.",
    "The root cause has been confirmed.",
    "The evidence confirms that the vulnerability exists.",
])
def test_a_hypothesis_presented_as_confirmed_is_rejected(text):
    assert "confirmation_claim" in codes_for(problem_addressed=text)


@pytest.mark.parametrize("text", [
    "The SQL injection is not confirmed.",
    "No evidence confirms the vulnerability.",
    "The root cause cannot be confirmed from the available evidence.",
])
def test_hedged_statements_about_confirmation_are_allowed(text):
    assert "confirmation_claim" not in codes_for(problem_addressed=text)


@pytest.mark.parametrize("text", [
    "The attacker successfully exploited the search endpoint.",
    "The database was accessed through the search request.",
    "Data was exfiltrated after the request.",
    "The application was compromised.",
    "Successful exploitation occurred at 11:07.",
    "Privilege escalation occurred afterwards.",
])
def test_unsupported_claims_of_compromise_are_rejected(text):
    assert "unsupported_compromise_claim" in codes_for(problem_addressed=text)


@pytest.mark.parametrize("text", [
    "There is no evidence of successful exploitation.",
    "Successful exploitation is not confirmed.",
    "Data exfiltration was not observed, so the application was not shown to be compromised.",
])
def test_hedged_statements_about_impact_are_allowed(text):
    assert "unsupported_compromise_claim" not in codes_for(problem_addressed=text)


def test_confirmation_is_allowed_only_when_phase_9_confirmed_it():
    from tests.rag_helpers import HYPOTHESIS, build_context, root_cause_doc
    from tests.remediation_helpers import answered, configuration
    from remediation.inputs import build_planner_input

    confirmed = dict(HYPOTHESIS, status="confirmed")
    context = build_context(root_cause=root_cause_doc(confirmed))
    assert context.hypotheses[0].is_confirmed
    planner_input = build_planner_input(context, answered(context), configuration())
    plan = good_plan(planner_input, good_proposal(problem_addressed="The SQL injection is confirmed."))
    assert "confirmation_claim" not in check(plan, planner_input).codes


# ------------------------------------------------------------
# unsupported claims of remediation success
# ------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "This change guarantees that injection is impossible.",
    "The fix completely prevents the attack.",
    "This eliminates the vulnerability.",
    "The vulnerability has been fixed.",
    "The application is now secure.",
    "The service is no longer vulnerable.",
    "The remediation succeeded.",
    "This leaves zero residual risk.",
    "The endpoint was successfully remediated.",
])
def test_claims_that_a_fix_works_are_rejected(text):
    assert "remediation_success_claim" in codes_for(expected_security_effect=text)


@pytest.mark.parametrize("text", [
    "This change does not guarantee that injection is impossible.",
    "It cannot eliminate the risk entirely.",
    "No guarantee is given.",
    "This is intended to reduce the chance that request text alters a query.",
    "It will not eliminate the vulnerability by itself.",
])
def test_cautious_wording_about_effect_is_allowed(text):
    assert "remediation_success_claim" not in codes_for(expected_security_effect=text)


# ------------------------------------------------------------
# invented identifiers, paths and commands
# ------------------------------------------------------------

def test_an_invented_cve_is_rejected():
    assert "unsupported_identifier" in codes_for(description="This addresses CVE-2099-0001 in the handler.")


def test_a_cwe_not_stated_by_the_hypothesis_or_a_cited_passage_is_rejected():
    assert "unsupported_identifier" in codes_for(description="This also covers CWE-79.")


def test_identifiers_from_the_hypothesis_and_cited_passages_are_allowed():
    codes = codes_for(description="This follows the CWE-89 hypothesis and the passage on CWE-1.")
    assert "unsupported_identifier" not in codes


def test_an_identifier_of_an_uncited_passage_is_not_allowed():
    # CAPEC-1 is a real usable passage but this proposal cites only CWE-1's passage
    assert "unsupported_identifier" in codes_for(description="See CAPEC-1 for the attack pattern.")


@pytest.mark.parametrize("text", [
    "Edit `/etc/nginx/nginx.conf` and reload the server.",
    "Run `docker run --privileged app` to test.",
    "Set `read_only: true` in the compose file.",
    "Change the setting at /var/lib/app/config.yaml.",
    "Fetch https://evil.example/patch.sh first.",
])
def test_invented_paths_commands_and_settings_are_rejected(text):
    assert "ungrounded_literal" in codes_for(implementation_steps=[{"text": text, "origin": "general_practice"}])


def test_paths_that_occur_in_the_supplied_material_are_allowed():
    step = {"text": "Review how `/search` and /api/search build their queries.", "origin": "general_practice"}
    assert "ungrounded_literal" not in codes_for(implementation_steps=[step])


def test_an_invented_affected_asset_path_is_rejected():
    area = {"state": "identified", "description": "The component at /srv/legacy/search.py (event application-4)."}
    assert "ungrounded_literal" in codes_for(affected_area=area)


# ------------------------------------------------------------
# grounding and completeness rules
# ------------------------------------------------------------

def test_a_proposal_without_a_reference_is_rejected():
    assert "missing_citation" in codes_for(reference_ids=[])


def test_a_proposal_tied_to_evidence_must_name_its_hypothesis():
    assert "missing_hypothesis_link" in codes_for(hypothesis_ids=[])


def test_general_hardening_without_evidence_is_allowed():
    area = {"state": "unknown", "description": "The affected component is not named by the supplied material."}
    assert check(single(observed_evidence_ids=[], hypothesis_ids=[], affected_area=area)).passed


def test_an_identified_area_needs_a_basis():
    area = {"state": "identified", "description": "The search handler."}
    assert "unsupported_affected_area" in codes_for(observed_evidence_ids=[], hypothesis_ids=[], affected_area=area)


def test_unknown_and_not_applicable_areas_are_legal_without_a_basis():
    for state in ("unknown", "not_applicable"):
        area = {"state": state, "description": "Cannot be determined from the supplied material."}
        assert check(single(observed_evidence_ids=[], hypothesis_ids=[], affected_area=area)).passed


def test_side_effects_and_steps_are_required():
    assert "missing_side_effects" in codes_for(operational_side_effects=[])
    assert "missing_steps" in codes_for(implementation_steps=[])


@pytest.mark.parametrize("disruption, reversibility", [
    ("high", "reversible"), ("unknown", "reversible"), ("low", "irreversible"), ("low", "unknown"),
    ("medium", "partially_reversible"),
])
def test_disruptive_or_irreversible_proposals_must_list_preconditions(disruption, reversibility):
    assert "missing_precondition" in codes_for(disruption=disruption, reversibility=reversibility, preconditions=[])


def test_low_disruption_reversible_proposals_need_no_preconditions():
    assert check(single(disruption="low", reversibility="reversible", preconditions=[])).passed


def test_high_confidence_is_refused_for_an_unconfirmed_hypothesis():
    assert "confidence_exceeds_support" in codes_for(confidence={"level": "high", "rationale": "Very sure."})


def test_medium_confidence_needs_a_step_stated_by_a_cited_reference():
    steps = [{"text": "Validate the parameter.", "origin": "general_practice"}]
    assert "confidence_exceeds_support" in codes_for(
        implementation_steps=steps, confidence={"level": "medium", "rationale": "Reasonable."}
    )


# ------------------------------------------------------------
# missing or contradictory infrastructure configuration
# ------------------------------------------------------------

def test_with_unavailable_configuration_a_proposal_must_admit_it_depends_on_it():
    planner_input = make_input(config=UNAVAILABLE)
    codes = codes_for(planner_input, configuration_dependent=False, unresolved_questions=[])
    assert "configuration_dependency_missing" in codes and "missing_unresolved_question" in codes


def test_with_unavailable_configuration_an_honest_proposal_passes():
    planner_input = make_input(config=UNAVAILABLE)
    plan = good_plan(planner_input, good_proposal(configuration_dependent=True))
    assert check(plan, planner_input).passed


def test_with_unavailable_configuration_a_configuration_fact_cannot_be_invented():
    planner_input = make_input(config=UNAVAILABLE)
    assert "unknown_configuration_id" in codes_for(planner_input, configuration_dependent=True, configuration_fact_ids=["CFG-1"])


def _conflicting():
    return build_configuration_context(None, (
        ConfigurationFactInput("container:test-api", "runs_as_user", "root", single_valued=True),
        ConfigurationFactInput("container:test-api", "runs_as_user", "appuser", single_valued=True),
    ))


def test_conflicting_configuration_facts_cannot_be_cited():
    config = _conflicting()
    planner_input = make_input(config=config)
    for fact_id in config.conflicting_fact_ids:
        codes = codes_for(planner_input, configuration_dependent=True, configuration_fact_ids=[fact_id])
        assert "conflicting_configuration_cited" in codes
        assert "conflicting_configuration_cited" in codes_for(
            planner_input, configuration_dependent=True, rationale=f"Because of {fact_id}.")


def test_with_conflicting_configuration_a_proposal_must_admit_the_dependency():
    planner_input = make_input(config=_conflicting())
    assert "configuration_dependency_missing" in codes_for(planner_input, configuration_dependent=False)


def test_a_configuration_fact_can_be_cited_when_available():
    assert check(single(configuration_fact_ids=["CFG-1"])).passed
    assert "prose_citation_mismatch" in codes_for(rationale="Per CFG-1 the container is attached to a network.")


# ------------------------------------------------------------
# plan-level consistency
# ------------------------------------------------------------

def test_the_wrong_incident_is_rejected():
    plan = single()
    plan["incident_id"] = "incident_002"
    assert "wrong_incident" in check(plan).codes


def test_outcome_and_proposals_must_agree():
    plan = single()
    plan["proposals"] = []
    assert "no_proposals" in check(plan).codes
    plan = single()
    plan["outcome"] = "insufficient_evidence"
    plan["insufficient_reason"] = "Not enough."
    assert "insufficient_with_proposals" in check(plan).codes
    plan = single()
    plan.update(outcome="insufficient_evidence", proposals=[], insufficient_reason="  ")
    assert "missing_reason" in check(plan).codes


def test_a_valid_insufficient_evidence_answer_passes():
    plan = single()
    plan.update(outcome="insufficient_evidence", proposals=[], insufficient_reason="The passages do not fit the incident.")
    assert check(plan).passed


def test_duplicate_proposals_are_rejected():
    assert "duplicate_proposal" in check(good_plan(None, good_proposal(), good_proposal())).codes
    other = good_proposal(description="A different description of the same thing.")
    assert "duplicate_proposal" in check(good_plan(None, good_proposal(), other)).codes  # same title


def test_too_many_proposals_are_rejected():
    proposals = [good_proposal(title=f"Proposal number {i}") for i in range(9)]
    assert "too_many_items" in check(good_plan(None, *proposals)).codes


def test_plan_level_text_is_screened_too():
    plan = single()
    plan["limitations"] = ["The application was compromised and the fix guarantees safety."]
    codes = check(plan).codes
    assert "unsupported_compromise_claim" in codes and "remediation_success_claim" in codes


def test_every_issue_is_reported_not_just_the_first():
    codes = codes_for(
        observed_evidence_ids=["nginx-404"], reference_ids=[], expected_security_effect="This guarantees safety.",
        description="CVE-2099-1 is addressed."
    )
    assert {"unknown_evidence_id", "missing_citation", "remediation_success_claim", "unsupported_identifier"} <= set(codes)


def test_validation_does_not_mutate_its_inputs():
    plan = single()
    before = copy.deepcopy(plan)
    check(plan)
    assert plan == before
    draft = parse_plan(json.dumps(plan))
    assert isinstance(draft, PlanDraft)
