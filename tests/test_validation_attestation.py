"""Phase 13B.2 preparation: the remediation-attestation contract and its binding to plan, scenario, lab, states and
operator authorization. Missing, inconsistent, stale and mismatched attestations and identities are all rejected."""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import ValidationError

from tests.validation_helpers import (
    attestation_for,
    authorization,
    full_request,
    insufficient_context_result,
    planned_result,
    run_fake,
)
from tests.validation_prep_helpers import (
    NOW,
    POST,
    PRE,
    assess_bound,
    attestation,
    identity_assessment,
    observation,
    ts,
)
from validation.attestation import (
    ATTESTATION_MAX_AGE_SECONDS,
    CLAIM_REASON_CODES,
    EVIDENCE_REASON_CODES,
    RemediationAttestation,
    assess_remediation_attestation,
    plan_binding_sha256,
)
from validation.inputs import intake_plan
from validation.lab_identity import CLOCK_SKEW_SECONDS, assess_lab_identity
from validation.scenario import SQLI_SCENARIO_ID


def ids(result=None) -> list[str]:
    return [p.proposal_id for p in (result or planned_result()).plan.proposals]


# ------------------------------------------------------------
# the versioned contract
# ------------------------------------------------------------

def test_the_attestation_is_versioned_strict_frozen_and_never_claims_verification():
    att = attestation()
    assert att.contract_version == "1.0" and att.claim_kind == "operator_claim" and att.verified_by_phase13 is False
    with pytest.raises(ValidationError):
        att.lab_id = "other"
    for bad in ({"contract_version": "2.0"}, {"verified_by_phase13": True}, {"claim_kind": "verified"},
                {"unexpected": 1}, {"attested_at_utc": "yesterday"}, {"plan_binding_sha256": "abc"},
                {"post_remediation_state_id": "has space"}):
        with pytest.raises(ValidationError):
            attestation(**bad)


def test_the_pre_and_post_states_must_be_different_states():
    with pytest.raises(ValidationError, match="different states"):
        attestation(post_remediation_state_id=PRE)


@pytest.mark.parametrize("applied", [[], ["a", "a"]])
def test_an_attestation_names_applied_proposals_at_least_once_and_without_duplicates(applied):
    with pytest.raises(ValidationError):
        attestation(applied_proposal_ids=applied)


def test_every_binding_field_is_required():
    required = ("incident_id", "scenario_id", "lab_id", "plan_binding_sha256", "applied_proposal_ids",
                "pre_remediation_state_id", "post_remediation_state_id", "attested_by", "authorization_id",
                "attested_at_utc", "attestation_id")
    assert set(required) <= {n for n, f in RemediationAttestation.model_fields.items() if f.is_required()}


def test_the_reason_code_lists_have_no_duplicates():
    assert len(set(CLAIM_REASON_CODES)) == len(CLAIM_REASON_CODES)
    assert len(set(EVIDENCE_REASON_CODES)) == len(EVIDENCE_REASON_CODES)


# ------------------------------------------------------------
# the plan binding
# ------------------------------------------------------------

def test_the_plan_binding_is_deterministic_and_none_without_a_usable_plan():
    first = plan_binding_sha256(intake_plan(planned_result()))
    assert first == plan_binding_sha256(intake_plan(planned_result())) and len(first) == 64
    assert plan_binding_sha256(intake_plan(insufficient_context_result())) is None
    assert plan_binding_sha256(intake_plan(None)) is None


def test_the_plan_binding_changes_when_a_proposal_or_its_grounding_changes():
    intake = intake_plan(planned_result())
    base = plan_binding_sha256(intake)
    changed_rank = dataclasses.replace(intake, proposals=(
        intake.proposals[0].model_copy(update={"priority_rank": 99}), *intake.proposals[1:]))
    changed_evidence = dataclasses.replace(intake, proposals=(
        intake.proposals[0].model_copy(update={"observed_evidence_ids": ["other-evidence"]}), *intake.proposals[1:]))
    dropped = dataclasses.replace(intake, proposals=intake.proposals[:-1])
    assert len({base, plan_binding_sha256(changed_rank), plan_binding_sha256(changed_evidence),
                plan_binding_sha256(dropped)}) == 4


def test_the_plan_binding_does_not_depend_on_proposal_order():
    intake = intake_plan(planned_result())
    reordered = dataclasses.replace(intake, proposals=tuple(reversed(intake.proposals)))
    assert plan_binding_sha256(reordered) == plan_binding_sha256(intake)


# ------------------------------------------------------------
# a consistent, evidenced claim
# ------------------------------------------------------------

def test_a_fully_bound_attestation_with_independent_state_evidence_is_ready():
    result = assess_bound()
    assert result.claim_reason_codes == [] and result.evidence_reason_codes == []
    assert result.claim_consistent and result.lab_state_evidenced and result.ready_for_post_remediation_check
    assert result.verified_by_phase13 is False and result.contract_version == "1.0"


def test_only_the_listed_proposals_are_applied_the_rest_stay_proposed():
    all_ids = ids()
    result = assess_bound()
    assert result.applied_proposal_ids == all_ids[:1]
    assert result.not_applied_proposal_ids == all_ids[1:] and len(all_ids) == 3
    assert result.proposed_ids_treated_as_applied is False
    both = assess_bound(att=attestation(applied_proposal_ids=all_ids[:2]))
    assert both.applied_proposal_ids == all_ids[:2] and both.not_applied_proposal_ids == all_ids[2:]


def test_the_legacy_inputs_carry_exactly_the_attested_ids_and_states():
    result = assess_bound()
    legacy = result.legacy_attestation
    assert legacy.applied_proposal_ids == ids()[:1] and legacy.attested_by == "test-operator"
    assert (legacy.pre_remediation_lab_state_id, legacy.post_remediation_lab_state_id) == (PRE, POST)
    assert result.scope_identity.lab_state_id == POST


# ------------------------------------------------------------
# missing and malformed
# ------------------------------------------------------------

def test_a_missing_attestation_means_not_ready_and_no_legacy_input():
    result = assess_bound(att=None)
    assert result.claim_reason_codes == ["attestation_missing"] and result.legacy_attestation is None
    assert result.ready_for_post_remediation_check is False and result.applied_proposal_ids == []


@pytest.mark.parametrize("value", ["ATT-1", 7, {"attestation_id": "x"}, object()])
def test_a_malformed_attestation_is_rejected(value):
    result = assess_bound(att=value)
    assert result.claim_reason_codes == ["attestation_malformed"] and result.legacy_attestation is None


def test_the_old_minimal_attestation_type_is_not_accepted_by_the_bound_contract():
    old = attestation_for(planned_result())
    assert assess_bound(att=old).claim_reason_codes == ["attestation_malformed"]


def test_an_unusable_plan_means_no_attestation_can_be_bound():
    result = assess_bound(insufficient_context_result(), att=attestation(planned_result()))
    assert result.claim_reason_codes == ["plan_unusable"] and result.ready_for_post_remediation_check is False


# ------------------------------------------------------------
# inconsistent and mismatched claims
# ------------------------------------------------------------

@pytest.mark.parametrize("overrides,code", [
    ({"incident_id": "incident_002"}, "incident_mismatch"),
    ({"scenario_id": "other-scenario"}, "scenario_mismatch"),
    ({"plan_binding_sha256": "0" * 64}, "plan_binding_mismatch"),
    ({"applied_proposal_ids": ["REM-not-in-the-plan"]}, "applied_proposals_not_in_plan"),
    ({"authorization_id": "AUTH-OTHER"}, "authorization_id_mismatch"),
    ({"attested_by": "someone-else"}, "attester_not_authorized_operator"),
    ({"lab_id": "lab-2"}, "authorization_lab_mismatch"),
])
def test_each_binding_mismatch_makes_the_claim_inconsistent(overrides, code):
    result = assess_bound(att=attestation(**overrides))
    assert code in result.claim_reason_codes
    assert result.claim_consistent is False and result.ready_for_post_remediation_check is False
    assert result.legacy_attestation is None and result.applied_proposal_ids == []


def test_a_partially_valid_applied_list_is_rejected_whole_not_trimmed():
    mixed = ids()[:1] + ["REM-invented"]
    result = assess_bound(att=attestation(applied_proposal_ids=mixed))
    assert "applied_proposals_not_in_plan" in result.claim_reason_codes
    assert result.applied_proposal_ids == [] and result.legacy_attestation is None


def test_an_attestation_for_another_scenario_is_rejected_even_if_the_authorization_matches_it():
    other = authorization(scenario_id="other-scenario")
    result = assess_bound(att=attestation(scenario_id="other-scenario"), auth=other)
    assert "scenario_mismatch" in result.claim_reason_codes


def test_an_attestation_for_another_incident_is_rejected_even_if_the_authorization_matches_it():
    result = assess_bound(att=attestation(incident_id="incident_002"), auth=authorization(incident_id="incident_002"))
    assert "incident_mismatch" in result.claim_reason_codes


@pytest.mark.parametrize("auth,code", [
    (None, "authorization_missing"),
    ("AUTH-1", "authorization_missing"),
    (authorization(authorization_id="AUTH-9"), "authorization_id_mismatch"),
    (authorization(operator_id="another-operator"), "attester_not_authorized_operator"),
    (authorization(lab_id="lab-2"), "authorization_lab_mismatch"),
    (authorization(scenario_id="other-scenario"), "authorization_scenario_mismatch"),
    (authorization(incident_id="incident_002"), "authorization_incident_mismatch"),
])
def test_the_operator_authorization_must_match_the_attestation(auth, code):
    result = assess_bound(auth=auth)
    assert code in result.claim_reason_codes and result.ready_for_post_remediation_check is False


def test_a_stale_plan_makes_an_old_attestation_unusable():
    intake_changed = dataclasses.replace(intake_plan(planned_result()), proposals=intake_plan(planned_result()).proposals[:-1])
    result = assess_remediation_attestation(
        attestation(), intake_changed, incident_id="incident_001", scenario_id=SQLI_SCENARIO_ID,
        authorization=authorization(), identity=identity_assessment(), now=NOW)
    assert "plan_binding_mismatch" in result.claim_reason_codes


# ------------------------------------------------------------
# stale and future attestations
# ------------------------------------------------------------

def test_an_attestation_exactly_at_the_maximum_age_is_current_and_one_second_older_is_stale():
    assert assess_bound(att=attestation(attested_at_utc=ts(-ATTESTATION_MAX_AGE_SECONDS)),
                        identity=identity_assessment(observation(observed_at_utc=ts(-60)))).claim_consistent
    stale = assess_bound(att=attestation(attested_at_utc=ts(-ATTESTATION_MAX_AGE_SECONDS - 1)),
                         identity=identity_assessment(observation(observed_at_utc=ts(-60))))
    assert "attestation_stale" in stale.claim_reason_codes and stale.legacy_attestation is None


def test_an_attestation_from_the_future_is_rejected_beyond_the_clock_skew():
    ok = assess_bound(att=attestation(attested_at_utc=ts(CLOCK_SKEW_SECONDS)),
                      identity=identity_assessment(observation(observed_at_utc=ts(CLOCK_SKEW_SECONDS))))
    assert ok.claim_consistent
    future = assess_bound(att=attestation(attested_at_utc=ts(CLOCK_SKEW_SECONDS + 1)))
    assert "attestation_from_future" in future.claim_reason_codes


def test_the_age_limit_can_be_tightened():
    result = assess_bound(max_age_seconds=100)  # the default attestation is 300 s old
    assert "attestation_stale" in result.claim_reason_codes


def test_a_naive_now_is_a_programming_error():
    with pytest.raises(TypeError):
        assess_bound(now=None)


# ------------------------------------------------------------
# lab-state evidence: the claim alone is never enough
# ------------------------------------------------------------

def test_without_any_lab_identity_the_claim_may_be_consistent_but_the_state_is_not_evidenced():
    result = assess_bound(identity=None)
    assert result.claim_consistent and result.legacy_attestation is not None
    assert result.evidence_reason_codes == ["lab_identity_missing"]
    assert result.lab_state_evidenced is False and result.ready_for_post_remediation_check is False
    assert result.scope_identity is None


def test_with_the_default_registry_there_is_no_independent_evidence_so_nothing_is_ever_ready():
    unverifiable = assess_lab_identity(observation(), now=NOW)  # default, empty registry: verifier unavailable
    result = assess_bound(identity=unverifiable)
    assert result.claim_consistent is True
    assert "lab_identity_not_independently_verified" in result.evidence_reason_codes
    assert result.ready_for_post_remediation_check is False


def test_an_operator_asserted_identity_is_not_state_evidence():
    asserted = identity_assessment(observation(verification={"claim": "operator_asserted", "source_id": "test-operator"}))
    assert "lab_identity_not_independently_verified" in assess_bound(identity=asserted).evidence_reason_codes


def test_a_stale_identity_observation_is_not_state_evidence():
    stale = identity_assessment(observation(observed_at_utc=ts(-5000)))
    result = assess_bound(identity=stale)
    assert {"lab_identity_stale", "lab_identity_not_independently_verified"} <= set(result.evidence_reason_codes)
    assert result.scope_identity is None


def test_an_observation_of_another_lab_is_rejected():
    other = identity_assessment(observation(lab_id="lab-2"))
    assert "lab_identity_lab_mismatch" in assess_bound(identity=other).evidence_reason_codes


def test_an_observed_state_that_is_still_the_pre_remediation_state_is_called_out():
    pre = identity_assessment(observation(lab_state_id=PRE))
    codes = assess_bound(identity=pre).evidence_reason_codes
    assert "lab_state_is_pre_remediation" in codes and "lab_state_mismatch" not in codes


def test_an_observed_state_that_is_neither_attested_state_is_a_mismatch():
    other = identity_assessment(observation(lab_state_id="STATE-UNKNOWN-7"))
    codes = assess_bound(identity=other).evidence_reason_codes
    assert "lab_state_mismatch" in codes and "lab_state_is_pre_remediation" not in codes


def test_an_observation_made_before_the_attestation_does_not_evidence_the_attested_state():
    early = identity_assessment(observation(observed_at_utc=ts(-400)))  # the attestation is from -300
    assert "lab_observation_predates_attestation" in assess_bound(identity=early).evidence_reason_codes


def test_evidence_problems_never_hide_a_good_claim_and_claim_problems_never_become_ready():
    bad_evidence = assess_bound(identity=None)
    assert bad_evidence.claim_consistent and not bad_evidence.ready_for_post_remediation_check
    bad_claim = assess_bound(att=attestation(lab_id="lab-2"))
    assert bad_claim.lab_state_evidenced is False and not bad_claim.ready_for_post_remediation_check


# ------------------------------------------------------------
# hand-off to the unchanged Phase 13B.1 runner (still simulation only, still fail closed)
# ------------------------------------------------------------

def request_from(assessment):
    return full_request(application=assessment.legacy_attestation, lab_identity=assessment.scope_identity)


def test_a_ready_assessment_drives_the_runner_but_a_fake_run_stays_a_simulation():
    assessment = assess_bound()
    report, transport = run_fake(request=request_from(assessment), vulnerable=False)
    assert report.status == "objective_not_observed" and report.execution.transport_kind == "fake"
    assert report.lab_state.pre_remediation_state_id == PRE and report.lab_state.post_remediation_state_id == POST
    assert report.remediation.application.applied_proposal_ids == ids()[:1]
    assert len(transport.requests) == 2


def test_with_the_default_registry_the_same_inputs_are_blocked_by_the_scope_policy_and_nothing_is_sent():
    assessment = assess_bound(identity=assess_lab_identity(observation(), now=NOW))
    report, transport = run_fake(request=request_from(assessment))
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert "lab_identity_not_independent" in report.scope.reason_codes
    assert transport.requests == [] and report.execution.attempted is False


def test_a_claim_that_failed_binding_hands_the_runner_no_attestation_so_nothing_is_tested():
    assessment = assess_bound(att=attestation(plan_binding_sha256="1" * 64))
    report, transport = run_fake(request=request_from(assessment))
    assert (report.status, report.reason_code) == ("not_validated", "remediation_application_not_attested")
    assert transport.requests == [] and report.lab_state.post_remediation_state_id is None


def test_a_state_the_lab_no_longer_shows_is_blocked_by_the_scope_policy_even_if_the_claim_is_consistent():
    assessment = assess_bound(identity=identity_assessment(observation(lab_state_id=PRE)))
    report, transport = run_fake(request=request_from(assessment))
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert "lab_state_mismatch" in report.scope.reason_codes and transport.requests == []


# ------------------------------------------------------------
# nothing about the locks or the scenario changed
# ------------------------------------------------------------

def test_the_locks_and_the_scenario_review_state_are_unchanged_by_this_milestone():
    from validation import lab_identity, orchestrator, runner
    from validation.scenario import default_registry
    from validation.scope import ScopePolicy, ScopeRequest

    assert runner.LIVE_TRANSPORT_ENABLED is False and orchestrator.LIVE_EXECUTION_IMPLEMENTED is False
    assert lab_identity.IDENTITY_VERIFIER_AVAILABLE is False
    scenario = default_registry().get(SQLI_SCENARIO_ID)
    assert scenario.eligible_for_execution is False and scenario.review.status == "pending_human_review"
    assert len(scenario.open_blocking_ambiguities()) == 5
    assert scenario.probe.wire_encoding == "unspecified" and scenario.probe.control_value is None
    assert ScopePolicy().evaluate(ScopeRequest()).execution_permitted is False
