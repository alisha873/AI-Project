"""Phase 13A: the adapter from the real Phase 12 result, and the remediation-application attestation."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from remediation.contract import RemediationPlan
from remediation.planner import RemediationPlanResult
from tests.validation_helpers import (
    attestation_for,
    insufficient_context_result,
    invalid_response_result,
    llm_error_result,
    model_insufficient_result,
    planned_result,
    ungrounded_result,
)
from validation.inputs import (
    KNOWN_PLAN_STATUSES,
    RemediationApplicationAttestation,
    assess_attestation,
    intake_plan,
)


# ------------------------------------------------------------
# a real, usable plan
# ------------------------------------------------------------

def test_a_real_planned_result_is_usable_and_its_proposals_stay_proposed():
    result = planned_result()
    intake = intake_plan(result)
    assert intake.usable and intake.reason_code is None
    assert [p.proposal_id for p in intake.proposals] == [p.proposal_id for p in result.plan.proposals]
    assert all(p.proposal_status == "proposed" and p.executed is False for p in intake.proposals)
    assert intake.upstream.plan_result_status == "planned" and intake.upstream.plan_outcome == "proposals"
    assert intake.incident_id == "incident_001"


def test_identifiers_are_preserved_in_their_own_categories():
    result = planned_result()
    intake = intake_plan(result)
    by_id = {p.proposal_id: p for p in result.plan.proposals}
    for ref in intake.proposals:
        original = by_id[ref.proposal_id]
        assert ref.observed_evidence_ids == original.observed_evidence_ids
        assert ref.hypothesis_ids == original.hypothesis_ids
        assert ref.configuration_fact_ids == original.configuration_fact_ids
        assert ref.reference_ids == original.reference_ids
        assert ref.reference_chunk_ids == [r.chunk_id for r in original.references]
        assert ref.priority_rank == original.priority_rank
    flat = [i for ref in intake.proposals for i in ref.observed_evidence_ids]
    assert "application-4" in flat and "nginx-19" in flat
    # a reference chunk id is never presented as evidence and an event id never as a reference
    assert not any("#" in item for ref in intake.proposals for item in ref.observed_evidence_ids)
    assert all("#" in item for ref in intake.proposals for item in ref.reference_chunk_ids)


def test_hypothesis_status_is_copied_not_upgraded():
    intake = intake_plan(planned_result())
    assert intake.upstream.hypotheses
    assert all(h.is_confirmed is False for h in intake.upstream.hypotheses)


def test_the_serialized_phase_12_result_is_accepted_and_gives_the_same_references():
    result = planned_result()
    from_object = intake_plan(result)
    from_dict = intake_plan(result.to_dict())
    assert from_dict.usable and from_dict.proposals == from_object.proposals


def test_a_bare_valid_plan_is_accepted_and_an_insufficient_bare_plan_is_not():
    result = planned_result()
    assert intake_plan(result.plan).usable
    bare = intake_plan(insufficient_context_result().plan)
    assert bare.usable is False and bare.reason_code == "remediation_plan_insufficient_context"


def test_phase_12_free_text_is_never_copied_into_the_intake():
    result = planned_result()
    intake = intake_plan(result)
    blob = repr(intake)
    for proposal in result.plan.proposals:
        assert proposal.validation.procedure not in blob
        assert proposal.title not in blob and proposal.description not in blob


# ------------------------------------------------------------
# missing, malformed, rejected, unavailable, insufficient
# ------------------------------------------------------------

def test_a_missing_plan_is_reported_as_missing():
    intake = intake_plan(None)
    assert (intake.usable, intake.reason_code) == (False, "remediation_plan_missing")
    assert intake.proposals == ()


@pytest.mark.parametrize("bad", ["a plan", 5, [], ["plan"], object(), b"{}", {}, {"status": "planned"},
                                 {"status": "weird", "plan": None}, {"status": ["planned"]}, {"status": None}])
def test_unsupported_or_malformed_inputs_are_malformed(bad):
    intake = intake_plan(bad)
    assert (intake.usable, intake.reason_code) == (False, "remediation_plan_malformed")
    assert intake.proposals == ()


@pytest.mark.parametrize("make,reason", [
    (llm_error_result, "remediation_plan_unavailable"),
    (invalid_response_result, "remediation_plan_rejected"),
    (ungrounded_result, "remediation_plan_rejected"),
    (insufficient_context_result, "remediation_plan_insufficient_context"),
    (model_insufficient_result, "remediation_plan_insufficient_context"),
])
def test_every_failed_or_insufficient_phase_12_outcome_has_its_own_reason_and_no_proposals(make, reason):
    result = make()
    for supplied in (result, result.to_dict()):
        intake = intake_plan(supplied)
        assert (intake.usable, intake.reason_code) == (False, reason)
        assert intake.proposals == ()
        assert intake.upstream.plan_result_status == result.status


def test_every_known_phase_12_status_is_handled():
    assert set(KNOWN_PLAN_STATUSES) == {
        "planned", "not_planned_insufficient_context", "model_declared_insufficient", "llm_error",
        "rejected_invalid_response", "rejected_grounding"}


def test_no_remediation_is_invented_for_a_rejected_plan_even_if_its_raw_text_looks_valid():
    result = ungrounded_result()
    assert result.raw_response  # the rejected model text exists...
    intake = intake_plan(result)
    assert intake.usable is False and intake.proposals == ()  # ...and is never turned into proposals


def test_a_rejected_or_failed_result_that_carries_a_plan_is_malformed():
    for result in (llm_error_result(), ungrounded_result(), invalid_response_result()):
        data = result.to_dict()
        data["plan"] = planned_result().plan.model_dump(mode="json")
        assert intake_plan(data).reason_code == "remediation_plan_malformed"


def test_a_planned_result_must_carry_a_valid_plan_and_a_passed_validation():
    good = planned_result().to_dict()
    for change in (
        lambda d: d.update(plan=None),
        lambda d: d.update(plan={"schema_version": "1.0"}),
        lambda d: d["plan"].update(advisory_only=False),
        lambda d: d["plan"].update(execution_performed=True),
        lambda d: d["plan"]["proposals"][0].update(executed=True),
        lambda d: d["plan"]["proposals"][0].update(status="applied"),
        lambda d: d["plan"]["proposals"][0].update(unexpected_key=1),
        lambda d: d["plan"].update(proposals=[]),
        lambda d: d["plan"].update(outcome="insufficient_evidence"),
        lambda d: d.update(validation=None),
        lambda d: d["validation"].update(passed=False),
        lambda d: d["validation"].update(passed="yes"),
        lambda d: d.update(incident_id="incident_999"),
        lambda d: d.update(finding_index=7),
        lambda d: d["plan"]["proposals"].append(copy.deepcopy(d["plan"]["proposals"][0])),
    ):
        data = copy.deepcopy(good)
        change(data)
        intake = intake_plan(data)
        assert (intake.usable, intake.reason_code) == (False, "remediation_plan_malformed"), data.keys()


def test_an_insufficient_result_that_carries_proposals_is_malformed():
    data = insufficient_context_result().to_dict()
    data["plan"] = planned_result().plan.model_dump(mode="json")
    assert intake_plan(data).reason_code == "remediation_plan_malformed"


def test_a_plan_built_around_validation_with_model_construct_is_rechecked():
    result = planned_result()
    forged = RemediationPlan.model_construct(**{**result.plan.__dict__, "advisory_only": False})
    assert intake_plan(forged).reason_code == "remediation_plan_malformed"


def test_a_result_object_with_a_nonsense_status_is_malformed():
    result = planned_result()
    odd = RemediationPlanResult(incident_id=result.incident_id, finding_index=result.finding_index,
                                status="mystery", message="m", plan=result.plan, validation=result.validation)  # type: ignore[arg-type]
    assert intake_plan(odd).reason_code == "remediation_plan_malformed"


# ------------------------------------------------------------
# attestation: a proposal is not an applied remediation
# ------------------------------------------------------------

def test_no_attestation_means_not_attested_and_records_no_lab_state_ids():
    intake = intake_plan(planned_result())
    assessment = assess_attestation(None, intake)
    assert assessment.accepted is False
    assert assessment.application.state == "unknown"
    assert assessment.pre_state_id is None and assessment.post_state_id is None


def test_a_valid_attestation_is_recorded_but_never_verified():
    result = planned_result()
    assessment = assess_attestation(attestation_for(result), intake_plan(result))
    assert assessment.accepted is True
    application = assessment.application
    assert application.state == "independently_attested" and application.verified_by_phase13 is False
    assert application.applied_proposal_ids == [result.plan.proposals[0].proposal_id]
    assert (assessment.pre_state_id, assessment.post_state_id) == ("STATE-PRE-1", "STATE-POST-1")


def test_an_attestation_for_unknown_duplicate_or_no_proposals_is_rejected():
    result = planned_result()
    intake = intake_plan(result)
    real = result.plan.proposals[0].proposal_id
    for ids in (["REM-not-in-the-plan"], [real, "REM-not-in-the-plan"], [real, real], []):
        assessment = assess_attestation(attestation_for(result, applied_proposal_ids=ids), intake)
        assert assessment.accepted is False and assessment.application.state == "unknown"
        assert assessment.post_state_id is None  # a rejected claim never produces a lab-state id


def test_an_attestation_without_a_usable_plan_or_of_the_wrong_type_is_rejected():
    result = planned_result()
    unusable = intake_plan(llm_error_result())
    assert assess_attestation(attestation_for(result), unusable).accepted is False
    for wrong in ("attested", {"attestation_id": "A"}, 1, True):
        assert assess_attestation(wrong, intake_plan(result)).accepted is False


def test_the_attestation_model_is_strict():
    with pytest.raises(ValidationError):
        RemediationApplicationAttestation(attestation_id="A", attested_by="op", applied_proposal_ids=["P"])  # no post state
    with pytest.raises(ValidationError):
        RemediationApplicationAttestation(attestation_id="A", attested_by="op", applied_proposal_ids="P",  # type: ignore[arg-type]
                                          post_remediation_lab_state_id="S")
    with pytest.raises(ValidationError):
        RemediationApplicationAttestation(attestation_id="", attested_by="op", applied_proposal_ids=["P"],
                                          post_remediation_lab_state_id="S")
