"""Phase 13A: the Phase 14 hand-off view and the rule for when a result is usable as post-remediation evidence."""

from __future__ import annotations

import json

import pytest

from tests.validation_helpers import (
    approved_registry, blocked_report, executed_dict, full_policy, full_request, gate_without_identity_rules,
    planned_result,
)
from validation.contract import ValidationReport
from validation.handoff import HANDOFF_VERSION, Phase14Handoff, build_phase14_handoff, serialize
from validation.orchestrator import ValidationRequest, build_validation_report
from validation.scenario import SQLI_SCENARIO_ID

NOT_OBSERVED = ("objective_not_observed", "objective_not_observed_under_tested_conditions")
ACHIEVED = ("objective_achieved", "objective_observed")


def handoff_of(data: dict) -> Phase14Handoff:
    return build_phase14_handoff(ValidationReport.model_validate(data))


def test_the_handoff_has_exactly_the_documented_fields():
    assert set(Phase14Handoff.model_fields) == {
        "handoff_version", "contract_version", "validation_run_id", "incident_id",
        "status", "reason_code", "usable_as_post_remediation_evidence", "not_usable_reasons",
        "identity_source_id", "identity_independently_verified", "identity_live_suitable",
        "scenario_id", "scenario_definition_sha256", "oracle_id", "tested_state",
        "pre_remediation_state_id", "post_remediation_state_id", "requested_origin",
        "check_attempted", "check_completed", "transport_kind", "requests_sent", "observation_ids",
        "condition_results",
        "proposal_ids", "applied_proposal_ids", "application_state", "attestation_id", "attested_by",
        "application_verified_by_phase13", "unmet_preconditions",
        "observed_evidence_ids", "hypothesis_ids", "configuration_fact_ids", "reference_ids", "reference_chunk_ids",
        "interpretation_limits", "remediation_effectiveness_proven", "general_security_guarantee",
    }
    assert HANDOFF_VERSION == "1.1"  # 13B.3 migration: adds the identity fields and the not-usable reasons


def test_a_blocked_report_hands_over_what_was_missing_and_is_not_evidence_of_effectiveness():
    report = blocked_report()
    handoff = build_phase14_handoff(report)
    assert handoff.usable_as_post_remediation_evidence is False
    assert (handoff.status, handoff.reason_code) == ("blocked", "execution_not_implemented_phase_13a")
    assert handoff.unmet_preconditions == ["live_execution_available"]
    assert (handoff.check_attempted, handoff.check_completed, handoff.requests_sent) == (False, False, 0)
    assert handoff.observation_ids == [] and handoff.tested_state == "not_tested"
    assert set(handoff.condition_results.values()) == {"not_evaluated"}
    assert handoff.application_state == "independently_attested"
    assert handoff.application_verified_by_phase13 is False
    assert handoff.remediation_effectiveness_proven is False and handoff.general_security_guarantee is False
    assert handoff.interpretation_limits == report.limitations


def test_proposals_are_listed_separately_from_the_applied_proposals():
    plan = planned_result()
    report = build_validation_report(full_request(plan), approved_registry(), full_policy())
    handoff = build_phase14_handoff(report)
    assert handoff.proposal_ids == [p.proposal_id for p in plan.plan.proposals]
    assert handoff.applied_proposal_ids == [plan.plan.proposals[0].proposal_id]
    assert len(handoff.proposal_ids) > len(handoff.applied_proposal_ids)  # proposed is not applied


def test_upstream_identifiers_stay_in_separate_categories_and_are_deduplicated_in_order():
    plan = planned_result()
    report = build_validation_report(full_request(plan), approved_registry(), full_policy())
    handoff = build_phase14_handoff(report)
    expected_events = []
    for proposal in plan.plan.proposals:  # Phase 12 order of proposals
        for item in proposal.observed_evidence_ids:
            if item not in expected_events:
                expected_events.append(item)
    assert handoff.observed_evidence_ids == expected_events
    assert len(set(handoff.observed_evidence_ids)) == len(handoff.observed_evidence_ids)
    assert handoff.hypothesis_ids == ["H1"]
    assert all("#" in chunk for chunk in handoff.reference_chunk_ids) and handoff.reference_chunk_ids
    assert not set(handoff.observed_evidence_ids) & set(handoff.reference_chunk_ids)
    assert not set(handoff.hypothesis_ids) & set(handoff.observed_evidence_ids)


def test_no_check_no_plan_no_scenario_are_all_unusable_and_say_what_is_missing():
    handoff = build_phase14_handoff(build_validation_report(ValidationRequest("incident_001", SQLI_SCENARIO_ID)))
    assert handoff.usable_as_post_remediation_evidence is False
    assert "plan_usable" in handoff.unmet_preconditions and handoff.status == "not_validated"
    unknown = build_phase14_handoff(build_validation_report(ValidationRequest("incident_001", "nope")))
    assert unknown.scenario_definition_sha256 is None and unknown.oracle_id is None and unknown.condition_results == {}


# ------------------------------------------------------------
# which results a Judge may rely on (reports built by hand, as a future live runner would)
# ------------------------------------------------------------

@pytest.mark.parametrize("pair", [NOT_OBSERVED, ACHIEVED])
def test_a_conclusive_post_remediation_attested_result_is_usable(pair, monkeypatch):
    gate_without_identity_rules(monkeypatch)  # the identity/issuance rules are tested un-stubbed elsewhere
    handoff = handoff_of(executed_dict(*pair))
    assert handoff.usable_as_post_remediation_evidence is True
    assert handoff.check_attempted and handoff.check_completed and handoff.requests_sent == 2
    assert handoff.observation_ids == ["OBS-1"] and handoff.tested_state == "post_remediation"
    assert handoff.post_remediation_state_id == "STATE-POST-1"
    assert handoff.remediation_effectiveness_proven is False, "even a usable result proves nothing universal"


def test_achieved_means_the_objective_still_works_and_is_still_usable_evidence_against_the_remediation(monkeypatch):
    gate_without_identity_rules(monkeypatch)
    handoff = handoff_of(executed_dict(*ACHIEVED))
    assert handoff.status == "objective_achieved" and handoff.usable_as_post_remediation_evidence is True


def test_inconclusive_results_are_never_usable():
    for reason in ("timeout", "lab_unreachable", "unexpected_response", "control_check_failed"):
        assert handoff_of(executed_dict("inconclusive", reason)).usable_as_post_remediation_evidence is False


@pytest.mark.parametrize("pair", [NOT_OBSERVED, ACHIEVED])
def test_a_fake_transport_result_is_a_simulation_and_never_usable_evidence(pair):
    data = executed_dict(*pair)
    data["execution"]["transport_kind"] = "fake"
    with pytest.raises(ValueError, match="simulation"):
        ValidationReport.model_validate(data)  # a fake run must say so
    data["limitations"].append("SIMULATION: produced by a fake transport.")
    handoff = handoff_of(data)
    assert handoff.transport_kind == "fake"
    assert handoff.status == pair[0] and handoff.usable_as_post_remediation_evidence is False


def test_a_live_transport_result_records_that_it_was_live():
    assert handoff_of(executed_dict(*NOT_OBSERVED)).transport_kind == "live"


def test_a_pre_remediation_baseline_is_not_post_remediation_evidence():
    data = executed_dict(*ACHIEVED, tested_state="pre_remediation")
    data["lab_state"]["pre_remediation_state_id"] = "STATE-PRE-1"
    handoff = handoff_of(data)
    assert handoff.tested_state == "pre_remediation" and handoff.usable_as_post_remediation_evidence is False


def test_a_report_that_is_not_a_report_is_refused():
    for bad in (None, {}, "report", blocked_report().model_dump()):
        with pytest.raises(TypeError):
            build_phase14_handoff(bad)  # type: ignore[arg-type]


def test_the_handoff_is_deterministic_serializable_and_round_trips():
    report = blocked_report()
    first, second = build_phase14_handoff(report), build_phase14_handoff(report)
    assert first == second and serialize(first) == serialize(second)
    assert Phase14Handoff.model_validate(json.loads(serialize(first))) == first
    assert serialize(first).endswith("\n") and list(json.loads(serialize(first))) == sorted(json.loads(serialize(first)))


def test_the_handoff_constants_can_not_be_changed():
    data = build_phase14_handoff(blocked_report()).model_dump()
    for field in ("remediation_effectiveness_proven", "general_security_guarantee", "application_verified_by_phase13"):
        bad = dict(data, **{field: True})
        with pytest.raises(ValueError):
            Phase14Handoff.model_validate(bad)
