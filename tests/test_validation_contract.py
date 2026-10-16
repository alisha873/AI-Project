"""Phase 13A: the validation-report contract, its status/reason model and the invalid combinations."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from tests.validation_helpers import blocked_report, executed_dict
from validation.orchestrator import ValidationRequest, build_validation_report
from validation.scenario import SQLI_SCENARIO_ID
from validation.contract import (
    ALL_REASON_CODES,
    CONTRACT_VERSION,
    INPUT_PRECONDITIONS,
    PERMISSION_PRECONDITIONS,
    PRECONDITIONS,
    REASONS_BY_STATUS,
    STATUSES,
    ApplicationRecord,
    ExecutionRecord,
    LabStateRecord,
    PreconditionFinding,
    ScenarioReference,
    ValidationReport,
)

NOT_OBSERVED = ("objective_not_observed", "objective_not_observed_under_tested_conditions")
ACHIEVED = ("objective_achieved", "objective_observed")


def report_from(data: dict) -> ValidationReport:
    return ValidationReport.model_validate(data)


def broken(data: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        report_from(data)


# ------------------------------------------------------------
# the status model
# ------------------------------------------------------------

def test_the_five_statuses_and_their_reason_codes_are_exactly_these():
    assert set(STATUSES) == {"not_validated", "blocked", "inconclusive", "objective_achieved", "objective_not_observed"}
    assert set(REASONS_BY_STATUS) == set(STATUSES)
    assert len(ALL_REASON_CODES) == len(set(ALL_REASON_CODES)), "a reason code belongs to exactly one status"
    assert REASONS_BY_STATUS["objective_achieved"] == ("objective_observed",)


def test_failure_modes_never_map_to_a_conclusive_status():
    for reason in ("timeout", "lab_unreachable", "unexpected_response", "control_check_failed",
                   "evidence_insufficient", "lab_state_unknown"):
        assert reason in REASONS_BY_STATUS["inconclusive"]
        assert reason not in REASONS_BY_STATUS["objective_not_observed"]
    for status in ("objective_achieved", "objective_not_observed"):
        assert not any(r in REASONS_BY_STATUS[status] for r in REASONS_BY_STATUS["inconclusive"])


def test_preconditions_split_into_input_and_permission_groups():
    assert INPUT_PRECONDITIONS + PERMISSION_PRECONDITIONS == PRECONDITIONS
    assert PERMISSION_PRECONDITIONS == ("scope_conditions_met", "live_execution_available")


def test_every_reason_code_of_another_status_is_rejected_for_every_status():
    bases = {
        "not_validated": build_validation_report(ValidationRequest("incident_001", SQLI_SCENARIO_ID))
        .model_dump(mode="json"),
        "blocked": blocked_report().model_dump(mode="json"),
        "inconclusive": executed_dict("inconclusive", "timeout"),
        "objective_achieved": executed_dict(*ACHIEVED),
        "objective_not_observed": executed_dict(*NOT_OBSERVED),
    }
    assert set(bases) == set(STATUSES)
    for status, base in bases.items():
        report_from(base)  # each base is itself valid
        for other_status, reasons in REASONS_BY_STATUS.items():
            if other_status == status:
                continue
            for reason in reasons:
                data = copy.deepcopy(base)
                data["reason_code"] = reason
                broken(data, "not allowed for status")


def test_the_three_executed_examples_construct():
    for status, reason in (NOT_OBSERVED, ACHIEVED, ("inconclusive", "timeout")):
        assert report_from(executed_dict(status, reason)).status == status


def test_the_contract_is_versioned_and_the_version_is_fixed():
    assert blocked_report().contract_version == CONTRACT_VERSION == "1.0"
    data = blocked_report().model_dump(mode="json")
    data["contract_version"] = "2.0"
    broken(data, "contract_version")


# ------------------------------------------------------------
# an absent or failed check is never a success
# ------------------------------------------------------------

@pytest.mark.parametrize("status,reason", [NOT_OBSERVED, ACHIEVED])
def test_a_conclusive_result_without_an_attempted_check_is_unrepresentable(status, reason):
    data = executed_dict(status, reason)
    data["execution"] = {"attempted": False, "completed": False, "requests_sent": 0}
    broken(data, "requires an attempted check")


@pytest.mark.parametrize("status,reason", [NOT_OBSERVED, ACHIEVED])
def test_a_conclusive_result_without_completion_or_evidence_is_unrepresentable(status, reason):
    data = executed_dict(status, reason)
    data["execution"]["completed"] = False
    data["execution"]["finished_at_utc"] = None
    broken(data, "completed check and observed evidence")
    data = executed_dict(status, reason)
    data["evidence"] = []
    broken(data, "completed check and observed evidence")


def test_a_timeout_or_missing_lab_can_only_be_inconclusive_never_not_observed():
    data = executed_dict(*NOT_OBSERVED)
    data["execution"].update(completed=False, finished_at_utc=None)
    broken(data, "completed check")
    assert report_from(executed_dict("inconclusive", "timeout")).status == "inconclusive"
    assert report_from(executed_dict("inconclusive", "lab_unreachable")).status == "inconclusive"


def test_objective_not_observed_after_remediation_requires_an_attested_application():
    data = executed_dict(*NOT_OBSERVED)
    data["remediation"]["application"] = {"state": "unknown"}
    broken(data, "attested remediation application")
    data["remediation"]["application"] = {"state": "not_applicable"}
    broken(data, "attested remediation application")


def test_a_conclusive_result_needs_a_known_tested_state_and_its_state_id():
    for state in ("not_tested", "unknown"):
        data = executed_dict(*NOT_OBSERVED, tested_state=state)
        broken(data, "known tested lab state")
    data = executed_dict(*NOT_OBSERVED)
    data["lab_state"]["post_remediation_state_id"] = None
    broken(data, "post-remediation lab-state id")


def test_a_pre_remediation_baseline_needs_no_attestation_but_is_not_post_remediation_evidence():
    data = executed_dict(*ACHIEVED, tested_state="pre_remediation")
    data["remediation"]["application"] = {"state": "not_applicable"}
    data["lab_state"]["pre_remediation_state_id"] = "STATE-PRE-1"
    assert report_from(data).lab_state.tested_state == "pre_remediation"


def test_a_conclusive_result_needs_every_condition_evaluated():
    data = executed_dict(*NOT_OBSERVED)
    data["oracle"]["conditions"][0]["result"] = "not_evaluated"
    broken(data, "every oracle condition evaluated")
    data = executed_dict(*ACHIEVED)
    for condition in data["oracle"]["conditions"]:
        condition["result"] = "not_met"
    broken(data, "condition observed as met")


def test_a_check_can_only_have_run_if_the_scope_policy_permitted_it_and_the_lab_identity_was_verified():
    data = executed_dict(*NOT_OBSERVED)
    data["scope"]["execution_permitted"] = False
    broken(data, "scope policy permitted")
    data = executed_dict(*NOT_OBSERVED)
    data["target"]["lab_identity_verified"] = False
    broken(data, "verified lab identity")
    data = executed_dict("inconclusive", "timeout")
    data["preconditions"][0].update(satisfied=False, evaluated=True, reason_code="remediation_plan_missing")
    broken(data, "every precondition was met")


# ------------------------------------------------------------
# not_validated / blocked mean that nothing ran
# ------------------------------------------------------------

@pytest.mark.parametrize("change", [
    lambda d: d["execution"].update(attempted=True, runner_id="r", started_at_utc="2026-10-16T12:00:00Z"),
    lambda d: d["execution"].update(attempted=True, runner_id="r", started_at_utc="2026-10-16T12:00:00Z",
                                    transport="t", transport_kind="fake"),
    lambda d: d["execution"].update(transport_kind="fake"),
    lambda d: d.update(evidence=[{"observation_id": "o", "kind": "k", "summary": "s",
                                  "captured_at_utc": "2026-10-16T12:00:00Z", "attributes": {}}]),
    lambda d: d["lab_state"].update(tested_state="unknown"),
    lambda d: d["oracle"]["conditions"][0].update(result="met"),
])
def test_an_offline_report_can_not_carry_execution_evidence_a_tested_state_or_condition_results(change):
    data = blocked_report().model_dump(mode="json")
    change(data)
    broken(data, "nothing ran|no condition|requires|can not exist|unmet|mean|execution metadata|transport")


def test_a_blocked_report_can_not_carry_an_execution_permission():
    data = blocked_report().model_dump(mode="json")
    data["scope"]["execution_permitted"] = True
    broken(data, "execution permission")


def test_the_reason_code_must_be_that_of_the_first_unmet_precondition():
    data = blocked_report().model_dump(mode="json")
    data["reason_code"] = "scope_policy_denied"
    broken(data, "first unmet precondition")


def test_blocked_requires_every_input_precondition_met_and_not_validated_requires_an_input_failure():
    data = blocked_report().model_dump(mode="json")
    data["status"], data["reason_code"] = "not_validated", "execution_not_implemented_phase_13a"
    broken(data, "not allowed|input precondition")
    data = blocked_report().model_dump(mode="json")
    data["preconditions"][4].update(satisfied=False, reason_code="remediation_application_not_attested")
    broken(data, "first unmet precondition|input preconditions are all met")


def test_an_unevaluated_precondition_counts_as_unmet_and_has_no_reason_code():
    item = PreconditionFinding(precondition="scenario_eligible", satisfied=False, evaluated=False, detail="x")
    assert item.reason_code is None
    with pytest.raises(ValidationError):
        PreconditionFinding(precondition="scenario_eligible", satisfied=True, evaluated=False, detail="x")
    with pytest.raises(ValidationError):
        PreconditionFinding(precondition="scenario_eligible", satisfied=False, detail="x")  # needs a reason
    with pytest.raises(ValidationError):
        PreconditionFinding(precondition="scenario_eligible", satisfied=True, reason_code="scenario_unknown", detail="x")
    with pytest.raises(ValidationError):
        PreconditionFinding(precondition="scenario_eligible", satisfied=False, reason_code="timeout", detail="x")


# ------------------------------------------------------------
# components
# ------------------------------------------------------------

def test_execution_metadata_can_not_exist_without_an_attempt():
    assert ExecutionRecord().attempted is False
    for bad in ({"completed": True}, {"requests_sent": 1}, {"runner_id": "r"}, {"started_at_utc": "2026-10-16T12:00:00Z"}):
        with pytest.raises(ValidationError):
            ExecutionRecord(**bad)
    with pytest.raises(ValidationError):  # an attempt must record its runner and start time
        ExecutionRecord(attempted=True)
    with pytest.raises(ValidationError):  # timestamps must carry a time zone
        ExecutionRecord(attempted=True, runner_id="r", started_at_utc="2026-10-16T12:00:00")
    with pytest.raises(ValidationError):
        ExecutionRecord(attempted=True, runner_id="r", started_at_utc="not a time")
    with pytest.raises(ValidationError):  # completed needs a finish time and a request
        ExecutionRecord(attempted=True, completed=True, runner_id="r", started_at_utc="2026-10-16T12:00:00Z")


def test_an_attestation_needs_its_details_and_details_need_an_attested_state():
    with pytest.raises(ValidationError):
        ApplicationRecord(state="independently_attested")
    with pytest.raises(ValidationError):
        ApplicationRecord(state="unknown", attested_by="someone")
    ok = ApplicationRecord(state="independently_attested", attestation_id="A", attested_by="op",
                           applied_proposal_ids=["P"])
    assert ok.verified_by_phase13 is False
    with pytest.raises(ValidationError):
        ApplicationRecord(state="unknown", verified_by_phase13=True)


def test_an_attested_proposal_must_be_among_the_proposals_evaluated():
    data = blocked_report().model_dump(mode="json")
    data["remediation"]["application"]["applied_proposal_ids"] = ["REM-not-in-plan"]
    broken(data, "not among the proposals")


def test_an_unregistered_scenario_carries_no_definition_facts_and_eligibility_needs_approval():
    assert ScenarioReference(scenario_id="x", registered=False).eligible_for_execution is False
    with pytest.raises(ValidationError):
        ScenarioReference(scenario_id="x", registered=False, family="sql_injection")
    with pytest.raises(ValidationError):
        ScenarioReference(scenario_id="x", registered=True)
    data = blocked_report().scenario.model_dump()
    data.update(review_status="pending_human_review", eligible_for_execution=True)
    with pytest.raises(ValidationError):
        ScenarioReference(**data)


def test_lab_state_ids_must_match_the_tested_state():
    with pytest.raises(ValidationError):
        LabStateRecord(tested_state="pre_remediation")
    with pytest.raises(ValidationError):
        LabStateRecord(tested_state="post_remediation", pre_remediation_state_id="only-pre")
    assert LabStateRecord().tested_state == "not_tested"


# ------------------------------------------------------------
# strictness and the constant markers
# ------------------------------------------------------------

def test_required_fields_unknown_keys_and_coercion_are_refused():
    data = blocked_report().model_dump(mode="json")
    for key in ("validation_run_id", "incident_id", "scenario", "remediation", "status", "reason_code",
                "execution", "scope", "preconditions", "limitations"):
        missing = copy.deepcopy(data)
        del missing[key]
        with pytest.raises(ValidationError):
            report_from(missing)
    extra = copy.deepcopy(data)
    extra["score"] = 0.9
    broken(extra, "score")
    coerced = copy.deepcopy(data)
    coerced["execution"]["attempted"] = "false"
    broken(coerced, "attempted")


@pytest.mark.parametrize("field", ["remediation_effectiveness_proven", "general_security_guarantee",
                                   "ground_truth_used", "llm_used"])
def test_the_report_can_never_claim_effectiveness_a_guarantee_ground_truth_or_a_model(field):
    data = blocked_report().model_dump(mode="json")
    data[field] = True
    broken(data, field)


def test_the_remediation_can_never_be_marked_applied_by_phase_13_or_executed():
    data = blocked_report().model_dump(mode="json")
    data["remediation"]["applied_by_phase13"] = True
    broken(data, "applied_by_phase13")
    data = blocked_report().model_dump(mode="json")
    data["remediation"]["proposals"][0]["executed"] = True
    broken(data, "executed")
    data = blocked_report().model_dump(mode="json")
    data["remediation"]["proposals"][0]["proposal_status"] = "applied"
    broken(data, "proposal_status")


def test_a_report_without_limitations_is_refused_and_there_is_no_numeric_score_field():
    data = blocked_report().model_dump(mode="json")
    data["limitations"] = []
    broken(data, "limitations")
    names = set(ValidationReport.model_fields)
    assert not any("score" in n or "probability" in n or "confidence" in n or "risk" in n for n in names)


def test_the_report_is_immutable():
    report = blocked_report()
    with pytest.raises(ValidationError):
        report.status = "objective_not_observed"  # type: ignore[misc]
