"""Phase 13B.1: the offline runner, driven only by fake transports. No test here touches a network."""

from __future__ import annotations

import socket
import subprocess
from datetime import datetime, timezone

import pytest

from tests.validation_helpers import (
    approved_registry,
    attestation_for,
    authorization,
    fixed_clock,
    full_policy,
    full_request,
    identity,
    lab_body,
    planned_result,
    run_fake,
    simulated_lab,
)
from validation.contract import ValidationReport
from validation.handoff import build_phase14_handoff, serialize
from validation.oracle import ClassifierRegistry, OracleOutcome, SqliSearchClassifier
from validation.orchestrator import ValidationRequestError
from validation.runner import (
    LIVE_TRANSPORT_ENABLED,
    SIMULATION_STATEMENT,
    run_controlled_validation,
)
from validation.scenario import default_registry
from validation.scope import ScopePolicy
from validation.transport import ScriptedTransport, ScriptExhausted, TransportResponse

EMPTY = TransportResponse.ok(lab_body([]))


def assert_nothing_was_sent(report: ValidationReport, transport) -> None:
    assert transport.requests == []
    assert report.execution.attempted is False and report.execution.requests_sent == 0
    assert report.execution.transport_kind == "none"
    assert report.evidence == [] and report.lab_state.tested_state == "not_tested"
    assert report.scope.execution_permitted is False


# ------------------------------------------------------------
# a fully in-order request, a fake lab: the three conclusions
# ------------------------------------------------------------

def test_a_vulnerable_simulated_lab_gives_objective_achieved():
    report, transport = run_fake(vulnerable=True)
    assert (report.status, report.reason_code) == ("objective_achieved", "objective_observed")
    assert report.execution.attempted and report.execution.completed and report.execution.requests_sent == 2
    assert [r.role for r in transport.requests] == ["control", "probe"]
    assert report.lab_state.tested_state == "post_remediation"
    assert report.lab_state.post_remediation_state_id == "STATE-POST-1"
    assert report.target.lab_identity_verified is True and report.target.requested_origin == "http://127.0.0.1:8080"


def test_a_patched_simulated_lab_gives_objective_not_observed_under_tested_conditions():
    report, _ = run_fake(vulnerable=False)
    assert (report.status, report.reason_code) == (
        "objective_not_observed", "objective_not_observed_under_tested_conditions")
    assert report.remediation_effectiveness_proven is False and report.general_security_guarantee is False


def test_every_conclusive_result_evaluated_every_condition():
    for vulnerable in (True, False):
        report, _ = run_fake(vulnerable=vulnerable)
        assert all(c.result != "not_evaluated" for c in report.oracle.conditions)


# ------------------------------------------------------------
# a fake run is a simulation and can never be evidence
# ------------------------------------------------------------

@pytest.mark.parametrize("vulnerable", [True, False])
def test_a_fake_run_is_labelled_a_simulation_and_is_not_usable_for_phase_14(vulnerable):
    report, _ = run_fake(vulnerable=vulnerable)
    assert report.execution.transport_kind == "fake"
    assert SIMULATION_STATEMENT in report.limitations and SIMULATION_STATEMENT in report.validity_conditions
    assert report.validation_run_id.endswith("-fake")
    handoff = build_phase14_handoff(report)
    assert handoff.transport_kind == "fake" and handoff.usable_as_post_remediation_evidence is False
    assert handoff.check_attempted and handoff.check_completed


def test_the_report_without_the_simulation_statement_is_rejected_by_the_contract():
    report, _ = run_fake(vulnerable=True)
    data = report.model_dump(mode="json")
    data["limitations"] = [t for t in data["limitations"] if "SIMULATION" not in t]
    data["validity_conditions"] = [t for t in data["validity_conditions"] if "SIMULATION" not in t]
    with pytest.raises(ValueError, match="simulation"):
        ValidationReport.model_validate(data)


def test_the_report_round_trips_through_the_versioned_contract_and_is_deterministic():
    first, _ = run_fake(vulnerable=True)
    second, _ = run_fake(vulnerable=True)
    assert ValidationReport.model_validate(first.model_dump(mode="json")) == first
    assert serialize(first) == serialize(second)
    # 13B.3 migration: a report that ran a check is revision 1.1 and records how the identity was established
    assert first.contract_version == "1.1" and first.identity is not None
    assert first.identity.live_suitable is False and first.identity.authoritative_registry is False
    assert first.llm_used is False and first.ground_truth_used is False


def test_the_two_transport_kinds_get_different_run_ids():
    fake_report, _ = run_fake(vulnerable=True)
    blocked = run_controlled_validation(full_request(), approved_registry(), full_policy(False),
                                        ScriptedTransport(), clock=fixed_clock())
    assert fake_report.validation_run_id != blocked.validation_run_id


# ------------------------------------------------------------
# evidence is a summary, never a raw body
# ------------------------------------------------------------

def test_evidence_holds_status_hash_length_and_counts_but_no_body():
    report, _ = run_fake(vulnerable=True)
    assert [e.kind for e in report.evidence] == ["control_exchange", "probe_exchange"]
    control, probe = report.evidence
    assert control.attributes["http_status"] == "200" and len(control.attributes["body_sha256"]) == 64
    assert probe.attributes["probe_record_count"] == "3" and probe.attributes["records_outside_literal_match"] == "3"
    text = serialize(report)
    assert "first note" not in text and "hello world" not in text, "raw response bodies are never stored"


def test_the_clock_is_injectable_so_timestamps_are_deterministic():
    report, _ = run_fake(vulnerable=False)
    assert report.execution.started_at_utc == "2026-10-16T12:00:00Z"
    assert report.execution.finished_at_utc == "2026-10-16T12:00:01Z"
    assert all(e.captured_at_utc == "2026-10-16T12:00:01Z" for e in report.evidence)


def test_the_default_clock_produces_a_valid_report():
    transport = ScriptedTransport(simulated_lab(False), simulated_lab(False))
    fresh = identity(observed_at=datetime.now(timezone.utc))  # the real clock is used, so the observation is current
    report = run_controlled_validation(full_request(lab_identity=fresh), approved_registry(), full_policy(), transport)
    assert report.status == "objective_not_observed" and report.execution.started_at_utc.endswith("Z")


# ------------------------------------------------------------
# failure handling: always inconclusive, never a conclusion
# ------------------------------------------------------------

@pytest.mark.parametrize("probe_answer,reason", [
    (TransportResponse.timeout(), "timeout"),
    (TransportResponse.unreachable(), "lab_unreachable"),
    (TransportResponse.error(), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=500), "unexpected_response"),
    (TransportResponse.ok("<html>waf block</html>"), "unexpected_response"),
    (TransportResponse.ok('{"data": "nope"}'), "unexpected_response"),
])
def test_a_failing_probe_is_inconclusive_and_incomplete(probe_answer, reason):
    report, transport = run_fake(ScriptedTransport(EMPTY, probe_answer))
    assert (report.status, report.reason_code) == ("inconclusive", reason)
    assert report.execution.attempted and report.execution.requests_sent == 2 and len(transport.requests) == 2
    # "completed" only says that every intended request got an HTTP answer; it never implies a conclusion
    assert report.execution.completed is (probe_answer.outcome == "response")
    assert build_phase14_handoff(report).usable_as_post_remediation_evidence is False


@pytest.mark.parametrize("control_answer,reason", [
    (TransportResponse.timeout(), "timeout"),
    (TransportResponse.unreachable(), "lab_unreachable"),
    (TransportResponse.error(), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=503), "control_check_failed"),
    (TransportResponse.ok("garbage"), "control_check_failed"),
])
def test_a_failing_control_means_the_attack_shaped_probe_is_never_sent(control_answer, reason):
    transport = ScriptedTransport(control_answer)  # a second send would raise ScriptExhausted
    report, _ = run_fake(transport)
    assert (report.status, report.reason_code) == ("inconclusive", reason)
    assert [r.role for r in transport.requests] == ["control"]
    assert report.execution.requests_sent == 1 and report.execution.completed is False
    assert [e.kind for e in report.evidence] == ["control_exchange"]
    assert report.oracle.conditions[0].result == "not_met"
    assert all(c.result == "not_evaluated" for c in report.oracle.conditions[1:])


@pytest.mark.parametrize("script", [
    (RuntimeError("boom"),),
    (EMPTY, ValueError("boom")),
    ("not a response",),
    (EMPTY, None),
    (EMPTY, {"outcome": "response", "status_code": 200, "body": "{}"}),
    (EMPTY, lambda request: TransportResponse.ok("x" * 200_000)),  # a body over the cap can not even be built
])
def test_a_misbehaving_transport_is_inconclusive_not_a_crash_and_not_a_conclusion(script):
    report, _ = run_fake(ScriptedTransport(*script))
    assert report.status == "inconclusive"
    assert report.reason_code in ("unexpected_response", "control_check_failed")
    assert report.execution.completed is False
    assert report.remediation_effectiveness_proven is False


def test_running_out_of_scripted_answers_is_a_loud_test_error_not_a_verdict():
    with pytest.raises(ScriptExhausted):
        run_fake(ScriptedTransport(EMPTY))


def test_an_inconclusive_result_is_never_usable_by_phase_14():
    report, _ = run_fake(ScriptedTransport(EMPTY, TransportResponse.timeout()))
    assert build_phase14_handoff(report).usable_as_post_remediation_evidence is False


class RaisingClassifier(SqliSearchClassifier):
    def classify(self, scenario, control, probe):
        raise RuntimeError("classifier bug")


class OverreachingClassifier(SqliSearchClassifier):
    """Claims a conclusion without having evaluated every condition."""

    def classify(self, scenario, control, probe):
        base = super().classify(scenario, control, probe)
        conditions = [c.model_copy(update={"result": "not_evaluated"}) if c.condition_id == "probe_response_parseable"
                      else c for c in base.conditions]
        return OracleOutcome(status="objective_not_observed",
                             reason_code="objective_not_observed_under_tested_conditions", conditions=conditions)


class WrongConditionsClassifier(SqliSearchClassifier):
    def classify(self, scenario, control, probe):
        return OracleOutcome(status="objective_achieved", reason_code="objective_observed", conditions=[])


class WrongReasonClassifier(SqliSearchClassifier):
    def classify(self, scenario, control, probe):
        base = super().classify(scenario, control, probe)
        return OracleOutcome(status="inconclusive", reason_code="objective_observed", conditions=base.conditions)


@pytest.mark.parametrize("classifier", [RaisingClassifier, OverreachingClassifier, WrongConditionsClassifier,
                                        WrongReasonClassifier])
def test_an_unsound_classifier_result_is_downgraded_to_inconclusive_insufficient_evidence(classifier):
    report, _ = run_fake(vulnerable=False, classifiers=ClassifierRegistry(classifier()))
    assert (report.status, report.reason_code) == ("inconclusive", "evidence_insufficient")


# ------------------------------------------------------------
# preconditions: nothing is sent unless everything is in order
# ------------------------------------------------------------

def test_the_built_in_scenario_is_not_eligible_so_nothing_is_ever_sent_for_it():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, registry=default_registry())
    assert (report.status, report.reason_code) == ("not_validated", "scenario_not_eligible")
    assert_nothing_was_sent(report, transport)
    assert report.scenario.eligible_for_execution is False
    assert len(report.scenario.ambiguity_codes) == 5


def test_an_approved_scenario_without_a_registered_classifier_is_not_run():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, classifiers=ClassifierRegistry())
    assert (report.status, report.reason_code) == ("not_validated", "scenario_not_eligible")
    assert "classifier" in next(p for p in report.preconditions if p.precondition == "scenario_eligible").detail
    assert_nothing_was_sent(report, transport)


@pytest.mark.parametrize("plan", [None, "nonsense", {}, 7])
def test_a_missing_or_unusable_plan_means_not_validated(plan):
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, request=full_request(plan_result=plan, application=None))
    assert report.status == "not_validated"
    assert_nothing_was_sent(report, transport)


def test_a_proposed_remediation_is_not_an_applied_one_without_an_attestation_nothing_is_tested():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, request=full_request(application=None))
    assert (report.status, report.reason_code) == ("not_validated", "remediation_application_not_attested")
    assert report.remediation.application.state != "independently_attested"
    assert report.lab_state.post_remediation_state_id is None
    assert_nothing_was_sent(report, transport)


def test_an_attestation_for_proposals_the_plan_never_made_is_rejected():
    result = planned_result()
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    bad = attestation_for(result, applied_proposal_ids=["PROP-DOES-NOT-EXIST"])
    report, _ = run_fake(transport, request=full_request(result, application=bad))
    assert (report.status, report.reason_code) == ("not_validated", "remediation_application_not_attested")
    assert_nothing_was_sent(report, transport)


@pytest.mark.parametrize("overrides,expected_code", [
    ({"lab_identity": None}, "lab_identity_missing"),
    ({"authorization": None}, "operator_authorization_missing"),
    ({"target_url": None}, "target_missing"),
    ({"target_url": "http://localhost:8080"}, "target_not_approved"),
    ({"target_url": "http://127.0.0.1:9999"}, "target_not_approved"),
    ({"target_url": "http://example.com:8080"}, "target_not_approved"),
    ({"authorization": authorization(live_execution_authorized=False)}, "operator_authorization_not_live"),
    ({"authorization": authorization(lab_id="other-lab")}, "operator_authorization_mismatch"),
    ({"authorization": authorization(incident_id="incident_002")}, "operator_authorization_mismatch"),
    ({"lab_identity": identity(established_independently=False)}, "lab_identity_not_independent"),
    ({"lab_identity": identity(observed_network="other_net")}, "lab_identity_mismatch"),
    ({"lab_identity": identity(compose_sha256="d" * 64)}, "lab_identity_mismatch"),
])
def test_rejected_scope_inputs_block_the_run_and_send_nothing(overrides, expected_code):
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, request=full_request(**overrides))
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert_nothing_was_sent(report, transport)
    assert expected_code in report.scope.reason_codes


@pytest.mark.parametrize("state,expected_code", [
    ("STATE-SOMETHING-ELSE", "lab_state_mismatch"),
    ("STATE-PRE-1", "lab_state_mismatch"),  # the lab still looks pre-remediation
    (None, "lab_state_unverified"),
])
def test_the_independently_observed_lab_state_must_equal_the_attested_post_remediation_state(state, expected_code):
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, request=full_request(lab_identity=identity(lab_state_id=state)))
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert expected_code in report.scope.reason_codes
    assert_nothing_was_sent(report, transport)


def test_a_configured_localhost_url_alone_authorizes_nothing():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    request = full_request(lab_identity=None, authorization=None, application=None)
    report = run_controlled_validation(request, approved_registry(), ScopePolicy(), transport, clock=fixed_clock())
    assert report.status == "not_validated"
    assert transport.requests == []
    report = run_controlled_validation(full_request(lab_identity=None, authorization=None), approved_registry(),
                                       None, transport, clock=fixed_clock())
    assert report.status == "blocked" and transport.requests == []


def test_the_default_policy_denies_everything():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report = run_controlled_validation(full_request(), approved_registry(), None, transport, clock=fixed_clock())
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert "no_approved_lab" in report.scope.reason_codes and transport.requests == []


def test_the_policys_own_execution_switch_is_an_independent_lock():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, policy=full_policy(execution_enabled=False))
    assert (report.status, report.reason_code) == ("blocked", "execution_not_enabled")
    assert report.scope.scope_conditions_met is True and report.scope.execution_permitted is False
    assert_nothing_was_sent(report, transport)


class LooksLiveTransport:
    """Declares itself live. It must never be called while the live lock is closed."""

    kind = "live"
    transport_id = "pretend-live"

    def __init__(self) -> None:
        self.requests: list = []

    def send(self, request):
        self.requests.append(request)
        raise AssertionError("a live transport must never be called in Phase 13B.1")


def test_a_live_transport_is_refused_while_the_live_lock_is_closed():
    assert LIVE_TRANSPORT_ENABLED is False
    transport = LooksLiveTransport()
    report, _ = run_fake(transport)
    assert (report.status, report.reason_code) == ("blocked", "live_transport_not_enabled")
    assert_nothing_was_sent(report, transport)


@pytest.mark.parametrize("transport", [None, object(), "http://127.0.0.1:8080", 5])
def test_a_missing_or_invalid_transport_blocks_the_run(transport):
    report = run_controlled_validation(full_request(), approved_registry(), full_policy(), transport,
                                       clock=fixed_clock())
    assert (report.status, report.reason_code) == ("blocked", "transport_unavailable")
    assert report.execution.attempted is False


def test_a_transport_with_an_unknown_kind_is_not_accepted():
    class Odd:
        kind = "simulated-live"
        transport_id = "odd"

        def send(self, request):
            raise AssertionError("must not be called")

    report = run_controlled_validation(full_request(), approved_registry(), full_policy(), Odd(),
                                       clock=fixed_clock())
    assert (report.status, report.reason_code) == ("blocked", "transport_unavailable")


def test_the_blocking_order_is_inputs_then_scope_then_transport_then_live_lock_then_the_switch():
    live = LooksLiveTransport()
    # inputs first: no attestation outranks everything else
    report = run_controlled_validation(full_request(application=None), approved_registry(),
                                       full_policy(False), live, clock=fixed_clock())
    assert report.reason_code == "remediation_application_not_attested"
    # scope next
    report = run_controlled_validation(full_request(authorization=None), approved_registry(),
                                       full_policy(False), live, clock=fixed_clock())
    assert report.reason_code == "scope_policy_denied"
    # then an invalid transport, then the live lock, then the policy switch
    report = run_controlled_validation(full_request(), approved_registry(), full_policy(False), None,
                                       clock=fixed_clock())
    assert report.reason_code == "transport_unavailable"
    report = run_controlled_validation(full_request(), approved_registry(), full_policy(False), live,
                                       clock=fixed_clock())
    assert report.reason_code == "live_transport_not_enabled"
    report = run_controlled_validation(full_request(), approved_registry(), full_policy(False),
                                       ScriptedTransport(), clock=fixed_clock())
    assert report.reason_code == "execution_not_enabled"
    assert live.requests == []


def test_a_blocked_runner_report_still_conforms_to_the_contract_and_handoff():
    report = run_controlled_validation(full_request(), approved_registry(), full_policy(False),
                                       ScriptedTransport(), clock=fixed_clock())
    assert ValidationReport.model_validate(report.model_dump(mode="json")) == report
    handoff = build_phase14_handoff(report)
    assert handoff.usable_as_post_remediation_evidence is False and handoff.transport_kind == "none"
    assert "No check ran for this report" in " ".join(report.limitations)


def test_a_full_fake_run_creates_no_socket_resolves_no_name_and_starts_no_process(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network or process access attempted")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "gethostbyname", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)
    for vulnerable in (True, False):
        report, _ = run_fake(vulnerable=vulnerable)
        assert report.execution.transport_kind == "fake"
    report, _ = run_fake(ScriptedTransport(TransportResponse.timeout()))
    assert report.status == "inconclusive"


def test_the_phase_13a_orchestrator_still_refuses_to_execute_even_for_an_approved_scenario():
    from validation.orchestrator import build_validation_report
    report = build_validation_report(full_request(), approved_registry(), full_policy())
    assert (report.status, report.reason_code) == ("blocked", "execution_not_implemented_phase_13a")
    assert report.execution.attempted is False


def test_a_malformed_request_object_is_a_programming_error():
    with pytest.raises(ValidationRequestError):
        run_controlled_validation("not a request", approved_registry(), full_policy(), ScriptedTransport())
    with pytest.raises(ValidationRequestError):
        run_controlled_validation(None)
