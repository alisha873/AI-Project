"""Pre-remediation observation mode. Offline only: fake transports, no network, no approvals.

`control_value` is reviewed procedure metadata. The Incident 1 sequence sends the three baseline
searches and, only when they are healthy, the probe. It does not send the control value. The built-in
scenario stays ineligible until reviewers resolve that field and the other blocking ambiguities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from judge.agent import JudgeAgent
from judge.contract import CLAIM_APPLICATION_SECURE, CLAIM_REMEDIATION_APPLIED
from scripts import run_controlled_validation as cli
from tests.validation_helpers import (
    INCIDENT,
    TARGET,
    SearchLabTransport,
    approved_registry,
    approved_registry_with_baselines,
    attestation_for,
    authorization,
    fixed_clock,
    full_policy,
    identity,
    model_insufficient_result,
    planned_result,
    run_fake_baselines,
)
from validation import seals
from validation.comparison import compare_baseline_and_post
from validation.contract import ValidationReport
from validation.evidence_gate import observation_gate_failures
from validation.handoff import build_phase14_handoff
from validation.orchestrator import ValidationRequest
from validation.runner import run_controlled_validation, run_pre_remediation_observation
from validation.scenario import procedure_digest, sqli_search_scenario
from validation.transport import TransportResponse
CONTROL_VALUE = "zzqx-no-such-record"
PROBE_QUERY = "message=%27%20OR%20%271%27%3D%271"


class _Recording:
    def __init__(self, kind="fake"):
        self.kind = kind
        self.transport_id = f"recording-{kind}"
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        raise AssertionError("no request may be sent")


def _request(**overrides) -> ValidationRequest:
    data = {
        "incident_id": INCIDENT,
        "scenario_id": "sqli-search-incident-001",
        "target_url": TARGET,
        "lab_identity": identity(lab_state_id="STATE-PRE-OBS"),
        "authorization": authorization(),
    }
    data.update(overrides)
    return ValidationRequest(**data)


def _run(transport=None, **overrides):
    transport = transport if transport is not None else SearchLabTransport(True)
    report = run_pre_remediation_observation(
        _request(**overrides), approved_registry_with_baselines(), full_policy(), transport,
        clock=fixed_clock(),
    )
    return report, transport


def test_control_value_is_procedure_metadata_and_the_real_scenario_stays_ineligible():
    scenario = sqli_search_scenario()
    assert scenario.probe.control_value is None
    assert scenario.eligible_for_execution is False
    current = procedure_digest(scenario)
    changed = scenario.model_dump()
    changed["probe"]["control_value"] = "a-reviewer-must-choose-this"
    assert procedure_digest(changed) != current


def test_a_successful_observation_sends_baselines_then_the_probe_and_not_the_control_value():
    report, transport = _run(SearchLabTransport(True))
    queries = [item.query for item in transport.requests]
    assert queries == ["message=hello", "message=security", "message=autoshield", PROBE_QUERY]
    assert CONTROL_VALUE not in "".join(queries)
    assert [item.attack_shaped for item in transport.requests] == [False, False, False, True]
    assert report.report_kind == "pre_remediation_observation"
    assert report.status == "objective_achieved"
    assert report.lab_state.tested_state == "pre_remediation"
    assert report.lab_state.pre_remediation_state_id == "STATE-PRE-OBS"
    assert report.lab_state.post_remediation_state_id is None
    assert report.remediation.proposals == []
    assert report.remediation.application.state == "not_applicable"
    assert report.upstream.plan_result_status is None
    assert report.execution.attempted is True and report.execution.requests_sent == 4
    assert seals.is_report_issued(seals.report_key(report.model_dump(mode="json"))) is True
    handoff = build_phase14_handoff(report)
    assert handoff.usable_as_post_remediation_evidence is False
    assert handoff.usable_as_pre_remediation_observation is False  # fake transport is a simulation


def test_the_judge_does_not_treat_a_fake_observation_as_remediation_evidence():
    report, _ = _run(SearchLabTransport(False))
    assert report.status == "objective_not_observed"
    agent = JudgeAgent(registry=approved_registry_with_baselines(), clock=fixed_clock())
    decision = agent.evaluate(report)
    assert decision.evidence_provenance == "mocked"
    assert decision.decision == "supported"
    assert decision.usable_as_post_remediation_evidence is False
    assert decision.usable_as_pre_remediation_observation is False
    assert decision.live_evidence is False and decision.eligible_for_downstream is False
    assert decision.remediation_effectiveness_proven is False
    secure = agent.evaluate(report, claim_id=CLAIM_APPLICATION_SECURE)
    applied = agent.evaluate(report, claim_id=CLAIM_REMEDIATION_APPLIED)
    assert secure.decision == "not_supported" and applied.decision == "not_supported"
    assert secure.eligible_for_downstream is False and applied.eligible_for_downstream is False
    assert "pre-remediation" in " ".join(decision.limitations).lower()


def test_comparison_refuses_an_observation_as_remediation_evidence():
    observed, _ = _run(SearchLabTransport(True))
    post, _ = run_fake_baselines(vulnerable=False)
    result = compare_baseline_and_post(observed, post)
    assert result.comparable is False
    assert "observation_is_not_remediation_evidence" in result.not_comparable_reasons
    assert result.usable_as_comparison_evidence is False
    assert result.interpretation == "no_inference_possible"
    assert result.remediation_effectiveness_proven is False


def test_a_failed_baseline_never_sends_the_probe():
    from validation.transport import ScriptedTransport
    transport = ScriptedTransport(
        TransportResponse(outcome="response", status_code=500, body="{}"),
        TransportResponse(outcome="response", status_code=500, body="{}"),
        TransportResponse(outcome="response", status_code=500, body="{}"),
    )
    report, _ = _run(transport)
    assert report.status == "inconclusive"
    assert report.execution.requests_sent == 3
    assert all(item.attack_shaped is False for item in transport.requests)
    assert len(transport.requests) == 3


@pytest.mark.parametrize("overrides,reason", [
    ({"plan_result": planned_result()}, "remediation_claim_present"),
    ({"plan_result": model_insufficient_result()}, "remediation_claim_present"),
    ({"application": attestation_for(planned_result())}, "remediation_claim_present"),
    ({"lab_identity": None}, "scope_policy_denied"),
    ({"authorization": None}, "scope_policy_denied"),
    ({"target_url": "http://127.0.0.1:9090"}, "scope_policy_denied"),
    ({"lab_identity": identity(
        lab_state_id="STATE-STALE-OBS", observed_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )}, "scope_policy_denied"),
])
def test_blocked_observation_inputs_send_zero_requests(overrides, reason):
    transport = _Recording()
    report = run_pre_remediation_observation(
        _request(**overrides), approved_registry_with_baselines(), full_policy(), transport,
        clock=fixed_clock(),
    )
    assert transport.requests == []
    assert report.execution.attempted is False and report.execution.requests_sent == 0
    assert report.reason_code == reason
    assert report.report_kind == "pre_remediation_observation"


def test_the_unapproved_builtin_scenario_and_a_missing_baseline_sequence_send_nothing():
    transport = _Recording()
    pending = run_pre_remediation_observation(
        _request(), clock=fixed_clock(), transport=transport,
    )
    assert transport.requests == []
    assert pending.reason_code == "scenario_not_eligible"
    transport = _Recording()
    missing = run_pre_remediation_observation(
        _request(), approved_registry(), full_policy(), transport, clock=fixed_clock(),
    )
    assert transport.requests == []
    assert missing.reason_code == "observation_baselines_required"


def test_a_live_transport_without_opt_in_or_a_live_suitable_identity_sends_nothing():
    transport = _Recording(kind="live")
    refused = run_pre_remediation_observation(
        _request(), approved_registry_with_baselines(), full_policy(), transport,
        clock=fixed_clock(), live_opt_in=False,
    )
    assert transport.requests == []
    assert refused.reason_code == "live_transport_not_enabled"
    transport = _Recording(kind="live")
    unsuitable = run_pre_remediation_observation(
        _request(), approved_registry_with_baselines(), full_policy(), transport,
        clock=fixed_clock(), live_opt_in=True,
    )
    assert transport.requests == []
    assert unsuitable.execution.attempted is False
    assert "lab_identity_not_live_suitable" in unsuitable.scope.reason_codes


def test_the_post_remediation_path_still_requires_a_plan_and_sends_nothing_without_one():
    transport = _Recording()
    report = run_controlled_validation(
        _request(), approved_registry_with_baselines(), full_policy(), transport, clock=fixed_clock(),
    )
    assert transport.requests == []
    assert report.reason_code == "remediation_plan_missing"
    assert report.report_kind == "post_remediation_validation"


def test_forged_or_contradictory_observation_fields_are_rejected():
    report, _ = _run(SearchLabTransport(True))
    dumped = report.model_dump(mode="json")
    dumped["remediation"]["proposals"] = [{
        "proposal_id": "REM-forged", "priority_rank": 1,
        "observed_evidence_ids": [], "hypothesis_ids": [], "configuration_fact_ids": [],
        "reference_ids": [], "reference_chunk_ids": [],
    }]
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(dumped)
    dumped = report.model_dump(mode="json")
    dumped["remediation"]["application"]["state"] = "independently_attested"
    dumped["remediation"]["application"]["attestation_id"] = "ATT-forged"
    dumped["remediation"]["application"]["attested_by"] = "someone"
    dumped["remediation"]["application"]["applied_proposal_ids"] = ["REM-forged"]
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(dumped)
    dumped = report.model_dump(mode="json")
    dumped["lab_state"]["post_remediation_state_id"] = "STATE-POST-FORGED"
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(dumped)
    dumped = report.model_dump(mode="json")
    dumped["upstream"]["plan_result_status"] = "planned"
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(dumped)
    dumped = report.model_dump(mode="json")
    dumped["report_kind"] = "post_remediation_validation"
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(dumped)


def test_a_saved_report_is_not_observation_evidence_after_issuance_is_cleared(monkeypatch):
    report, _ = _run(SearchLabTransport(True))
    reloaded = ValidationReport.model_validate(json.loads(json.dumps(report.model_dump(mode="json"))))
    monkeypatch.setattr(seals, "_REPORTS", {})
    assert observation_gate_failures(reloaded)
    assert build_phase14_handoff(reloaded).usable_as_pre_remediation_observation is False
    assert build_phase14_handoff(reloaded).usable_as_post_remediation_evidence is False
    decision = JudgeAgent(registry=approved_registry_with_baselines(), clock=fixed_clock()).evaluate(reloaded)
    assert decision.evidence_provenance == "mocked"
    assert decision.live_evidence is False and decision.eligible_for_downstream is False


def test_a_hand_edited_live_transport_flag_is_not_observation_evidence():
    report, _ = _run(SearchLabTransport(True))
    dumped = report.model_dump(mode="json")
    dumped["execution"]["transport_kind"] = "live"
    edited = ValidationReport.model_validate(dumped)
    assert observation_gate_failures(edited)
    decision = JudgeAgent(registry=approved_registry_with_baselines(), clock=fixed_clock()).evaluate(edited)
    assert decision.evidence_provenance == "unverified"
    assert decision.usable_as_post_remediation_evidence is False
    assert decision.usable_as_pre_remediation_observation is False
    assert decision.decision == "inconclusive"
    assert decision.eligible_for_downstream is False


def test_a_stale_approval_digest_cannot_make_the_builtin_scenario_eligible():
    data = sqli_search_scenario().model_dump()
    for item in data["ambiguities"]:
        item["resolved"] = True
    data["probe"]["wire_encoding"] = "percent_encoded_utf8_query"
    data["probe"]["control_value"] = CONTROL_VALUE
    data["review"]["status"] = "approved"
    data["review"]["reviewer"] = "lead-reviewer"
    digest = procedure_digest(data)
    data["review"]["approvals"] = [
        {
            "name": name, "approved": True, "confirms_procedure": True, "date": "2026-10-16",
            "scenario_id": data["scenario_id"], "procedure_sha256": digest,
        }
        for name in ("Alisha Ranjan Kumar", "Jihan Chhabra")
    ]
    data["review"]["approvals"][1]["procedure_sha256"] = "0" * 64
    from validation.scenario import ScenarioDefinition
    with pytest.raises(ValueError, match="different procedure digest"):
        ScenarioDefinition.model_validate(data)
    assert sqli_search_scenario().review.approvals == []


def test_the_cli_observation_flag_sends_nothing(tmp_path, monkeypatch):
    sent = []

    class Recording:
        kind = "live"
        transport_id = "cli-live"

        def send(self, request):
            sent.append(request)
            raise AssertionError("cli must not send")

    monkeypatch.setattr("validation_live.http_transport.LiveHttpTransport", lambda *a, **k: Recording())
    monkeypatch.setenv("AUTOSHIELD_LIVE_VALIDATION", "1")
    output = tmp_path / "observation.json"
    code = cli.main([
        "incident_001", "--pre-remediation-observation", "--live",
        "--i-accept-authorized-live-validation", "--target", TARGET, "--output", str(output),
    ])
    assert code == 3
    assert sent == []
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution"]["attempted"] is False
    assert payload["execution"]["requests_sent"] == 0
    assert payload["report_kind"] == "pre_remediation_observation"
    assert payload["execution_mode"] == "observation-blocked"


@pytest.mark.parametrize("extra,env_value", [
    ([], ""),
    (["--live"], "1"),
    (["--live", "--i-accept-authorized-live-validation"], ""),
])
def test_incomplete_observation_opt_in_constructs_no_transport(tmp_path, monkeypatch, extra, env_value):
    constructed = []

    class Boom:
        def __init__(self, *args, **kwargs):
            constructed.append(args)
            raise AssertionError("transport must not be constructed")

        def send(self, request):
            raise AssertionError("cli must not send")

    monkeypatch.setattr("validation_live.http_transport.LiveHttpTransport", Boom)
    monkeypatch.setenv("AUTOSHIELD_LIVE_VALIDATION", env_value)
    output = tmp_path / "observation.json"
    code = cli.main([
        "incident_001", "--pre-remediation-observation", *extra, "--target", TARGET, "--output", str(output),
    ])
    assert code == 3
    assert constructed == []
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution"]["attempted"] is False
    assert payload["execution"]["requests_sent"] == 0
    assert payload["execution_mode"] == "observation-refused-opt-in"
    assert payload["report_kind"] == "pre_remediation_observation"
