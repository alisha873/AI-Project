"""Phase 14: deterministic judge decisions from saved validation evidence.

No network, model, Docker, or live probe. Fake-transport reports are mocked assessments.
"""

from __future__ import annotations

import ast
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence.schema import JudgeDecision
from judge.agent import JudgeAgent, JudgeInputError
from judge.contract import (
    CLAIM_APPLICATION_SECURE,
    CLAIM_OBJECTIVE_NOT_OBSERVED,
    CLAIM_OBJECTIVE_OBSERVED,
    CLAIM_REMEDIATION_APPLIED,
    DECISIONS,
    JudgeAssessment,
    serialize,
    to_pipeline_decision,
)
from scripts.run_judge import main
from tests.validation_helpers import (
    approved_registry,
    blocked_report,
    executed_dict,
    gate_without_identity_rules,
    llm_error_result,
    run_fake,
)
from validation.contract import ValidationReport
from validation.handoff import build_phase14_handoff
from validation.orchestrator import ValidationRequest, build_validation_report
from validation.scenario import SQLI_SCENARIO_ID, default_registry

ROOT = Path(__file__).resolve().parents[1]
WHEN = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
ACHIEVED = ("objective_achieved", "objective_observed")
NOT_OBSERVED = ("objective_not_observed", "objective_not_observed_under_tested_conditions")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("the judge tests must not open a socket")

    monkeypatch.setattr(socket, "socket", boom)
    monkeypatch.setattr(socket, "create_connection", boom)


def clock():
    return WHEN


def judged_fake(*, vulnerable: bool):
    registry = approved_registry()
    report, _transport = run_fake(vulnerable=vulnerable, registry=registry)
    return report, JudgeAgent(registry=registry, clock=clock)


def test_valid_mocked_evidence_can_support_the_narrow_observed_claim():
    report, agent = judged_fake(vulnerable=True)
    result = agent.evaluate(report)
    assert result.decision == "supported"
    assert result.claim_id == CLAIM_OBJECTIVE_OBSERVED
    assert result.evidence_sufficiency == "sufficient"
    assert result.supporting_evidence_ids
    assert result.evidence_provenance == "mocked"
    assert result.live_evidence is False and result.eligible_for_downstream is False
    assert result.usable_as_post_remediation_evidence is False
    assert "mocked_simulation" in result.reason_codes
    assert result.security_guarantee is False and result.general_security_guarantee is False
    assert any("not a live security finding" in item for item in result.limitations)
    handoff = build_phase14_handoff(report)
    assert result.handoff_version == handoff.handoff_version
    assert result.upstream_status == report.status
    assert set(result.supporting_evidence_ids) <= {item.evidence_id for item in result.evidence_index}


def test_valid_evidence_that_shows_the_objective_does_not_support_the_opposite_claim():
    report, agent = judged_fake(vulnerable=True)
    result = agent.evaluate(report, claim_id=CLAIM_OBJECTIVE_NOT_OBSERVED)
    assert result.decision == "not_supported"
    assert result.contradicting_evidence_ids
    assert result.supporting_evidence_ids == []
    assert result.evidence_provenance == "mocked"
    assert result.eligible_for_downstream is False
    assert "claim_not_supported_by_evidence" in result.reason_codes


def test_objective_not_observed_is_supported_only_as_a_narrow_claim_and_not_as_security():
    report, agent = judged_fake(vulnerable=False)
    result = agent.evaluate(report)
    assert report.status == "objective_not_observed"
    assert result.decision == "supported"
    assert result.claim_id == CLAIM_OBJECTIVE_NOT_OBSERVED
    assert "does not show that the application is secure" in result.claim
    assert any("not proof that the application is secure" in item for item in result.limitations)
    secure = agent.evaluate(report, claim_id=CLAIM_APPLICATION_SECURE)
    assert secure.decision == "not_supported"
    assert secure.security_guarantee is False
    assert secure.remediation_effectiveness_proven is False
    assert "security_claim_not_supported" in secure.reason_codes


def test_a_missing_or_malformed_report_is_not_evaluated_and_does_not_echo_secrets():
    agent = JudgeAgent(clock=clock)
    missing = agent.evaluate_payload(None)
    assert missing.decision == "not_evaluated" and missing.reason_codes == ["report_missing"]
    assert missing.evidence_provenance == "unverified" and missing.eligible_for_downstream is False
    malformed = agent.evaluate_payload({"status": "objective_achieved", "incident_id": "incident_001",
                                        "authorization": "Bearer super-secret-token"})
    assert malformed.decision == "not_evaluated"
    assert malformed.incident_id == "incident_001"
    assert "report_invalid" in malformed.reason_codes
    assert "super-secret-token" not in serialize(malformed)
    assert agent.evaluate_payload(["not", "a", "report"]).decision == "not_evaluated"


def test_dangling_evidence_ids_are_rejected_and_not_cited_as_support():
    report, agent = judged_fake(vulnerable=True)
    result = agent.evaluate(report, cited_evidence_ids=["DOES-NOT-EXIST"])
    assert result.decision == "inconclusive"
    assert "dangling_evidence" in result.reason_codes
    assert "DOES-NOT-EXIST" not in result.supporting_evidence_ids
    assert "DOES-NOT-EXIST" not in result.contradicting_evidence_ids
    assert all(item.evidence_id != "DOES-NOT-EXIST" for item in result.evidence_index)
    assert any("DOES-NOT-EXIST" in item for item in result.missing_evidence)


def test_contradictory_condition_results_are_not_resolved_in_favour_of_the_recorded_status():
    data = executed_dict(*ACHIEVED)
    for condition in data["oracle"]["conditions"]:
        if condition["condition_id"] == "record_outside_literal_match":
            condition["result"] = "not_met"
    report = ValidationReport.model_validate(data)
    result = JudgeAgent(registry=approved_registry(), clock=clock).evaluate(report)
    assert result.decision == "inconclusive"
    assert "rules_conflict" in result.reason_codes
    assert result.decision != "supported"


def test_disagreeing_observations_are_kept_as_a_contradiction():
    registry = approved_registry()
    report, _transport = run_fake(vulnerable=True, registry=registry)
    data = report.model_dump(mode="json")
    extra = dict(data["evidence"][-1])
    extra["observation_id"] = extra["observation_id"] + "-conflict"
    extra["attributes"] = dict(extra["attributes"], http_status="500")
    data["evidence"].append(extra)
    conflicted = ValidationReport.model_validate(data)
    result = JudgeAgent(registry=registry, clock=clock).evaluate(conflicted)
    assert result.decision == "inconclusive"
    assert "observations_contradict" in result.reason_codes
    assert result.live_evidence is False


def test_a_blocked_report_is_blocked_and_is_not_a_successful_remediation():
    result = JudgeAgent(clock=clock).evaluate(blocked_report())
    assert result.decision == "blocked"
    assert result.evidence_provenance == "dry_run"
    assert result.eligible_for_downstream is False and result.live_evidence is False
    assert result.supporting_evidence_ids == []
    assert "blocked_execution" in result.reason_codes
    assert any("not a successful remediation" in result.rationale for _ in (0,))
    assert "blocked" in result.rationale or "blocked" == result.decision


def test_not_validated_and_a_dry_run_wrapper_are_not_evaluated():
    report = build_validation_report(ValidationRequest("incident_001", SQLI_SCENARIO_ID))
    result = JudgeAgent(clock=clock).evaluate(report)
    assert report.status == "not_validated"
    assert result.decision == "not_evaluated"
    assert result.evidence_provenance == "dry_run"
    payload = report.model_dump(mode="json")
    payload["execution_mode"] = "dry-run"
    wrapped = JudgeAgent(clock=clock).evaluate_payload(payload)
    assert wrapped.decision == "not_evaluated"
    assert wrapped.execution_mode == "dry-run"
    assert wrapped.evidence_provenance == "dry_run"
    assert wrapped.live_evidence is False


def test_a_provider_error_plan_is_not_a_validation_result():
    request = ValidationRequest("incident_001", SQLI_SCENARIO_ID, plan_result=llm_error_result())
    report = build_validation_report(request)
    result = JudgeAgent(clock=clock).evaluate(report)
    assert result.decision == "not_evaluated"
    assert result.upstream_plan_status == "llm_error"
    assert "provider error" in result.rationale.lower() or "not a validation result" in result.rationale.lower()


def test_an_unconfirmed_scenario_definition_is_inconclusive_rather_than_copied_through():
    report, _transport = run_fake(vulnerable=True, registry=approved_registry())
    result = JudgeAgent(registry=default_registry(), clock=clock).evaluate(report)
    assert result.decision == "inconclusive"
    assert "definition_unconfirmed" in result.reason_codes
    assert result.evidence_provenance == "mocked"
    assert result.live_evidence is False


def test_a_conclusive_report_with_no_objective_is_inconclusive():
    data = executed_dict(*ACHIEVED)
    data["objective"] = None
    report = ValidationReport.model_validate(data)
    result = JudgeAgent(registry=approved_registry(), clock=clock).evaluate(report)
    assert result.decision == "inconclusive"
    assert "definition_unconfirmed" in result.reason_codes


def test_mocked_evidence_is_never_labelled_live_even_if_the_identity_gate_is_stubbed(monkeypatch):
    gate_without_identity_rules(monkeypatch)
    registry = approved_registry()
    report, _transport = run_fake(vulnerable=True, registry=registry)
    result = JudgeAgent(registry=registry, clock=clock).evaluate(report)
    assert result.evidence_provenance == "mocked"
    assert result.live_evidence is False
    assert result.eligible_for_downstream is False
    assert result.decision == "supported"


def test_a_hand_built_live_report_is_not_live_until_the_evidence_gate_passes():
    report = ValidationReport.model_validate(executed_dict(*ACHIEVED))
    result = JudgeAgent(registry=approved_registry(), clock=clock).evaluate(report)
    assert result.evidence_provenance == "unverified"
    assert result.live_evidence is False
    assert result.decision == "inconclusive"
    assert result.eligible_for_downstream is False
    assert "evidence_gate_unsatisfied" in result.reason_codes


def test_when_the_evidence_gate_passes_a_live_result_is_eligible_for_downstream_use(monkeypatch):
    gate_without_identity_rules(monkeypatch)
    report = ValidationReport.model_validate(executed_dict(*ACHIEVED))
    result = JudgeAgent(registry=approved_registry(), clock=clock).evaluate(report)
    assert result.evidence_provenance == "live"
    assert result.live_evidence is True
    assert result.decision == "supported"
    assert result.eligible_for_downstream is True
    assert result.usable_as_post_remediation_evidence is True
    pipeline = to_pipeline_decision(result)
    assert pipeline.live_evidence is True and pipeline.eligible_for_downstream is True
    assert pipeline.evidence_provenance == "live"


def test_a_proposed_remediation_is_never_treated_as_executed_remediation():
    report, agent = judged_fake(vulnerable=True)
    result = agent.evaluate(report, claim_id=CLAIM_REMEDIATION_APPLIED)
    assert result.decision == "inconclusive"
    assert result.remediation_application_verified is False
    assert result.decision != "supported"
    proposal_ids = {item.evidence_id for item in result.evidence_index if item.role == "proposal"}
    assert proposal_ids
    assert proposal_ids.isdisjoint(result.supporting_evidence_ids)
    blocked = JudgeAgent(clock=clock).evaluate(blocked_report(), claim_id=CLAIM_REMEDIATION_APPLIED)
    assert blocked.decision == "blocked"
    assert blocked.supporting_evidence_ids == []
    hypotheses = [item for item in result.evidence_index if item.role == "hypothesis"]
    assert hypotheses and all(item.used_as == "excluded" for item in hypotheses)
    assert all(item.evidence_id not in result.supporting_evidence_ids for item in hypotheses)


def test_confidence_cannot_override_evidence_and_invalid_confidence_is_rejected():
    blocked = JudgeAgent(clock=clock).evaluate(blocked_report(), caller_confidence=0.99)
    assert blocked.decision == "blocked"
    assert blocked.caller_confidence_ignored is True and blocked.confidence_used is False
    assert to_pipeline_decision(blocked).confidence == 0.0
    report, agent = judged_fake(vulnerable=True)
    rejected = agent.evaluate(report, caller_confidence=1.5)
    assert rejected.decision == "not_evaluated"
    assert rejected.caller_confidence_rejected is True
    assert agent.evaluate(report, caller_confidence=True).decision == "not_evaluated"
    assert agent.evaluate(report, caller_confidence="high").decision == "not_evaluated"
    payload = report.model_dump(mode="json")
    payload["confidence"] = 0.99
    assert agent.evaluate_payload(payload).decision == "not_evaluated"
    with pytest.raises(ValidationError):
        JudgeDecision(decision="blocked", confidence=0.9)
    with pytest.raises(ValidationError):
        JudgeDecision(
            decision="supported", confidence=0.8, evidence_sufficiency="insufficient",
            supporting_evidence=["OBS-1"], evidence_provenance="mocked",
        )


def test_the_assessment_is_versioned_stable_and_round_trips():
    report, agent = judged_fake(vulnerable=True)
    first, second = agent.evaluate(report), agent.evaluate(report)
    assert first == second
    text = serialize(first)
    assert text.endswith("\n")
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)
    assert parsed["schema_version"] == "1.0"
    assert JudgeAssessment.model_validate(parsed) == first
    dumped = first.model_dump()
    dumped["confidence"] = 0.5
    with pytest.raises(ValidationError):
        JudgeAssessment.model_validate(dumped)
    pipeline = to_pipeline_decision(first)
    assert pipeline.schema_version == "1.0"
    assert pipeline.decision == "supported"
    assert pipeline.confidence == 0.0
    assert set(JudgeDecision.model_fields["decision"].annotation.__args__) == set(DECISIONS)


def test_secrets_in_an_observation_are_not_copied_into_the_decision():
    registry = approved_registry()
    report, _transport = run_fake(vulnerable=True, registry=registry)
    data = report.model_dump(mode="json")
    data["evidence"][0]["summary"] = "Authorization: Bearer super-secret-token"
    data["evidence"][0]["attributes"]["authorization"] = "Bearer super-secret-token"
    tainted = ValidationReport.model_validate(data)
    result = JudgeAgent(registry=registry, clock=clock).evaluate(tainted)
    blob = serialize(result)
    assert "super-secret-token" not in blob
    assert "Bearer" not in blob
    assert result.decision == "supported"
    assert result.evidence_provenance == "mocked"


def test_a_non_report_is_refused_by_the_typed_entry_point():
    with pytest.raises(JudgeInputError):
        JudgeAgent(clock=clock).evaluate({"status": "blocked"})  # type: ignore[arg-type]


def test_the_cli_judges_a_saved_dry_run_and_a_missing_file_without_a_network(tmp_path):
    report = build_validation_report(ValidationRequest("incident_001", SQLI_SCENARIO_ID))
    source = tmp_path / "validation_report.json"
    payload = report.model_dump(mode="json")
    payload["execution_mode"] = "dry-run"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "judge_decision.json"
    assert main(["--report", str(source), "--output", str(output)]) == 0
    assessment = JudgeAssessment.model_validate(json.loads(output.read_text(encoding="utf-8")))
    assert assessment.decision == "not_evaluated"
    assert assessment.evidence_provenance == "dry_run"
    assert assessment.network_used is False and assessment.llm_used is False
    assert assessment.live_evidence is False
    missing = tmp_path / "missing.json"
    destination = tmp_path / "missing_judge.json"
    assert main(["--report", str(missing), "--output", str(destination)]) == 0
    assert JudgeAssessment.model_validate(json.loads(destination.read_text(encoding="utf-8"))).decision == "not_evaluated"
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2


def test_judge_modules_do_not_import_a_network_a_model_or_docker():
    forbidden = ("socket", "ssl", "requests", "urllib", "httpx", "docker", "llm", "groq", "openai", "subprocess")
    paths = list((ROOT / "judge").glob("*.py")) + [ROOT / "scripts" / "run_judge.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"


def test_downstream_flags_follow_provenance():
    mocked_report, agent = judged_fake(vulnerable=True)
    mocked = agent.evaluate(mocked_report)
    assert (mocked.eligible_for_downstream, mocked.live_evidence, mocked.evidence_provenance) == (False, False, "mocked")
    dry = JudgeAgent(clock=clock).evaluate(
        build_validation_report(ValidationRequest("incident_001", SQLI_SCENARIO_ID))
    )
    assert (dry.eligible_for_downstream, dry.live_evidence, dry.evidence_provenance) == (False, False, "dry_run")
    pipeline = to_pipeline_decision(mocked)
    assert pipeline.eligible_for_downstream is False and pipeline.live_evidence is False
