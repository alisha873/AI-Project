"""
Shared helpers for the Phase 13A validation tests (not a test module).

All data is SYNTHETIC or built by the real Phase 12 code with a scripted fake model: remediation plans come from
`plan_remediation(...)` driven by `FakeLLM`, so the tests exercise the actual Phase 12 contract. Nothing here
touches a network, Docker, a file or a real model. `RecordingTransport` only records calls.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from urllib.parse import parse_qs

from remediation.planner import RemediationPlanResult, plan_remediation
from tests.remediation_helpers import FakeLLM, good_plan, good_proposal, make_input, plan_json
from validation.inputs import RemediationApplicationAttestation
from validation.lab_identity import LabIdentityObservation, VerifierEntry, VerifierRegistry, assess_lab_identity
from validation.orchestrator import ValidationRequest, build_validation_report
from validation.runner import run_controlled_validation
from validation.transport import ScriptedTransport, TransportResponse
from validation.scenario import (
    SQLI_SCENARIO_ID,
    ScenarioDefinition,
    ScenarioRegistry,
    sqli_search_scenario,
)
from validation.scope import ApprovedLab, LabIdentityEvidence, OperatorAuthorization, ScopePolicy

INCIDENT = "incident_001"
TARGET = "http://127.0.0.1:8080"
CONTAINERS = [
    "autoshield-nginx", "autoshield-juice-shop", "autoshield-api",
    "autoshield-postgres", "autoshield-redis", "autoshield-linux-admin",
]
COMPOSE_SHA = "c" * 64


class RecordingTransport:
    """A transport that records every call. A test passes it to prove Phase 13A never calls it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send(self, request):
        self.calls.append(dict(request))
        raise AssertionError("Phase 13A must never use a transport")


LAB_RECORDS = ["first note", "second note", "hello world"]
PROBE_TEXT = "' OR '1'='1"


def lab_body(messages: list[str]) -> str:
    return json.dumps({"data": [{"id": i + 1, "message": m} for i, m in enumerate(messages)]})


def simulated_lab(vulnerable: bool):
    """An in-memory stand-in for the search endpoint (a pure function of the request; no network)."""
    def respond(request) -> TransportResponse:
        value = parse_qs(request.query, keep_blank_values=True)["message"][0]
        matches = list(LAB_RECORDS) if (vulnerable and PROBE_TEXT in value) else [m for m in LAB_RECORDS if value in m]
        return TransportResponse.ok(lab_body(matches))
    return respond


def fixed_clock():
    moments = iter(datetime(2026, 10, 16, 12, 0, s, tzinfo=timezone.utc) for s in range(60))
    return lambda: next(moments)


def run_fake(transport=None, *, vulnerable=False, request=None, policy=None, registry=None, **kwargs):
    transport = transport if transport is not None else ScriptedTransport(
        simulated_lab(vulnerable), simulated_lab(vulnerable))
    report = run_controlled_validation(
        request or full_request(), registry or approved_registry(), policy or full_policy(), transport,
        clock=kwargs.pop("clock", fixed_clock()), **kwargs)
    return report, transport


def planned_result() -> RemediationPlanResult:
    result = plan_remediation(make_input(), FakeLLM(plan_json(good_plan())))
    assert result.status == "planned", result.message
    return result


def insufficient_context_result() -> RemediationPlanResult:
    result = plan_remediation(make_input(with_answer=False), FakeLLM("{}"))
    assert result.status == "not_planned_insufficient_context"
    return result


def model_insufficient_result() -> RemediationPlanResult:
    plan = good_plan()
    plan.update(outcome="insufficient_evidence", insufficient_reason="Not enough material.", proposals=[])
    result = plan_remediation(make_input(), FakeLLM(plan_json(plan)))
    assert result.status == "model_declared_insufficient", result.message
    return result


def llm_error_result() -> RemediationPlanResult:
    result = plan_remediation(make_input(), FakeLLM(RuntimeError("provider down")))
    assert result.status == "llm_error"
    return result


def invalid_response_result() -> RemediationPlanResult:
    result = plan_remediation(make_input(), FakeLLM("this is not json"))
    assert result.status == "rejected_invalid_response"
    return result


def ungrounded_result() -> RemediationPlanResult:
    plan = good_plan(None, good_proposal(observed_evidence_ids=["fabricated-9"]))
    result = plan_remediation(make_input(), FakeLLM(plan_json(plan)))
    assert result.status == "rejected_grounding", result.status
    return result


def approved_scenario(*, with_baseline_searches: bool = False) -> ScenarioDefinition:
    """The built-in scenario as a human reviewer might one day approve it (a TEST fixture, not a real review).

    By default the reviewed baseline searches are dropped, so this generic fixture exercises the single-control flow
    that the rest of the suite relies on. `with_baseline_searches=True` keeps the three reviewed baselines, so the
    runner uses the baseline-gated Incident 001 flow."""
    data = copy.deepcopy(sqli_search_scenario().model_dump())
    for item in data["ambiguities"]:
        item["resolved"] = True
    data["review"] = {"status": "approved", "reviewer": "test-reviewer", "note": "Synthetic approval for a test."}
    data["probe"]["wire_encoding"] = "percent_encoded_utf8_query"
    data["probe"]["control_value"] = "zzqx-no-such-record"
    if not with_baseline_searches:
        data["probe"]["baseline_searches"] = []
    return ScenarioDefinition.model_validate(data)


def approved_scenario_with_baselines() -> ScenarioDefinition:
    return approved_scenario(with_baseline_searches=True)


def approved_registry() -> ScenarioRegistry:
    return ScenarioRegistry([approved_scenario()])


def approved_registry_with_baselines() -> ScenarioRegistry:
    return ScenarioRegistry([approved_scenario_with_baselines()])


# A deterministic in-memory search table where each record's message contains exactly one reviewed baseline term, so
# a benign search for that term returns it (confirming the LIKE '%term%' semantics). It is SYNTHETIC: it stands in for
# the lab's records, which are runtime evidence and are never invented as real.
BASELINE_LAB_RECORDS = {1: "hello world", 2: "security advisory", 3: "autoshield status"}


class SearchLabTransport:
    """A deterministic fake of the /search endpoint that answers ANY number of requests (no network, no files).

    A benign request returns the records whose message contains the submitted value (substring/LIKE semantics); when
    `vulnerable` is True the probe text instead returns the whole table, as the injection would."""

    kind = "fake"

    def __init__(self, vulnerable: bool, records: dict[int, str] | None = None,
                 transport_id: str = "fake-search-lab") -> None:
        self.transport_id = transport_id
        self.vulnerable = vulnerable
        self.table = dict(records if records is not None else BASELINE_LAB_RECORDS)
        self.requests: list = []

    def send(self, request) -> TransportResponse:
        self.requests.append(request)
        value = parse_qs(request.query, keep_blank_values=True)["message"][0]
        if self.vulnerable and PROBE_TEXT in value:
            matched = list(self.table.items())
        else:
            matched = [(i, m) for i, m in self.table.items() if value in m]
        return TransportResponse.ok(json.dumps({"data": [{"id": i, "message": m} for i, m in matched]}))


def run_fake_baselines(*, vulnerable=False, records=None, request=None, policy=None, registry=None, **kwargs):
    """Run the baseline-gated Incident 001 flow against the in-memory search fake."""
    transport = SearchLabTransport(vulnerable, records)
    report = run_controlled_validation(
        request or full_request(), registry or approved_registry_with_baselines(), policy or full_policy(), transport,
        clock=kwargs.pop("clock", fixed_clock()), **kwargs)
    return report, transport


def approved_lab(**overrides) -> ApprovedLab:
    data = {
        "approval_id": "APPR-1", "lab_id": "lab-1", "approved_by": "test-operator",
        "host": "127.0.0.1", "port": 8080, "expected_network": "autoshield_net",
        "expected_containers": list(CONTAINERS), "compose_sha256": COMPOSE_SHA,
        "approved_scenario_ids": [SQLI_SCENARIO_ID],
    }
    data.update(overrides)
    return ApprovedLab(**data)


IDENTITY_OBSERVED_AT = datetime(2026, 10, 16, 11, 59, 0, tzinfo=timezone.utc)  # fresh for fixed_clock() (12:00:00)


def identity(**overrides) -> LabIdentityEvidence:
    """Lab identity evidence for tests, produced by the REAL assessment path (`assess_lab_identity`) with a
    TEST-DOUBLE verifier. It is independent and fresh (so simulation runs proceed) but NEVER live-suitable.

    Two overrides build a deliberately HAND-MADE claim instead (no issuance record, so the scope policy denies it
    as `lab_identity_not_assessed`): `established_independently=False`, and `lab_state_id=None` (an observation
    always has a state id, so evidence without one can not come from the assessment path).
    `observed_at` (a datetime) moves the observation time, for tests that use the real clock."""
    claimed = overrides.pop("established_independently", True)
    source = overrides.pop("established_by", "test-identity-probe")
    state = overrides.pop("lab_state_id", "STATE-POST-1")
    observed_at = overrides.pop("observed_at", IDENTITY_OBSERVED_AT)
    data = {
        "lab_id": "lab-1", "observed_network": "autoshield_net", "observed_containers": list(reversed(CONTAINERS)),
        "compose_sha256": COMPOSE_SHA,
    }
    data.update(overrides)
    if not claimed or state is None:
        return LabIdentityEvidence(**data, lab_state_id=state, established_by=source,
                                   established_independently=claimed)
    observation = LabIdentityObservation(
        observation_id="OBS-ID-T", **data, lab_state_id=state,
        observed_at_utc=observed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        verification={"claim": "verifier_attested", "source_id": source},
    )
    result = assess_lab_identity(
        observation, VerifierRegistry(VerifierEntry(verifier_id=source, kind="test_double")), now=observed_at)
    assert result.independently_verified and result.scope_identity is not None, result.reason_codes
    return result.scope_identity


def authorization(**overrides) -> OperatorAuthorization:
    data = {
        "authorization_id": "AUTH-1", "operator_id": "test-operator", "lab_id": "lab-1",
        "scenario_id": SQLI_SCENARIO_ID, "incident_id": INCIDENT, "live_execution_authorized": True,
    }
    data.update(overrides)
    return OperatorAuthorization(**data)


def attestation_for(result: RemediationPlanResult, **overrides) -> RemediationApplicationAttestation:
    ids = [proposal.proposal_id for proposal in result.plan.proposals]
    data = {
        "attestation_id": "ATT-1", "attested_by": "test-operator", "applied_proposal_ids": ids[:1],
        "post_remediation_lab_state_id": "STATE-POST-1", "pre_remediation_lab_state_id": "STATE-PRE-1",
    }
    data.update(overrides)
    return RemediationApplicationAttestation(**data)


def full_request(result: RemediationPlanResult | None = None, **overrides) -> ValidationRequest:
    """A request in which EVERYTHING is in order (so only the Phase 13A lock stands in the way)."""
    result = result or planned_result()
    data = {
        "incident_id": INCIDENT, "scenario_id": SQLI_SCENARIO_ID, "plan_result": result,
        "application": attestation_for(result), "target_url": TARGET, "lab_identity": identity(),
        "authorization": authorization(),
    }
    data.update(overrides)
    return ValidationRequest(**data)


def full_policy(execution_enabled: bool = True) -> ScopePolicy:
    return ScopePolicy([approved_lab()], execution_enabled=execution_enabled)


def blocked_report():
    """The most advanced report Phase 13A can produce: everything in order, execution still locked."""
    report = build_validation_report(full_request(), approved_registry(), full_policy())
    assert report.status == "blocked", report.reason_code
    return report


def executed_dict(status: str, reason: str, tested_state: str = "post_remediation") -> dict:
    """A report dict as a FUTURE live runner might produce it (built here by hand for contract tests only)."""
    data = blocked_report().model_dump(mode="json")
    data["status"], data["reason_code"] = status, reason
    for item in data["preconditions"]:
        item.update(satisfied=True, evaluated=True, reason_code=None)
    data["scope"].update(scope_conditions_met=True, execution_permitted=True)
    data["target"]["lab_identity_verified"] = True
    data["lab_state"]["tested_state"] = tested_state
    data["execution"] = {
        "attempted": True, "completed": status in ("objective_achieved", "objective_not_observed"),
        "requests_sent": 2, "runner_id": "test-runner", "transport": "test-transport",
        "transport_kind": "live",
        "started_at_utc": "2026-10-16T12:00:00Z", "finished_at_utc": "2026-10-16T12:00:01Z",
    }
    data["evidence"] = [{
        "observation_id": "OBS-1", "kind": "probe_response", "summary": "Synthetic observation.",
        "captured_at_utc": "2026-10-16T12:00:01Z", "attributes": {"http_status": "200"},
    }]
    for condition in data["oracle"]["conditions"]:
        condition["result"] = "met" if condition["condition_id"] != "record_outside_literal_match" else (
            "met" if status == "objective_achieved" else "not_met")
    return data


# The evidence-gate codes about HOW the identity was established and who issued the report (Phase 13B.3). A hand-built
# test report can never satisfy these, so tests of the OTHER usability rules (status, completion, post state,
# attestation, live transport) use this stub: the real gate minus exactly these codes. The identity rules themselves are
# tested, un-stubbed, in tests/test_validation_identity_gate.py.
IDENTITY_GATE_CODES = (
    "legacy_report_without_identity_provenance", "identity_not_live_suitable", "report_not_issued_by_runner",
)


def gate_without_identity_rules(monkeypatch) -> None:
    import validation.comparison as comparison_module
    import validation.handoff as handoff_module
    from validation.evidence_gate import evidence_gate_failures as real

    def stub(report, *, tested_state="post_remediation"):
        return [code for code in real(report, tested_state=tested_state) if code not in IDENTITY_GATE_CODES]

    monkeypatch.setattr(handoff_module, "evidence_gate_failures", stub)
    monkeypatch.setattr(comparison_module, "evidence_gate_failures", stub)