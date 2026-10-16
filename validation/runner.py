"""
Phase 13B.1 - the OFFLINE controlled-validation runner: `run_controlled_validation(...)`.

It drives a `ValidationTransport` through the first validation scenario and turns what came back into a versioned
`ValidationReport`. In Phase 13B.1 the only transport that can ever be accepted is a FAKE one:

  * a fake-transport result is a SIMULATION. Its report says so, records `transport_kind == "fake"`, and the
    Phase 14 handoff never treats it as evidence (`usable_as_post_remediation_evidence` stays False);
  * a `live` transport is refused with `live_transport_not_enabled` while `LIVE_TRANSPORT_ENABLED` is False, and no
    live transport exists in this build.

This module does no I/O itself: no sockets, no DNS, no subprocess, no Docker, no files. It only calls
`transport.send(...)`, which for a fake is an in-memory function.

Preconditions are the Phase 13A order (`validation.orchestrator.assess`), with these 13B.1 differences:

  * `scenario_eligible` additionally needs the scenario to be buildable into requests and an oracle classifier to be
    registered for its oracle. A scenario that is not approved is never run, whatever else is in order;
  * `scope_conditions_met` additionally compares the independently observed lab state with the post-remediation
    state the operator attested (`lab_state_unverified` / `lab_state_mismatch`);
  * `live_execution_available` checks, in this order: a valid transport (`transport_unavailable`), the live-transport
    lock (`live_transport_not_enabled`), then the scope policy's own execution switch (`execution_not_enabled`).
    A configured localhost URL alone therefore never causes a request.

Execution: the benign CONTROL request goes first. If the control fails, the attack-shaped probe is NOT sent and the
result is inconclusive. Every transport failure (timeout, unreachable, error, an exception, a wrong return type) is
converted to an inconclusive result; nothing a transport does can raise out of this function or become
`objective_not_observed`. The tested state is always `post_remediation`, and the attestation (which Phase 13 does
not verify) is required before any check is made. Evidence is a summary (status, hash, length, counts), never a raw
response body.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable

from validation import seals
from validation.evidence_gate import report_digest
from validation.contract import (
    CONCLUSIVE_STATUSES,
    CONTRACT_VERSION_IDENTITY,
    REASONS_BY_STATUS,
    IdentityProvenance,
    STANDARD_LIMITATIONS,
    ConditionRecord,
    ExecutionRecord,
    LabStateRecord,
    ObservationEvidence,
    OracleRecord,
    RemediationRecord,
    ScopeRecord,
    ValidationReport,
)
from validation.oracle import ClassifierRegistry, OracleClassifier, OracleOutcome, default_classifiers
from validation.orchestrator import (
    Assessment,
    ValidationRequest,
    ValidationRequestError,
    assess,
    not_attempted_report,
    objective_record,
    report_run_id,
    target_record,
)
from validation.probe_requests import ProbeRequest, build_baseline_requests, build_probe_requests, check_constructible
from validation.scenario import ScenarioDefinition, ScenarioRegistry, default_registry
from validation.scope import ScopeDecision, ScopePolicy, issued_identity_for
from validation.inputs import assess_attestation, intake_plan
from validation.transport import TransportResponse, ValidationTransport

# The second lock for live traffic (the first is the scope policy's `execution_enabled`). Nothing in 13B.1 can turn
# a live transport on; Phase 13B.2 is where this is decided, deliberately and separately.
LIVE_TRANSPORT_ENABLED = False

RUNNER_ID = "autoshield-validation-runner-13b1"

SIMULATION_STATEMENT = (
    "SIMULATION: this result came from a deterministic in-memory fake transport. It says nothing about any real "
    "lab, target or remediation and must not be used as validation evidence."
)
CHECK_SCOPE_LIMITATION = (
    "This check sent one benign control request and one probe request. It covers that probe only; other payloads, "
    "encodings, endpoints and lab states were not tested."
)
BASELINE_CHECK_SCOPE_LIMITATION = (
    "This check sent the reviewed controlled baseline searches, validated them, and then sent one probe request. It "
    "covers that probe only; other payloads, encodings, endpoints and lab states were not tested."
)
PHASE_13B1_NOTE = (
    "No check ran for this report, no traffic was sent and nothing here demonstrates that any remediation is "
    "effective."
)

Clock = Callable[[], datetime]


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------
# readiness and the live gate
# ------------------------------------------------------------------

def _readiness(classifiers: ClassifierRegistry) -> Callable[[ScenarioDefinition], str | None]:
    def check(scenario: ScenarioDefinition) -> str | None:
        problem = check_constructible(scenario)
        if problem is not None:
            return f"its requests can not be built ({problem})"
        if classifiers.get(scenario.oracle) is None:
            return (f"no oracle classifier is registered for {scenario.oracle.oracle_id!r} "
                    f"version {scenario.oracle.version!r}")
        return None
    return check


def _transport_is_valid(transport: object) -> bool:
    return (isinstance(transport, ValidationTransport) and transport.kind in ("fake", "live")
            and isinstance(transport.transport_id, str) and bool(transport.transport_id.strip()))


def _live_gate(transport: object) -> Callable[[ScopeDecision], tuple[bool, str, str]]:
    def gate(decision: ScopeDecision) -> tuple[bool, str, str]:
        if not _transport_is_valid(transport):
            return False, "No valid transport (id, kind and send) was supplied.", "transport_unavailable"
        if transport.kind == "live" and not LIVE_TRANSPORT_ENABLED:  # type: ignore[union-attr]
            return (False, "A live transport was supplied but live transports are not enabled in this build.",
                    "live_transport_not_enabled")
        if not decision.execution_permitted:
            reason = "execution_not_enabled" if "execution_disabled" in decision.reason_codes else "scope_policy_denied"
            return False, "The scope policy does not permit execution (its execution switch is off).", reason
        return True, "A valid transport is available and the scope policy permits execution.", ""
    return gate


# ------------------------------------------------------------------
# sending
# ------------------------------------------------------------------

def _send(transport: ValidationTransport, request: ProbeRequest) -> TransportResponse:
    """Send one request. Whatever the transport does, the answer is a `TransportResponse`."""
    try:
        answer = transport.send(request)
    except Exception:  # noqa: BLE001 - a failing transport must never escape as a crash or a conclusive verdict
        return TransportResponse.error()
    return answer if isinstance(answer, TransportResponse) else TransportResponse.error()


def _evidence(run_id: str, role: str, response: TransportResponse, when: str, facts: dict[str, str],
              *, index: int | None = None) -> ObservationEvidence:
    attributes = {"outcome": response.outcome, "role": role}
    if response.outcome == "response":
        body = response.body or ""
        attributes.update(
            http_status=str(response.status_code), body_length=str(len(body)),
            body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        )
    attributes.update(facts)
    summary = (f"The {role} request received HTTP {response.status_code}." if response.outcome == "response"
               else f"The {role} request failed: {response.outcome}.")
    observation_id = f"{run_id}:{role}" if index is None else f"{run_id}:{role}:{index}"
    return ObservationEvidence(observation_id=observation_id, kind=f"{role}_exchange", summary=summary,
                               captured_at_utc=when, attributes=attributes)


def _request_facts(request: ProbeRequest, scenario: ScenarioDefinition) -> dict[str, str]:
    """The exact request, recorded for reproducibility (method, endpoint, parameter, encoded value, request line).
    These are the reviewed request fields, not secrets; no raw response body or credential is included."""
    _, _, encoded_value = request.query.partition("=")
    return {
        "request_method": request.method, "request_path": request.path,
        "request_parameter": scenario.endpoint.parameter, "request_encoded_value": encoded_value,
        "request_target": request.target,
    }


def _checked(outcome: object, scenario: ScenarioDefinition, completed: bool) -> OracleOutcome:
    """Accept a classifier's outcome only if it is internally sound; otherwise downgrade to inconclusive."""
    ids = [c.condition_id for c in scenario.oracle.conditions]
    fallback = OracleOutcome(
        status="inconclusive", reason_code="evidence_insufficient",
        conditions=[ConditionRecord(condition_id=i) for i in ids],
    )
    if not isinstance(outcome, OracleOutcome):
        return fallback
    if outcome.reason_code not in REASONS_BY_STATUS[outcome.status] or outcome.status in ("not_validated", "blocked"):
        return fallback
    if [c.condition_id for c in outcome.conditions] != ids:
        return fallback
    if outcome.status in CONCLUSIVE_STATUSES and (
        not completed or any(c.result == "not_evaluated" for c in outcome.conditions)
    ):
        return OracleOutcome(status="inconclusive", reason_code="evidence_insufficient",
                             conditions=outcome.conditions, facts=outcome.facts)
    return outcome


# ------------------------------------------------------------------
# the runner
# ------------------------------------------------------------------

def run_controlled_validation(
    request: ValidationRequest,
    registry: ScenarioRegistry | None = None,
    policy: ScopePolicy | None = None,
    transport: object = None,
    *,
    classifiers: ClassifierRegistry | None = None,
    clock: Clock | None = None,
    runner_id: str = RUNNER_ID,
) -> ValidationReport:
    """Run the offline validation and report it. Sends nothing unless every precondition holds; with a fake
    transport the result is labelled a simulation. Raises `ValidationRequestError` only for a malformed request."""
    if not isinstance(request, ValidationRequest):
        raise ValidationRequestError("request must be a ValidationRequest")
    registry = registry if registry is not None else default_registry()
    policy = policy if policy is not None else ScopePolicy()  # deny by default
    classifiers = classifiers if classifiers is not None else default_classifiers()

    clock = clock or _utc_clock
    now = clock()  # the single "now" of the run: identity freshness is judged against it, and it is `started_at`
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValidationRequestError("clock must return a timezone-aware datetime")

    attestation = assess_attestation(request.application, intake_plan(request.plan_result))
    assessment = assess(
        request, registry, policy,
        readiness=_readiness(classifiers), live_gate=_live_gate(transport),
        expected_lab_state_id=attestation.post_state_id if attestation.accepted else None,
        now=now, forbidden_verifier_ids=(runner_id,),
        # While the live lock is closed the refusal is `live_transport_not_enabled` (the established blocking order);
        # the moment it opens, a live transport needs a live-suitable identity or the scope policy denies it.
        require_live_suitable_identity=bool(getattr(transport, "kind", None) == "live" and LIVE_TRANSPORT_ENABLED),
    )
    if not assessment.all_met:
        return not_attempted_report(assessment, PHASE_13B1_NOTE)

    assert assessment.scenario is not None and assessment.decision.origin is not None
    return _execute(assessment, transport, classifiers, clock, runner_id, now)  # type: ignore[arg-type]


def _identity_provenance(assessment: Assessment, now: datetime) -> IdentityProvenance:
    """Describe the issuance record behind the identity that just passed the scope policy."""
    evidence = assessment.request.lab_identity
    issued = issued_identity_for(evidence, now) if evidence is not None else None
    assert evidence is not None and issued is not None  # the scope policy denied otherwise
    return IdentityProvenance(
        source_id=issued.source_id, observed_at_utc=issued.observed_at_utc, checked_at_utc=_stamp(now),
        lab_state_id=evidence.lab_state_id, live_suitable=issued.live_suitable,
        authoritative_registry=issued.authoritative,
    )


def _execute_baselines(
    transport: ValidationTransport, scenario: ScenarioDefinition, origin: str, probe_request: ProbeRequest,
    classifier: OracleClassifier, run_id: str, clock: Clock,
) -> tuple[list[ObservationEvidence], OracleOutcome, int, bool, str]:
    """The reviewed Incident 001 control flow: send and validate the controlled baseline searches first, and only
    then (if every baseline is usable) send the attack-shaped probe. Returns evidence, the checked oracle outcome,
    the number of requests sent, whether the run completed, and the finish timestamp."""
    baseline_requests = build_baseline_requests(scenario, origin)
    terms = [baseline.term for baseline in scenario.probe.baseline_searches]
    baseline_responses: dict[str, TransportResponse] = {}
    exchanges: list[tuple[int, ProbeRequest, TransportResponse]] = []
    for index, (term, request) in enumerate(zip(terms, baseline_requests)):
        response = _send(transport, request)
        baseline_responses[term] = response
        exchanges.append((index, request, response))
    sent = len(baseline_requests)

    probe: TransportResponse | None = None
    if classifier.baselines_healthy(scenario, baseline_responses):  # the probe is ineligible until the baselines pass
        sent += 1
        probe = _send(transport, probe_request)
    finished = _stamp(clock())

    completed = (bool(baseline_responses)
                 and all(r.outcome == "response" for r in baseline_responses.values())
                 and probe is not None and probe.outcome == "response")
    try:
        raw = classifier.classify(scenario, None, probe, baselines=baseline_responses)
    except Exception:  # noqa: BLE001 - a broken classifier becomes "no conclusion", never a verdict
        raw = None
    outcome = _checked(raw, scenario, completed)

    evidence = [_evidence(run_id, "control", response, finished, _request_facts(request, scenario), index=index)
                for index, request, response in exchanges]
    if probe is not None:
        probe_facts = dict(outcome.facts)
        probe_facts.update(_request_facts(probe_request, scenario))
        evidence.append(_evidence(run_id, "probe", probe, finished, probe_facts))
    return evidence, outcome, sent, completed, finished


def _execute(assessment: Assessment, transport: ValidationTransport, classifiers: ClassifierRegistry,
             clock: Clock, runner_id: str, now: datetime) -> ValidationReport:
    scenario, decision = assessment.scenario, assessment.decision
    assert scenario is not None and decision.origin is not None
    classifier: OracleClassifier | None = classifiers.get(scenario.oracle)
    assert classifier is not None  # guaranteed by the readiness check
    pair = build_probe_requests(scenario, decision.origin)  # raises loudly, before anything is sent, if unsound
    run_id = f"{report_run_id(assessment)}-{transport.kind}"
    started = _stamp(now)
    baseline_mode = bool(scenario.probe.baseline_searches)

    if baseline_mode:
        evidence, outcome, sent, completed, finished = _execute_baselines(
            transport, scenario, decision.origin, pair.probe, classifier, run_id, clock)
    else:
        sent = 1
        control = _send(transport, pair.control)
        probe: TransportResponse | None = None
        if classifier.control_healthy(scenario, control):
            sent = 2
            probe = _send(transport, pair.probe)
        finished = _stamp(clock())

        completed = control.outcome == "response" and probe is not None and probe.outcome == "response"
        try:
            raw = classifier.classify(scenario, control, probe)
        except Exception:  # noqa: BLE001 - a broken classifier becomes "no conclusion", never a verdict
            raw = None
        outcome = _checked(raw, scenario, completed)

        control_facts = {k: v for k, v in outcome.facts.items() if k.startswith("control_")}
        control_facts.update(_request_facts(pair.control, scenario))
        evidence = [_evidence(run_id, "control", control, finished, control_facts)]
        if probe is not None:
            probe_facts = {k: v for k, v in outcome.facts.items() if not k.startswith("control_")}
            probe_facts.update(_request_facts(pair.probe, scenario))
            evidence.append(_evidence(run_id, "probe", probe, finished, probe_facts))

    attestation = assessment.attestation
    check_limitation = BASELINE_CHECK_SCOPE_LIMITATION if baseline_mode else CHECK_SCOPE_LIMITATION
    limitations = [*STANDARD_LIMITATIONS, check_limitation, *scenario.limitations]
    validity = [
        "Valid only for the lab state the operator attested as the post-remediation state and the independent "
        "identity check observed. Phase 13 recorded that attestation; it did not verify it.",
        "Valid only for the scenario definition, oracle version and request encoding recorded in this report.",
    ]
    if transport.kind == "fake":
        limitations.insert(0, SIMULATION_STATEMENT)
        validity.insert(0, SIMULATION_STATEMENT)

    report = ValidationReport(
        contract_version=CONTRACT_VERSION_IDENTITY,
        identity=_identity_provenance(assessment, now),
        validation_run_id=run_id,
        incident_id=assessment.request.incident_id,
        scenario=assessment.scenario_ref,
        objective=objective_record(scenario),
        target=target_record(scenario, decision.origin, verified=True),
        lab_state=LabStateRecord(tested_state="post_remediation",
                                 pre_remediation_state_id=attestation.pre_state_id,
                                 post_remediation_state_id=attestation.post_state_id),
        remediation=RemediationRecord(proposals=list(assessment.intake.proposals),
                                      application=attestation.application),
        status=outcome.status,
        reason_code=outcome.reason_code,
        oracle=OracleRecord(oracle_id=scenario.oracle.oracle_id, oracle_version=scenario.oracle.version,
                            conditions=outcome.conditions),
        execution=ExecutionRecord(
            attempted=True, completed=completed, requests_sent=sent, runner_id=runner_id,
            transport=transport.transport_id, transport_kind=transport.kind,
            started_at_utc=started, finished_at_utc=finished,
        ),
        evidence=evidence,
        scope=ScopeRecord(policy_version=decision.policy_version, scope_conditions_met=True,
                          execution_permitted=True, reason_codes=list(decision.reason_codes)),
        preconditions=assessment.findings,
        upstream=assessment.intake.upstream,
        limitations=limitations,
        validity_conditions=validity,
    )
    # The ONLY place a report is issued: `validation.evidence_gate` accepts no report the runner did not produce.
    seals.issue_report(report_digest(report))
    return report
