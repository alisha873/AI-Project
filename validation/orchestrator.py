"""
Phase 13A - the offline validation orchestrator: `build_validation_report(...)`.

It turns the upstream inputs into a `ValidationReport` WITHOUT performing any I/O. In Phase 13A it can only
ever produce `not_validated` or `blocked`: no check is executed, no request is sent, no remediation is
applied, no model is called, no file is read or written, and nothing touches Docker.

Fixed checking order (the first unmet precondition is the report's primary reason):

  input preconditions (-> not_validated)
    1. plan_usable                          a Phase 12 result that passes `validation.inputs.intake_plan`
    2. incident_matches_plan                the plan is for the incident being validated
    3. scenario_known                       the scenario id is in the registry
    4. scenario_eligible                    approved by a human reviewer, no open blocking ambiguity
    5. remediation_application_attested     someone other than Phase 13 claims the remediation was applied
  permission preconditions (-> blocked)
    6. scope_conditions_met                 the `ScopePolicy` found the target, identity and authorization in order
    7. live_execution_available             always unmet here: `LIVE_EXECUTION_IMPLEMENTED` is False

Two independent locks keep execution off: the scope policy denies by default, and this module refuses to
execute regardless of the policy because no runner exists. Even a fully approved request is therefore
`blocked` with reason `execution_not_implemented_phase_13a`. `transport` is a reserved seam for Phase 13B:
it is accepted so a test can prove it is never called, and it is never used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from validation.contract import (
    INPUT_PRECONDITIONS,
    STANDARD_LIMITATIONS,
    ConditionRecord,
    ExecutionRecord,
    LabStateRecord,
    ObjectiveRecord,
    OracleRecord,
    PreconditionFinding,
    RemediationRecord,
    ScenarioReference,
    ScopeRecord,
    TargetRecord,
    ValidationReport,
)
from validation.inputs import (
    AttestationAssessment,
    PlanIntake,
    RemediationApplicationAttestation,
    assess_attestation,
    intake_plan,
)
from validation.scenario import ScenarioDefinition, ScenarioRegistry, default_registry
from validation.scope import (
    LabIdentityEvidence,
    OperatorAuthorization,
    ScopeDecision,
    ScopePolicy,
    ScopeRequest,
)

LIVE_EXECUTION_IMPLEMENTED = False


class ValidationRequestError(TypeError):
    """The request itself is malformed (a programming error), as opposed to unusable upstream data."""


@dataclass(frozen=True)
class ValidationRequest:
    incident_id: str
    scenario_id: str
    plan_result: object = None
    application: RemediationApplicationAttestation | None = None
    target_url: object = None
    lab_identity: LabIdentityEvidence | None = None
    authorization: OperatorAuthorization | None = None


def _finding(precondition: str, ok: bool, detail: str, reason: str | None = None,
             evaluated: bool = True) -> PreconditionFinding:
    return PreconditionFinding(
        precondition=precondition, satisfied=ok, evaluated=evaluated,
        reason_code=None if ok or not evaluated else reason, detail=detail,
    )


def _scenario_reference(scenario_id: str, scenario: ScenarioDefinition | None) -> ScenarioReference:
    if scenario is None:
        return ScenarioReference(scenario_id=scenario_id, registered=False)
    return ScenarioReference(
        scenario_id=scenario.scenario_id,
        registered=True,
        definition_version=scenario.definition_version,
        definition_sha256=scenario.definition_sha256(),
        family=scenario.family,
        review_status=scenario.review.status,
        eligible_for_execution=scenario.eligible_for_execution,
        ambiguity_codes=scenario.open_blocking_ambiguities(),
    )


def _run_id(incident_id: str, scenario: ScenarioReference, proposal_ids: list[str], plan_status: str | None,
            attestation_id: str | None, origin: str | None) -> str:
    identity = {
        "incident": incident_id, "scenario": scenario.scenario_id, "definition": scenario.definition_sha256,
        "proposals": sorted(proposal_ids), "plan_status": plan_status, "attestation": attestation_id,
        "origin": origin,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"VAL-{incident_id}-{digest}"


@dataclass(frozen=True)
class Assessment:
    """Everything the fixed precondition check found. Shared by the 13A offline report and the 13B runner."""

    request: ValidationRequest
    scenario: ScenarioDefinition | None
    intake: PlanIntake
    attestation: AttestationAssessment
    decision: ScopeDecision
    findings: list[PreconditionFinding]
    scenario_ref: ScenarioReference

    @property
    def all_met(self) -> bool:
        return all(item.satisfied for item in self.findings)


# (ok, detail, reason_code) for the last precondition, given the scope decision
LiveGate = Callable[[ScopeDecision], tuple[bool, str, str]]
ScenarioReadiness = Callable[[ScenarioDefinition], "str | None"]


def _phase_13a_gate(decision: ScopeDecision) -> tuple[bool, str, str]:
    return (
        LIVE_EXECUTION_IMPLEMENTED,
        "Live execution exists." if LIVE_EXECUTION_IMPLEMENTED
        else "Live validation is not implemented in Phase 13A; nothing is executed.",
        "execution_not_implemented_phase_13a",
    )


def assess(
    request: ValidationRequest,
    registry: ScenarioRegistry | None = None,
    policy: ScopePolicy | None = None,
    *,
    readiness: ScenarioReadiness | None = None,
    live_gate: LiveGate | None = None,
    expected_lab_state_id: str | None = None,
    now: datetime | None = None,
    forbidden_verifier_ids: Iterable[str] = (),
    require_live_suitable_identity: bool = False,
) -> Assessment:
    """Run the fixed precondition order. Never raises for bad upstream DATA; raises `ValidationRequestError`
    only if the request object itself is malformed. `readiness` (13B runner only) may add a reason why an
    otherwise approved scenario still can not be run; `live_gate` replaces the 13A "not implemented" gate."""
    if not isinstance(request, ValidationRequest):
        raise ValidationRequestError("request must be a ValidationRequest")
    for name in ("incident_id", "scenario_id"):
        value = getattr(request, name)
        if not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > 200:
            raise ValidationRequestError(f"{name} must be a non-empty string of at most 200 characters "
                                         "without surrounding whitespace")
    registry = registry if registry is not None else default_registry()
    policy = policy if policy is not None else ScopePolicy()  # deny by default

    intake = intake_plan(request.plan_result)
    scenario = registry.get(request.scenario_id)
    attestation = assess_attestation(request.application, intake)
    # The identity verifier may never be the operator, the attester or the runner (checked at use time, here, so a
    # caller-chosen `distinct_from` at assessment time can not weaken it).
    forbidden = {str(item) for item in forbidden_verifier_ids}
    if request.authorization is not None:
        forbidden.add(getattr(request.authorization, "operator_id", ""))
    if request.application is not None:
        forbidden.add(getattr(request.application, "attested_by", ""))
    forbidden.discard("")
    decision = policy.evaluate(ScopeRequest(
        target_url=request.target_url, scenario=scenario, incident_id=request.incident_id,
        identity=request.lab_identity, authorization=request.authorization,
        expected_lab_state_id=expected_lab_state_id,
        now=now, forbidden_verifier_ids=tuple(sorted(forbidden)),
        require_live_suitable_identity=require_live_suitable_identity,
    ))

    findings: list[PreconditionFinding] = []
    findings.append(_finding("plan_usable", intake.usable, intake.detail, intake.reason_code))
    if intake.usable:
        matches = intake.incident_id == request.incident_id
        findings.append(_finding(
            "incident_matches_plan", matches,
            "The plan is for the incident being validated." if matches
            else "The plan was made for a different incident.", "incident_mismatch"))
    else:
        findings.append(_finding("incident_matches_plan", False,
                                 "Not evaluated: there is no usable plan.", evaluated=False))
    findings.append(_finding(
        "scenario_known", scenario is not None,
        "The scenario is registered." if scenario else "The scenario id is not in the registry.",
        "scenario_unknown"))
    if scenario is None:
        findings.append(_finding("scenario_eligible", False, "Not evaluated: the scenario is unknown.",
                                 evaluated=False))
    else:
        eligible = scenario.eligible_for_execution
        problem = readiness(scenario) if (eligible and readiness is not None) else None
        findings.append(_finding(
            "scenario_eligible", eligible and problem is None,
            ("The scenario is approved and has no open blocking ambiguity." if problem is None
             else f"The scenario is approved but can not be run: {problem}.") if eligible
            else "The scenario is not approved for execution (review status "
                 f"{scenario.review.status!r}; open blocking ambiguities: "
                 f"{len(scenario.open_blocking_ambiguities())}).",
            "scenario_not_eligible"))
    if intake.usable:
        findings.append(_finding("remediation_application_attested", attestation.accepted, attestation.detail,
                                 "remediation_application_not_attested"))
    else:
        findings.append(_finding("remediation_application_attested", False,
                                 "Not evaluated: there is no usable plan.", evaluated=False))
    findings.append(_finding(
        "scope_conditions_met", decision.scope_conditions_met,
        "The scope policy found the target, scenario, lab identity and authorization in order."
        if decision.scope_conditions_met
        else "The scope policy denied: " + ", ".join(c for c in decision.reason_codes if c != "execution_disabled")
        + ".", "scope_policy_denied"))
    gate_ok, gate_detail, gate_reason = (live_gate or _phase_13a_gate)(decision)
    findings.append(_finding("live_execution_available", gate_ok, gate_detail, gate_reason))

    return Assessment(
        request=request, scenario=scenario, intake=intake, attestation=attestation, decision=decision,
        findings=findings, scenario_ref=_scenario_reference(request.scenario_id, scenario),
    )


def objective_record(scenario: ScenarioDefinition) -> ObjectiveRecord:
    return ObjectiveRecord(
        summary=scenario.objective_summary,
        source_kind=scenario.objective_source.kind,
        source_ref=scenario.objective_source.path,
        source_sha256=scenario.objective_source.sha256,
        obtained_by=scenario.objective_source.obtained_by,
    )


def target_record(scenario: ScenarioDefinition, origin: str | None, *, verified: bool) -> TargetRecord:
    endpoint = scenario.endpoint
    return TargetRecord(
        method=endpoint.method, path=endpoint.path, parameter=endpoint.parameter,
        entry_service=endpoint.entry_service, target_service=endpoint.target_service,
        data_service=endpoint.data_service, requested_origin=origin, lab_identity_verified=verified,
    )


def report_run_id(assessment: Assessment) -> str:
    request, intake = assessment.request, assessment.intake
    return _run_id(request.incident_id, assessment.scenario_ref, [p.proposal_id for p in intake.proposals],
                   intake.upstream.plan_result_status, assessment.attestation.application.attestation_id,
                   assessment.decision.origin)


def build_validation_report(
    request: ValidationRequest,
    registry: ScenarioRegistry | None = None,
    policy: ScopePolicy | None = None,
    transport: object = None,  # reserved; never used by the 13A orchestrator (the 13B runner owns transports)
) -> ValidationReport:
    """Build the honest offline report for `request`. It never executes anything."""
    return not_attempted_report(assess(request, registry, policy))


def not_attempted_report(assessment: Assessment, phase_note: str | None = None) -> ValidationReport:
    """The `not_validated` / `blocked` report for an assessment with at least one unmet precondition."""
    request, scenario, intake = assessment.request, assessment.scenario, assessment.intake
    attestation, decision, findings = assessment.attestation, assessment.decision, assessment.findings
    first_unmet = next(item for item in findings if not item.satisfied)
    status = "not_validated" if first_unmet.precondition in INPUT_PRECONDITIONS else "blocked"
    reason_code = first_unmet.reason_code
    assert reason_code is not None  # the first unmet precondition is always an evaluated one

    scenario_ref = _scenario_reference(request.scenario_id, scenario)
    accepted_ids = (attestation.pre_state_id, attestation.post_state_id) if attestation.accepted else (None, None)
    limitations = [*STANDARD_LIMITATIONS,
                   phase_note or "Phase 13A performs no live validation: no check ran, no traffic was sent "
                                 "and nothing here demonstrates that any remediation is effective."]
    if scenario is not None:
        limitations.extend(scenario.limitations)

    return ValidationReport(
        validation_run_id=report_run_id(assessment),
        incident_id=request.incident_id,
        scenario=scenario_ref,
        objective=None if scenario is None else objective_record(scenario),
        target=None if scenario is None else target_record(scenario, decision.origin, verified=False),
        lab_state=LabStateRecord(tested_state="not_tested", pre_remediation_state_id=accepted_ids[0],
                                 post_remediation_state_id=accepted_ids[1]),
        remediation=RemediationRecord(proposals=list(intake.proposals), application=attestation.application),
        status=status,
        reason_code=reason_code,
        oracle=None if scenario is None else OracleRecord(
            oracle_id=scenario.oracle.oracle_id, oracle_version=scenario.oracle.version,
            conditions=[ConditionRecord(condition_id=c.condition_id) for c in scenario.oracle.conditions]),
        execution=ExecutionRecord(),
        evidence=[],
        scope=ScopeRecord(policy_version=decision.policy_version,
                          scope_conditions_met=decision.scope_conditions_met,
                          # a permission only counts if this build can act on it (it can not)
                          execution_permitted=decision.execution_permitted and LIVE_EXECUTION_IMPLEMENTED,
                          reason_codes=list(decision.reason_codes)),
        preconditions=findings,
        upstream=intake.upstream,
        limitations=limitations,
        validity_conditions=[
            "No check ran, so this report is valid only as a statement of WHY no validation result exists.",
            "It reflects the inputs supplied at that time; any change to the plan, scenario, attestation or "
            "scope configuration requires a new report.",
        ],
    )
