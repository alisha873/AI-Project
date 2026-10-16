"""
Phase 13 -> Phase 14 hand-off (pure; no I/O).

Phase 14 (the Judge) must decide using ACTUAL post-remediation validation evidence, never because a model says
a remediation is safe. `build_phase14_handoff(report)` flattens a `ValidationReport` into the fields a Judge
needs, keeping apart:

  what was TESTED       scenario, oracle, tested lab state, requested origin
  what was OBSERVED     whether a check ran and completed, requests sent, observation ids, condition results
  the OUTCOME           status and machine-readable reason code
  the REMEDIATION       proposed ids, which ids an operator attests were applied, and that Phase 13 did not verify it
  what is MISSING       every unmet precondition and the primary reason
  the upstream IDS      observed evidence, hypotheses, configuration facts, reference ids and chunk ids, in
                        separate lists exactly as Phase 12 carried them
  the LIMITS            why this is not a general security guarantee

`usable_as_post_remediation_evidence` is True ONLY if `validation.evidence_gate` passes: a conclusive result
(`objective_achieved` or `objective_not_observed`) against a post-remediation lab state whose remediation
application is attested, through a LIVE transport, with identity provenance from a live-suitable verifier in the
repository's own registry, in a report the runner itself issued. Anything else (no check ran, blocked, inconclusive,
a pre-remediation baseline, a simulation, a legacy 1.0 report, a hand-built report) is not evidence that a
remediation worked and the Judge must not treat it as such; `not_usable_reasons` says why. In this build no live
transport and no live-suitable verifier exist, so it is never True. Phase 14 itself is not implemented here.
"""

from __future__ import annotations

import json
from typing import Literal

from validation.contract import Identifier, Text, ValidationReport, _Strict
from validation.evidence_gate import evidence_gate_failures

# 1.1 (Phase 13B.3): usability also needs identity provenance and runner issuance; adds the three identity fields
# and the reason list. 1.0 handoffs are not produced any more and are not reinterpreted.
HANDOFF_VERSION = "1.1"


class Phase14Handoff(_Strict):
    handoff_version: Literal["1.1"] = HANDOFF_VERSION
    contract_version: Identifier
    validation_run_id: Identifier
    incident_id: Identifier

    # outcome
    status: Identifier
    reason_code: Identifier
    usable_as_post_remediation_evidence: bool
    # Why it is not usable (empty only when usable); stable codes from `validation.evidence_gate`.
    not_usable_reasons: list[Identifier]

    # how the lab identity behind a check was established (None when no check ran or for a 1.0 report)
    identity_source_id: Identifier | None
    identity_independently_verified: bool
    identity_live_suitable: bool

    # what was tested
    scenario_id: Identifier
    scenario_definition_sha256: Identifier | None
    oracle_id: Identifier | None
    tested_state: Identifier
    pre_remediation_state_id: Identifier | None
    post_remediation_state_id: Identifier | None
    requested_origin: str | None

    # what was observed
    check_attempted: bool
    check_completed: bool
    transport_kind: Identifier
    requests_sent: int
    observation_ids: list[Identifier]
    condition_results: dict[str, str]

    # remediation
    proposal_ids: list[Identifier]
    applied_proposal_ids: list[Identifier]
    application_state: Identifier
    attestation_id: Identifier | None
    attested_by: Identifier | None
    application_verified_by_phase13: Literal[False] = False

    # what is missing
    unmet_preconditions: list[Identifier]

    # upstream identifiers, one list per category (never merged)
    observed_evidence_ids: list[Identifier]
    hypothesis_ids: list[Identifier]
    configuration_fact_ids: list[Identifier]
    reference_ids: list[Identifier]
    reference_chunk_ids: list[Identifier]

    # interpretation
    interpretation_limits: list[Text]
    remediation_effectiveness_proven: Literal[False] = False
    general_security_guarantee: Literal[False] = False


def _ordered_unique(groups: list[list[str]]) -> list[str]:
    seen: dict[str, None] = {}
    for group in groups:
        for item in group:
            seen.setdefault(item, None)
    return list(seen)


def build_phase14_handoff(report: ValidationReport) -> Phase14Handoff:
    if not isinstance(report, ValidationReport):
        raise TypeError("report must be a ValidationReport")
    proposals = report.remediation.proposals
    application = report.remediation.application
    # One gate for everything (status, completion, post state, attestation, live transport, identity provenance from a
    # live-suitable repository verifier, and that the runner itself issued this exact report).
    gate_failures = evidence_gate_failures(report, tested_state="post_remediation")
    identity = report.identity
    return Phase14Handoff(
        not_usable_reasons=gate_failures,
        identity_source_id=identity.source_id if identity else None,
        identity_independently_verified=bool(identity and identity.independently_verified),
        identity_live_suitable=bool(identity and identity.live_suitable and identity.authoritative_registry),
        contract_version=report.contract_version,
        validation_run_id=report.validation_run_id,
        incident_id=report.incident_id,
        status=report.status,
        reason_code=report.reason_code,
        usable_as_post_remediation_evidence=not gate_failures,
        scenario_id=report.scenario.scenario_id,
        scenario_definition_sha256=report.scenario.definition_sha256,
        oracle_id=report.oracle.oracle_id if report.oracle else None,
        tested_state=report.lab_state.tested_state,
        pre_remediation_state_id=report.lab_state.pre_remediation_state_id,
        post_remediation_state_id=report.lab_state.post_remediation_state_id,
        requested_origin=report.target.requested_origin if report.target else None,
        check_attempted=report.execution.attempted,
        check_completed=report.execution.completed,
        transport_kind=report.execution.transport_kind,
        requests_sent=report.execution.requests_sent,
        observation_ids=[item.observation_id for item in report.evidence],
        condition_results=({c.condition_id: c.result for c in report.oracle.conditions} if report.oracle else {}),
        proposal_ids=[p.proposal_id for p in proposals],
        applied_proposal_ids=list(application.applied_proposal_ids),
        application_state=application.state,
        attestation_id=application.attestation_id,
        attested_by=application.attested_by,
        unmet_preconditions=[item.precondition for item in report.preconditions if not item.satisfied],
        observed_evidence_ids=_ordered_unique([p.observed_evidence_ids for p in proposals]),
        hypothesis_ids=_ordered_unique([p.hypothesis_ids for p in proposals]),
        configuration_fact_ids=_ordered_unique([p.configuration_fact_ids for p in proposals]),
        reference_ids=_ordered_unique([p.reference_ids for p in proposals]),
        reference_chunk_ids=_ordered_unique([p.reference_chunk_ids for p in proposals]),
        interpretation_limits=list(report.limitations),
    )


def serialize(model: ValidationReport | Phase14Handoff) -> str:
    """Deterministic JSON: sorted keys, fixed indent, no timestamps added, same input gives the same bytes."""
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
