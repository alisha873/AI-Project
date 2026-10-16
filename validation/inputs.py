"""
Phase 13 - the adapter from the Phase 12 remediation-plan result to Phase 13's own inputs (pure; no I/O).

Phase 12 is advisory and FALLIBLE, and its contract is left exactly as it is. This adapter reads only its
public, versioned interface: a `RemediationPlanResult` (or the dictionary its `to_dict()` produces, as stored
in a Phase 12 report) or a bare `RemediationPlan`. It re-validates the plan with the real Phase 12 model and
fails closed. It never repairs, completes or invents a plan.

Outcomes (`PlanIntake.reason_code` when `usable` is False; all are `not_validated` reasons):
  remediation_plan_missing               nothing was supplied.
  remediation_plan_malformed             wrong type, unknown status, inconsistent status/plan/validation,
                                         a plan that does not satisfy the Phase 12 schema, duplicate ids.
  remediation_plan_rejected              the planner's own output was rejected (invalid or ungrounded).
  remediation_plan_unavailable           the planner could not produce a plan (the model call failed).
  remediation_plan_insufficient_context  an explicit insufficient-evidence outcome (no proposals).

Only identifiers and statuses are copied into the Phase 13 records. Phase 12's free text (including the
planner's free-text `validation.procedure`) is deliberately NOT copied, interpreted or executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from remediation.contract import PlanOutcome, RemediationPlan, RemediationProposal
from remediation.planner import RemediationPlanResult
from validation.contract import (
    ApplicationRecord,
    HypothesisReference,
    Identifier,
    ProposalReference,
    UpstreamRecord,
    _Strict,
)

KNOWN_PLAN_STATUSES = (
    "planned",
    "not_planned_insufficient_context",
    "model_declared_insufficient",
    "llm_error",
    "rejected_invalid_response",
    "rejected_grounding",
)
_REJECTED = ("rejected_invalid_response", "rejected_grounding")
_INSUFFICIENT = ("not_planned_insufficient_context", "model_declared_insufficient")
PLAN_ONLY = "plan_only"  # a bare RemediationPlan was supplied; there is no result status to report


class RemediationApplicationAttestation(_Strict):
    """A CLAIM, made by an authorized operator or a separately identified component, that remediation was
    applied to the lab. Phase 13 records it, checks it against the plan's proposal ids, and does not verify it
    (`ApplicationRecord.verified_by_phase13` is always False). It never applies anything itself."""

    attestation_id: Identifier
    attested_by: Identifier
    applied_proposal_ids: list[Identifier]
    post_remediation_lab_state_id: Identifier
    pre_remediation_lab_state_id: Identifier | None = None


@dataclass(frozen=True)
class AttestationAssessment:
    accepted: bool
    detail: str
    application: ApplicationRecord = field(default_factory=ApplicationRecord)
    pre_state_id: str | None = None
    post_state_id: str | None = None


def assess_attestation(attestation: object, intake: PlanIntake) -> AttestationAssessment:
    """Accept an attestation only if the plan is usable and it names a non-empty subset of the plan's proposals.

    A rejected or absent attestation leaves the application state `unknown` and records no lab-state ids.
    """
    if not intake.usable:
        return AttestationAssessment(False, "There is no usable plan, so no attestation can be matched.")
    if attestation is None:
        return AttestationAssessment(False, "No attestation that the remediation was applied was supplied; a "
                                            "proposal is not an applied remediation.")
    if not isinstance(attestation, RemediationApplicationAttestation):
        return AttestationAssessment(False, "The attestation has an unsupported type.")
    known = {proposal.proposal_id for proposal in intake.proposals}
    applied = attestation.applied_proposal_ids
    if not applied or len(set(applied)) != len(applied) or not set(applied) <= known:
        return AttestationAssessment(False, "The attestation names proposal ids that are not in the plan.")
    return AttestationAssessment(
        True,
        "An attestation names proposals from the plan; Phase 13 records it and does not verify it.",
        ApplicationRecord(
            state="independently_attested",
            attestation_id=attestation.attestation_id,
            attested_by=attestation.attested_by,
            applied_proposal_ids=list(applied),
        ),
        pre_state_id=attestation.pre_remediation_lab_state_id,
        post_state_id=attestation.post_remediation_lab_state_id,
    )


@dataclass(frozen=True)
class PlanIntake:
    usable: bool
    reason_code: str | None
    detail: str
    incident_id: str | None = None
    proposals: tuple[ProposalReference, ...] = ()
    upstream: UpstreamRecord = field(default_factory=UpstreamRecord)


def _fail(reason: str, detail: str, upstream: UpstreamRecord | None = None) -> PlanIntake:
    return PlanIntake(False, reason, detail, upstream=upstream or UpstreamRecord())


def _proposal_reference(proposal: RemediationProposal) -> ProposalReference:
    return ProposalReference(
        proposal_id=proposal.proposal_id,
        priority_rank=proposal.priority_rank,
        observed_evidence_ids=list(proposal.observed_evidence_ids),
        hypothesis_ids=list(proposal.hypothesis_ids),
        configuration_fact_ids=list(proposal.configuration_fact_ids),
        reference_ids=list(proposal.reference_ids),
        reference_chunk_ids=[reference.chunk_id for reference in proposal.references],
    )


def _hypotheses(plan: RemediationPlan) -> list[HypothesisReference]:
    seen: dict[str, HypothesisReference] = {}
    for proposal in plan.proposals:
        for item in proposal.linked_hypotheses:
            seen.setdefault(
                item.hypothesis_id,
                HypothesisReference(hypothesis_id=item.hypothesis_id, status=item.status,
                                    is_confirmed=item.is_confirmed),
            )
    return list(seen.values())


def _split(plan_input: object) -> tuple[str, Any, Any, Any, Any] | None:
    """(status, plan, validation_passed, incident_id, finding_index) from a supported input, else None."""
    if isinstance(plan_input, RemediationPlanResult):
        passed = None if plan_input.validation is None else plan_input.validation.passed
        return plan_input.status, plan_input.plan, passed, plan_input.incident_id, plan_input.finding_index
    if isinstance(plan_input, dict):
        validation = plan_input.get("validation")
        passed = validation.get("passed") if isinstance(validation, dict) else None
        return (plan_input.get("status"), plan_input.get("plan"), passed,
                plan_input.get("incident_id"), plan_input.get("finding_index"))
    if isinstance(plan_input, RemediationPlan):
        return PLAN_ONLY, plan_input, True, plan_input.incident_id, plan_input.finding_index
    return None


def _parse_plan(plan: object) -> RemediationPlan | None:
    """Re-validate with the real Phase 12 model (an instance built with model_construct is checked again)."""
    try:
        if isinstance(plan, RemediationPlan):
            return RemediationPlan.model_validate(plan.model_dump())
        if isinstance(plan, dict):
            return RemediationPlan.model_validate(plan)
    except ValidationError:
        return None
    return None


def intake_plan(plan_input: object) -> PlanIntake:
    if plan_input is None:
        return _fail("remediation_plan_missing", "No remediation plan result was supplied.")
    parts = _split(plan_input)
    if parts is None:
        return _fail("remediation_plan_malformed", "The remediation plan input has an unsupported type.")
    status, plan, validation_passed, incident_id, finding_index = parts
    if status != PLAN_ONLY and status not in KNOWN_PLAN_STATUSES:
        return _fail("remediation_plan_malformed", "The remediation plan result has an unknown status.")
    upstream = UpstreamRecord(
        plan_result_status=status,
        finding_index=finding_index if isinstance(finding_index, int) and not isinstance(finding_index, bool)
        else None,
    )

    if status == "llm_error" or status in _REJECTED:
        if plan is not None:
            return _fail("remediation_plan_malformed", f"A {status} result must not carry a plan.", upstream)
        reason = "remediation_plan_unavailable" if status == "llm_error" else "remediation_plan_rejected"
        return _fail(reason, f"The planner result is {status}; there is no usable plan.", upstream)

    parsed = _parse_plan(plan)
    if parsed is None:
        return _fail("remediation_plan_malformed",
                     "The plan is missing or does not satisfy the Phase 12 plan contract.", upstream)
    upstream = upstream.model_copy(update={
        "plan_outcome": parsed.outcome, "plan_schema_version": parsed.schema_version,
        "finding_index": parsed.finding_index,
    })
    if not isinstance(incident_id, str) or incident_id != parsed.incident_id:
        return _fail("remediation_plan_malformed", "The result and its plan name different incidents.", upstream)
    if isinstance(finding_index, int) and finding_index != parsed.finding_index:
        return _fail("remediation_plan_malformed", "The result and its plan name different findings.", upstream)

    outcome: PlanOutcome = parsed.outcome
    if status in _INSUFFICIENT or (status == PLAN_ONLY and outcome == "insufficient_evidence"):
        if parsed.proposals or outcome != "insufficient_evidence":
            return _fail("remediation_plan_malformed",
                         "An insufficient-evidence result must not carry proposals.", upstream)
        return _fail("remediation_plan_insufficient_context",
                     "The planner reported insufficient context; there is no remediation to validate.", upstream)

    # status "planned" (or a bare plan with proposals)
    if validation_passed is not True:
        return _fail("remediation_plan_malformed",
                     "A planned result must carry a passed Phase 12 grounding validation.", upstream)
    ids = [proposal.proposal_id for proposal in parsed.proposals]
    if outcome != "proposals" or not ids or len(set(ids)) != len(ids):
        return _fail("remediation_plan_malformed",
                     "A planned result needs at least one proposal and unique proposal ids.", upstream)
    return PlanIntake(
        usable=True,
        reason_code=None,
        detail="The Phase 12 plan satisfies the plan contract; its proposals are PROPOSED, not applied.",
        incident_id=parsed.incident_id,
        proposals=tuple(_proposal_reference(p) for p in parsed.proposals),
        upstream=upstream.model_copy(update={"hypotheses": _hypotheses(parsed)}),
    )
