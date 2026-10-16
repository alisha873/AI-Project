"""
Phase 14 decision contract (pure data; no I/O).

A `JudgeAssessment` is what the Judge Agent hands to the dashboard and to later risk components. It is not a
security certificate. `supported` means the named claim is justified by the evidence that was actually checked.
It does not mean the application is secure, that residual risk is zero, or that a proposal was applied.

Decision vocabulary
-------------------
supported       Sufficient, schema-valid, internally consistent evidence supports the named claim.
not_supported   Sufficient valid evidence does not support the named claim. This is not proof of the opposite
                claim beyond the evidence that was checked, and it is not proof that a system is secure.
inconclusive    Evidence is missing, malformed, contradictory, ambiguous, or not usable for the claim.
blocked         Authorization, scope, approval, or another safety gate prevented execution. A blocked test is
                not a successful remediation.
not_evaluated   No valid assessment was performed (no report, a report that never ran, or input that was rejected).

`confidence` is intentionally absent. The pipeline stub `JudgeDecision.confidence` remains only as a
non-probability placeholder (always 0.0 from this agent) and cannot change a decision.

Provenance
----------
mocked      A fake-transport simulation. Never live evidence.
dry_run     No check was executed.
live        The Phase 13 evidence gate accepted the report as post-remediation evidence. This build's gate
            does not accept reports, so normal operation does not produce this value.
unverified  A check was recorded, but it is not a labelled simulation and it did not pass the evidence gate.
            It is not live evidence.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from evidence.schema import JudgeDecision
from validation.contract import Identifier, Text, _Strict

JUDGE_SCHEMA_VERSION = "1.0"

Decision = Literal["supported", "not_supported", "inconclusive", "blocked", "not_evaluated"]
DECISIONS = ("supported", "not_supported", "inconclusive", "blocked", "not_evaluated")

Provenance = Literal["mocked", "dry_run", "live", "unverified"]
Sufficiency = Literal["sufficient", "insufficient", "not_applicable"]
EvidenceRole = Literal[
    "validation_observation",
    "oracle_condition",
    "observed_incident_evidence",
    "hypothesis",
    "configuration",
    "reference",
    "proposal",
]
UsedAs = Literal["supporting", "contradicting", "excluded", "context"]

ExecutionMode = Literal["dry-run", "live-refused-opt-in", "live-refused-target", "live-refused-origin", "live"]
EXECUTION_MODES = ("dry-run", "live-refused-opt-in", "live-refused-target", "live-refused-origin", "live")

# Machine-readable reasons the judge itself assigns. These are not Phase 13 reason codes.
JudgeReason = Literal[
    "report_missing",
    "report_invalid",
    "confidence_rejected",
    "claim_unknown",
    "blocked_execution",
    "validation_not_performed",
    "upstream_inconclusive",
    "rules_confirmed",
    "rules_conflict",
    "definition_unconfirmed",
    "dangling_evidence",
    "observations_contradict",
    "evidence_id_ambiguous",
    "execution_mode_conflict",
    "evidence_gate_unsatisfied",
    "claim_not_supported_by_evidence",
    "security_claim_not_supported",
    "proposal_not_executed",
    "attestation_not_verified",
    "mocked_simulation",
]

CLAIM_OBJECTIVE_OBSERVED = "objective_observed_under_tested_conditions"
CLAIM_OBJECTIVE_NOT_OBSERVED = "objective_not_observed_under_tested_conditions"
CLAIM_REMEDIATION_APPLIED = "remediation_applied"
CLAIM_APPLICATION_SECURE = "application_is_secure"
CLAIM_NOT_ASSESSED = "not_assessed"

CLAIM_TEXT = {
    CLAIM_OBJECTIVE_OBSERVED: (
        "The original attack objective was observed under the tested conditions only."
    ),
    CLAIM_OBJECTIVE_NOT_OBSERVED: (
        "The original attack objective was not observed under the tested conditions. "
        "This does not show that the application is secure or that the objective is impossible."
    ),
    CLAIM_REMEDIATION_APPLIED: (
        "The remediation proposals were applied, and that application was verified."
    ),
    CLAIM_APPLICATION_SECURE: (
        "The application is secure, or the tested vulnerability has been eliminated."
    ),
    CLAIM_NOT_ASSESSED: "No claim was assessed.",
}

_SECRET = re.compile(
    r"(?i)\b(bearer|authorization|api[_-]?key|password|passwd|secret|token|cookie|session|credential)\b\s*[:=]?\s*\S+"
)


def redact_text(value: str) -> str:
    """Remove secret-shaped substrings. The replacement does not include the original value."""
    return _SECRET.sub("[redacted]", value)


class EvidenceRecord(_Strict):
    """One identifier the judge actually saw, with its role and how the decision used it."""

    evidence_id: Identifier
    role: EvidenceRole
    source: Identifier
    used_as: UsedAs
    exclusion_reason: Identifier | None = None

    @model_validator(mode="after")
    def _reason(self) -> "EvidenceRecord":
        if self.used_as == "excluded" and not self.exclusion_reason:
            raise ValueError("excluded evidence needs a reason")
        if self.used_as != "excluded" and self.exclusion_reason is not None:
            raise ValueError("only excluded evidence has an exclusion reason")
        return self


class JudgeAssessment(_Strict):
    """Versioned, immutable judge result. Unknown keys and coerced types are rejected."""

    schema_version: Literal["1.0"] = JUDGE_SCHEMA_VERSION
    assessed_at_utc: str
    incident_id: Identifier | None = None
    scenario_id: Identifier | None = None
    validation_run_id: Identifier | None = None
    claim_id: Identifier
    claim: Text
    decision: Decision
    rationale: Text
    reason_codes: list[JudgeReason] = Field(min_length=1)
    supporting_evidence_ids: list[Identifier] = []
    contradicting_evidence_ids: list[Identifier] = []
    missing_evidence: list[Text] = []
    evidence_sufficiency: Sufficiency
    evidence_index: list[EvidenceRecord] = []
    upstream_status: Identifier | None = None
    upstream_reason_code: Identifier | None = None
    upstream_plan_status: Identifier | None = None
    handoff_version: Identifier | None = None
    usable_as_post_remediation_evidence: bool = False
    not_usable_reasons: list[Identifier] = []
    transport_kind: Identifier | None = None
    execution_mode: ExecutionMode | None = None
    limitations: list[Text] = Field(min_length=1)
    upstream_limitations: list[Text] = []
    eligible_for_downstream: bool
    evidence_provenance: Provenance
    live_evidence: bool
    caller_confidence_ignored: bool = False
    caller_confidence_rejected: bool = False
    # Constants: this agent never does these, so a result cannot say that it did.
    llm_used: Literal[False] = False
    network_used: Literal[False] = False
    confidence_used: Literal[False] = False
    security_guarantee: Literal[False] = False
    remediation_effectiveness_proven: Literal[False] = False
    remediation_application_verified: Literal[False] = False
    general_security_guarantee: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "JudgeAssessment":
        _parse_utc(self.assessed_at_utc)
        if self.caller_confidence_ignored and self.caller_confidence_rejected:
            raise ValueError("a confidence value cannot be both ignored and rejected")
        if self.confidence_used or self.llm_used or self.network_used:
            raise ValueError("this agent does not use confidence, a model, or the network")
        if self.security_guarantee or self.remediation_effectiveness_proven or self.general_security_guarantee:
            raise ValueError("a judge result cannot claim a security guarantee")
        if self.remediation_application_verified:
            raise ValueError("phase 13 does not verify that a remediation was applied")
        self._provenance()
        self._decision()
        self._citations()
        if self.claim_id == CLAIM_APPLICATION_SECURE and self.decision == "supported":
            raise ValueError("the claim that the application is secure cannot be supported")
        if self.claim_id == CLAIM_REMEDIATION_APPLIED and self.decision == "supported":
            raise ValueError("a proposal or an attestation is not verified application")
        return self

    def _provenance(self) -> None:
        if self.evidence_provenance == "live":
            if not (self.live_evidence and self.usable_as_post_remediation_evidence):
                raise ValueError("live provenance requires the phase 13 evidence gate")
            if self.not_usable_reasons:
                raise ValueError("live evidence has no not-usable reasons")
        else:
            if self.live_evidence or self.eligible_for_downstream:
                raise ValueError("only live evidence can be labelled live or sent downstream")
        if self.evidence_provenance == "mocked" and self.live_evidence:
            raise ValueError("mocked evidence cannot be labelled live")
        if self.usable_as_post_remediation_evidence and self.evidence_provenance != "live":
            raise ValueError("usable post-remediation evidence is live provenance")
        if self.evidence_provenance == "live" and self.decision in ("supported", "not_supported"):
            if not self.eligible_for_downstream:
                raise ValueError("a live supported or not_supported result is eligible for downstream use")

    def _decision(self) -> None:
        if self.decision in ("supported", "not_supported"):
            if self.evidence_sufficiency != "sufficient":
                raise ValueError(f"{self.decision} requires sufficient evidence")
            if self.evidence_provenance not in ("mocked", "live"):
                raise ValueError(f"{self.decision} requires mocked or live provenance")
        elif self.decision == "inconclusive":
            if self.evidence_sufficiency != "insufficient":
                raise ValueError("inconclusive evidence is insufficient")
            if self.eligible_for_downstream:
                raise ValueError("an inconclusive result is not eligible for downstream use")
        else:
            if self.evidence_sufficiency != "not_applicable":
                raise ValueError(f"{self.decision} has no sufficiency score")
            if self.eligible_for_downstream:
                raise ValueError(f"{self.decision} is not eligible for downstream use")
        if self.decision == "supported" and not self.supporting_evidence_ids:
            raise ValueError("supported requires at least one supporting evidence id")
        if self.decision == "not_supported" and not self.contradicting_evidence_ids:
            raise ValueError("not_supported requires at least one contradicting evidence id")
        if self.decision in ("blocked", "not_evaluated") and (
            self.supporting_evidence_ids or self.contradicting_evidence_ids
        ):
            raise ValueError(f"{self.decision} does not cite evidence as support or contradiction")

    def _citations(self) -> None:
        known = [item.evidence_id for item in self.evidence_index]
        if len(known) != len(set(known)):
            raise ValueError("an evidence id appears more than once")
        by_id = {item.evidence_id: item for item in self.evidence_index}
        overlap = set(self.supporting_evidence_ids) & set(self.contradicting_evidence_ids)
        if overlap:
            raise ValueError("an evidence id cannot both support and contradict the same decision")
        for evidence_id in self.supporting_evidence_ids:
            record = by_id.get(evidence_id)
            if record is None or record.used_as != "supporting":
                raise ValueError(f"supporting id {evidence_id} is not in the evidence index as supporting")
        for evidence_id in self.contradicting_evidence_ids:
            record = by_id.get(evidence_id)
            if record is None or record.used_as != "contradicting":
                raise ValueError(f"contradicting id {evidence_id} is not in the evidence index as contradicting")
        derived_support = [item.evidence_id for item in self.evidence_index if item.used_as == "supporting"]
        derived_contra = [item.evidence_id for item in self.evidence_index if item.used_as == "contradicting"]
        if derived_support != list(self.supporting_evidence_ids) or derived_contra != list(self.contradicting_evidence_ids):
            raise ValueError("evidence id lists must match the evidence index")


def _parse_utc(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("assessed_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("assessed_at_utc must carry a time zone")


def serialize(assessment: JudgeAssessment) -> str:
    """Deterministic JSON. The same assessment always produces the same bytes."""
    if not isinstance(assessment, JudgeAssessment):
        raise TypeError("assessment must be a JudgeAssessment")
    return json.dumps(assessment.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def to_pipeline_decision(assessment: JudgeAssessment) -> JudgeDecision:
    """Project a judge assessment onto the pipeline `JudgeDecision` stub.

    `confidence` is always 0.0. That means no score was computed. It is not a probability and it is not an
    input to the decision.
    """
    if not isinstance(assessment, JudgeAssessment):
        raise TypeError("assessment must be a JudgeAssessment")
    return JudgeDecision(
        decision=assessment.decision,
        confidence=0.0,
        reasoning=assessment.rationale,
        rationale=assessment.rationale,
        supporting_evidence=list(assessment.supporting_evidence_ids),
        schema_version=assessment.schema_version,
        incident_id=assessment.incident_id or "",
        scenario_id=assessment.scenario_id or "",
        claim_id=assessment.claim_id,
        claim=assessment.claim,
        contradicting_evidence=list(assessment.contradicting_evidence_ids),
        missing_evidence=list(assessment.missing_evidence),
        evidence_sufficiency=assessment.evidence_sufficiency,
        upstream_status=assessment.upstream_status or "",
        upstream_reason_code=assessment.upstream_reason_code or "",
        limitations=list(assessment.limitations),
        eligible_for_downstream=assessment.eligible_for_downstream,
        evidence_provenance=assessment.evidence_provenance,
        live_evidence=assessment.live_evidence,
    )
