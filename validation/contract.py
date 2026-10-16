"""
Phase 13 - the versioned validation-report contract.

A `ValidationReport` is what Phase 13 hands to later phases (Phase 14, the dashboard). It answers, in
machine-readable fields, six separate questions that must never be conflated:

1. What remediation was PROPOSED?          `remediation.proposals` (copied from Phase 12, always "proposed").
2. Was it actually APPLIED, and who says so? `remediation.application` (attested / unknown / not applicable).
3. Did a check actually RUN, against which lab state?  `execution`, `lab_state`.
4. Was the original objective OBSERVED?    `status == "objective_achieved"`.
5. Was it NOT observed (under the tested conditions only)?  `status == "objective_not_observed"`.
6. Could nothing be concluded?             `not_validated`, `blocked`, `inconclusive`.

Status model (the exact semantics; `REASONS_BY_STATUS` lists the allowed machine-readable reasons)
-------------------------------------------------------------------------------------------------
not_validated           No check was attempted because an INPUT precondition is unmet (no usable remediation
                        plan, unknown or ineligible scenario, remediation application not attested).
blocked                 Every input precondition is met but execution is PROHIBITED (the scope policy denied
                        it, or live execution does not exist in this build).
inconclusive            A check was attempted but no conclusion can be drawn (lab unreachable, timeout,
                        unexpected response, failed control, insufficient evidence, unknown lab state).
objective_achieved      A completed check OBSERVED the original objective succeed in the tested lab state.
objective_not_observed  A completed check did NOT observe the objective, under the tested conditions only.

An absent result is never a success: `not_validated`, `blocked` and `inconclusive` carry no
evidence of effectiveness, a timeout or a missing lab is `inconclusive`, and even `objective_not_observed`
is NOT proof that the objective is unachievable or that the application is secure (see
`STANDARD_LIMITATIONS`). Invalid combinations (for example `objective_not_observed` with no executed check,
no evidence or no attestation of the remediation) are rejected when the report is constructed.

This module is data only. It performs no I/O and creates no network objects.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

CONTRACT_VERSION = "1.0"
# Revision 1.1 (Phase 13B.3) adds the optional `identity` provenance record. Revision 1.0 reports stay valid and are
# never reinterpreted: a 1.0 report that ran a check simply has no identity provenance, so it can never be usable
# post-remediation evidence (see `validation.evidence_gate`). `CONTRACT_VERSION` stays the default for reports that
# ran nothing.
CONTRACT_VERSION_IDENTITY = "1.1"
SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_VERSION_IDENTITY)

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

ValidationStatus = Literal[
    "not_validated", "blocked", "inconclusive", "objective_achieved", "objective_not_observed"
]
STATUSES = ("not_validated", "blocked", "inconclusive", "objective_achieved", "objective_not_observed")
NOT_ATTEMPTED_STATUSES = ("not_validated", "blocked")
CONCLUSIVE_STATUSES = ("objective_achieved", "objective_not_observed")

REASONS_BY_STATUS: dict[str, tuple[str, ...]] = {
    "not_validated": (
        "remediation_plan_missing",
        "remediation_plan_malformed",
        "remediation_plan_rejected",
        "remediation_plan_unavailable",
        "remediation_plan_insufficient_context",
        "incident_mismatch",
        "scenario_unknown",
        "scenario_not_eligible",
        "remediation_application_not_attested",
    ),
    "blocked": (
        "scope_policy_denied",
        "execution_not_implemented_phase_13a",
        "execution_not_enabled",
        "live_transport_not_enabled",
        "transport_unavailable",
    ),
    "inconclusive": (
        "lab_unreachable",
        "timeout",
        "unexpected_response",
        "control_check_failed",
        "evidence_insufficient",
        "lab_state_unknown",
    ),
    "objective_achieved": ("objective_observed",),
    "objective_not_observed": ("objective_not_observed_under_tested_conditions",),
}
ALL_REASON_CODES = tuple(code for codes in REASONS_BY_STATUS.values() for code in codes)

# The fixed order in which input and policy preconditions are checked (the first unmet one is the primary
# reason of a not_validated / blocked report). The first six are INPUT preconditions (-> not_validated);
# the last two are PERMISSION preconditions (-> blocked).
PRECONDITIONS = (
    "plan_usable",
    "incident_matches_plan",
    "scenario_known",
    "scenario_eligible",
    "remediation_application_attested",
    "scope_conditions_met",
    "live_execution_available",
)
INPUT_PRECONDITIONS = PRECONDITIONS[:5]
PERMISSION_PRECONDITIONS = PRECONDITIONS[5:]
Precondition = Literal[
    "plan_usable",
    "incident_matches_plan",
    "scenario_known",
    "scenario_eligible",
    "remediation_application_attested",
    "scope_conditions_met",
    "live_execution_available",
]

TransportKind = Literal["none", "fake", "live"]
TestedState = Literal["not_tested", "unknown", "pre_remediation", "post_remediation"]
AttestationState = Literal["independently_attested", "unknown", "not_applicable"]
ConditionResult = Literal["met", "not_met", "not_evaluated"]
ReviewStatus = Literal["pending_human_review", "approved", "rejected"]

STANDARD_LIMITATIONS = (
    "A validation result describes only the lab state, scenario, request and conditions that were actually "
    "tested. It is not a universal security guarantee and it does not mean residual risk is zero.",
    "A failed or absent attack observation is not proof that the objective is unachievable: other payloads, "
    "encodings, endpoints or states were not tested.",
    "A proposed remediation is not an applied remediation. Phase 13 does not apply remediation; whether and "
    "how one was applied is stated by the attestation recorded here and is not verified by Phase 13.",
    "The free-text validation procedure written by the Phase 12 planner is never interpreted or executed.",
    "The original objective comes from an operator-side metadata file, not from a model and not from the "
    "evaluation answer key. It describes the attack that was generated in the lab, which may not be what a "
    "real attacker would try.",
)


class _Strict(BaseModel):
    # strict: no coercion; unknown keys are errors; instances are immutable (same convention as Phase 12).
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _parse_utc(value: str, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must carry a time zone (use UTC)")


# ============================================================
# COMPONENTS
# ============================================================

class ScenarioReference(_Strict):
    """Which approved scenario definition was used. `registered=False` means it was not found: no facts."""

    scenario_id: Identifier
    registered: bool
    definition_version: Identifier | None = None
    definition_sha256: Sha256 | None = None
    family: Identifier | None = None
    review_status: ReviewStatus | None = None
    eligible_for_execution: bool = False
    ambiguity_codes: list[Identifier] = []

    @model_validator(mode="after")
    def _consistent(self) -> "ScenarioReference":
        facts = (self.definition_version, self.definition_sha256, self.family, self.review_status)
        if not self.registered:
            if any(item is not None for item in facts) or self.eligible_for_execution or self.ambiguity_codes:
                raise ValueError("an unregistered scenario can not carry definition facts")
        elif any(item is None for item in facts):
            raise ValueError("a registered scenario must state its version, hash, family and review status")
        if self.eligible_for_execution and (self.review_status != "approved" or self.ambiguity_codes):
            raise ValueError("a scenario is eligible only when approved and free of open ambiguities")
        return self


class ObjectiveRecord(_Strict):
    """The original attack objective and how it was obtained. Never taken from the answer key."""

    summary: Text
    source_kind: Identifier
    source_ref: Identifier
    source_sha256: Sha256
    obtained_by: Identifier
    ground_truth_consulted: Literal[False] = False


class TargetRecord(_Strict):
    """The authorized lab identity and endpoint named by the scenario, and the target the caller supplied."""

    method: Identifier
    path: Identifier
    parameter: Identifier
    entry_service: Identifier
    target_service: Identifier
    data_service: Identifier
    requested_origin: str | None = None  # normalized scheme://host:port, only if it parsed
    lab_identity_verified: bool = False


class IdentityProvenance(_Strict):
    """How the lab identity behind a check that RAN was established (revision 1.1).

    This record is descriptive. It is written by the runner from the issuance record that `assess_lab_identity`
    created; it is never what makes a report trustworthy (a hand-written copy proves nothing, which is why
    `validation.evidence_gate` also requires that the runner issued the whole report)."""

    source_id: Identifier  # the verifier that made the observation
    observed_at_utc: str
    checked_at_utc: str  # the time the run checked that the observation was still fresh
    lab_state_id: Identifier | None = None
    independently_verified: Literal[True] = True
    live_suitable: bool
    authoritative_registry: bool

    @model_validator(mode="after")
    def _consistent(self) -> "IdentityProvenance":
        _parse_utc(self.observed_at_utc, "observed_at_utc")
        _parse_utc(self.checked_at_utc, "checked_at_utc")
        if self.live_suitable and not self.authoritative_registry:
            raise ValueError("only the repository's own verifier registry can make an identity live-suitable")
        return self


class LabStateRecord(_Strict):
    """Operator-supplied lab-state identifiers. Never invented; absent unless supplied and accepted."""

    tested_state: TestedState = "not_tested"
    pre_remediation_state_id: Identifier | None = None
    post_remediation_state_id: Identifier | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "LabStateRecord":
        if self.tested_state == "pre_remediation" and self.pre_remediation_state_id is None:
            raise ValueError("a pre-remediation test needs the pre-remediation lab-state id")
        if self.tested_state == "post_remediation" and self.post_remediation_state_id is None:
            raise ValueError("a post-remediation test needs the post-remediation lab-state id")
        return self


class ProposalReference(_Strict):
    """One Phase 12 proposal: identifiers only, with the four id categories kept in separate lists."""

    proposal_id: Identifier
    priority_rank: int
    proposal_status: Literal["proposed"] = "proposed"
    executed: Literal[False] = False
    observed_evidence_ids: list[Identifier]
    hypothesis_ids: list[Identifier]
    configuration_fact_ids: list[Identifier]
    reference_ids: list[Identifier]
    reference_chunk_ids: list[Identifier]


class HypothesisReference(_Strict):
    """A Phase 9 hypothesis exactly as Phase 12 carried it; never upgraded."""

    hypothesis_id: Identifier
    status: Identifier
    is_confirmed: bool


class ApplicationRecord(_Strict):
    """Whether the remediation was applied, and who says so. Phase 13 never applies and never verifies it."""

    state: AttestationState = "unknown"
    attestation_id: Identifier | None = None
    attested_by: Identifier | None = None
    applied_proposal_ids: list[Identifier] = []
    verified_by_phase13: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "ApplicationRecord":
        attested = self.state == "independently_attested"
        if attested and (not self.attestation_id or not self.attested_by or not self.applied_proposal_ids):
            raise ValueError("an attestation needs its id, its author and the applied proposal ids")
        if not attested and (self.attestation_id or self.attested_by or self.applied_proposal_ids):
            raise ValueError("attestation details are only allowed when the state is independently_attested")
        return self


class RemediationRecord(_Strict):
    proposals: list[ProposalReference]
    application: ApplicationRecord
    applied_by_phase13: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "RemediationRecord":
        known = {proposal.proposal_id for proposal in self.proposals}
        if not set(self.application.applied_proposal_ids) <= known:
            raise ValueError("an attested proposal id is not among the proposals evaluated")
        return self


class ConditionRecord(_Strict):
    condition_id: Identifier
    result: ConditionResult = "not_evaluated"


class OracleRecord(_Strict):
    oracle_id: Identifier
    oracle_version: Identifier
    conditions: list[ConditionRecord]


class ExecutionRecord(_Strict):
    """Execution metadata. Every field stays at its 'nothing ran' value unless a check really ran.

    `transport_kind` says WHAT answered: `fake` is a deterministic in-memory stand-in (a SIMULATION that says
    nothing about any real lab) and `live` is a real transport. Only a `live` result can ever be used as
    post-remediation evidence (see `validation.handoff`)."""

    attempted: bool = False
    completed: bool = False
    requests_sent: int = 0
    runner_id: Identifier | None = None
    transport: Identifier | None = None
    transport_kind: TransportKind = "none"
    started_at_utc: str | None = None
    finished_at_utc: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "ExecutionRecord":
        details = (self.runner_id, self.transport, self.started_at_utc, self.finished_at_utc)
        if not self.attempted:
            if (self.completed or self.requests_sent or self.transport_kind != "none"
                    or any(item is not None for item in details)):
                raise ValueError("execution metadata can not exist when no check was attempted")
            return self
        if self.transport_kind == "none" or self.transport is None:
            raise ValueError("an attempted check must record its transport and whether it was fake or live")
        if self.requests_sent < 0:
            raise ValueError("requests_sent can not be negative")
        if self.runner_id is None or self.started_at_utc is None:
            raise ValueError("an attempted check must record its runner and start time")
        _parse_utc(self.started_at_utc, "started_at_utc")
        if self.finished_at_utc is not None:
            _parse_utc(self.finished_at_utc, "finished_at_utc")
        if self.completed and (self.finished_at_utc is None or self.requests_sent < 1):
            raise ValueError("a completed check must record a finish time and at least one request")
        return self


class ObservationEvidence(_Strict):
    """Something an executed check actually observed. Summaries and hashes, not raw response bodies."""

    observation_id: Identifier
    kind: Identifier
    summary: Text
    captured_at_utc: str
    attributes: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> "ObservationEvidence":
        _parse_utc(self.captured_at_utc, "captured_at_utc")
        return self


class ScopeRecord(_Strict):
    """The scope-policy decision. `execution_permitted` is a permission, not proof that anything ran."""

    policy_version: Identifier
    scope_conditions_met: bool
    execution_permitted: bool
    reason_codes: list[Identifier]


class PreconditionFinding(_Strict):
    """One checked precondition. `evaluated=False` means an earlier precondition already failed, so this one
    could not be judged; it counts as unmet (never as satisfied) and has no reason code of its own."""

    precondition: Precondition
    satisfied: bool
    evaluated: bool = True
    reason_code: Identifier | None = None
    detail: Text

    @model_validator(mode="after")
    def _consistent(self) -> "PreconditionFinding":
        if self.satisfied and (self.reason_code is not None or not self.evaluated):
            raise ValueError("a satisfied precondition was evaluated and has no reason code")
        if not self.satisfied and self.evaluated and self.reason_code is None:
            raise ValueError("an evaluated, unmet precondition needs a reason code")
        if not self.evaluated and self.reason_code is not None:
            raise ValueError("an unevaluated precondition has no reason code")
        if self.reason_code is not None and self.reason_code not in (
            *REASONS_BY_STATUS["not_validated"], *REASONS_BY_STATUS["blocked"]
        ):
            raise ValueError(f"{self.reason_code!r} is not a precondition reason code")
        return self


class UpstreamRecord(_Strict):
    """What Phase 12 reported, as a status only. No Phase 12 free text is copied."""

    plan_result_status: Identifier | None = None
    plan_outcome: Identifier | None = None
    plan_schema_version: Identifier | None = None
    finding_index: int | None = None
    hypotheses: list[HypothesisReference] = []


# ============================================================
# THE REPORT
# ============================================================

class ValidationReport(_Strict):
    contract_version: Literal["1.0", "1.1"] = CONTRACT_VERSION
    validation_run_id: Identifier
    incident_id: Identifier
    scenario: ScenarioReference
    objective: ObjectiveRecord | None
    target: TargetRecord | None
    lab_state: LabStateRecord
    remediation: RemediationRecord
    status: ValidationStatus
    reason_code: Identifier
    oracle: OracleRecord | None
    execution: ExecutionRecord
    evidence: list[ObservationEvidence]
    scope: ScopeRecord
    preconditions: list[PreconditionFinding]
    upstream: UpstreamRecord
    limitations: list[Text]
    validity_conditions: list[Text]
    # Revision 1.1 only: how the identity behind a check that ran was established. Always None in revision 1.0.
    identity: IdentityProvenance | None = None
    # constants: Phase 13 never does any of these, so the report can never say it did
    remediation_effectiveness_proven: Literal[False] = False
    general_security_guarantee: Literal[False] = False
    ground_truth_used: Literal[False] = False
    llm_used: Literal[False] = False

    @model_validator(mode="after")
    def _status_rules(self) -> "ValidationReport":
        if self.reason_code not in REASONS_BY_STATUS[self.status]:
            raise ValueError(f"reason code {self.reason_code!r} is not allowed for status {self.status!r}")
        if not self.limitations:
            raise ValueError("a report must state its limitations")
        if self.status in NOT_ATTEMPTED_STATUSES:
            self._check_not_attempted()
        else:
            self._check_attempted()
        if self.execution.transport_kind == "fake" and not any(
            "simulat" in item.lower() for item in (*self.limitations, *self.validity_conditions)
        ):
            raise ValueError("a result from a fake transport must state that it is a simulation")
        if self.contract_version == CONTRACT_VERSION and self.identity is not None:
            raise ValueError("identity provenance belongs to contract revision 1.1")
        if self.contract_version == CONTRACT_VERSION_IDENTITY:
            if self.execution.attempted and self.identity is None:
                raise ValueError("a revision 1.1 report that ran a check must record its identity provenance")
            if not self.execution.attempted and self.identity is not None:
                raise ValueError("a report that ran nothing has no identity provenance")
        return self

    def _check_not_attempted(self) -> None:
        if self.execution.attempted or self.evidence or self.lab_state.tested_state != "not_tested":
            raise ValueError(f"{self.status} means nothing ran: no execution, no evidence, no tested state")
        if any(c.result != "not_evaluated" for c in (self.oracle.conditions if self.oracle else [])):
            raise ValueError("no condition can be evaluated when nothing ran")
        if self.scope.execution_permitted and self.status == "blocked":
            raise ValueError("a blocked report can not carry an execution permission")
        unmet = [item for item in self.preconditions if not item.satisfied]
        if not unmet or not unmet[0].evaluated or unmet[0].reason_code != self.reason_code:
            raise ValueError("the reason code must be that of the first unmet precondition")
        input_unmet = any(item.precondition in INPUT_PRECONDITIONS for item in unmet)
        if self.status == "not_validated" and not input_unmet:
            raise ValueError("not_validated needs an unmet input precondition")
        if self.status == "blocked" and input_unmet:
            raise ValueError("blocked is only for reports whose input preconditions are all met")

    def _check_attempted(self) -> None:
        if not self.execution.attempted:
            raise ValueError(f"{self.status} requires an attempted check")
        if not (self.scope.scope_conditions_met and self.scope.execution_permitted):
            raise ValueError("a check can only have run if the scope policy permitted it")
        if any(not item.satisfied for item in self.preconditions):
            raise ValueError("a check can only have run if every precondition was met")
        if self.target is None or not self.target.lab_identity_verified:
            raise ValueError("a check can only have run against a verified lab identity")
        if self.status == "inconclusive":
            return
        # a conclusive result: it must be complete, evidenced and tied to a known lab state
        if not self.execution.completed or not self.evidence:
            raise ValueError("a conclusive result needs a completed check and observed evidence")
        if self.lab_state.tested_state not in ("pre_remediation", "post_remediation"):
            raise ValueError("a conclusive result needs a known tested lab state")
        if self.oracle is None or not self.oracle.conditions or any(
            c.result == "not_evaluated" for c in self.oracle.conditions
        ):
            raise ValueError("a conclusive result needs every oracle condition evaluated")
        if self.lab_state.tested_state == "post_remediation" and (
            self.remediation.application.state != "independently_attested"
        ):
            raise ValueError("a post-remediation result needs an attested remediation application")
        if self.status == "objective_achieved" and not any(c.result == "met" for c in self.oracle.conditions):
            raise ValueError("objective_achieved needs at least one condition observed as met")
