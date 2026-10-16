"""
Phase 12 - the versioned remediation-plan contract.

Two families of models live here, deliberately kept apart:

1. DRAFT models (`PlanDraft`, `ProposalDraft`, ...). This is the ONLY shape the language model is asked
   to produce. They hold the model's own words and the identifiers it cites, nothing else. They only
   check structure (types, allowed values, lengths, no unknown keys); the grounding rules that need
   the incident context live in `remediation.grounding`.

2. OUTPUT models (`RemediationPlan`, `RemediationProposal`, ...). This is what AutoShield stores and
   hands to later phases. Everything that must be trustworthy is filled in by deterministic code, never
   by the model: the proposal ID, the priority rank and its basis, the hypothesis status (copied from
   Phase 9), the reference provenance (copied from the verified Phase 11D context) and the
   "proposed, not executed, human review required" markers.

Evidence categories stay separate in the contract
-------------------------------------------------
- `observed_evidence_ids`   incident events (for example `nginx-19`). OBSERVED.
- `hypothesis_ids`          Phase 9 findings (`H1`). INFERRED, never confirmed by the planner.
- `reference_ids`           retrieved CWE/CAPEC chunk ids. GENERAL KNOWLEDGE, never incident evidence.
- `configuration_fact_ids`  configuration facts (`CFG-1`). CONFIGURED, not observed activity.

"Unknown" is a legal value wherever a fact may be unavailable (`affected_area.state`, severity,
disruption, reversibility); nothing forces the model to invent one.

This is an advisory contract. A plan is a set of PROPOSALS: nothing in it is executed and every
proposal requires human review.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

PLAN_SCHEMA_VERSION = "1.0"

MAX_TEXT_CHARS = 2000
MAX_PROPOSALS = 8
MAX_LIST_ITEMS = 12

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]

PlanOutcome = Literal["proposals", "insufficient_evidence"]
AreaState = Literal["identified", "unknown", "not_applicable"]
Level = Literal["low", "medium", "high", "unknown"]
Reversibility = Literal["reversible", "partially_reversible", "irreversible", "unknown"]
ConfidenceLevel = Literal["low", "medium", "high"]
StepOrigin = Literal["cited_reference", "general_practice"]
EvidenceLinkage = Literal["incident_linked", "general_hardening"]

AREA_STATES = ("identified", "unknown", "not_applicable")
LEVELS = ("low", "medium", "high", "unknown")
REVERSIBILITIES = ("reversible", "partially_reversible", "irreversible", "unknown")
CONFIDENCE_LEVELS = ("low", "medium", "high")
STEP_ORIGINS = ("cited_reference", "general_practice")
OUTCOMES = ("proposals", "insufficient_evidence")

CONFIDENCE_MEANING = (
    "Confidence is the model's own ordinal judgement (low, medium, high) of how well the cited incident "
    "evidence, hypotheses, configuration facts and reference passages support THIS proposal. It is not a "
    "probability, it is not calibrated against any measurement, and it says nothing about whether the "
    "proposal would work. 'high' is refused by validation unless a linked hypothesis is confirmed and "
    "configuration facts are available; 'medium' needs at least one step stated by a cited reference."
)

PRIORITY_RULE = (
    "Proposals are ordered by, in turn: (1) evidence linkage, incident_linked before general_hardening; "
    "(2) severity, high before medium before low before unknown; (3) disruption, low before medium before "
    "high before unknown; (4) reversibility, reversible before partially_reversible before unknown before "
    "irreversible; (5) the order in which the model returned them."
)

PRIORITY_CRITERIA = (
    "priority_rank is an ORDINAL position under a fixed, documented rule; it is not a risk score, a "
    "probability or a measured exposure.",
    PRIORITY_RULE,
    "The severity, disruption and reversibility ratings are the model's own, uncalibrated judgements. "
    "The evidence linkage is computed from the citations: a proposal is incident_linked only if it cites "
    "at least one observed evidence id.",
)

PLAN_CAVEATS = (
    "This plan is advisory. Every proposal is a proposal only: nothing was executed, no file or "
    "container was changed and no test was launched.",
    "Hypotheses stay hypotheses. A proposal does not confirm a root cause, and reference passages are "
    "general knowledge that neither confirm nor refute what happened in the incident.",
    "Reference passages come from a small curated CWE/CAPEC subset and cannot be treated as "
    "comprehensive Docker, Juice Shop or configuration-hardening guidance.",
    "Grounding validation checks that identifiers exist and sit in the right category and screens some "
    "high-risk wording; it does not prove that a recommendation is correct, sufficient or safe.",
    "Whether a proposal reduces risk has not been tested; validation belongs to a later phase and its "
    "success under tested conditions would not be a universal security guarantee.",
)


class _Strict(BaseModel):
    # strict: no coercion ("yes" is not True, 1 is not "1"); extra keys are errors; instances are immutable.
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


# ============================================================
# DRAFT MODELS (what the language model may produce)
# ============================================================

class AffectedArea(_Strict):
    state: AreaState
    description: Text


class SeverityAssessment(_Strict):
    level: Level
    rationale: Text


class ImplementationStep(_Strict):
    text: Text
    origin: StepOrigin


class ValidationProcedure(_Strict):
    procedure: Text
    expected_result: Text


class ConfidenceAssessment(_Strict):
    level: ConfidenceLevel
    rationale: Text


class ProposalDraft(_Strict):
    title: Text
    description: Text
    problem_addressed: Text
    affected_area: AffectedArea
    observed_evidence_ids: list[Identifier]
    hypothesis_ids: list[Identifier]
    configuration_fact_ids: list[Identifier]
    reference_ids: list[Identifier]
    rationale: Text
    reference_support_limits: Text
    severity: SeverityAssessment
    expected_security_effect: Text
    operational_side_effects: list[Text]
    implementation_steps: list[ImplementationStep]
    configuration_dependent: bool
    disruption: Level
    reversibility: Reversibility
    preconditions: list[Text]
    dependencies: list[Text]
    unresolved_questions: list[Text]
    validation: ValidationProcedure
    confidence: ConfidenceAssessment


class PlanDraft(_Strict):
    schema_version: Literal["1.0"]
    incident_id: Identifier
    outcome: PlanOutcome
    insufficient_reason: str
    proposals: list[ProposalDraft]
    unresolved_questions: list[Text]
    limitations: list[Text]


# ============================================================
# OUTPUT MODELS (what AutoShield stores; derived fields are filled by code)
# ============================================================

class ReferenceCitation(_Strict):
    """A retrieved passage used by a proposal, copied from the verified Phase 11D context."""

    reference_id: str
    doc_id: str
    chunk_id: str
    ordinal: int
    char_start: int
    char_end: int
    retrieval_rank: int
    source: str
    source_id: str
    source_version: str
    doc_type: str
    title: str
    source_url: str | None
    license: str | None
    chunk_sha256: str
    document_content_sha256: str
    provenance: dict[str, str]
    relevance_label: Literal["relevant"]
    basis: Literal["general_reference_knowledge"] = "general_reference_knowledge"


class LinkedHypothesis(_Strict):
    """A Phase 9 finding exactly as Phase 9 reported it; never upgraded by the planner."""

    hypothesis_id: str
    status: str
    is_confirmed: bool
    basis: Literal["inferred_hypothesis", "confirmed_finding"]


class PriorityBasis(_Strict):
    evidence_linkage: EvidenceLinkage
    severity: Level
    disruption: Level
    reversibility: Reversibility
    ordering_rule: str = PRIORITY_RULE


class RemediationProposal(ProposalDraft):
    proposal_id: str
    priority_rank: int
    priority_basis: PriorityBasis
    linked_hypotheses: list[LinkedHypothesis]
    references: list[ReferenceCitation]
    status: Literal["proposed"] = "proposed"
    executed: Literal[False] = False
    human_review_required: Literal[True] = True


class ExcludedReference(_Strict):
    reference_id: str
    label: Literal["not_relevant", "unclear", "unassessed"]


class ConfigurationConflictSummary(_Strict):
    subject: str
    attribute: str
    fact_ids: list[str]
    values: list[str]


class InputSummary(_Strict):
    """What the planner was shown, by category. Identifiers only; full text stays in the context."""

    retriever: str
    is_production_retriever: bool
    observed_evidence_ids: list[str]
    unresolved_evidence_ids: list[str]
    hypotheses: list[LinkedHypothesis]
    usable_reference_ids: list[str]
    excluded_references: list[ExcludedReference]
    relevance_source: Literal["validated_phase_11d_answer", "unavailable"]
    configuration_status: Literal["available", "unavailable", "conflicting"]
    configuration_fact_ids: list[str]
    configuration_conflicts: list[ConfigurationConflictSummary]


class InsufficiencyReason(_Strict):
    code: str
    message: str


class RemediationPlan(_Strict):
    schema_version: Literal["1.0"] = PLAN_SCHEMA_VERSION
    incident_id: str
    finding_index: int
    outcome: PlanOutcome
    insufficient_reasons: list[InsufficiencyReason]
    proposals: list[RemediationProposal]
    unresolved_questions: list[str]
    limitations: list[str]
    input_summary: InputSummary
    priority_criteria: list[str]
    confidence_meaning: str
    caveats: list[str]
    advisory_only: Literal[True] = True
    execution_performed: Literal[False] = False
    ground_truth_used: Literal[False] = False
