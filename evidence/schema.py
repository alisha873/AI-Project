from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# LAYER 1 — EVIDENCE
# ============================================================

class EvidenceEvent(BaseModel):
    """
    Normalized representation of a single forensic event.
    """

    event_id: str
    timestamp: datetime

    source: str
    event_type: str

    source_entity: str | None = None
    destination_entity: str | None = None

    action: str

    attributes: dict[str, Any] = Field(default_factory=dict)


class IncidentMetadata(BaseModel):
    """
    Ground-truth metadata for a controlled incident.

    IMPORTANT:
    This information is used for evaluation only and must
    never be exposed to the AI during inference.
    """

    incident_id: str
    scenario: str
    attack_type: str

    initial_access: str
    affected_services: list[str]

    expected_attack_path: list[str]

    root_cause: str
    expected_remediation: str


class Incident(BaseModel):
    """
    Normalized representation of an entire security incident.
    """

    incident_id: str

    events: list[EvidenceEvent] = Field(default_factory=list)

    assets: list[str] = Field(default_factory=list)

    network_artifacts: list[str] = Field(default_factory=list)

    configurations: list[str] = Field(default_factory=list)


# ============================================================
# LAYER 2 — ATTACK GRAPH
# ============================================================

class GraphNode(BaseModel):
    """
    Represents an entity in the Docker/system environment.
    """

    node_id: str
    node_type: Literal["container", "service", "ip", "port", "process", "user"]
    name: str

    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    source_artifacts: list[str] = Field(default_factory=list)
    basis: Literal["observed", "configured"] = "observed"


class GraphEdge(BaseModel):
    """
    Represents a relationship between two entities.
    """

    source: str
    target: str
    edge_type: Literal[
        "communicates_with",
        "connects_to",
        "accesses",
        "executes",
        "authenticates_to",
        "hosts",
        "forwards",
        "depends_on",
    ]

    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    source_artifacts: list[str] = Field(default_factory=list)
    basis: Literal["observed", "configured"] = "observed"

class EvidenceExtraction(BaseModel):
    """
    Structured output produced by the Evidence Extraction Agent.

    The agent identifies evidence relevant to the incident but does
    not diagnose the attack, determine root cause, or propose
    remediation.
    """

    model_config = ConfigDict(extra="forbid")

    incident_id: str

    relevant_events: list[str] = Field(default_factory=list)

    relevant_configuration: list[str] = Field(default_factory=list)

    relevant_network_events: list[str] = Field(default_factory=list)

    relevant_network_artifacts: list[str] = Field(default_factory=list)

    evidence_summary: str = ""

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

class IncidentEvidenceContext(BaseModel):
    """
    Controlled, agent-safe representation of an incident's evidence.

    Ground-truth evaluation artifacts are intentionally excluded.
    """

    incident_id: str

    events: list[EvidenceEvent] = Field(default_factory=list)

    configuration_artifacts: list[str] = Field(default_factory=list)

    network_artifacts: list[str] = Field(default_factory=list)

    evidence_artifacts: list[str] = Field(default_factory=list)

class AttackGraph(BaseModel):
    """
    Represents the reconstructed system/attack graph.
    """

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

    attack_path: list[str] = Field(default_factory=list)


# ============================================================
# LAYER 3 — INCIDENT RECONSTRUCTION
# ============================================================

class TimelineEvent(BaseModel):
    """
    Represents one event in the reconstructed attack timeline.
    """

    timestamp: datetime
    event: str

    evidence_ids: list[str] = Field(default_factory=list)


class AttackReconstruction(BaseModel):
    """
    AI-generated reconstruction of what happened during
    the incident.
    """

    timeline: list[TimelineEvent] = Field(default_factory=list)

    attack_path: list[str] = Field(default_factory=list)

    affected_assets: list[str] = Field(default_factory=list)

    root_cause: str = ""

    attack_techniques: list[str] = Field(default_factory=list)

    supporting_evidence: list[str] = Field(default_factory=list)

    confidence: float = 0.0


# ============================================================
# LAYER 4 — REMEDIATION
# ============================================================

class RemediationChange(BaseModel):
    """
    Represents one concrete change proposed by the
    Remediation Planner.
    """

    target: str
    change_type: str
    description: str

    before: str | None = None
    after: str | None = None


class Remediation(BaseModel):
    """
    AI-generated remediation proposal.
    """

    finding: str = ""

    proposed_fix: str = ""

    changes: list[RemediationChange] = Field(default_factory=list)

    evidence: list[str] = Field(default_factory=list)

    references: list[str] = Field(default_factory=list)

    confidence: float = 0.0


# ============================================================
# LAYER 5 — VALIDATION / RED TEAM
# ============================================================

class ValidationCheck(BaseModel):
    """
    Represents one security validation performed after
    remediation.
    """

    check_id: str

    description: str

    status: str

    evidence: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """
    Result produced by the Penetration Testing /
    Adversarial Validation layer.
    """

    attack_objective: str = ""

    before_status: str = ""

    after_status: str = ""

    passed: bool = False

    checks: list[ValidationCheck] = Field(default_factory=list)

    evidence: list[str] = Field(default_factory=list)

    residual_vulnerabilities: list[str] = Field(default_factory=list)


# ============================================================
# LAYER 6 — JUDGE
# ============================================================

class JudgeDecision(BaseModel):
    """
    Final remediation decision produced by the Judge Agent.
    """

    decision: str

    confidence: float = 0.0

    reasoning: str = ""

    supporting_evidence: list[str] = Field(default_factory=list)


# ============================================================
# LAYER 7 — RISK
# ============================================================

class RiskAssessment(BaseModel):
    """
    Uncertainty-aware residual risk assessment.

    Eventually produced using the Bayesian Neural Network.
    """

    residual_risk: float = 0.0

    uncertainty: float = 0.0

    confidence_interval: list[float] = Field(default_factory=list)

    risk_level: str = ""

    contributing_factors: list[str] = Field(default_factory=list)


# ============================================================
# LAYER 8 — EXPLAINABILITY
# ============================================================

class FeatureImportance(BaseModel):
    """
    One feature's contribution to a Judge/Risk decision.
    """

    feature: str

    contribution: float

    direction: str


class ExplainabilityResult(BaseModel):
    """
    SHAP-based explanation of the security decision.
    """

    decision: str

    features: list[FeatureImportance] = Field(default_factory=list)

    explanation: str = ""


# ============================================================
# LAYER 9 — FINAL SECURITY ASSESSMENT
# ============================================================

class SecurityAssessment(BaseModel):
    """
    Final structured output of the AutoShield pipeline.
    """

    incident_id: str

    reconstruction: AttackReconstruction

    remediation: Remediation

    validation: ValidationResult

    judge: JudgeDecision

    risk: RiskAssessment

    explainability: ExplainabilityResult