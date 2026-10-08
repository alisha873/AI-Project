from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


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

    This information is used for evaluation and must not be
    exposed to the AI during inference.
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
    Normalized incident representation consumed by the
    downstream AI pipeline.
    """

    incident_id: str

    events: list[EvidenceEvent] = Field(default_factory=list)

    assets: list[str] = Field(default_factory=list)

    network_artifacts: list[str] = Field(default_factory=list)

    configurations: list[str] = Field(default_factory=list)