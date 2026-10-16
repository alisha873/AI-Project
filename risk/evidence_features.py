"""Conservative conversion of remediation and validation records to risk features."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evidence.schema import Remediation, ValidationResult
from risk.features import RiskFeatures


class FeatureProvenance(BaseModel):
    """Trace a derived value to the evidence that supports it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature: str
    value: bool | int | None
    source_fields: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str


class EvidenceFeatureResult(BaseModel):
    """Feature record plus provenance and fields requiring verification."""

    model_config = ConfigDict(extra="forbid")

    features: RiskFeatures
    provenance: list[FeatureProvenance]
    unresolved_features: list[str]


def _unique(values: list[str]) -> list[str]:
    """Preserve order while deduplicating evidence references."""
    return list(dict.fromkeys(value for value in values if value.strip()))


def build_evidence_features(
    remediation: Remediation,
    validation: ValidationResult,
    *,
    attack_blocked: bool | None = None,
    attack_path_closed: bool | None = None,
    remaining_exposed_ports: int | None = None,
    remaining_vulnerabilities: int | None = None,
    patch_applied: bool | None = None,
    configuration_change: bool | None = None,
) -> EvidenceFeatureResult:
    """Build features without promoting proposals or missing data to facts.

    Optional overrides must come from independently verified evidence.
    This function does not decide whether a status string proves success.
    """

    validation_refs = _unique(
        list(validation.evidence)
        + [ref for check in validation.checks for ref in check.evidence]
    )
    remediation_refs = _unique(remediation.evidence)

    # Count unique references, not repeated mentions of the same artifact.
    evidence_count = len(validation_refs)

    features = RiskFeatures(
        attack_blocked=attack_blocked,
        attack_path_closed=attack_path_closed,
        remaining_exposed_ports=remaining_exposed_ports,
        remaining_vulnerabilities=remaining_vulnerabilities,
        patch_applied=patch_applied,
        configuration_change=configuration_change,
        validation_evidence_count=evidence_count,
    )

    provenance = [
        FeatureProvenance(
            feature="attack_blocked",
            value=attack_blocked,
            source_fields=["explicit_verified_input.attack_blocked"],
            evidence_refs=validation_refs if attack_blocked is not None else [],
            rationale=(
                "Supplied as an explicit verified input."
                if attack_blocked is not None
                else "Unknown: validation status alone is not interpreted as proof."
            ),
        ),
        FeatureProvenance(
            feature="attack_path_closed",
            value=attack_path_closed,
            source_fields=["explicit_verified_input.attack_path_closed"],
            evidence_refs=validation_refs if attack_path_closed is not None else [],
            rationale=(
                "Supplied as an explicit verified input."
                if attack_path_closed is not None
                else "Unknown: no automatic inference from generic status strings."
            ),
        ),
        FeatureProvenance(
            feature="remaining_exposed_ports",
            value=remaining_exposed_ports,
            source_fields=["explicit_verified_input.remaining_exposed_ports"],
            evidence_refs=[],
            rationale=(
                "Supplied from a verified port inventory."
                if remaining_exposed_ports is not None
                else "Unknown: no verified structured port count was supplied."
            ),
        ),
        FeatureProvenance(
            feature="remaining_vulnerabilities",
            value=remaining_vulnerabilities,
            source_fields=["explicit_verified_input.remaining_vulnerabilities"],
            evidence_refs=[],
            rationale=(
                "Supplied from a verified vulnerability assessment."
                if remaining_vulnerabilities is not None
                else "Unknown: residual vulnerability strings are not assumed to be a complete count."
            ),
        ),
        FeatureProvenance(
            feature="patch_applied",
            value=patch_applied,
            source_fields=["explicit_verified_input.patch_applied"],
            evidence_refs=remediation_refs if patch_applied is not None else [],
            rationale=(
                "Supplied as an explicitly verified applied-patch state."
                if patch_applied is not None
                else "Unknown: a remediation proposal does not prove application."
            ),
        ),
        FeatureProvenance(
            feature="configuration_change",
            value=configuration_change,
            source_fields=["explicit_verified_input.configuration_change"],
            evidence_refs=remediation_refs if configuration_change is not None else [],
            rationale=(
                "Supplied as an explicitly verified configuration-change state."
                if configuration_change is not None
                else "Unknown: proposed changes do not prove a deployed change."
            ),
        ),
        FeatureProvenance(
            feature="validation_evidence_count",
            value=evidence_count,
            source_fields=[
                "ValidationResult.evidence",
                "ValidationCheck.evidence",
            ],
            evidence_refs=validation_refs,
            rationale="Count of unique non-empty validation evidence references.",
        ),
    ]

    unresolved = [
        item.feature for item in provenance if item.value is None
    ]

    # Keep this variable explicit: remediation references are tracked in
    # provenance but are not counted as validation evidence.
    _ = remediation_refs

    return EvidenceFeatureResult(
        features=features,
        provenance=provenance,
        unresolved_features=unresolved,
    )
