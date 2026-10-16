"""Typed, validated remediation-risk features for AutoShield."""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class RiskFeatures(BaseModel):
    """Features describing remediation and its validation evidence.

    None means unknown or not yet verified. It never means False or zero.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "attack_blocked",
        "attack_path_closed",
        "remaining_exposed_ports",
        "remaining_vulnerabilities",
        "patch_applied",
        "configuration_change",
        "validation_evidence_count",
    )

    attack_blocked: bool | None = None
    attack_path_closed: bool | None = None
    remaining_exposed_ports: int | None = Field(default=None, ge=0)
    remaining_vulnerabilities: int | None = Field(default=None, ge=0)
    patch_applied: bool | None = None
    configuration_change: bool | None = None
    validation_evidence_count: int | None = Field(default=None, ge=0)

    def to_model_array(self) -> np.ndarray:
        """Return one numeric row in the canonical feature order.

        Refuses incomplete data rather than silently treating unknown
        values as negative findings.
        """
        values = self.model_dump()
        missing = [name for name in self.FEATURE_NAMES if values[name] is None]
        if missing:
            raise ValueError(
                "Cannot create a model input with unknown features: "
                + ", ".join(missing)
            )

        row = [
            float(values[name])
            for name in self.FEATURE_NAMES
        ]
        return np.asarray(row, dtype=float)

    @classmethod
    def feature_names(cls) -> list[str]:
        return list(cls.FEATURE_NAMES)


def features_to_matrix(
    records: list[RiskFeatures],
) -> tuple[np.ndarray, list[str]]:
    """Convert complete feature records into a model matrix and column names."""
    if not records:
        raise ValueError("At least one RiskFeatures record is required.")

    matrix = np.vstack([record.to_model_array() for record in records])
    return matrix, RiskFeatures.feature_names()
