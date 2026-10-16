from evidence.schema import (
    Remediation,
    ValidationCheck,
    ValidationResult,
)
from risk.evidence_features import build_evidence_features


def test_proposal_does_not_automatically_mean_patch_was_applied():
    remediation = Remediation(
        proposed_fix="Apply security patch",
        evidence=["change-request-7"],
    )
    validation = ValidationResult(
        checks=[
            ValidationCheck(
                check_id="check-1",
                description="Post-remediation check",
                status="passed",
                evidence=["test-log-1"],
            )
        ]
    )

    result = build_evidence_features(remediation, validation)

    assert result.features.patch_applied is None
    assert "patch_applied" in result.unresolved_features
    patch_record = next(
        item for item in result.provenance
        if item.feature == "patch_applied"
    )
    assert "does not prove application" in patch_record.rationale


def test_validation_evidence_count_counts_unique_references():
    remediation = Remediation()
    validation = ValidationResult(
        evidence=["log-1", "log-1"],
        checks=[
            ValidationCheck(
                check_id="check-1",
                description="First check",
                status="passed",
                evidence=["log-1", "log-2"],
            )
        ],
    )

    result = build_evidence_features(remediation, validation)

    assert result.features.validation_evidence_count == 2


def test_verified_inputs_are_kept_and_have_provenance():
    result = build_evidence_features(
        Remediation(evidence=["change-1"]),
        ValidationResult(evidence=["test-1"]),
        attack_blocked=True,
        attack_path_closed=True,
        remaining_exposed_ports=0,
        remaining_vulnerabilities=1,
        patch_applied=True,
        configuration_change=False,
    )

    assert result.features.attack_blocked is True
    assert result.features.remaining_exposed_ports == 0
    assert result.features.patch_applied is True
    assert result.unresolved_features == []
    assert len(result.provenance) == 7
