import numpy as np
import pytest
from pydantic import ValidationError

from risk.features import RiskFeatures, features_to_matrix


def complete_features(**overrides):
    values = {
        "attack_blocked": True,
        "attack_path_closed": True,
        "remaining_exposed_ports": 1,
        "remaining_vulnerabilities": 2,
        "patch_applied": True,
        "configuration_change": False,
        "validation_evidence_count": 3,
    }
    values.update(overrides)
    return RiskFeatures(**values)


def test_feature_vector_uses_canonical_order():
    features = complete_features()
    row = features.to_model_array()

    assert row.shape == (7,)
    assert row.dtype == float
    assert np.array_equal(row, [1, 1, 1, 2, 1, 0, 3])
    assert RiskFeatures.feature_names() == [
        "attack_blocked",
        "attack_path_closed",
        "remaining_exposed_ports",
        "remaining_vulnerabilities",
        "patch_applied",
        "configuration_change",
        "validation_evidence_count",
    ]


def test_unknown_features_are_not_silently_converted():
    features = RiskFeatures(attack_blocked=None)

    with pytest.raises(ValueError, match="unknown features"):
        features.to_model_array()


@pytest.mark.parametrize(
    "field,value",
    [
        ("remaining_exposed_ports", -1),
        ("remaining_vulnerabilities", -1),
        ("validation_evidence_count", -1),
    ],
)
def test_counts_cannot_be_negative(field, value):
    with pytest.raises(ValidationError):
        complete_features(**{field: value})


def test_unexpected_features_are_rejected():
    with pytest.raises(ValidationError):
        complete_features(unreviewed_security_claim=True)


def test_batch_conversion_returns_names_and_matrix():
    records = [
        complete_features(),
        complete_features(remaining_exposed_ports=0),
    ]

    matrix, names = features_to_matrix(records)

    assert matrix.shape == (2, 7)
    assert len(names) == 7
    assert matrix[1, 2] == 0


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="At least one"):
        features_to_matrix([])
