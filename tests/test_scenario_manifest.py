import pytest
from pydantic import ValidationError

from evaluation.scenario_manifest import ScenarioManifest


def valid_manifest():
    return {
        "schema_version": "1.0",
        "scenario_id": "scenario_002",
        "scenario_family": "sql_injection",
        "description": "Controlled SQL injection in the isolated lab",
        "environment": "isolated_docker_lab",
        "execution": {
            "method": "scripted",
            "target_allowlist": ["127.0.0.1:8080"],
        },
        "ground_truth_reference": "ground_truth/scenario_002.json",
    }


def test_valid_manifest_passes_validation():
    manifest = ScenarioManifest.model_validate(valid_manifest())

    assert manifest.scenario_id == "scenario_002"
    assert manifest.execution.target_allowlist == ["127.0.0.1:8080"]


@pytest.mark.parametrize(
    "changes",
    [
        {"scenario_id": "incident_002"},
        {"environment": "production"},
        {"execution": {"method": "scripted", "target_allowlist": []}},
        {"unexpected_field": "must be rejected"},
    ],
)
def test_invalid_manifest_is_rejected(changes):
    data = valid_manifest()
    data.update(changes)

    with pytest.raises(ValidationError):
        ScenarioManifest.model_validate(data)


def test_unexpected_nested_execution_field_is_rejected():
    data = valid_manifest()
    data["execution"]["unexpected"] = True

    with pytest.raises(ValidationError):
        ScenarioManifest.model_validate(data)

def test_empty_target_allowlist_entry_is_rejected():
    data = valid_manifest()
    data["execution"]["target_allowlist"] = [""]

    with pytest.raises(ValidationError):
        ScenarioManifest.model_validate(data)