"""Phase 13B.1: the oracle rule engine and the SQL-injection classifier (achieved / not observed / inconclusive)."""

from __future__ import annotations

import copy
import json

import pytest

from tests.validation_helpers import PROBE_TEXT, approved_scenario, lab_body
from validation.oracle import (
    ClassifierRegistry,
    OracleOutcome,
    SqliSearchClassifier,
    apply_rules,
    default_classifiers,
    parse_search_records,
)
from validation.scenario import ScenarioDefinition
from validation.transport import TransportResponse

SCENARIO = approved_scenario()
CLASSIFIER = SqliSearchClassifier()
OK_EMPTY = TransportResponse.ok(lab_body([]))
ATTACK_ROWS = TransportResponse.ok(lab_body(["first note", "second note"]))


def classify(control, probe):
    return CLASSIFIER.classify(SCENARIO, control, probe)


def results(outcome: OracleOutcome) -> dict[str, str]:
    return {c.condition_id: c.result for c in outcome.conditions}


# ------------------------------------------------------------
# the three kinds of conclusion
# ------------------------------------------------------------

def test_records_outside_the_literal_match_mean_the_objective_was_achieved():
    outcome = classify(OK_EMPTY, ATTACK_ROWS)
    assert (outcome.status, outcome.reason_code) == ("objective_achieved", "objective_observed")
    assert results(outcome) == {"control_request_succeeds": "met", "probe_response_parseable": "met",
                                "record_outside_literal_match": "met"}
    assert outcome.facts["records_outside_literal_match"] == "2"


def test_a_healthy_control_and_an_empty_probe_result_means_not_observed():
    outcome = classify(OK_EMPTY, OK_EMPTY)
    assert (outcome.status, outcome.reason_code) == (
        "objective_not_observed", "objective_not_observed_under_tested_conditions")
    assert results(outcome)["record_outside_literal_match"] == "not_met"


def test_records_that_all_contain_the_literal_are_not_an_alteration():
    body = TransportResponse.ok(lab_body([f"note {PROBE_TEXT}", PROBE_TEXT]))
    assert classify(OK_EMPTY, body).status == "objective_not_observed"


def test_one_record_outside_the_match_is_enough():
    body = TransportResponse.ok(lab_body([f"note {PROBE_TEXT}", "unrelated"]))
    assert classify(OK_EMPTY, body).status == "objective_achieved"


@pytest.mark.parametrize("probe,reason", [
    (TransportResponse.timeout(), "timeout"),
    (TransportResponse.unreachable(), "lab_unreachable"),
    (TransportResponse.error(), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=500), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=302), "unexpected_response"),
    (TransportResponse.ok("<html>blocked</html>"), "unexpected_response"),
    (TransportResponse.ok("{}"), "unexpected_response"),
    (TransportResponse.ok(""), "unexpected_response"),
])
def test_a_probe_failure_is_inconclusive_never_not_observed(probe, reason):
    outcome = classify(OK_EMPTY, probe)
    assert (outcome.status, outcome.reason_code) == ("inconclusive", reason)
    assert results(outcome)["probe_response_parseable"] == "not_met"
    assert results(outcome)["record_outside_literal_match"] == "not_evaluated"


@pytest.mark.parametrize("control,reason", [
    (TransportResponse.timeout(), "timeout"),
    (TransportResponse.unreachable(), "lab_unreachable"),
    (TransportResponse.error(), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=503), "control_check_failed"),
    (TransportResponse.ok("not json"), "control_check_failed"),
])
def test_a_control_failure_is_inconclusive_and_nothing_else_is_judged(control, reason):
    outcome = classify(control, None)
    assert (outcome.status, outcome.reason_code) == ("inconclusive", reason)
    assert results(outcome) == {"control_request_succeeds": "not_met", "probe_response_parseable": "not_evaluated",
                                "record_outside_literal_match": "not_evaluated"}


def test_a_missing_probe_after_a_healthy_control_is_insufficient_evidence():
    outcome = classify(OK_EMPTY, None)
    assert (outcome.status, outcome.reason_code) == ("inconclusive", "evidence_insufficient")


def test_without_a_control_only_the_declared_achieved_rule_can_fire_and_not_observed_never_can():
    # The 'achieved' rule (as reviewed) does not need the control; the runner still always sends the control first.
    outcome = classify(None, ATTACK_ROWS)
    assert outcome.status == "objective_achieved"
    assert results(outcome)["control_request_succeeds"] == "not_evaluated"
    assert classify(None, OK_EMPTY).status == "inconclusive", "no control, so 'not observed' can not be claimed"


# ------------------------------------------------------------
# parsing
# ------------------------------------------------------------

@pytest.mark.parametrize("body", [
    None, 5, "", "nope", "[]", "{}", '{"data": 3}', '{"data": [1]}', '{"data": [{"id": 1}]}',
    '{"data": [{"message": 3}]}', '{"data": [{"message": "ok"}, "x"]}',
])
def test_a_malformed_body_does_not_parse(body):
    assert parse_search_records(body) is None


def test_a_well_formed_body_parses_and_extra_fields_are_ignored():
    body = json.dumps({"data": [{"id": 1, "message": "a", "extra": True}], "other": 1})
    assert parse_search_records(body) == ["a"]
    assert parse_search_records(json.dumps({"data": []})) == []


def test_pathologically_nested_json_is_inconclusive_not_a_crash():
    deep = "[" * 100_000 + "]" * 100_000
    assert parse_search_records(deep) is None


# ------------------------------------------------------------
# the rule engine is declarative and strict
# ------------------------------------------------------------

def test_the_engine_applies_the_scenarios_own_rules():
    oracle = SCENARIO.oracle
    met = {"control_request_succeeds": "met", "probe_response_parseable": "met",
           "record_outside_literal_match": "met"}
    assert apply_rules(oracle, met).status == "objective_achieved"
    assert apply_rules(oracle, {**met, "record_outside_literal_match": "not_met"}).status == "objective_not_observed"


def test_a_not_evaluated_condition_never_supports_a_conclusion():
    oracle = SCENARIO.oracle
    for blank in oracle.conditions:
        partial = {c.condition_id: "met" for c in oracle.conditions}
        partial[blank.condition_id] = "not_evaluated"
        rule = apply_rules(oracle, partial)
        assert rule is None or rule.status != "objective_not_observed"
    assert apply_rules(oracle, {}) is None


def test_the_verdict_comes_from_the_scenario_rules_not_from_the_classifier_code():
    data = copy.deepcopy(SCENARIO.model_dump())
    data["oracle"]["rules"] = [data["oracle"]["rules"][1]]  # drop the 'achieved' rule
    scenario = ScenarioDefinition.model_validate(data)
    outcome = CLASSIFIER.classify(scenario, OK_EMPTY, ATTACK_ROWS)
    assert outcome.status == "inconclusive", "with no rule for it, even attack-shaped records conclude nothing"


# ------------------------------------------------------------
# control health and the registry
# ------------------------------------------------------------

def test_the_control_is_healthy_only_for_a_parseable_200():
    assert CLASSIFIER.control_healthy(SCENARIO, OK_EMPTY) is True
    for bad in (TransportResponse.timeout(), TransportResponse.ok("x"),
                TransportResponse.ok(lab_body([]), status_code=404)):
        assert CLASSIFIER.control_healthy(SCENARIO, bad) is False


def test_the_registry_finds_the_classifier_by_oracle_id_and_version():
    registry = default_classifiers()
    assert registry.get(SCENARIO.oracle) is not None
    data = copy.deepcopy(SCENARIO.model_dump())
    data["oracle"]["version"] = "9.9"
    assert registry.get(ScenarioDefinition.model_validate(data).oracle) is None


def test_registering_twice_is_refused_and_an_empty_registry_knows_nothing():
    with pytest.raises(ValueError):
        ClassifierRegistry(SqliSearchClassifier(), SqliSearchClassifier())
    assert ClassifierRegistry().get(SCENARIO.oracle) is None


def test_the_outcome_model_rejects_an_unknown_status():
    with pytest.raises(ValueError):
        OracleOutcome(status="great_success", reason_code="x", conditions=[])
