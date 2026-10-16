"""Phase 13B.1: scenario-specific request construction (pure data, fail closed)."""

from __future__ import annotations

import copy

import pytest

from tests.validation_helpers import PROBE_TEXT, approved_scenario
from validation.probe_requests import (
    CONSTRUCTION_CODES,
    REQUEST_TIMEOUT_SECONDS,
    ProbeRequest,
    RequestConstructionError,
    build_probe_requests,
    check_constructible,
)
from validation.scenario import ScenarioDefinition, sqli_search_scenario

ORIGIN = "http://127.0.0.1:8080"


def with_change(**probe_changes) -> ScenarioDefinition:
    data = copy.deepcopy(approved_scenario().model_dump())
    data["probe"].update(probe_changes)
    return ScenarioDefinition.model_validate(data)


def fails_with(code: str, scenario, origin=ORIGIN) -> None:
    with pytest.raises(RequestConstructionError) as caught:
        build_probe_requests(scenario, origin)
    assert caught.value.code == code and code in CONSTRUCTION_CODES


# ------------------------------------------------------------
# the built-in scenario can not build requests
# ------------------------------------------------------------

def test_the_built_in_scenario_is_ineligible_so_it_can_not_build_requests():
    scenario = sqli_search_scenario()
    assert scenario.eligible_for_execution is False
    assert check_constructible(scenario) == "scenario_not_eligible"
    fails_with("scenario_not_eligible", scenario)


def test_the_built_in_scenario_still_has_five_blocking_ambiguities_and_no_invented_values():
    scenario = sqli_search_scenario()
    assert len(scenario.open_blocking_ambiguities()) == 5
    assert "control_value_unspecified" in scenario.open_blocking_ambiguities()
    assert scenario.probe.wire_encoding == "unspecified" and scenario.probe.control_value is None


def test_approval_can_not_be_reached_by_resolving_ambiguities_alone():
    data = copy.deepcopy(sqli_search_scenario().model_dump())
    for item in data["ambiguities"]:
        item["resolved"] = True
    data["review"] = {"status": "approved", "reviewer": "r", "note": "n"}
    with pytest.raises(ValueError, match="wire encoding and its control value"):
        ScenarioDefinition.model_validate(data)


# ------------------------------------------------------------
# a (test-only) approved scenario builds exactly two requests
# ------------------------------------------------------------

def test_an_approved_scenario_builds_a_control_and_a_probe_request():
    pair = build_probe_requests(approved_scenario(), ORIGIN)
    assert (pair.control.role, pair.probe.role) == ("control", "probe")
    for item in (pair.control, pair.probe):
        assert isinstance(item, ProbeRequest) and item.method == "GET"
        assert item.origin == ORIGIN and item.path == "/api/search"
        assert item.timeout_seconds == REQUEST_TIMEOUT_SECONDS
    assert pair.control.attack_shaped is False and pair.probe.attack_shaped is True


def test_the_query_is_percent_encoded_with_only_unreserved_characters_left():
    pair = build_probe_requests(approved_scenario(), ORIGIN)
    assert pair.control.query == "message=zzqx-no-such-record"
    assert pair.probe.query == "message=%27%20OR%20%271%27%3D%271"
    assert all(ch.isalnum() or ch in "-._~%=" for ch in pair.probe.query)
    assert pair.probe.target == "/api/search?message=%27%20OR%20%271%27%3D%271"


def test_the_probe_text_comes_from_the_scenario_not_from_the_caller():
    assert approved_scenario().probe.parameter_value_candidate == PROBE_TEXT


def test_unicode_and_reserved_characters_are_encoded_as_utf8_bytes():
    pair = build_probe_requests(with_change(parameter_value_candidate="a&b=c#d é"), ORIGIN)
    assert pair.probe.query == "message=a%26b%3Dc%23d%20%C3%A9"


def test_request_ids_are_stable_and_there_are_no_headers_or_extra_hosts():
    pair = build_probe_requests(approved_scenario(), ORIGIN)
    assert pair.control.request_id == "sqli-search-incident-001:control"
    assert set(ProbeRequest.model_fields) == {
        "request_id", "role", "method", "origin", "path", "query", "timeout_seconds", "attack_shaped"}


def test_construction_is_deterministic():
    first = build_probe_requests(approved_scenario(), ORIGIN)
    assert first == build_probe_requests(approved_scenario(), ORIGIN)


# ------------------------------------------------------------
# fail closed
# ------------------------------------------------------------

def test_an_unspecified_wire_encoding_is_not_guessed():
    scenario = approved_scenario()
    data = copy.deepcopy(scenario.model_dump())
    data["probe"]["wire_encoding"] = "unspecified"
    with pytest.raises(ValueError):  # an approved scenario can not even exist in this state
        ScenarioDefinition.model_validate(data)


def test_a_reopened_review_blocks_construction():
    data = copy.deepcopy(approved_scenario().model_dump())
    data["review"]["status"] = "pending_human_review"
    fails_with("scenario_not_eligible", ScenarioDefinition.model_validate(data))


@pytest.mark.parametrize("value", ["x" * 201, "a\nb", "a\x00b"])
def test_an_unusable_probe_value_is_refused(value):
    fails_with("probe_value_unusable", with_change(parameter_value_candidate=value))


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_probe_value_can_not_even_be_stored(value):
    with pytest.raises(ValueError):
        with_change(parameter_value_candidate=value)


@pytest.mark.parametrize("value", ["x" * 201, "a\tb", "a\x7fb"])
def test_an_unusable_control_value_is_refused(value):
    fails_with("control_value_unspecified", with_change(control_value=value))


def test_a_blank_control_value_can_not_make_a_scenario_approvable():
    with pytest.raises(ValueError, match="wire encoding and its control value"):
        with_change(control_value=None)


@pytest.mark.parametrize("origin", [
    None, 42, "", "localhost", "http://127.0.0.1:8080/", "http://127.0.0.1:8080/x",
    "https://127.0.0.1:8080", "http://user@127.0.0.1:8080", "http://127.0.0.1", "HTTP://127.0.0.1:8080",
    "http://127.0.0.1:8080?a=b",
])
def test_only_a_normalized_origin_is_accepted(origin):
    fails_with("origin_not_approved_form", approved_scenario(), origin)


def test_a_non_scenario_is_refused():
    fails_with("scenario_not_eligible", {"scenario_id": "x"})
    fails_with("scenario_not_eligible", None)


def test_non_get_methods_and_odd_paths_are_refused():
    data = copy.deepcopy(approved_scenario().model_dump())
    data["endpoint"]["method"] = "GET"
    for path in ("api/search", "/api/search?x=1", "/api/ search", "/api\\search"):
        mutated = copy.deepcopy(data)
        mutated["endpoint"]["path"] = path
        try:
            scenario = ScenarioDefinition.model_validate(mutated)
        except ValueError:
            continue
        assert check_constructible(scenario) == "endpoint_malformed", path
