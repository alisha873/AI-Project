"""Phase 13A: the fail-closed scope policy (pure decisions; nothing is resolved, connected or probed)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.validation_helpers import (
    COMPOSE_SHA,
    INCIDENT,
    TARGET,
    approved_lab,
    approved_scenario,
    authorization,
    identity,
)
from validation.scenario import sqli_search_scenario
from validation.scope import (
    SCOPE_REASON_CODES,
    ApprovedLab,
    ScopePolicy,
    ScopeRequest,
    normalized_origin,
    parse_target_origin,
)


def request(**overrides) -> ScopeRequest:
    data = {"target_url": TARGET, "scenario": approved_scenario(), "incident_id": INCIDENT,
            "identity": identity(), "authorization": authorization()}
    data.update(overrides)
    return ScopeRequest(**data)


def policy(**kwargs) -> ScopePolicy:
    return ScopePolicy([approved_lab()], **kwargs)


# ------------------------------------------------------------
# default deny
# ------------------------------------------------------------

def test_a_policy_with_no_approved_lab_denies_everything_even_a_perfect_looking_request():
    decision = ScopePolicy().evaluate(request())
    assert decision.execution_permitted is False and decision.scope_conditions_met is False
    assert "no_approved_lab" in decision.reason_codes and "execution_disabled" in decision.reason_codes


def test_an_empty_request_is_denied_with_every_reason():
    decision = ScopePolicy().evaluate(ScopeRequest())
    assert decision.execution_permitted is False
    assert {"no_approved_lab", "target_missing", "scenario_missing", "lab_identity_missing",
            "operator_authorization_missing", "execution_disabled"} <= set(decision.reason_codes)


def test_everything_in_order_still_does_not_permit_execution_while_the_switch_is_off():
    decision = policy().evaluate(request())
    assert decision.scope_conditions_met is True
    assert decision.execution_permitted is False
    assert decision.reason_codes == ("execution_disabled",)
    assert decision.lab_id == "lab-1" and decision.origin == "http://127.0.0.1:8080"


def test_only_a_true_boolean_turns_the_switch_on_and_then_everything_must_still_pass():
    assert policy(execution_enabled=True).evaluate(request()).execution_permitted is True
    for truthy in (1, "yes", "true", [1]):
        assert ScopePolicy([approved_lab()], execution_enabled=truthy).evaluate(request()).execution_permitted is False
    assert policy(execution_enabled=True).evaluate(request(authorization=None)).execution_permitted is False


def test_every_reason_code_is_a_known_code():
    decisions = [
        ScopePolicy().evaluate(ScopeRequest()),
        policy().evaluate(request(target_url="http://10.1.2.3:8080")),
        policy().evaluate(request(target_url="ftp://127.0.0.1:8080")),
        policy().evaluate(request(scenario=sqli_search_scenario(), identity=identity(established_independently=False),
                                  authorization=authorization(live_execution_authorized=False, lab_id="other"))),
    ]
    for decision in decisions:
        assert set(decision.reason_codes) <= set(SCOPE_REASON_CODES)


# ------------------------------------------------------------
# targets
# ------------------------------------------------------------

@pytest.mark.parametrize("target", [
    "http://example.com:8080",           # public name
    "http://93.184.216.34:8080",         # public address
    "http://8.8.8.8:8080",
    "http://127.0.0.1:9999",             # right host, wrong port
    "http://127.0.0.2:8080",             # loopback is not a lab
    "http://localhost:8080",             # a name is not the approved literal address
    "http://10.0.0.5:8080",              # a private address is not automatically safe
    "http://192.168.1.10:8080",
    "http://172.19.0.2:8080",
    "http://127.1:8080",                 # shorthand forms
    "http://2130706433:8080",            # decimal-encoded 127.0.0.1
    "http://0x7f.0.0.1:8080",
    "http://127.0.0.1.:8080",            # trailing dot
    "http://[::1]:8080",
])
def test_public_third_party_unapproved_and_obfuscated_targets_are_denied(target):
    decision = policy(execution_enabled=True).evaluate(request(target_url=target))
    assert decision.execution_permitted is False
    assert decision.scope_conditions_met is False
    assert {"target_not_approved", "target_malformed"} & set(decision.reason_codes)


@pytest.mark.parametrize("target", [
    "", "   ", "127.0.0.1:8080", "https://127.0.0.1:8080", "http://127.0.0.1", "http://:8080",
    "http://user:pw@127.0.0.1:8080", "http://127.0.0.1:8080/api/search", "http://127.0.0.1:8080/?q=1",
    "http://127.0.0.1:8080#x", "http://127.0.0.1:99999", "http://127.0.0.1:abc", "http://127.0.0.1:8080\n",
    "http://127.0.0.1:8080 ", "http://127.0.0.1\\:8080", "http://127.0.0.1%2e:8080", "javascript:alert(1)",
    "http://" + "a" * 300 + ":8080", "http://127.0.0.1:8080@evil.example",
])
def test_malformed_targets_are_denied_as_malformed(target):
    assert parse_target_origin(target) is None
    decision = policy(execution_enabled=True).evaluate(request(target_url=target))
    assert "target_malformed" in decision.reason_codes and decision.execution_permitted is False


@pytest.mark.parametrize("target", [None, 8080, ["http://127.0.0.1:8080"], {"url": TARGET}, b"http://127.0.0.1:8080"])
def test_a_target_that_is_not_a_string_is_denied(target):
    decision = policy(execution_enabled=True).evaluate(request(target_url=target))
    assert decision.execution_permitted is False
    assert {"target_missing", "target_malformed"} & set(decision.reason_codes)


def test_the_approved_origin_parses_and_normalizes():
    assert parse_target_origin("http://127.0.0.1:8080") == ("127.0.0.1", 8080)
    assert parse_target_origin("http://127.0.0.1:8080/") == ("127.0.0.1", 8080)
    assert normalized_origin("127.0.0.1", 8080) == "http://127.0.0.1:8080"


def test_a_url_alone_grants_nothing():
    # right address and port but no identity, no authorization, no eligible scenario
    decision = policy(execution_enabled=True).evaluate(
        ScopeRequest(target_url=TARGET, scenario=sqli_search_scenario(), incident_id=INCIDENT))
    assert decision.execution_permitted is False
    assert {"scenario_not_eligible", "lab_identity_missing", "operator_authorization_missing"} <= set(decision.reason_codes)


# ------------------------------------------------------------
# approved lab configuration
# ------------------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {"host": "8.8.8.8"}, {"host": "93.184.216.34"}, {"host": "example.com"}, {"host": "0.0.0.0"},
    {"host": "224.0.0.1"}, {"host": "127.1"}, {"host": "2130706433"}, {"host": "LOCALHOST"},
    {"port": 0}, {"port": 70000}, {"expected_containers": []}, {"expected_containers": ["a", "a"]},
    {"approved_scenario_ids": []}, {"compose_sha256": "short"},
])
def test_an_approved_lab_can_not_be_public_ambiguous_or_incomplete(overrides):
    with pytest.raises(ValidationError):
        approved_lab(**overrides)


def test_private_and_loopback_addresses_are_allowed_only_as_explicit_configuration():
    for host in ("127.0.0.1", "10.0.0.5", "192.168.1.10", "localhost", "::1"):
        assert approved_lab(host=host).host == host
    # ...and being allowed in a configuration does not match any other target
    decision = ScopePolicy([approved_lab(host="10.0.0.5")], execution_enabled=True).evaluate(request())
    assert "target_not_approved" in decision.reason_codes


def test_a_lab_must_approve_the_scenario():
    other = ScopePolicy([approved_lab(approved_scenario_ids=["something-else"])], execution_enabled=True)
    decision = other.evaluate(request())
    assert "scenario_not_approved_for_lab" in decision.reason_codes and decision.execution_permitted is False


def test_the_policy_only_accepts_approved_lab_objects():
    with pytest.raises(TypeError):
        ScopePolicy([{"host": "127.0.0.1", "port": 8080}])  # type: ignore[list-item]


# ------------------------------------------------------------
# scenario eligibility
# ------------------------------------------------------------

def test_an_unreviewed_scenario_is_denied_whatever_else_is_true():
    decision = policy(execution_enabled=True).evaluate(request(scenario=sqli_search_scenario()))
    assert "scenario_not_eligible" in decision.reason_codes
    assert decision.scope_conditions_met is False and decision.execution_permitted is False


def test_a_missing_scenario_is_denied():
    decision = policy(execution_enabled=True).evaluate(request(scenario=None))
    assert "scenario_missing" in decision.reason_codes and decision.execution_permitted is False


# ------------------------------------------------------------
# lab identity (independent of the address)
# ------------------------------------------------------------

@pytest.mark.parametrize("override,code", [
    ({"established_independently": False}, "lab_identity_not_independent"),
    ({"lab_id": "another-lab"}, "lab_identity_mismatch"),
    ({"observed_network": "bridge"}, "lab_identity_mismatch"),
    ({"observed_containers": ["autoshield-api"]}, "lab_identity_mismatch"),
    ({"observed_containers": ["autoshield-api", "evil"]}, "lab_identity_mismatch"),
    ({"compose_sha256": "d" * 64}, "lab_identity_mismatch"),
    ({"compose_sha256": None}, "lab_identity_mismatch"),
])
def test_a_wrong_or_not_independent_identity_is_denied(override, code):
    decision = policy(execution_enabled=True).evaluate(request(identity=identity(**override)))
    assert code in decision.reason_codes and decision.execution_permitted is False


def test_missing_identity_is_denied_and_a_matching_address_does_not_substitute_for_it():
    decision = policy(execution_enabled=True).evaluate(request(identity=None))
    assert "lab_identity_missing" in decision.reason_codes and decision.execution_permitted is False


def test_container_order_does_not_matter_but_the_set_must_match_exactly():
    assert policy(execution_enabled=True).evaluate(request()).execution_permitted is True  # identity() is reversed


def test_a_lab_without_a_pinned_compose_hash_does_not_require_one():
    lab = approved_lab(compose_sha256=None)
    decision = ScopePolicy([lab], execution_enabled=True).evaluate(request(identity=identity(compose_sha256=None)))
    assert decision.execution_permitted is True
    assert COMPOSE_SHA  # the pinned variant is covered above


def test_the_lab_state_is_only_compared_when_an_expected_state_is_given():
    enabled = policy(execution_enabled=True)
    # 13B.3 migration: an independently assessed identity always carries the observed state id (an observation requires
    # one), so evidence without a state can only be hand-made and is denied as not assessed.
    assert enabled.evaluate(request(identity=identity(lab_state_id="ANY"))).execution_permitted is True
    assert enabled.evaluate(request()).execution_permitted is True
    stateless = enabled.evaluate(request(identity=identity(lab_state_id=None)))
    assert stateless.execution_permitted is False and "lab_identity_not_assessed" in stateless.reason_codes


def test_an_expected_lab_state_must_equal_the_independently_observed_one():
    enabled = policy(execution_enabled=True)
    same = enabled.evaluate(request(identity=identity(lab_state_id="S-1"), expected_lab_state_id="S-1"))
    assert same.execution_permitted is True and same.reason_codes == ()
    other = enabled.evaluate(request(identity=identity(lab_state_id="S-2"), expected_lab_state_id="S-1"))
    assert other.reason_codes == ("lab_state_mismatch",) and other.execution_permitted is False
    unknown = enabled.evaluate(request(identity=identity(lab_state_id=None), expected_lab_state_id="S-1"))
    assert "lab_state_unverified" in unknown.reason_codes and unknown.execution_permitted is False
    assert "lab_identity_not_assessed" in unknown.reason_codes  # a stateless claim can not come from an assessment


def test_a_missing_identity_does_not_also_claim_a_lab_state_result():
    decision = policy(execution_enabled=True).evaluate(request(identity=None, expected_lab_state_id="S-1"))
    assert "lab_identity_missing" in decision.reason_codes
    assert not {"lab_state_mismatch", "lab_state_unverified"} & set(decision.reason_codes)


def test_identity_for_an_unapproved_target_is_never_accepted():
    decision = policy(execution_enabled=True).evaluate(request(target_url="http://10.9.9.9:8080"))
    assert "lab_identity_mismatch" in decision.reason_codes and decision.execution_permitted is False


# ------------------------------------------------------------
# operator authorization
# ------------------------------------------------------------

@pytest.mark.parametrize("override,code", [
    ({"live_execution_authorized": False}, "operator_authorization_not_live"),
    ({"lab_id": "another-lab"}, "operator_authorization_mismatch"),
    ({"scenario_id": "another-scenario"}, "operator_authorization_mismatch"),
    ({"incident_id": "incident_002"}, "operator_authorization_mismatch"),
])
def test_an_authorization_for_something_else_or_without_live_permission_is_denied(override, code):
    decision = policy(execution_enabled=True).evaluate(request(authorization=authorization(**override)))
    assert code in decision.reason_codes and decision.execution_permitted is False


def test_missing_authorization_is_denied():
    decision = policy(execution_enabled=True).evaluate(request(authorization=None))
    assert "operator_authorization_missing" in decision.reason_codes and decision.execution_permitted is False


def test_the_decision_is_deterministic():
    assert policy().evaluate(request()) == policy().evaluate(request())
    assert ApprovedLab.model_json_schema()  # the configuration is a plain, inspectable schema
