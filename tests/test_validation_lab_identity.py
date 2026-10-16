"""Phase 13B.2 preparation: the versioned lab-identity contract and its honest verification levels."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from tests.validation_helpers import TARGET, approved_lab, approved_scenario, authorization
from tests.validation_prep_helpers import (
    NOW,
    RUNNER,
    VERIFIER,
    double_registry,
    identity_assessment,
    observation,
    ts,
)
from validation import lab_identity
from validation.lab_identity import (
    CLOCK_SKEW_SECONDS,
    IDENTITY_MAX_AGE_SECONDS,
    IDENTITY_REASON_CODES,
    IDENTITY_VERIFIER_AVAILABLE,
    VerifierEntry,
    VerifierRegistry,
    assess_lab_identity,
    default_verifier_registry,
    parse_utc,
)
from validation.scope import ScopePolicy, ScopeRequest


# ------------------------------------------------------------
# honest defaults: no independent verifier exists
# ------------------------------------------------------------

def test_no_independent_verifier_is_available_in_this_build():
    assert IDENTITY_VERIFIER_AVAILABLE is False
    assert default_verifier_registry().is_empty


@pytest.mark.parametrize("claim,source", [("verifier_attested", VERIFIER), ("verifier_attested", "docker-inspector"),
                                          ("operator_asserted", "test-operator")])
def test_with_the_default_registry_nothing_is_ever_independently_verified(claim, source):
    obs = observation(verification={"claim": claim, "source_id": source})
    result = assess_lab_identity(obs, now=NOW)  # default registry
    assert result.independently_verified is False and result.live_suitable is False
    assert result.level in ("verifier_unavailable", "operator_asserted")


def test_a_verifier_claim_with_no_registered_verifier_is_reported_as_unavailable_not_as_verified():
    result = assess_lab_identity(observation(), VerifierRegistry(), now=NOW)
    assert result.level == "verifier_unavailable" and result.reason_codes == ["verifier_unavailable"]
    assert result.scope_identity.established_independently is False


# ------------------------------------------------------------
# the versioned model
# ------------------------------------------------------------

def test_the_observation_is_versioned_strict_and_frozen():
    obs = observation()
    assert obs.contract_version == "1.0"
    with pytest.raises(ValidationError):
        obs.lab_id = "other"
    for bad in ({"contract_version": "2.0"}, {"unexpected": 1}, {"lab_state_id": 7}):
        with pytest.raises(ValidationError):
            observation(**bad)


@pytest.mark.parametrize("value", [
    "2026-10-16 12:00:00", "2026-10-16T12:00:00", "2026-10-16T12:00:00+00:00", "2026-13-16T12:00:00Z",
    "2026-10-32T12:00:00Z", "2026-10-16T25:00:00Z", "yesterday", "", "2026-10-16T12:00:00.5Z",
])
def test_only_a_canonical_utc_timestamp_is_accepted(value):
    with pytest.raises(ValidationError):
        observation(observed_at_utc=value)
    with pytest.raises(ValueError):
        parse_utc(value)


@pytest.mark.parametrize("state", ["", " ", "has space", "-leading", "a/b", "x" * 129, "ünï", "a;b"])
def test_a_lab_state_id_must_be_a_plain_token(state):
    with pytest.raises(ValidationError):
        observation(lab_state_id=state)


@pytest.mark.parametrize("containers", [[], ["a", "a"]])
def test_an_observation_lists_its_containers_once_each_and_at_least_one(containers):
    with pytest.raises(ValidationError):
        observation(observed_containers=containers)


def test_a_verification_claim_must_be_one_of_the_known_claims():
    with pytest.raises(ValidationError):
        observation(verification={"claim": "independently_verified", "source_id": "x"})
    with pytest.raises(ValidationError):
        observation(verification={"claim": "verifier_attested"})


def test_a_test_double_can_never_be_suitable_for_live_and_a_registry_refuses_duplicates():
    with pytest.raises(ValidationError):
        VerifierEntry(verifier_id="v", kind="test_double", suitable_for_live=True)
    entry = VerifierEntry(verifier_id="v", kind="test_double")
    with pytest.raises(ValueError):
        VerifierRegistry(entry, entry)
    with pytest.raises(TypeError):
        VerifierRegistry("v")


def test_every_reason_code_is_a_known_code():
    cases = [
        assess_lab_identity(None, now=NOW), assess_lab_identity("x", now=NOW),
        assess_lab_identity(observation(), now=NOW),
        assess_lab_identity(observation(verification={"claim": "operator_asserted", "source_id": "o"}), now=NOW),
        assess_lab_identity(observation(), double_registry(), now=NOW, distinct_from=[VERIFIER]),
        assess_lab_identity(observation(verification={"claim": "verifier_attested", "source_id": "zzz"}),
                            double_registry(), now=NOW),
        assess_lab_identity(observation(observed_at_utc=ts(-5000)), double_registry(), now=NOW),
        assess_lab_identity(observation(observed_at_utc=ts(5000)), double_registry(), now=NOW),
    ]
    for case in cases:
        assert set(case.reason_codes) <= set(IDENTITY_REASON_CODES)


# ------------------------------------------------------------
# the levels
# ------------------------------------------------------------

def test_a_missing_or_malformed_observation_is_none_supplied():
    for value, code in ((None, "identity_missing"), ({"lab_id": "x"}, "identity_malformed"), ("lab-1", "identity_malformed")):
        result = assess_lab_identity(value, double_registry(), now=NOW)
        assert result.level == "none_supplied" and result.reason_codes == [code]
        assert result.scope_identity is None and result.independently_verified is False


def test_an_operator_asserted_observation_is_never_independent_even_with_a_registry():
    obs = observation(verification={"claim": "operator_asserted", "source_id": "test-operator"})
    result = assess_lab_identity(obs, double_registry(), now=NOW)
    assert result.level == "operator_asserted" and "identity_operator_asserted_only" in result.reason_codes
    assert result.independently_verified is False
    assert result.scope_identity.established_independently is False


def test_an_unregistered_verifier_does_not_count():
    obs = observation(verification={"claim": "verifier_attested", "source_id": "mystery-tool"})
    result = assess_lab_identity(obs, double_registry(), now=NOW)
    assert result.level == "verifier_unregistered" and result.reason_codes == ["verifier_not_registered"]
    assert result.independently_verified is False


@pytest.mark.parametrize("excluded", [RUNNER, "test-operator", VERIFIER])
def test_a_verifier_that_is_the_runner_or_the_operator_is_not_independent(excluded):
    registry = VerifierRegistry(VerifierEntry(verifier_id=excluded, kind="test_double"))
    obs = observation(verification={"claim": "verifier_attested", "source_id": excluded})
    result = assess_lab_identity(obs, registry, now=NOW, distinct_from=[RUNNER, "test-operator", VERIFIER])
    assert result.level == "verifier_not_independent" and result.independently_verified is False


def test_a_registered_distinct_fresh_verifier_is_independent_but_a_test_double_is_never_live_suitable():
    result = identity_assessment()
    assert result.level == "verifier_registered" and result.reason_codes == []
    assert result.independently_verified is True and result.live_suitable is False
    assert result.scope_identity.established_independently is True and result.scope_identity.established_by == VERIFIER


def test_only_a_registry_entry_declared_suitable_for_live_can_be_live_suitable():
    # 13B.3 migration: a registry the CALLER built is never authoritative, so even an entry that declares itself
    # live-suitable does not make the assessment live-suitable. Only the repository's own registry can.
    entry = VerifierEntry(verifier_id=VERIFIER, kind="read_only_environment_inspector", suitable_for_live=True)
    caller_built = identity_assessment(registry=VerifierRegistry(entry))
    assert caller_built.independently_verified is True and caller_built.live_suitable is False
    assert caller_built.authoritative_registry is False
    registry = VerifierRegistry(VerifierEntry(verifier_id=VERIFIER, kind="read_only_environment_inspector"))
    assert identity_assessment(registry=registry).live_suitable is False


def test_an_entry_in_the_repository_registry_that_is_live_suitable_makes_the_assessment_live_suitable(monkeypatch):
    # synthetic: a stand-in for a future, reviewed verifier; nothing is registered in the real build
    import validation.lab_identity as module
    entry = VerifierEntry(verifier_id=VERIFIER, kind="read_only_environment_inspector", suitable_for_live=True)
    monkeypatch.setattr(module, "_BUILTIN_VERIFIERS", (entry,))
    result = assess_lab_identity(observation(), module.default_verifier_registry(), now=NOW)
    assert result.live_suitable is True and result.authoritative_registry is True


def test_the_assessment_carries_the_observed_lab_state_for_later_binding():
    result = identity_assessment()
    assert (result.lab_id, result.lab_state_id, result.observation_id) == ("lab-1", "STATE-POST-1", "OBS-ID-1")


# ------------------------------------------------------------
# freshness
# ------------------------------------------------------------

def test_an_observation_exactly_at_the_maximum_age_is_fresh_and_one_second_older_is_stale():
    assert identity_assessment(observation(observed_at_utc=ts(-IDENTITY_MAX_AGE_SECONDS))).fresh is True
    stale = identity_assessment(observation(observed_at_utc=ts(-IDENTITY_MAX_AGE_SECONDS - 1)))
    assert stale.fresh is False and "identity_stale" in stale.reason_codes
    assert stale.independently_verified is False and stale.scope_identity is None


def test_a_future_observation_beyond_the_clock_skew_is_rejected():
    assert identity_assessment(observation(observed_at_utc=ts(CLOCK_SKEW_SECONDS))).fresh is True
    future = identity_assessment(observation(observed_at_utc=ts(CLOCK_SKEW_SECONDS + 1)))
    assert future.fresh is False and "identity_from_future" in future.reason_codes
    assert future.independently_verified is False and future.scope_identity is None


def test_the_age_limit_is_adjustable_but_never_disabled_by_a_falsy_value():
    old = observation(observed_at_utc=ts(-100))
    assert identity_assessment(old, max_age_seconds=200).fresh is True
    assert identity_assessment(old, max_age_seconds=50).fresh is False
    assert identity_assessment(old, max_age_seconds=0).fresh is False


@pytest.mark.parametrize("bad_now", [None, "2026-10-16T12:00:00Z", datetime(2026, 10, 16, 12, 0, 0), 5])
def test_now_must_be_an_aware_datetime_supplied_by_the_caller(bad_now):
    with pytest.raises(TypeError):
        assess_lab_identity(observation(), double_registry(), now=bad_now)


def test_the_module_never_reads_a_clock():
    import inspect
    source = inspect.getsource(lab_identity)
    assert "datetime.now" not in source and "utcnow" not in source and "time.time" not in source


# ------------------------------------------------------------
# the bridge to the Phase 13A scope policy (unchanged and still fail closed)
# ------------------------------------------------------------

def decide(assessment):
    policy = ScopePolicy([approved_lab()], execution_enabled=True)
    return policy.evaluate(ScopeRequest(target_url=TARGET, scenario=approved_scenario(), incident_id="incident_001",
                                        identity=assessment.scope_identity, authorization=authorization()))


def test_the_scope_policy_accepts_only_what_is_independently_verified_and_fresh():
    assert decide(identity_assessment()).execution_permitted is True
    unverified = identity_assessment(observation(verification={"claim": "operator_asserted", "source_id": "o"}))
    assert "lab_identity_not_independent" in decide(unverified).reason_codes
    assert decide(identity_assessment(observation(observed_at_utc=ts(-5000)))).reason_codes == (
        "lab_identity_missing",)


def test_with_the_default_registry_the_scope_policy_always_denies_for_independence():
    decision = decide(assess_lab_identity(observation(), now=NOW))
    assert decision.execution_permitted is False and "lab_identity_not_independent" in decision.reason_codes


def test_a_mismatching_observation_is_still_denied_by_the_scope_policy():
    wrong = identity_assessment(observation(observed_network="bridge"))
    assert wrong.independently_verified is True  # independence is not correctness
    assert "lab_identity_mismatch" in decide(wrong).reason_codes
