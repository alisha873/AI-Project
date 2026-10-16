"""Phase 13B.3: identity gate integration and closing the legacy trust paths. Offline only.

Before 13B.3 a caller could write `LabIdentityEvidence(established_independently=True)` (or a whole
`ValidationReport` that "came from a live transport") and the scope policy, the runner and the Phase 14 handoff
would believe it. Each test below pins one bypass that was found, and the rule that now closes it.

Nothing here contacts anything: transports are in-memory scripts, the "live" ones are synthetic stand-ins that never
touch a socket, and the constants that keep live traffic off (`LIVE_TRANSPORT_ENABLED`, `LIVE_EXECUTION_IMPLEMENTED`)
are only patched inside tests that exercise the synthetic positive path, then restored."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

import validation.lab_identity as lab_identity_module
import validation.runner as runner_module
import validation.seals as seals
from tests.validation_helpers import (
    CONTAINERS, COMPOSE_SHA, IDENTITY_OBSERVED_AT, SQLI_SCENARIO_ID, approved_registry, attestation_for, authorization,
    blocked_report, executed_dict, full_policy, full_request, identity, lab_body, planned_result, run_fake,
    simulated_lab,
)
from tests.validation_prep_helpers import ACHIEVED, NOT_OBSERVED, baseline, post
from validation.comparison import compare_baseline_and_post
from validation.contract import IdentityProvenance, ValidationReport
from validation.evidence_gate import GATE_REASON_CODES, evidence_gate_failures, report_digest
from validation.handoff import build_phase14_handoff
from validation.lab_identity import (
    IDENTITY_VERIFIER_AVAILABLE, LabIdentityObservation, VerifierEntry, VerifierRegistry, assess_lab_identity,
    default_verifier_registry,
)
from validation.orchestrator import LIVE_EXECUTION_IMPLEMENTED, ValidationRequestError, assess
from validation.runner import RUNNER_ID, run_controlled_validation
from validation.scenario import default_registry
from validation.scope import LabIdentityEvidence, ScopePolicy, ScopeRequest
from validation.transport import ScriptedTransport, TransportResponse

OBSERVED = IDENTITY_OBSERVED_AT  # the default test identity was observed at 11:59:00Z
LIVE_VERIFIER = "repo-live-verifier-synthetic"


def clock_at(moment: datetime):
    """A clock whose first reading (the run's `now`) is `moment`, then one second later per reading."""
    readings = iter(moment + timedelta(seconds=n) for n in range(60))
    return lambda: next(readings)


class SyntheticLiveTransport(ScriptedTransport):
    """Claims `kind == "live"` but is an in-memory script: it can not reach anything."""

    kind = "live"


def synthetic_live(vulnerable: bool = False) -> SyntheticLiveTransport:
    return SyntheticLiveTransport(simulated_lab(vulnerable), simulated_lab(vulnerable), transport_id="synthetic-live")


def repository_verifier(*, live: bool = True) -> VerifierEntry:
    return VerifierEntry(verifier_id=LIVE_VERIFIER, kind="read_only_environment_inspector", suitable_for_live=live)


def observation(*, source: str = LIVE_VERIFIER, at: datetime = OBSERVED, **overrides) -> LabIdentityObservation:
    data = {
        "observation_id": "OBS-ID-G", "lab_id": "lab-1", "observed_network": "autoshield_net",
        "observed_containers": list(reversed(CONTAINERS)), "compose_sha256": COMPOSE_SHA,
        "lab_state_id": "STATE-POST-1", "observed_at_utc": at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verification": {"claim": "verifier_attested", "source_id": source},
    }
    data.update(overrides)
    return LabIdentityObservation(**data)


def request_for(evidence, **overrides) -> ScopeRequest:
    data = {
        "target_url": "http://127.0.0.1:8080", "scenario": approved_registry().get(SQLI_SCENARIO_ID),
        "incident_id": "incident_001", "identity": evidence, "authorization": authorization(),
        "now": datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc),
    }
    data.update(overrides)
    return ScopeRequest(**data)


def decide(evidence, **overrides):
    return full_policy().evaluate(request_for(evidence, **overrides))


def hand_built(**overrides) -> LabIdentityEvidence:
    """The legacy bypass: evidence that ASSERTS independence, built by hand (no assessment ever ran)."""
    data = {
        "lab_id": "lab-1", "observed_network": "autoshield_net", "observed_containers": list(CONTAINERS),
        "compose_sha256": COMPOSE_SHA, "lab_state_id": "STATE-POST-1", "established_by": "i-promise-i-verified",
        "established_independently": True,
    }
    data.update(overrides)
    return LabIdentityEvidence(**data)


@pytest.fixture
def live_world(monkeypatch):
    """A SYNTHETIC world with a reviewed, live-suitable repository verifier and the live lock open. The real build has
    neither: this only lets the positive path of the gate be exercised, with no I/O at all."""
    monkeypatch.setattr(lab_identity_module, "_BUILTIN_VERIFIERS", (repository_verifier(),))
    monkeypatch.setattr(runner_module, "LIVE_TRANSPORT_ENABLED", True)


def repository_identity(**overrides) -> LabIdentityEvidence:
    result = assess_lab_identity(observation(**overrides), default_verifier_registry(), now=OBSERVED)
    assert result.independently_verified and result.live_suitable and result.authoritative_registry
    return result.scope_identity


# ======================================================================
# B1: a caller-written independence claim
# ======================================================================

def test_hand_built_independent_evidence_is_denied_by_the_scope_policy():
    decision = decide(hand_built())
    assert decision.execution_permitted is False and decision.scope_conditions_met is False
    assert "lab_identity_not_assessed" in decision.reason_codes


def test_evidence_is_trusted_by_exact_content_and_a_single_changed_field_breaks_it():
    real = identity()
    forged = LabIdentityEvidence(**real.model_dump())  # an exact copy is the same content, so it IS the same evidence
    assert decide(forged).execution_permitted is True
    other_lab_state = real.model_copy(update={"lab_state_id": "STATE-POST-2"})  # one changed field: never issued
    assert "lab_identity_not_assessed" in decide(other_lab_state).reason_codes
    other_verifier = real.model_copy(update={"established_by": "somebody-else"})
    assert "lab_identity_not_assessed" in decide(other_verifier).reason_codes


def test_flipping_independence_on_an_assessed_but_unverified_identity_is_not_issued():
    result = assess_lab_identity(observation(source="unregistered-thing"), VerifierRegistry(
        VerifierEntry(verifier_id="someone-else", kind="test_double")), now=OBSERVED)
    assert result.independently_verified is False and result.scope_identity.established_independently is False
    flipped = result.scope_identity.model_copy(update={"established_independently": True})
    assert "lab_identity_not_assessed" in decide(flipped).reason_codes


def test_the_runner_never_sends_a_request_for_hand_built_independent_evidence():
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, request=full_request(lab_identity=hand_built(lab_state_id="STATE-POST-1")))
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert report.execution.attempted is False and transport.requests == []
    assert "lab_identity_not_assessed" in report.scope.reason_codes


def test_operator_assertion_alone_never_produces_independent_evidence():
    asserted = observation(verification={"claim": "operator_asserted", "source_id": "test-operator"})
    result = assess_lab_identity(asserted, VerifierRegistry(
        VerifierEntry(verifier_id="test-operator", kind="test_double")), now=OBSERVED)
    assert result.level == "operator_asserted" and result.independently_verified is False
    assert "lab_identity_not_independent" in decide(result.scope_identity).reason_codes


def test_the_gate_accepts_real_assessed_evidence_with_a_test_double_for_simulation_only():
    # the same assessed evidence that lets a SIMULATION run is never live-suitable
    evidence = identity()
    assert decide(evidence).execution_permitted is True
    live = decide(evidence, require_live_suitable_identity=True)
    assert live.execution_permitted is False and "lab_identity_not_live_suitable" in live.reason_codes


# ======================================================================
# B3: caller-built registries and caller-chosen conflict lists
# ======================================================================

def test_a_caller_built_registry_can_never_make_evidence_live_suitable():
    registry = VerifierRegistry(repository_verifier())  # even declaring itself live-suitable
    result = assess_lab_identity(observation(), registry, now=OBSERVED)
    assert result.independently_verified is True
    assert result.live_suitable is False and result.authoritative_registry is False
    assert "lab_identity_not_live_suitable" in decide(
        result.scope_identity, require_live_suitable_identity=True).reason_codes


def test_the_repository_registry_is_empty_so_nothing_is_live_suitable_in_this_build():
    assert lab_identity_module._BUILTIN_VERIFIERS == () and IDENTITY_VERIFIER_AVAILABLE is False
    registry = default_verifier_registry()
    assert registry.is_empty and registry.authoritative is True
    result = assess_lab_identity(observation(), now=OBSERVED)  # default registry
    assert (result.level, result.live_suitable, result.independently_verified) == (
        "verifier_unavailable", False, False)
    assert "lab_identity_not_independent" in decide(result.scope_identity).reason_codes


def test_a_live_suitable_repository_verifier_is_required_for_live_suitable_evidence(live_world):
    result = assess_lab_identity(observation(), default_verifier_registry(), now=OBSERVED)
    assert result.live_suitable is True
    # a test double can never be live-suitable even if the repository (wrongly) listed it as such
    with pytest.raises(ValidationError):
        VerifierEntry(verifier_id="x", kind="test_double", suitable_for_live=True)


def test_an_unregistered_verifier_is_not_independent(live_world):
    result = assess_lab_identity(observation(source="not-in-the-registry"), default_verifier_registry(), now=OBSERVED)
    assert result.level == "verifier_unregistered" and result.independently_verified is False
    assert "lab_identity_not_independent" in decide(result.scope_identity).reason_codes


@pytest.mark.parametrize("who", ["test-operator", RUNNER_ID])
def test_the_verifier_may_not_be_the_operator_or_the_runner_even_with_no_distinct_from_list(who):
    registry = VerifierRegistry(VerifierEntry(verifier_id=who, kind="test_double"))
    result = assess_lab_identity(observation(source=who), registry, now=OBSERVED)  # nothing passed as distinct_from
    assert result.independently_verified is True, "assessment alone does not know the operator or the runner"
    transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
    report, _ = run_fake(transport, request=full_request(lab_identity=result.scope_identity))
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert "lab_identity_verifier_conflict" in report.scope.reason_codes and transport.requests == []


def test_a_caller_supplied_runner_id_is_also_a_forbidden_verifier():
    registry = VerifierRegistry(VerifierEntry(verifier_id="custom-runner-7", kind="test_double"))
    evidence = assess_lab_identity(observation(source="custom-runner-7"), registry, now=OBSERVED).scope_identity
    report, _ = run_fake(request=full_request(lab_identity=evidence), runner_id="custom-runner-7")
    assert "lab_identity_verifier_conflict" in report.scope.reason_codes and report.execution.attempted is False


def test_the_verifier_may_not_be_the_remediation_attester():
    result = planned_result()
    registry = VerifierRegistry(VerifierEntry(verifier_id="attester-1", kind="test_double"))
    evidence = assess_lab_identity(observation(source="attester-1"), registry, now=OBSERVED).scope_identity
    request = full_request(result, lab_identity=evidence, application=attestation_for(result, attested_by="attester-1"))
    assessment = assess(request, approved_registry(), full_policy(), now=OBSERVED)
    assert "lab_identity_verifier_conflict" in assessment.decision.reason_codes


def test_the_verifier_may_not_be_a_caller_forbidden_id():
    decision = decide(identity(), forbidden_verifier_ids=("test-identity-probe",))
    assert "lab_identity_verifier_conflict" in decision.reason_codes and decision.execution_permitted is False


def test_an_independent_verifier_is_not_blocked_by_the_conflict_check():
    decision = decide(identity(), forbidden_verifier_ids=("test-operator", RUNNER_ID))
    assert decision.execution_permitted is True and decision.reason_codes == ()


# ======================================================================
# B4: stale or future observations, checked when the evidence is USED
# ======================================================================

@pytest.mark.parametrize("seconds,code", [
    (900 + 1, "lab_identity_stale"), (86_400, "lab_identity_stale"),
    (-60 - 1, "lab_identity_from_future"), (-86_400, "lab_identity_from_future"),
])
def test_evidence_that_was_fresh_when_assessed_is_rechecked_when_used(seconds, code):
    evidence = identity()  # assessed fresh (observation 11:59:00Z)
    moment = OBSERVED + timedelta(seconds=seconds)
    decision = decide(evidence, now=moment)
    assert decision.execution_permitted is False and code in decision.reason_codes


@pytest.mark.parametrize("seconds", [-60, 0, 450, 900])
def test_the_freshness_window_boundaries_are_inclusive(seconds):
    assert decide(identity(), now=OBSERVED + timedelta(seconds=seconds)).execution_permitted is True


def test_the_runner_denies_stale_and_future_evidence_and_sends_nothing():
    for seconds, code in ((901, "lab_identity_stale"), (-61, "lab_identity_from_future")):
        transport = ScriptedTransport(simulated_lab(True), simulated_lab(True))
        report, _ = run_fake(transport, clock=clock_at(OBSERVED + timedelta(seconds=seconds)))
        assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
        assert code in report.scope.reason_codes and transport.requests == []


def test_a_caller_can_tighten_but_never_extend_the_freshness_limit():
    stale = observation(source="v-1", at=OBSERVED - timedelta(seconds=901))
    registry = VerifierRegistry(VerifierEntry(verifier_id="v-1", kind="test_double"))
    result = assess_lab_identity(stale, registry, now=OBSERVED, max_age_seconds=10 ** 9)
    assert result.fresh is False and result.independently_verified is False
    tight = assess_lab_identity(observation(source="v-1", at=OBSERVED - timedelta(seconds=120)), registry,
                                now=OBSERVED, max_age_seconds=60)
    assert tight.fresh is False


def test_an_older_observation_assessed_later_does_not_shorten_a_newer_one():
    registry = VerifierRegistry(VerifierEntry(verifier_id="v-2", kind="test_double"))
    newer = assess_lab_identity(observation(source="v-2", at=OBSERVED + timedelta(hours=2)), registry,
                                now=OBSERVED + timedelta(hours=2)).scope_identity
    older = assess_lab_identity(observation(source="v-2", at=OBSERVED), registry, now=OBSERVED).scope_identity
    assert newer == older  # identical content shares the records of both observations
    assert decide(newer, now=OBSERVED + timedelta(minutes=5)).execution_permitted is True
    assert decide(newer, now=OBSERVED + timedelta(hours=2, minutes=5)).execution_permitted is True
    gap = decide(newer, now=OBSERVED + timedelta(hours=1))
    assert gap.execution_permitted is False and "lab_identity_stale" in gap.reason_codes


def test_use_time_freshness_is_unchecked_only_when_no_clock_is_given_and_never_for_a_live_check():
    evidence = identity()
    assert decide(evidence, now=None).execution_permitted is True  # the offline 13A path never sends anything
    live = decide(evidence, now=None, require_live_suitable_identity=True)
    assert "lab_identity_freshness_unchecked" in live.reason_codes and live.execution_permitted is False


def test_a_naive_run_clock_is_refused_rather_than_trusted():
    with pytest.raises(ValidationRequestError):
        run_controlled_validation(full_request(), approved_registry(), full_policy(),
                                  ScriptedTransport(), clock=lambda: datetime(2026, 10, 16, 12, 0, 0))


def test_issuance_records_are_bounded_and_eviction_only_removes_trust(monkeypatch):
    evidence = identity(lab_id="lab-1")
    assert decide(evidence).execution_permitted is True
    monkeypatch.setattr(seals, "_MAX_RECORDS", 1)
    identity(observed_network="other-net-1")  # a different key evicts the first
    assert "lab_identity_not_assessed" in decide(evidence).reason_codes


# ======================================================================
# B2: reports and the Phase 14 handoff
# ======================================================================

def live_report(vulnerable: bool = False) -> ValidationReport:
    report, transport = run_fake(synthetic_live(vulnerable), request=full_request(
        lab_identity=repository_identity()))
    assert report.execution.transport_kind == "live" and len(transport.requests) == 2
    return report


def test_a_synthetic_live_run_with_a_live_suitable_repository_verifier_passes_the_gate_end_to_end(live_world):
    report = live_report()
    assert report.contract_version == "1.1" and report.status == "objective_not_observed"
    assert report.identity.live_suitable is True and report.identity.authoritative_registry is True
    assert report.identity.source_id == LIVE_VERIFIER
    assert evidence_gate_failures(report) == []
    handoff = build_phase14_handoff(report)
    assert handoff.usable_as_post_remediation_evidence is True and handoff.not_usable_reasons == []
    assert (handoff.identity_source_id, handoff.identity_live_suitable) == (LIVE_VERIFIER, True)
    assert handoff.remediation_effectiveness_proven is False and handoff.general_security_guarantee is False


def test_with_the_live_lock_open_a_live_transport_still_needs_a_live_suitable_identity(monkeypatch):
    monkeypatch.setattr(runner_module, "LIVE_TRANSPORT_ENABLED", True)  # synthetic: the lock opened, nothing else
    transport = synthetic_live()
    report, _ = run_fake(transport, request=full_request(lab_identity=identity()))  # test-double identity
    assert (report.status, report.reason_code) == ("blocked", "scope_policy_denied")
    assert "lab_identity_not_live_suitable" in report.scope.reason_codes and transport.requests == []


def test_with_the_live_lock_closed_the_live_refusal_keeps_its_established_reason():
    transport = synthetic_live()
    report, _ = run_fake(transport, request=full_request(lab_identity=identity()))
    assert (report.status, report.reason_code) == ("blocked", "live_transport_not_enabled")
    assert transport.requests == []


def test_a_simulation_is_never_usable_evidence_even_with_real_assessed_identity():
    report, _ = run_fake(vulnerable=False)
    assert report.status == "objective_not_observed" and report.execution.transport_kind == "fake"
    handoff = build_phase14_handoff(report)
    assert handoff.usable_as_post_remediation_evidence is False
    assert "transport_not_live" in handoff.not_usable_reasons
    assert "identity_not_live_suitable" in handoff.not_usable_reasons
    assert "report_not_issued_by_runner" not in handoff.not_usable_reasons  # the runner did issue it
    assert report.identity.live_suitable is False


def test_a_legacy_hand_built_live_report_is_not_usable_evidence():
    legacy = ValidationReport.model_validate(executed_dict(*NOT_OBSERVED))  # the pre-13B.3 "usable" fixture
    assert legacy.contract_version == "1.0" and legacy.identity is None  # old reports still load unchanged
    handoff = build_phase14_handoff(legacy)
    assert handoff.usable_as_post_remediation_evidence is False
    assert "legacy_report_without_identity_provenance" in handoff.not_usable_reasons
    assert "report_not_issued_by_runner" in handoff.not_usable_reasons


def provenance(**overrides) -> dict:
    data = {"source_id": LIVE_VERIFIER, "observed_at_utc": "2026-10-16T11:59:00Z",
            "checked_at_utc": "2026-10-16T12:00:00Z", "lab_state_id": "STATE-POST-1",
            "independently_verified": True, "live_suitable": True, "authoritative_registry": True}
    data.update(overrides)
    return data


def test_a_hand_built_1_1_report_with_perfect_looking_provenance_is_still_not_issued():
    data = executed_dict(*NOT_OBSERVED)
    data.update(contract_version="1.1", identity=provenance())
    forged = ValidationReport.model_validate(data)
    assert forged.identity.live_suitable is True
    assert evidence_gate_failures(forged) == ["report_not_issued_by_runner"]
    assert build_phase14_handoff(forged).usable_as_post_remediation_evidence is False


def test_editing_an_issued_report_removes_its_issuance(live_world):
    report = live_report()
    assert evidence_gate_failures(report) == []
    for update in ({"status": "objective_achieved", "reason_code": "objective_observed"},
                   {"validation_run_id": "VAL-edited"}, {"incident_id": "incident_999"}):
        copy = report.model_copy(update=update)
        assert "report_not_issued_by_runner" in evidence_gate_failures(copy)
    tampered_identity = report.model_copy(update={"identity": report.identity.model_copy(
        update={"source_id": "someone-else"})})
    assert "report_not_issued_by_runner" in evidence_gate_failures(tampered_identity)


def test_a_byte_identical_reload_of_an_issued_report_in_the_same_process_is_the_same_report(live_world):
    report = live_report()
    reloaded = ValidationReport.model_validate(json.loads(json.dumps(report.model_dump(mode="json"))))
    assert report_digest(reloaded) == report_digest(report) and evidence_gate_failures(reloaded) == []


def test_issuance_is_not_persistent_so_a_report_unknown_to_this_process_is_not_usable(live_world, monkeypatch):
    report = live_report()
    monkeypatch.setattr(seals, "_REPORTS", {})  # what a fresh process looks like
    assert evidence_gate_failures(report) == ["report_not_issued_by_runner"]


def test_the_runner_issues_reports_for_checks_that_ran_and_not_for_blocked_runs():
    ran, _ = run_fake(vulnerable=False)
    assert seals.is_report_issued(report_digest(ran)) is True
    blocked = run_controlled_validation(full_request(), approved_registry(), full_policy(False), ScriptedTransport(),
                                        clock=clock_at(OBSERVED + timedelta(seconds=60)))
    assert blocked.status == "blocked" and seals.is_report_issued(report_digest(blocked)) is False
    assert blocked.identity is None and blocked.contract_version == "1.0"


def test_simulation_blocked_inconclusive_and_conclusive_stay_distinct(live_world):
    blocked = blocked_report()
    reasons = build_phase14_handoff(blocked).not_usable_reasons
    assert "status_not_conclusive" in reasons and "check_not_completed" in reasons
    timeout, _ = run_fake(ScriptedTransport(simulated_lab(False), TransportResponse.timeout()))
    assert timeout.status == "inconclusive" and timeout.identity is not None
    assert "status_not_conclusive" in build_phase14_handoff(timeout).not_usable_reasons
    simulated, _ = run_fake(vulnerable=True)
    assert simulated.status == "objective_achieved" and simulated.execution.transport_kind == "fake"
    assert "transport_not_live" in build_phase14_handoff(simulated).not_usable_reasons
    assert evidence_gate_failures(live_report(vulnerable=True)) == []  # a conclusive live "achieved" is usable evidence


def test_every_gate_code_is_documented_and_a_non_report_fails_everything():
    assert evidence_gate_failures(None) == list(GATE_REASON_CODES)
    assert evidence_gate_failures({"status": "objective_achieved"}) == list(GATE_REASON_CODES)
    assert len(set(GATE_REASON_CODES)) == len(GATE_REASON_CODES)


def test_no_report_can_be_usable_in_this_real_build():
    # no live transport exists and no verifier is registered: every report the runner can make is refused by the gate
    for vulnerable in (False, True):
        report, _ = run_fake(vulnerable=vulnerable)
        assert build_phase14_handoff(report).usable_as_post_remediation_evidence is False
    assert lab_identity_module._BUILTIN_VERIFIERS == () and runner_module.LIVE_TRANSPORT_ENABLED is False


# ======================================================================
# contract migration 1.0 -> 1.1
# ======================================================================

def test_the_migration_is_explicit_and_backward_compatible():
    old = ValidationReport.model_validate(blocked_report().model_dump(mode="json"))
    assert old.contract_version == "1.0" and old.identity is None  # a 1.0 report is never rewritten or reinterpreted
    legacy = ValidationReport.model_validate(executed_dict(*ACHIEVED))
    assert legacy.contract_version == "1.0" and legacy.execution.attempted is True and legacy.identity is None


def test_identity_provenance_belongs_to_revision_1_1_only():
    data = executed_dict(*NOT_OBSERVED)
    with pytest.raises(ValidationError):
        ValidationReport.model_validate({**data, "identity": provenance()})  # 1.0 with provenance
    with pytest.raises(ValidationError):
        ValidationReport.model_validate({**data, "contract_version": "1.1"})  # 1.1 that ran, no provenance
    blocked = blocked_report().model_dump(mode="json")
    with pytest.raises(ValidationError):
        ValidationReport.model_validate({**blocked, "contract_version": "1.1", "identity": provenance()})
    ValidationReport.model_validate({**data, "contract_version": "1.1", "identity": provenance()})  # consistent
    with pytest.raises(ValidationError):
        ValidationReport.model_validate({**data, "contract_version": "1.2", "identity": provenance()})


def test_provenance_can_not_claim_live_suitability_without_the_repository_registry():
    with pytest.raises(ValidationError):
        IdentityProvenance(**provenance(authoritative_registry=False))
    with pytest.raises(ValidationError):
        IdentityProvenance(**provenance(independently_verified=False))
    with pytest.raises(ValidationError):
        IdentityProvenance(**provenance(checked_at_utc="not-a-time"))
    IdentityProvenance(**provenance(live_suitable=False, authoritative_registry=False))


def test_the_runner_records_how_the_identity_was_established():
    report, _ = run_fake(vulnerable=False)
    identity_record = report.identity
    assert identity_record.source_id == "test-identity-probe" and identity_record.independently_verified is True
    assert identity_record.observed_at_utc == "2026-10-16T11:59:00Z"
    assert identity_record.checked_at_utc == report.execution.started_at_utc == "2026-10-16T12:00:00Z"
    assert (identity_record.live_suitable, identity_record.authoritative_registry) == (False, False)
    assert identity_record.lab_state_id == report.lab_state.post_remediation_state_id


# ======================================================================
# comparison
# ======================================================================

def issued(report: ValidationReport) -> ValidationReport:
    """Mark a hand-built report as runner-issued. This calls the issue function directly, which only code inside this
    process can do (a documented limit); it lets the comparison's positive path be tested without a live runner."""
    seals.issue_report(report_digest(report))
    return report


def with_identity(builder, **kwargs):
    return builder(contract_version="1.1", identity=provenance(), **kwargs)


def test_a_comparison_of_unissued_legacy_reports_is_never_usable_evidence():
    result = compare_baseline_and_post(baseline(ACHIEVED), post(NOT_OBSERVED))
    assert result.comparable is True and result.evidence_basis == "live"
    assert result.interpretation == "consistent_with_remediation_blocking_tested_probe"
    assert result.usable_as_comparison_evidence is False
    assert "baseline_legacy_report_without_identity_provenance" in result.not_usable_reasons
    assert "post_report_not_issued_by_runner" in result.not_usable_reasons


def test_a_comparison_needs_both_reports_to_pass_the_gate():
    good_base, good_post = (issued(with_identity(baseline, pair=ACHIEVED)),
                            issued(with_identity(post, pair=NOT_OBSERVED)))
    ok = compare_baseline_and_post(good_base, good_post)
    assert ok.usable_as_comparison_evidence is True and ok.not_usable_reasons == []
    assert ok.remediation_effectiveness_proven is False and ok.general_security_guarantee is False
    unissued_base = with_identity(baseline, pair=ACHIEVED, run="VAL-BASE-UNISSUED")  # same content shape, never issued
    only_post = compare_baseline_and_post(unissued_base, good_post)
    assert only_post.usable_as_comparison_evidence is False
    assert only_post.not_usable_reasons == ["baseline_report_not_issued_by_runner"]
    weak = issued(post(NOT_OBSERVED, run="VAL-POST-2", contract_version="1.1",
                       identity=provenance(live_suitable=False, authoritative_registry=False)))
    not_live = compare_baseline_and_post(good_base, weak)
    assert not_live.usable_as_comparison_evidence is False
    assert "post_identity_not_live_suitable" in not_live.not_usable_reasons


def test_not_usable_reasons_are_never_empty_when_a_comparison_is_unusable():
    for pair in ((baseline(ACHIEVED), post(NOT_OBSERVED)), (baseline(NOT_OBSERVED), post(NOT_OBSERVED)),
                 (blocked_report(), post(NOT_OBSERVED)), (None, None)):
        result = compare_baseline_and_post(*pair)
        assert result.usable_as_comparison_evidence is False and result.not_usable_reasons


# ======================================================================
# every code path is pinned, the locks are unchanged, the scenario is still pending review
# ======================================================================

VALIDATION_DIR = Path(__file__).resolve().parents[1] / "validation"


def calls_of(name: str) -> set[str]:
    found: set[str] = set()
    for path in VALIDATION_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if called == name:
                    found.add(path.name)
    return found


def test_identity_issuance_has_exactly_one_caller():
    assert calls_of("issue_identity") == {"lab_identity.py"}


def test_report_issuance_has_exactly_one_caller():
    assert calls_of("issue_report") == {"runner.py"}


def test_independent_evidence_is_constructed_only_by_the_assessment_path():
    constructors = set()
    for path in VALIDATION_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LabIdentityEvidence"):
                constructors.add(path.name)
    assert constructors == {"lab_identity.py"}


def test_the_builtin_verifier_list_is_assigned_in_one_place_and_empty():
    assigned = set()
    for path in VALIDATION_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(
                node, ast.AnnAssign) else []
            if any(getattr(t, "id", None) == "_BUILTIN_VERIFIERS" for t in targets):
                assigned.add(path.name)
    assert assigned == {"lab_identity.py"}


def test_the_live_locks_and_defaults_are_unchanged():
    assert runner_module.LIVE_TRANSPORT_ENABLED is False and LIVE_EXECUTION_IMPLEMENTED is False
    decision = ScopePolicy().evaluate(request_for(identity()))
    assert decision.execution_permitted is False and "execution_disabled" in decision.reason_codes


def test_the_first_scenario_is_still_pending_human_review_and_not_executable():
    scenario = default_registry().get(SQLI_SCENARIO_ID)
    assert scenario.review.status != "approved" and scenario.eligible_for_execution is False