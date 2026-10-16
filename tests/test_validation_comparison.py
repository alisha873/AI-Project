"""Phase 13B.2 preparation: comparing a pre-remediation baseline with a post-remediation report.

Reports are hand-built (the current runner only ever produces post-remediation simulations), so these tests pin the
comparison rules, not any real behavior of any lab."""

from __future__ import annotations

import json

import pytest

from tests.validation_helpers import gate_without_identity_rules
from tests.validation_prep_helpers import (
    ACHIEVED,
    INCONCLUSIVE,
    NOT_OBSERVED,
    baseline,
    post,
    report,
)
from validation.comparison import (
    COMPARISON_LIMITATIONS,
    NOT_COMPARABLE_REASONS,
    SIMULATION_NOTE,
    BaselineComparison,
    compare_baseline_and_post,
    serialize_comparison,
)
from validation.handoff import Phase14Handoff, build_phase14_handoff


def compare(b=None, p=None):
    return compare_baseline_and_post(b if b is not None else baseline(), p if p is not None else post())


# ------------------------------------------------------------
# the outcome table (live, comparable, conclusive pairs)
# ------------------------------------------------------------

@pytest.mark.parametrize("base,after,category,interpretation,usable", [
    (ACHIEVED, NOT_OBSERVED, "objective_not_observed_after_baseline_achieved",
     "consistent_with_remediation_blocking_tested_probe", True),
    (ACHIEVED, ACHIEVED, "objective_still_achievable", "remediation_did_not_stop_tested_probe", True),
    (NOT_OBSERVED, ACHIEVED, "objective_newly_observed", "unexpected_regression", True),
    (NOT_OBSERVED, NOT_OBSERVED, "objective_not_observed_in_both", "no_inference_possible", False),
])
def test_the_conclusive_outcome_table(base, after, category, interpretation, usable, monkeypatch):
    gate_without_identity_rules(monkeypatch)  # the identity/issuance rules are tested un-stubbed elsewhere
    result = compare(baseline(base), post(after))
    assert result.comparable and result.not_comparable_reasons == []
    assert (result.category, result.interpretation, result.evidence_basis) == (category, interpretation, "live")
    assert result.usable_as_comparison_evidence is usable


def test_even_the_best_outcome_never_proves_effectiveness_or_a_guarantee():
    result = compare()
    assert result.category == "objective_not_observed_after_baseline_achieved"
    assert result.remediation_effectiveness_proven is False and result.general_security_guarantee is False
    assert any("not a universal security guarantee" in item for item in result.limitations)
    assert any("changes other than the remediation" in item for item in result.limitations)
    with pytest.raises(ValueError):
        BaselineComparison(**{**result.model_dump(), "remediation_effectiveness_proven": True})


def test_not_observed_in_both_runs_says_nothing_about_the_remediation():
    result = compare(baseline(NOT_OBSERVED), post(NOT_OBSERVED))
    assert result.interpretation == "no_inference_possible" and result.usable_as_comparison_evidence is False


# ------------------------------------------------------------
# inconclusive results never count as success or failure
# ------------------------------------------------------------

@pytest.mark.parametrize("after", [ACHIEVED, NOT_OBSERVED, INCONCLUSIVE])
def test_an_inconclusive_baseline_supports_no_inference(after):
    result = compare(baseline(INCONCLUSIVE), post(after))
    assert result.comparable is True
    assert (result.category, result.interpretation) == ("baseline_inconclusive", "no_inference_possible")
    assert result.usable_as_comparison_evidence is False


@pytest.mark.parametrize("base", [ACHIEVED, NOT_OBSERVED])
def test_an_inconclusive_post_report_is_never_treated_as_a_fixed_objective(base):
    result = compare(baseline(base), post(INCONCLUSIVE))
    assert (result.category, result.interpretation) == ("post_inconclusive", "no_inference_possible")
    assert result.usable_as_comparison_evidence is False


@pytest.mark.parametrize("reason", ["timeout", "lab_unreachable", "unexpected_response", "control_check_failed"])
def test_every_inconclusive_reason_gives_the_same_non_conclusion(reason):
    result = compare(baseline(ACHIEVED), post(("inconclusive", reason)))
    assert result.category == "post_inconclusive" and result.interpretation == "no_inference_possible"


# ------------------------------------------------------------
# reports that did not run
# ------------------------------------------------------------

def blocked_like():
    from tests.validation_helpers import blocked_report
    return blocked_report()


def test_a_baseline_that_did_not_run_is_reported_as_such():
    result = compare(blocked_like(), post())
    assert (result.category, result.comparable) == ("baseline_not_run", False)
    assert result.usable_as_comparison_evidence is False and result.interpretation == "no_inference_possible"


def test_a_post_report_that_did_not_run_is_not_a_fixed_objective():
    result = compare(baseline(), blocked_like())
    assert (result.category, result.comparable) == ("post_not_run", False)
    assert result.usable_as_comparison_evidence is False


def test_two_reports_that_did_not_run_compare_to_nothing():
    result = compare(blocked_like(), blocked_like())
    assert result.category == "baseline_not_run" and result.usable_as_comparison_evidence is False


# ------------------------------------------------------------
# structural comparability: inconsistencies are reported, never repaired
# ------------------------------------------------------------

def rebuilt(report_obj, **changes):
    data = report_obj.model_dump(mode="json")
    for path, value in changes.items():
        target = data
        *parents, leaf = path.split("__")
        for key in parents:
            target = target[key]
        target[leaf] = value
    from validation.contract import ValidationReport
    return ValidationReport.model_validate(data)


def without_state(report_obj, field):
    """Bypass validation on purpose: the report contract already refuses a missing state id, so this exercises the
    comparison's own defensive check against a report that was damaged after it was built."""
    return report_obj.model_copy(update={"lab_state": report_obj.lab_state.model_copy(update={field: None})})


def test_the_report_contract_itself_refuses_a_tested_state_without_its_state_id():
    with pytest.raises(ValueError):
        post(post=None)
    with pytest.raises(ValueError):
        baseline(pre=None)
    # a post report with no pre-state id IS allowed by the contract; the comparison is what refuses to use it
    assert "pre_state_id_missing" in compare(baseline(), post(pre=None)).not_comparable_reasons
@pytest.mark.parametrize("make,reason", [
    (lambda: (rebuilt(baseline(), incident_id="incident_002"), post()), "incident_mismatch"),
    (lambda: (without_state(baseline(), "pre_remediation_state_id"), post()), "pre_state_id_missing"),
    (lambda: (baseline(), without_state(post(), "pre_remediation_state_id")), "pre_state_id_missing"),
    (lambda: (baseline(pre="STATE-OTHER"), post()), "pre_state_id_mismatch"),
    (lambda: (baseline(), without_state(post(), "post_remediation_state_id")), "post_state_id_missing"),
    (lambda: (baseline(), post(post="STATE-PRE-1")), "states_not_distinct"),
    (lambda: (baseline(), post(run="VAL-BASE")), "same_run"),
    (lambda: (baseline(), post(kind="fake")), "transport_kinds_differ"),
    (lambda: (baseline(kind="fake"), post()), "transport_kinds_differ"),
    (lambda: (baseline(), post(started="2026-10-16T11:00:00Z", finished="2026-10-16T11:00:01Z")),
     "baseline_not_before_post"),
])
def test_each_structural_inconsistency_makes_the_pair_not_comparable(make, reason):
    b, p = make()
    result = compare(b, p)
    assert reason in result.not_comparable_reasons
    assert (result.category, result.comparable, result.interpretation) == (
        "not_comparable", False, "no_inference_possible")
    assert result.usable_as_comparison_evidence is False
    assert set(result.not_comparable_reasons) <= set(NOT_COMPARABLE_REASONS)


def test_a_different_scenario_definition_or_oracle_or_origin_is_not_comparable():
    sha = rebuilt(post(), scenario__definition_sha256="f" * 64)
    assert "scenario_definition_mismatch" in compare(baseline(), sha).not_comparable_reasons
    oracle = rebuilt(post(), oracle__oracle_version="9.9")
    assert "oracle_mismatch" in compare(baseline(), oracle).not_comparable_reasons
    origin = rebuilt(post(), target__requested_origin="http://127.0.0.1:9999")
    assert "target_origin_mismatch" in compare(baseline(), origin).not_comparable_reasons
    other_id = rebuilt(post(), scenario__scenario_id="another-scenario")
    assert "scenario_mismatch" in compare(baseline(), other_id).not_comparable_reasons


def test_the_baseline_must_test_the_pre_remediation_state_and_the_post_report_the_post_state():
    assert "baseline_not_pre_remediation" in compare(post(run="VAL-X"), post()).not_comparable_reasons
    assert "post_not_post_remediation" in compare(baseline(), baseline(run="VAL-Y")).not_comparable_reasons
    reversed_pair = compare(post(NOT_OBSERVED, run="VAL-1"), baseline(ACHIEVED, run="VAL-2"))
    assert reversed_pair.comparable is False
    assert {"baseline_not_pre_remediation", "post_not_post_remediation"} <= set(reversed_pair.not_comparable_reasons)


def test_a_post_report_without_an_attested_application_is_not_comparable():
    unknown = rebuilt(post(INCONCLUSIVE), remediation__application={
        "state": "unknown", "applied_proposal_ids": [], "attestation_id": None, "attested_by": None,
        "verified_by_phase13": False})
    result = compare(baseline(), unknown)
    assert "remediation_not_attested" in result.not_comparable_reasons and result.category == "not_comparable"
    with pytest.raises(ValueError, match="attested remediation application"):
        rebuilt(post(), remediation__application=unknown.remediation.application.model_dump(mode="json"))


def test_missing_or_unparseable_timestamps_are_not_guessed_at():
    odd = rebuilt(post(), execution__started_at_utc="2026-10-16T12:10:00+00:00")
    assert "timestamps_missing" in compare(baseline(), odd).not_comparable_reasons


def test_several_problems_are_all_reported_in_a_stable_order():
    result = compare(baseline(pre="STATE-OTHER", kind="fake"), post(run="VAL-BASE", post="STATE-OTHER"))
    assert result.not_comparable_reasons == compare(
        baseline(pre="STATE-OTHER", kind="fake"), post(run="VAL-BASE", post="STATE-OTHER")).not_comparable_reasons
    assert {"same_run", "pre_state_id_mismatch", "states_not_distinct", "transport_kinds_differ"} <= set(
        result.not_comparable_reasons)


@pytest.mark.parametrize("bad", [None, "report", {}, 7])
def test_non_report_inputs_are_refused_not_crashed_on(bad):
    for left, right in ((bad, post()), (baseline(), bad), (bad, bad)):
        result = compare_baseline_and_post(left, right)
        assert result.category == "not_comparable" and result.not_comparable_reasons == ["report_type_invalid"]
    assert compare_baseline_and_post(bad, bad).baseline_run_id is None


# ------------------------------------------------------------
# simulations are never evidence
# ------------------------------------------------------------

@pytest.mark.parametrize("base,after", [(ACHIEVED, NOT_OBSERVED), (ACHIEVED, ACHIEVED), (NOT_OBSERVED, ACHIEVED)])
def test_a_comparison_of_two_simulated_reports_is_labelled_a_simulation_and_not_usable(base, after):
    result = compare(baseline(base, kind="fake"), post(after, kind="fake"))
    assert result.comparable is True and result.evidence_basis == "simulation"
    assert result.interpretation == "simulation_only" and result.usable_as_comparison_evidence is False
    assert result.limitations[0] == SIMULATION_NOTE


def test_simulated_inconclusive_results_are_also_not_evidence():
    result = compare(baseline(INCONCLUSIVE, kind="fake"), post(NOT_OBSERVED, kind="fake"))
    assert result.category == "baseline_inconclusive" and result.usable_as_comparison_evidence is False


def test_a_live_comparison_has_no_simulation_note():
    assert SIMULATION_NOTE not in compare().limitations
    assert compare().limitations == list(COMPARISON_LIMITATIONS)


# ------------------------------------------------------------
# the Phase 14 handoff is unchanged and the comparison is a separate artifact
# ------------------------------------------------------------

def test_the_per_report_handoff_semantics_are_unchanged_by_the_comparison(monkeypatch):
    gate_without_identity_rules(monkeypatch)
    live_post = build_phase14_handoff(post(NOT_OBSERVED))
    assert live_post.usable_as_post_remediation_evidence is True
    assert build_phase14_handoff(post(NOT_OBSERVED, kind="fake")).usable_as_post_remediation_evidence is False
    assert build_phase14_handoff(post(INCONCLUSIVE)).usable_as_post_remediation_evidence is False
    assert "comparison" not in " ".join(Phase14Handoff.model_fields)


def test_a_post_report_alone_is_still_a_handoff_candidate_but_the_comparison_adds_the_baseline_question(monkeypatch):
    gate_without_identity_rules(monkeypatch)
    # the handoff treats a live attested conclusive post report as usable; only the comparison asks whether the probe
    # ever worked before the remediation
    solo = build_phase14_handoff(post(NOT_OBSERVED))
    assert solo.usable_as_post_remediation_evidence is True
    assert compare(baseline(NOT_OBSERVED), post(NOT_OBSERVED)).usable_as_comparison_evidence is False


# ------------------------------------------------------------
# serialization
# ------------------------------------------------------------

def test_the_comparison_is_versioned_and_serializes_deterministically():
    first, second = compare(), compare()
    assert first.comparison_version == "1.1" and serialize_comparison(first) == serialize_comparison(second)
    data = json.loads(serialize_comparison(first))
    assert data["remediation_effectiveness_proven"] is False and data["general_security_guarantee"] is False
    assert data["baseline_run_id"] == "VAL-BASE" and data["post_run_id"] == "VAL-POST"
    assert data["applied_proposal_ids"] and data["pre_remediation_state_id"] == "STATE-PRE-1"
    assert "timestamp" not in json.dumps(sorted(data)).lower()


def test_the_comparison_never_copies_raw_response_content():
    text = serialize_comparison(compare())
    assert "body" not in json.loads(text) and "first note" not in text
    assert report  # helper import is part of the module surface used above
