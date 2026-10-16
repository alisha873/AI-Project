"""
Phase 13B.2 preparation - comparing a pre-remediation BASELINE report with a post-remediation report (pure; no I/O).

Purpose. A post-remediation `objective_not_observed` on its own can not show that a remediation helped: the probe may
simply never have worked in that lab. Comparing it with a baseline run in which the probe DID work is the only way to
say anything. This module defines that comparison and, just as important, what it refuses to say.

What it does NOT do: it never concludes that a remediation is effective (`remediation_effectiveness_proven` and
`general_security_guarantee` are always False), it never applies anything and it never runs a check. It only reads two
already-built `ValidationReport` objects. It does not change the per-report Phase 14 handoff, which stays exactly as it
was; a future handoff version may embed a comparison.

Step 1 - triage by status. A report that did not run (`not_validated` / `blocked`) is `baseline_not_run` /
`post_not_run`.

Step 2 - structural comparability (every problem is reported, none is repaired). The two reports must be two different
runs of the same incident, scenario (id and definition hash), oracle (id and version) and requested origin; the
baseline must test the pre-remediation state and the post report the post-remediation state; both must carry lab-state
ids that agree (the baseline's pre-remediation id equals the post report's) and that differ from each other; the post
report must carry an application attestation naming applied proposals; both must have used the same transport kind; and
the baseline must have finished before the post run started. Any failure gives `not_comparable`.

Step 3 - outcome. Inconclusive results are never read as success or as failure: a baseline that is `inconclusive` gives
`baseline_inconclusive`, and an inconclusive post report gives `post_inconclusive`; neither supports any inference.
For conclusive pairs:

    baseline            post                   category
    achieved            achieved               objective_still_achievable
    achieved            not_observed           objective_not_observed_after_baseline_achieved
    not_observed        achieved               objective_newly_observed
    not_observed        not_observed           objective_not_observed_in_both
    (any inconclusive on either side)          baseline_inconclusive / post_inconclusive

Only `objective_not_observed_after_baseline_achieved` is "consistent with the remediation having blocked this one
probe, under the tested conditions"; even then other changes between the runs are possible confounders. `not_observed`
in both says nothing about the remediation.

Simulations. If the reports came from a fake transport the category is still computed (so the logic is testable) but
the interpretation is `simulation_only` and `usable_as_comparison_evidence` is False. Mixed or missing transport kinds
make the pair `not_comparable`.

Usable evidence (13B.3). `usable_as_comparison_evidence` also requires both reports to pass
`validation.evidence_gate` (identity provenance from a live-suitable verifier in the repository's registry, and the
runner having issued the exact report). `not_usable_reasons` lists the failing gate codes, prefixed baseline_ / post_.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from validation.contract import CONCLUSIVE_STATUSES, Identifier, Text, ValidationReport, _Strict
from validation.evidence_gate import evidence_gate_failures
from validation.lab_identity import parse_utc

# 1.1 (Phase 13B.3): "usable" additionally needs both reports to pass the evidence gate; adds `not_usable_reasons`.
COMPARISON_VERSION = "1.1"

Category = Literal[
    "not_comparable",
    "baseline_not_run",
    "post_not_run",
    "baseline_inconclusive",
    "post_inconclusive",
    "objective_still_achievable",
    "objective_not_observed_after_baseline_achieved",
    "objective_newly_observed",
    "objective_not_observed_in_both",
]
Interpretation = Literal[
    "no_inference_possible",
    "consistent_with_remediation_blocking_tested_probe",
    "remediation_did_not_stop_tested_probe",
    "unexpected_regression",
    "simulation_only",
]
EvidenceBasis = Literal["live", "simulation", "none"]
NOT_COMPARABLE_REASONS = (
    "report_type_invalid",
    "same_run",
    "incident_mismatch",
    "scenario_mismatch",
    "scenario_definition_mismatch",
    "oracle_mismatch",
    "target_origin_mismatch",
    "baseline_not_pre_remediation",
    "post_not_post_remediation",
    "pre_state_id_missing",
    "pre_state_id_mismatch",
    "post_state_id_missing",
    "states_not_distinct",
    "remediation_not_attested",
    "transport_kinds_differ",
    "timestamps_missing",
    "baseline_not_before_post",
)
NOT_RUN_STATUSES = ("not_validated", "blocked")

_OUTCOMES: dict[tuple[str, str], tuple[str, str]] = {
    ("objective_achieved", "objective_achieved"): (
        "objective_still_achievable", "remediation_did_not_stop_tested_probe"),
    ("objective_achieved", "objective_not_observed"): (
        "objective_not_observed_after_baseline_achieved", "consistent_with_remediation_blocking_tested_probe"),
    ("objective_not_observed", "objective_achieved"): (
        "objective_newly_observed", "unexpected_regression"),
    ("objective_not_observed", "objective_not_observed"): (
        "objective_not_observed_in_both", "no_inference_possible"),
}

SIMULATION_NOTE = (
    "SIMULATION: at least one of these reports came from a fake transport, so this comparison says nothing about any "
    "real lab or remediation and is not evidence."
)
COMPARISON_LIMITATIONS = (
    "The baseline and the post-remediation check are separate runs. A difference between them can come from changes "
    "other than the remediation (data, restarts, timing, configuration); this comparison can not rule that out.",
    "It covers one scenario, one probe and one control only. It is not a universal security guarantee, it does not mean "
    "residual risk is zero, and it does not prove that any remediation is effective.",
    "An inconclusive, blocked or not-run report is never evidence that a remediation worked or failed.",
)


class BaselineComparison(_Strict):
    comparison_version: Literal["1.1"] = COMPARISON_VERSION
    baseline_run_id: Identifier | None
    post_run_id: Identifier | None
    incident_id: Identifier | None
    scenario_id: Identifier | None
    comparable: bool
    not_comparable_reasons: list[Identifier]
    category: Category
    interpretation: Interpretation
    evidence_basis: EvidenceBasis
    usable_as_comparison_evidence: bool
    not_usable_reasons: list[Identifier]  # empty only when usable (gate codes prefixed baseline_/post_)
    applied_proposal_ids: list[Identifier]
    pre_remediation_state_id: Identifier | None
    post_remediation_state_id: Identifier | None
    limitations: list[Text]
    remediation_effectiveness_proven: Literal[False] = False
    general_security_guarantee: Literal[False] = False


def _result(baseline: object, post: object, *, comparable: bool, reasons: list[str], category: str,
            interpretation: str = "no_inference_possible", basis: str = "none", usable: bool = False,
            not_usable: list[str] | None = None) -> BaselineComparison:
    base = baseline if isinstance(baseline, ValidationReport) else None
    after = post if isinstance(post, ValidationReport) else None
    limitations = list(COMPARISON_LIMITATIONS)
    if basis == "simulation":
        limitations.insert(0, SIMULATION_NOTE)
    return BaselineComparison(
        baseline_run_id=base.validation_run_id if base else None,
        post_run_id=after.validation_run_id if after else None,
        incident_id=(after or base).incident_id if (after or base) else None,
        scenario_id=(after or base).scenario.scenario_id if (after or base) else None,
        comparable=comparable, not_comparable_reasons=reasons, category=category,
        interpretation=interpretation, evidence_basis=basis, usable_as_comparison_evidence=usable,
        not_usable_reasons=[] if usable else (not_usable or [category]),
        applied_proposal_ids=list(after.remediation.application.applied_proposal_ids) if after else [],
        pre_remediation_state_id=after.lab_state.pre_remediation_state_id if after else None,
        post_remediation_state_id=after.lab_state.post_remediation_state_id if after else None,
        limitations=limitations,
    )


def _moment(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return parse_utc(value)
    except ValueError:
        return None


def _structural_problems(base: ValidationReport, post: ValidationReport) -> list[str]:
    problems: list[str] = []
    if base.validation_run_id == post.validation_run_id:
        problems.append("same_run")
    if base.incident_id != post.incident_id:
        problems.append("incident_mismatch")
    if base.scenario.scenario_id != post.scenario.scenario_id:
        problems.append("scenario_mismatch")
    if (base.scenario.definition_sha256 is None or base.scenario.definition_sha256 != post.scenario.definition_sha256):
        problems.append("scenario_definition_mismatch")
    if (base.oracle is None or post.oracle is None
            or (base.oracle.oracle_id, base.oracle.oracle_version) != (post.oracle.oracle_id, post.oracle.oracle_version)):
        problems.append("oracle_mismatch")
    origin_a = base.target.requested_origin if base.target else None
    origin_b = post.target.requested_origin if post.target else None
    if origin_a is None or origin_a != origin_b:
        problems.append("target_origin_mismatch")
    if base.lab_state.tested_state != "pre_remediation":
        problems.append("baseline_not_pre_remediation")
    if post.lab_state.tested_state != "post_remediation":
        problems.append("post_not_post_remediation")
    pre_base, pre_post = base.lab_state.pre_remediation_state_id, post.lab_state.pre_remediation_state_id
    if pre_base is None or pre_post is None:
        problems.append("pre_state_id_missing")
    elif pre_base != pre_post:
        problems.append("pre_state_id_mismatch")
    post_id = post.lab_state.post_remediation_state_id
    if post_id is None:
        problems.append("post_state_id_missing")
    elif post_id == pre_post or post_id == pre_base:
        problems.append("states_not_distinct")
    application = post.remediation.application
    if application.state != "independently_attested" or not application.applied_proposal_ids:
        problems.append("remediation_not_attested")
    if base.execution.transport_kind != post.execution.transport_kind:
        problems.append("transport_kinds_differ")
    ended, started = _moment(base.execution.finished_at_utc), _moment(post.execution.started_at_utc)
    if ended is None or started is None:
        problems.append("timestamps_missing")
    elif ended > started:
        problems.append("baseline_not_before_post")
    return problems


def compare_baseline_and_post(baseline: object, post: object) -> BaselineComparison:
    """Compare two reports. Never raises for bad DATA; never says a remediation works."""
    if not isinstance(baseline, ValidationReport) or not isinstance(post, ValidationReport):
        return _result(baseline, post, comparable=False, reasons=["report_type_invalid"], category="not_comparable")
    if baseline.status in NOT_RUN_STATUSES:
        return _result(baseline, post, comparable=False, reasons=[], category="baseline_not_run")
    if post.status in NOT_RUN_STATUSES:
        return _result(baseline, post, comparable=False, reasons=[], category="post_not_run")

    problems = _structural_problems(baseline, post)
    if problems:
        return _result(baseline, post, comparable=False, reasons=problems, category="not_comparable")

    basis = "simulation" if baseline.execution.transport_kind == "fake" else "live"
    if baseline.status not in CONCLUSIVE_STATUSES:
        category, interpretation = "baseline_inconclusive", "no_inference_possible"
    elif post.status not in CONCLUSIVE_STATUSES:
        category, interpretation = "post_inconclusive", "no_inference_possible"
    else:
        category, interpretation = _OUTCOMES[(baseline.status, post.status)]
    conclusive = baseline.status in CONCLUSIVE_STATUSES and post.status in CONCLUSIVE_STATUSES
    if basis == "simulation":
        interpretation = "simulation_only"
    # "usable" needs a live, conclusive pair AND an informative outcome (not_observed in both informs nothing) AND
    # both reports passing the evidence gate (identity provenance from a live-suitable verifier, issued by the runner)
    gate = [f"baseline_{code}" for code in evidence_gate_failures(baseline, tested_state="pre_remediation")]
    gate += [f"post_{code}" for code in evidence_gate_failures(post, tested_state="post_remediation")]
    uninformative = ["outcome_not_informative"] if interpretation == "no_inference_possible" else []
    usable = conclusive and basis == "live" and not gate and not uninformative
    return _result(baseline, post, comparable=True, reasons=[], category=category, interpretation=interpretation,
                   basis=basis, usable=usable, not_usable=[*gate, *uninformative])


def serialize_comparison(comparison: BaselineComparison) -> str:
    """Deterministic JSON (sorted keys, no timestamps added)."""
    return json.dumps(comparison.model_dump(mode="json"), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
