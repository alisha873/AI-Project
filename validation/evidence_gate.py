"""
Phase 13B.3 - the single gate that decides whether a report counts as live validation EVIDENCE (pure; no I/O).

Both the Phase 14 handoff (`usable_as_post_remediation_evidence`) and the baseline comparison
(`usable_as_comparison_evidence`) ask this module. A report passes only if EVERY condition holds; the failures are
returned as stable codes so a consumer can see why a report is not usable:

  status_not_conclusive                    only `objective_achieved` / `objective_not_observed` can be evidence
  check_not_completed                      the check did not finish (or did not run)
  not_post_remediation / not_pre_remediation   the tested state is not the one this use needs
  remediation_not_attested                 (post only) no attested application of remediation
  transport_not_live                       a fake transport gives a simulation; no transport gives nothing
  legacy_report_without_identity_provenance  a revision 1.0 report (or one with no identity record) says nothing
                                           about how the lab identity was established
  identity_not_live_suitable               the identity was not established by a live-suitable verifier from the
                                           repository's own registry (no such verifier exists in this build)
  report_not_issued_by_runner              this exact report was not produced by `validation.runner` in this
                                           process: a hand-built, edited, copied or reloaded report is never evidence

The last condition is what stops a caller-written report (with any flags set to taste) from impersonating a live,
independently verified run. It also means a report that was saved and reloaded is not usable until a signed,
persistent design exists (an open limitation). With no live transport and no registered verifier in this build, NO
report can pass this gate.
"""

from __future__ import annotations

from typing import Literal

from validation import seals
from validation.contract import CONCLUSIVE_STATUSES, CONTRACT_VERSION_IDENTITY, ValidationReport

GATE_REASON_CODES = (
    "status_not_conclusive",
    "check_not_completed",
    "not_post_remediation",
    "not_pre_remediation",
    "remediation_not_attested",
    "transport_not_live",
    "legacy_report_without_identity_provenance",
    "identity_not_live_suitable",
    "report_not_issued_by_runner",
)


def report_digest(report: ValidationReport) -> str:
    return seals.report_key(report.model_dump(mode="json"))


OBSERVATION_GATE_CODES = (
    "not_an_observation_report",
    "remediation_claim_present",
    "post_remediation_state_present",
    "post_remediation_gate_passed",
)


def evidence_gate_failures(
    report: object, *, tested_state: Literal["post_remediation", "pre_remediation"] = "post_remediation"
) -> list[str]:
    """Every reason `report` is not live evidence for `tested_state` (empty means it passes)."""
    if not isinstance(report, ValidationReport):
        return list(GATE_REASON_CODES)
    failures: list[str] = []
    if report.status not in CONCLUSIVE_STATUSES:
        failures.append("status_not_conclusive")
    if not (report.execution.attempted and report.execution.completed):
        failures.append("check_not_completed")
    if report.lab_state.tested_state != tested_state:
        failures.append("not_post_remediation" if tested_state == "post_remediation" else "not_pre_remediation")
    if tested_state == "post_remediation" and report.remediation.application.state != "independently_attested":
        failures.append("remediation_not_attested")
    if report.execution.transport_kind != "live":
        failures.append("transport_not_live")
    if report.contract_version != CONTRACT_VERSION_IDENTITY or report.identity is None:
        failures.append("legacy_report_without_identity_provenance")
    elif not (report.identity.live_suitable and report.identity.authoritative_registry):
        failures.append("identity_not_live_suitable")
    if not seals.is_report_issued(report_digest(report)):
        failures.append("report_not_issued_by_runner")
    return failures


def observation_gate_failures(report: object) -> list[str]:
    """Why `report` is not a pre-remediation observation.

    Passing the post-remediation gate, carrying proposals, or recording a post-remediation state keeps the
    report out of this category. An empty list means it is observation evidence only.
    """
    if not isinstance(report, ValidationReport):
        return ["not_an_observation_report", *GATE_REASON_CODES]
    failures: list[str] = []
    if report.report_kind != "pre_remediation_observation":
        failures.append("not_an_observation_report")
    failures.extend(evidence_gate_failures(report, tested_state="pre_remediation"))
    if report.remediation.proposals or report.remediation.application.state != "not_applicable":
        failures.append("remediation_claim_present")
    if report.lab_state.post_remediation_state_id is not None:
        failures.append("post_remediation_state_present")
    if not evidence_gate_failures(report, tested_state="post_remediation"):
        failures.append("post_remediation_gate_passed")
    return failures
