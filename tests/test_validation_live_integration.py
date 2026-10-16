"""
Operator procedure for a genuine authorized live integration test. This module NEVER starts Docker,
never changes lab configuration, and never transmits the probe.

A live end-to-end run is pending until all of the following are true in the operator's environment:

1. The isolated AutoShield lab is already running (started independently by the operator).
2. Both named reviewers have recorded explicit, dated, procedure-bound approvals for
   `sqli-search-incident-001` through the project's review process (not this test).
3. An independent, live-suitable lab-identity verifier is registered in the repository.
4. An ApprovedLab, operator authorization and remediation attestation are supplied for that lab.
5. AUTOSHIELD_LIVE_VALIDATION=1 is set AND the CLI is invoked with
   `--live --i-accept-authorized-live-validation`.

Until those exist, the test below only records that the live path stays blocked. It is not skipped
in a way that hides a failure: it asserts the current, honest blocked state.
"""

from __future__ import annotations

from validation.handoff import build_phase14_handoff
from validation.orchestrator import ValidationRequest, build_validation_report
from validation.scenario import SQLI_SCENARIO_ID, sqli_search_scenario
from validation.scope import ScopePolicy


def test_the_builtin_scenario_stays_ineligible_so_live_integration_is_not_performed():
    scenario = sqli_search_scenario()
    assert scenario.review.required_approvers == ["Alisha Ranjan Kumar", "Jihan Chhabra"]
    assert scenario.review.approvals == []
    assert scenario.eligible_for_execution is False
    report = build_validation_report(
        ValidationRequest(incident_id="incident_001", scenario_id=SQLI_SCENARIO_ID),
        policy=ScopePolicy(),
    )
    assert report.execution.attempted is False and report.execution.requests_sent == 0
    assert build_phase14_handoff(report).usable_as_post_remediation_evidence is False
