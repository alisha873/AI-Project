"""
Shared helpers for the Phase 13B.2-preparation tests (not a test module). Everything is synthetic and offline.

`double_registry()` registers a verifier of kind `test_double`: it exists only so tests can exercise the
"independent verifier" logic. It is NOT an identity verifier, can never be suitable for live use, and no production
code constructs it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.validation_helpers import (
    CONTAINERS,
    COMPOSE_SHA,
    INCIDENT,
    authorization,
    executed_dict,
    planned_result,
)
from validation.attestation import (
    BoundAttestationAssessment,
    RemediationAttestation,
    assess_remediation_attestation,
    plan_binding_sha256,
)
from validation.contract import ValidationReport
from validation.inputs import intake_plan
from validation.lab_identity import (
    IdentityAssessment,
    LabIdentityObservation,
    VerifierEntry,
    VerifierRegistry,
    assess_lab_identity,
)
from validation.scenario import SQLI_SCENARIO_ID

NOW = datetime(2026, 10, 16, 12, 0, 0, tzinfo=timezone.utc)
VERIFIER = "test-double-verifier"
RUNNER = "autoshield-validation-runner-13b1"
PRE, POST = "STATE-PRE-1", "STATE-POST-1"


def ts(seconds_from_now: float) -> str:
    return (NOW + timedelta(seconds=seconds_from_now)).strftime("%Y-%m-%dT%H:%M:%SZ")


def double_registry() -> VerifierRegistry:
    return VerifierRegistry(VerifierEntry(verifier_id=VERIFIER, kind="test_double"))


def observation(**overrides) -> LabIdentityObservation:
    data = {
        "observation_id": "OBS-ID-1", "lab_id": "lab-1", "observed_network": "autoshield_net",
        "observed_containers": list(reversed(CONTAINERS)), "compose_sha256": COMPOSE_SHA,
        "lab_state_id": POST, "observed_at_utc": ts(-120),
        "verification": {"claim": "verifier_attested", "source_id": VERIFIER},
    }
    data.update(overrides)
    return LabIdentityObservation(**data)


def identity_assessment(obs=None, registry=None, **kwargs) -> IdentityAssessment:
    return assess_lab_identity(obs if obs is not None else observation(),
                               registry if registry is not None else double_registry(),
                               now=kwargs.pop("now", NOW), distinct_from=kwargs.pop("distinct_from", (RUNNER, "test-operator")),
                               **kwargs)


def attestation(result=None, **overrides) -> RemediationAttestation:
    result = result or planned_result()
    intake = intake_plan(result)
    data = {
        "attestation_id": "ATT-2", "incident_id": INCIDENT, "scenario_id": SQLI_SCENARIO_ID, "lab_id": "lab-1",
        "plan_binding_sha256": plan_binding_sha256(intake),
        "applied_proposal_ids": [intake.proposals[0].proposal_id],
        "pre_remediation_state_id": PRE, "post_remediation_state_id": POST,
        "attested_by": "test-operator", "authorization_id": "AUTH-1", "attested_at_utc": ts(-300),
    }
    data.update(overrides)
    return RemediationAttestation(**data)


def assess_bound(result=None, att=..., identity=..., auth=..., **overrides) -> BoundAttestationAssessment:
    result = result or planned_result()
    kwargs = {
        "incident_id": INCIDENT, "scenario_id": SQLI_SCENARIO_ID,
        "authorization": authorization() if auth is ... else auth,
        "identity": identity_assessment() if identity is ... else identity,
        "now": NOW,
    }
    kwargs.update(overrides)
    return assess_remediation_attestation(attestation(result) if att is ... else att, intake_plan(result), **kwargs)


# ------------------------------------------------------------
# reports for the comparison tests (hand-built, as a future live runner might produce them)
# ------------------------------------------------------------

def report(status: str, reason: str, tested_state: str, *, kind: str = "live", run: str = "VAL-A",
           started: str = "2026-10-16T12:00:00Z", finished: str = "2026-10-16T12:00:01Z",
           pre: str | None = PRE, post: str | None = POST, **edits) -> ValidationReport:
    data = executed_dict(status, reason, tested_state)
    data["validation_run_id"] = run
    data["execution"].update(transport_kind=kind, started_at_utc=started, finished_at_utc=finished)
    data["lab_state"].update(pre_remediation_state_id=pre, post_remediation_state_id=post)
    if kind == "fake":
        data["limitations"].append("SIMULATION: produced by a fake transport.")
    for key, value in edits.items():
        data[key] = value
    return ValidationReport.model_validate(data)


ACHIEVED = ("objective_achieved", "objective_observed")
NOT_OBSERVED = ("objective_not_observed", "objective_not_observed_under_tested_conditions")
INCONCLUSIVE = ("inconclusive", "timeout")


def baseline(pair=ACHIEVED, **kwargs) -> ValidationReport:
    kwargs.setdefault("run", "VAL-BASE")
    return report(*pair, "pre_remediation", **kwargs)


def post(pair=NOT_OBSERVED, **kwargs) -> ValidationReport:
    kwargs.setdefault("run", "VAL-POST")
    kwargs.setdefault("started", "2026-10-16T12:10:00Z")
    kwargs.setdefault("finished", "2026-10-16T12:10:01Z")
    return report(*pair, "post_remediation", **kwargs)
