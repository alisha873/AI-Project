"""
Phase 13B.2 preparation - the versioned remediation-attestation contract and its binding checks (pure; no I/O).

A `RemediationAttestation` is a CLAIM by an authorized operator that specific remediation was applied to the lab. Phase
13 does not apply remediation and does not verify the claim (`verified_by_phase13` is always False). The contract makes
the claim *bind* everything it must agree with, so a claim written for something else can not be reused:

  incident          `incident_id`, equal to the incident being validated and to the plan's incident
  scenario          `scenario_id`, equal to the scenario being validated
  lab               `lab_id`, equal to the operator authorization and to the observed lab identity
  plan              `plan_binding_sha256`, a hash of the plan's proposals (ids and grounding references); a plan that
                    changed after the claim was made no longer matches (the claim is stale)
  applied proposals `applied_proposal_ids`, a non-empty, duplicate-free SUBSET of the plan's proposal ids. A proposal
                    id is never treated as applied unless it is listed here; `not_applied_proposal_ids` records the rest
  states            `pre_remediation_state_id` and `post_remediation_state_id`, which must differ
  authorization     `authorization_id`, and `attested_by` must be the authorized operator
  time              `attested_at_utc`, which must not be in the future and must not be older than `max_age_seconds`

Two separate questions, answered separately in `BoundAttestationAssessment`:

  claim_consistent       every binding above holds (the claim is well-formed and fits this plan, scenario, lab and
                         authorization). It says nothing about whether remediation was really applied.
  lab_state_evidenced    an independently verified, fresh lab-identity observation, made after the attestation, reports
                         the attested POST-remediation state id for the attested lab.

`ready_for_post_remediation_check` is both. With no independent verifier in this build, `lab_state_evidenced` is never
True with the default registry, so nothing is ever ready. The assessment hands the existing Phase 13A inputs
(`legacy_attestation`, `scope_identity`) to the runner; the scope policy still makes the final decision and its
behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import model_validator

from validation.contract import Identifier, Sha256, _Strict
from validation.inputs import PlanIntake, RemediationApplicationAttestation
from validation.lab_identity import (
    CLOCK_SKEW_SECONDS,
    IdentityAssessment,
    LabStateId,
    UtcTimestamp,
    parse_utc,
    require_aware,
)
from validation.scope import LabIdentityEvidence, OperatorAuthorization

ATTESTATION_CONTRACT_VERSION = "1.0"
ATTESTATION_MAX_AGE_SECONDS = 3600

CLAIM_REASON_CODES = (
    "attestation_missing",
    "attestation_malformed",
    "plan_unusable",
    "incident_mismatch",
    "scenario_mismatch",
    "plan_binding_mismatch",
    "applied_proposals_not_in_plan",
    "authorization_missing",
    "authorization_id_mismatch",
    "attester_not_authorized_operator",
    "authorization_lab_mismatch",
    "authorization_scenario_mismatch",
    "authorization_incident_mismatch",
    "attestation_stale",
    "attestation_from_future",
)
EVIDENCE_REASON_CODES = (
    "lab_identity_missing",
    "lab_identity_not_independently_verified",
    "lab_identity_stale",
    "lab_identity_lab_mismatch",
    "lab_state_is_pre_remediation",
    "lab_state_mismatch",
    "lab_observation_predates_attestation",
)


class RemediationAttestation(_Strict):
    contract_version: Literal["1.0"] = ATTESTATION_CONTRACT_VERSION
    attestation_id: Identifier
    incident_id: Identifier
    scenario_id: Identifier
    lab_id: Identifier
    plan_binding_sha256: Sha256
    applied_proposal_ids: list[Identifier]
    pre_remediation_state_id: LabStateId
    post_remediation_state_id: LabStateId
    attested_by: Identifier
    authorization_id: Identifier
    attested_at_utc: UtcTimestamp
    claim_kind: Literal["operator_claim"] = "operator_claim"
    verified_by_phase13: Literal[False] = False

    @model_validator(mode="after")
    def _consistent(self) -> "RemediationAttestation":
        applied = self.applied_proposal_ids
        if not applied or len(set(applied)) != len(applied):
            raise ValueError("an attestation names at least one applied proposal, each once")
        if self.pre_remediation_state_id == self.post_remediation_state_id:
            raise ValueError("the pre- and post-remediation lab states must be different states")
        return self


def plan_binding_sha256(intake: PlanIntake) -> str | None:
    """A hash of the plan's incident, finding and proposals (ids and their grounding references).

    None when there is no usable plan. Free text from Phase 12 is not part of it (it is never copied).
    """
    if not intake.usable:
        return None
    material = {
        "incident": intake.incident_id,
        "finding_index": intake.upstream.finding_index,
        "plan_schema_version": intake.upstream.plan_schema_version,
        "proposals": sorted((p.model_dump(mode="json") for p in intake.proposals),
                            key=lambda item: item["proposal_id"]),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


class BoundAttestationAssessment(_Strict):
    contract_version: Literal["1.0"] = ATTESTATION_CONTRACT_VERSION
    claim_consistent: bool
    lab_state_evidenced: bool
    ready_for_post_remediation_check: bool
    claim_reason_codes: list[Identifier]
    evidence_reason_codes: list[Identifier]
    applied_proposal_ids: list[Identifier] = []
    not_applied_proposal_ids: list[Identifier] = []
    # Phase 13A inputs; the first only when the claim is consistent, the second when a fresh identity was supplied
    legacy_attestation: RemediationApplicationAttestation | None = None
    scope_identity: LabIdentityEvidence | None = None
    proposed_ids_treated_as_applied: Literal[False] = False
    verified_by_phase13: Literal[False] = False


def _failed(codes: list[str]) -> BoundAttestationAssessment:
    return BoundAttestationAssessment(
        claim_consistent=False, lab_state_evidenced=False, ready_for_post_remediation_check=False,
        claim_reason_codes=codes, evidence_reason_codes=[],
    )


def _claim_checks(att: RemediationAttestation, intake: PlanIntake, incident_id: str, scenario_id: str,
                  authorization: object, now: datetime, max_age_seconds: int) -> list[str]:
    codes: list[str] = []
    if att.incident_id != incident_id or att.incident_id != intake.incident_id:
        codes.append("incident_mismatch")
    if att.scenario_id != scenario_id:
        codes.append("scenario_mismatch")
    if att.plan_binding_sha256 != plan_binding_sha256(intake):
        codes.append("plan_binding_mismatch")
    if not set(att.applied_proposal_ids) <= {p.proposal_id for p in intake.proposals}:
        codes.append("applied_proposals_not_in_plan")
    if not isinstance(authorization, OperatorAuthorization):
        codes.append("authorization_missing")
    else:
        if att.authorization_id != authorization.authorization_id:
            codes.append("authorization_id_mismatch")
        if att.attested_by != authorization.operator_id:
            codes.append("attester_not_authorized_operator")
        if att.lab_id != authorization.lab_id:
            codes.append("authorization_lab_mismatch")
        if att.scenario_id != authorization.scenario_id:
            codes.append("authorization_scenario_mismatch")
        if att.incident_id != authorization.incident_id:
            codes.append("authorization_incident_mismatch")
    age = (now - parse_utc(att.attested_at_utc)).total_seconds()
    if age < -CLOCK_SKEW_SECONDS:
        codes.append("attestation_from_future")
    elif age > max_age_seconds:
        codes.append("attestation_stale")
    return codes


def _evidence_checks(att: RemediationAttestation, identity: IdentityAssessment | None) -> list[str]:
    if identity is None or identity.observation_id is None:
        return ["lab_identity_missing"]
    codes: list[str] = []
    if not identity.fresh:
        codes.append("lab_identity_stale")
    if not identity.independently_verified:
        codes.append("lab_identity_not_independently_verified")
    if identity.lab_id != att.lab_id:
        codes.append("lab_identity_lab_mismatch")
    if identity.lab_state_id == att.pre_remediation_state_id:
        codes.append("lab_state_is_pre_remediation")
    elif identity.lab_state_id != att.post_remediation_state_id:
        codes.append("lab_state_mismatch")
    if identity.observed_at_utc is not None and parse_utc(identity.observed_at_utc) < parse_utc(att.attested_at_utc):
        codes.append("lab_observation_predates_attestation")
    return codes


def assess_remediation_attestation(
    attestation: object,
    intake: PlanIntake,
    *,
    incident_id: str,
    scenario_id: str,
    authorization: object,
    identity: IdentityAssessment | None,
    now: datetime,
    max_age_seconds: int = ATTESTATION_MAX_AGE_SECONDS,
) -> BoundAttestationAssessment:
    """Check an attestation against everything it must bind to. Never raises for bad DATA; raises TypeError only
    for a naive `now`. Evidence codes are only produced once there is a well-formed attestation to compare with."""
    moment = require_aware(now)
    if attestation is None:
        return _failed(["attestation_missing"])
    if not isinstance(attestation, RemediationAttestation):
        return _failed(["attestation_malformed"])
    if not intake.usable:
        return _failed(["plan_unusable"])

    claim_codes = _claim_checks(attestation, intake, incident_id, scenario_id, authorization, moment, max_age_seconds)
    evidence_codes = _evidence_checks(attestation, identity)
    consistent = not claim_codes
    evidenced = not evidence_codes
    proposals = [p.proposal_id for p in intake.proposals]
    applied = list(attestation.applied_proposal_ids)
    return BoundAttestationAssessment(
        claim_consistent=consistent,
        lab_state_evidenced=evidenced,
        ready_for_post_remediation_check=consistent and evidenced,
        claim_reason_codes=claim_codes,
        evidence_reason_codes=evidence_codes,
        applied_proposal_ids=applied if consistent else [],
        not_applied_proposal_ids=[pid for pid in proposals if pid not in set(applied)] if consistent else [],
        legacy_attestation=RemediationApplicationAttestation(
            attestation_id=attestation.attestation_id, attested_by=attestation.attested_by,
            applied_proposal_ids=applied,
            post_remediation_lab_state_id=attestation.post_remediation_state_id,
            pre_remediation_lab_state_id=attestation.pre_remediation_state_id,
        ) if consistent else None,
        scope_identity=identity.scope_identity if identity is not None else None,
    )
