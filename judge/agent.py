"""
Deterministic Phase 14 Judge Agent.

The agent reads a `ValidationReport` (or a saved JSON object that parses as one), rebuilds the Phase 14 handoff,
and decides whether a named claim is justified by that evidence. Oracle status is an input to check, not a
decision to copy. A caller-supplied confidence number is never an input to the decision.

No network, model, Docker, or live probe is used.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable

from pydantic import ValidationError

from judge.contract import (
    CLAIM_APPLICATION_SECURE,
    CLAIM_NOT_ASSESSED,
    CLAIM_OBJECTIVE_NOT_OBSERVED,
    CLAIM_OBJECTIVE_OBSERVED,
    CLAIM_REMEDIATION_APPLIED,
    CLAIM_TEXT,
    EXECUTION_MODES,
    EvidenceRecord,
    JudgeAssessment,
    redact_text,
)
from validation.contract import ValidationReport
from validation.handoff import build_phase14_handoff
from validation.oracle import apply_rules
from validation.scenario import ScenarioRegistry, default_registry

Clock = Callable[[], datetime]

_INCIDENT_ID = re.compile(r"incident_[0-9]{3}")
_SENSITIVE_KEY = re.compile(
    r"(?i)authorization|cookie|password|passwd|secret|token|api[_-]?key|session|credential|bearer"
)

_BASE_LIMITATIONS = (
    "This decision applies only to the named claim, scenario, lab state and conditions. "
    "It is not a universal security guarantee and it does not mean residual risk is zero.",
    "A result that the objective was not observed applies only to the tested conditions. "
    "It is not proof that the application is secure or that the objective is impossible.",
    "A proposed remediation is not an applied remediation. Phase 13 does not apply remediation "
    "and does not verify an attestation.",
    "No numerical confidence score is computed. A caller-supplied confidence value cannot change this decision.",
)
_PROVENANCE_LIMITATION = {
    "mocked": "The evidence provenance is mocked (a fake-transport simulation). This is not a live security finding.",
    "dry_run": "No check was executed. This is a dry run, not a live security finding.",
    "unverified": "The recorded check did not pass the Phase 13 evidence gate. It is not labelled as live evidence.",
    "live": (
        "The Phase 13 evidence gate accepted this report as post-remediation evidence. "
        "The conclusion is still limited to the tested conditions."
    ),
}
_NOT_VALIDATED = {
    "remediation_plan_missing": "No remediation plan was available.",
    "remediation_plan_malformed": "The remediation plan was malformed.",
    "remediation_plan_rejected": "The remediation plan or grounded answer was rejected and was not executed.",
    "remediation_plan_unavailable": "The remediation plan was unavailable. A provider error is not a validation result.",
    "remediation_plan_insufficient_context": "The planner did not have enough context to produce a plan.",
    "incident_mismatch": "The plan does not belong to this incident.",
    "scenario_unknown": "The scenario is not registered.",
    "scenario_not_eligible": "The scenario is not eligible for execution.",
    "remediation_application_not_attested": "Remediation application was not attested. The proposals were not executed.",
}
_ROLE_EXCLUSION = {
    "hypothesis": "hypothesis_is_not_observed_evidence",
    "configuration": "configuration_is_not_observed_evidence",
    "reference": "reference_is_not_observed_evidence",
    "proposal": "proposal_is_not_execution",
    "observed_incident_evidence": "incident_evidence_is_not_a_validation_observation",
    "validation_observation": "observation_not_used_for_this_claim",
    "oracle_condition": "condition_not_used_for_this_claim",
}


class _Missing:
    pass


_MISSING = _Missing()


class JudgeInputError(ValueError):
    """The caller passed a value this agent does not accept (not a malformed report)."""


class JudgeAgent:
    """Evaluate one validation report offline.

    `registry` is the scenario registry used to re-apply oracle rules. The default registry holds the
    repository's unapproved scenario. A report whose definition hash is not in the registry is inconclusive:
    the agent will not confirm a rule it cannot load. Tests that build an approved fixture pass that same
    registry here. The agent never treats a fixture registry as a human approval.
    """

    def __init__(self, registry: ScenarioRegistry | None = None, clock: Clock | None = None) -> None:
        if registry is not None and not isinstance(registry, ScenarioRegistry):
            raise JudgeInputError("registry must be a ScenarioRegistry")
        self.registry = registry if registry is not None else default_registry()
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def evaluate(
        self,
        report: ValidationReport,
        *,
        claim_id: str | None = None,
        cited_evidence_ids: list[str] | None = None,
        caller_confidence: object = _MISSING,
        execution_mode: str | None = None,
    ) -> JudgeAssessment:
        if not isinstance(report, ValidationReport):
            raise JudgeInputError("report must be a ValidationReport")
        return self._from_report(
            report,
            claim_id=claim_id,
            cited_evidence_ids=cited_evidence_ids,
            caller_confidence=caller_confidence,
            execution_mode=execution_mode,
        )

    def evaluate_payload(
        self,
        payload: object,
        *,
        claim_id: str | None = None,
        cited_evidence_ids: list[str] | None = None,
        caller_confidence: object = _MISSING,
    ) -> JudgeAssessment:
        """Assess a JSON object. A missing or invalid report becomes `not_evaluated`, not an exception."""
        rejected = _confidence_rejection(caller_confidence)
        if rejected is not None:
            return self._unevaluated(
                reason_codes=["confidence_rejected"],
                rationale=(
                    "The supplied confidence value is not a finite number from 0 to 1. "
                    "It was rejected and was not used. No assessment was made from it."
                ),
                missing=["A usable confidence value was not supplied, and confidence cannot replace evidence."],
                incident_id=_peek_incident(payload),
                claim_id=_known_claim(claim_id),
                confidence_rejected=True,
            )
        if payload is None:
            return self._unevaluated(
                reason_codes=["report_missing"],
                rationale="No validation report was supplied, so no claim was assessed.",
                missing=["A Phase 13 validation report is required."],
                claim_id=_known_claim(claim_id),
                confidence_ignored=caller_confidence is not _MISSING,
            )
        if not isinstance(payload, dict):
            return self._unevaluated(
                reason_codes=["report_invalid"],
                rationale="The validation report was not a JSON object, so no claim was assessed.",
                missing=["The validation report must be a JSON object matching the Phase 13 schema."],
                claim_id=_known_claim(claim_id),
                confidence_ignored=caller_confidence is not _MISSING,
            )
        data = dict(payload)
        mode = data.pop("execution_mode", None)
        if mode is not None and mode not in EXECUTION_MODES:
            return self._unevaluated(
                reason_codes=["report_invalid"],
                rationale="The report's execution_mode is not a known Phase 13 mode, so the report was not assessed.",
                missing=["execution_mode must be a known Phase 13 wrapper value."],
                incident_id=_peek_incident(payload),
                claim_id=_known_claim(claim_id),
                confidence_ignored=caller_confidence is not _MISSING,
            )
        try:
            report = ValidationReport.model_validate(data)
        except ValidationError:
            return self._unevaluated(
                reason_codes=["report_invalid"],
                rationale="The validation report did not match the Phase 13 schema, so no claim was assessed.",
                missing=["The validation report must match the Phase 13 schema. Fields were not coerced."],
                incident_id=_peek_incident(payload),
                claim_id=_known_claim(claim_id),
                confidence_ignored=caller_confidence is not _MISSING,
            )
        return self._from_report(
            report,
            claim_id=claim_id,
            cited_evidence_ids=cited_evidence_ids,
            caller_confidence=caller_confidence,
            execution_mode=mode if isinstance(mode, str) else None,
        )

    def _from_report(
        self,
        report: ValidationReport,
        *,
        claim_id: str | None,
        cited_evidence_ids: list[str] | None,
        caller_confidence: object,
        execution_mode: str | None,
    ) -> JudgeAssessment:
        rejected = _confidence_rejection(caller_confidence)
        ignored = caller_confidence is not _MISSING and rejected is None
        if rejected is not None:
            return self._unevaluated(
                reason_codes=["confidence_rejected"],
                rationale=(
                    "The supplied confidence value is not a finite number from 0 to 1. "
                    "It was rejected and was not used. No assessment was made from it."
                ),
                missing=["A usable confidence value was not supplied, and confidence cannot replace evidence."],
                incident_id=report.incident_id,
                claim_id=_known_claim(claim_id),
                confidence_rejected=True,
                report=report,
                execution_mode=execution_mode,
            )
        if claim_id is not None and claim_id not in CLAIM_TEXT:
            return self._unevaluated(
                reason_codes=["claim_unknown"],
                rationale="The requested claim is not in the judge vocabulary, so nothing was assessed.",
                missing=["The claim id is not one of the documented judge claims."],
                incident_id=report.incident_id,
                claim_id=CLAIM_NOT_ASSESSED,
                confidence_ignored=ignored,
                report=report,
                execution_mode=execution_mode,
            )
        cited = _cited(cited_evidence_ids)
        if cited is None:
            return self._unevaluated(
                reason_codes=["report_invalid"],
                rationale="The cited evidence ids were not a list of strings, so nothing was assessed.",
                missing=["Cited evidence ids must be a list of strings."],
                incident_id=report.incident_id,
                claim_id=_known_claim(claim_id),
                confidence_ignored=ignored,
                report=report,
                execution_mode=execution_mode,
            )
        return self._decide(
            report,
            claim_id=claim_id,
            cited=cited,
            confidence_ignored=ignored,
            execution_mode=execution_mode,
        )

    def _decide(
        self,
        report: ValidationReport,
        *,
        claim_id: str | None,
        cited: list[str],
        confidence_ignored: bool,
        execution_mode: str | None,
    ) -> JudgeAssessment:
        handoff = build_phase14_handoff(report)
        provenance = _provenance(report, handoff)
        rows, collision = _catalog_rows(report)
        common = dict(
            report=report,
            handoff=handoff,
            provenance=provenance,
            execution_mode=execution_mode,
            confidence_ignored=confidence_ignored,
            claim_id=claim_id or CLAIM_NOT_ASSESSED,
        )
        if execution_mode is not None and _mode_conflict(report, execution_mode):
            return self._finish(
                **common,
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    "The wrapper execution_mode disagrees with the report's execution record. "
                    "The result is inconclusive and was not treated as live evidence."
                ),
                reason_codes=["execution_mode_conflict"],
                rows=rows,
                missing=["The execution mode and the execution record describe different runs."],
            )
        if collision:
            return self._finish(
                **common,
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    "An evidence identifier is ambiguous or duplicated, so it cannot support a claim. "
                    "The result is inconclusive."
                ),
                reason_codes=["evidence_id_ambiguous"],
                rows=[],
                missing=["An evidence id is duplicated or used in more than one role."],
            )
        if _observations_conflict(report):
            return self._finish(
                **common,
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    "Observations of the same role disagree. The disagreement is recorded and neither "
                    "observation is chosen as the conclusion."
                ),
                reason_codes=["observations_contradict"],
                rows=rows,
                contradicting=_observation_ids(report),
                missing=["Observations of the same role record different HTTP statuses."],
            )
        dangling = [item for item in cited if item not in {row[0] for row in rows}]
        if dangling:
            shown = ", ".join(dangling[:5])
            return self._finish(
                **common,
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    "A cited evidence id is not in the validation report. Dangling ids are not treated as "
                    f"evidence. Missing ids: {shown}."
                ),
                reason_codes=["dangling_evidence"],
                rows=rows,
                missing=[f"Cited evidence id {redact_text(item)} is not in the validation report." for item in dangling],
            )
        if report.status == "blocked":
            unmet = [item.precondition for item in report.preconditions if not item.satisfied]
            return self._finish(
                **common,
                decision="blocked",
                sufficiency="not_applicable",
                rationale=(
                    f"Execution was blocked ({report.reason_code}). Unmet preconditions: {', '.join(unmet) or 'none'}. "
                    "A blocked test is not a successful remediation and does not show that the objective changed."
                ),
                reason_codes=["blocked_execution"],
                rows=rows,
                missing=[f"Unmet precondition: {name}." for name in unmet],
            )
        if report.status == "not_validated":
            sentence = _NOT_VALIDATED.get(report.reason_code, "Validation was not performed.")
            plan = report.upstream.plan_result_status
            plan_sentence = f" Upstream plan status: {plan}." if plan else ""
            codes = ["validation_not_performed"]
            if report.reason_code == "remediation_application_not_attested":
                codes.append("proposal_not_executed")
            return self._finish(
                **common,
                decision="not_evaluated",
                sufficiency="not_applicable",
                rationale=sentence + plan_sentence + " Proposed remediation was not executed, and no check was assessed.",
                reason_codes=codes,
                rows=rows,
                missing=[sentence],
            )
        if report.status == "inconclusive":
            return self._finish(
                **common,
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    f"The check did not produce a conclusion ({report.reason_code}). "
                    "An inconclusive validation is not evidence that the objective was achieved or removed."
                ),
                reason_codes=["upstream_inconclusive"],
                rows=rows,
                missing=[f"The validation outcome was inconclusive ({report.reason_code})."],
            )
        rule, rule_code = self._confirm(report)
        if rule is None:
            return self._finish(
                **dict(common, claim_id=claim_id or _default_claim(report.status)),
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    "The recorded outcome could not be confirmed from the scenario definition and the condition "
                    "results. The upstream status was not copied through as a decision."
                ),
                reason_codes=[rule_code],
                rows=rows,
                missing=["The scenario definition or the oracle rules do not confirm the recorded outcome."],
            )
        if provenance not in ("mocked", "live"):
            return self._finish(
                **dict(common, claim_id=claim_id or _default_claim(report.status)),
                decision="inconclusive",
                sufficiency="insufficient",
                rationale=(
                    "The condition results are consistent with the recorded outcome, but the Phase 13 evidence "
                    "gate did not accept the report. It is not labelled as live evidence and the claim is inconclusive."
                ),
                reason_codes=["evidence_gate_unsatisfied", "rules_confirmed"],
                rows=rows,
                missing=list(handoff.not_usable_reasons) or ["The report is not usable as post-remediation evidence."],
            )
        chosen = claim_id or _default_claim(report.status)
        return self._judge_claim(common, report, rows, rule, chosen)

    def _judge_claim(self, common: dict, report: ValidationReport, rows, rule, chosen: str) -> JudgeAssessment:
        observations = _observation_ids(report)
        rule_ids = observations + list(rule.requires_met) + list(rule.requires_not_met)
        extra = []
        if report.lab_state.tested_state == "pre_remediation":
            extra.append("The tested state is pre-remediation. This is not evidence that a remediation worked.")
        if chosen == CLAIM_APPLICATION_SECURE:
            return self._finish(
                **dict(common, claim_id=chosen),
                decision="not_supported",
                sufficiency="sufficient",
                rationale=(
                    "The claim that the application is secure, or that a vulnerability has been eliminated, "
                    "is not supported. The checked evidence covers one objective under the tested conditions only."
                ),
                reason_codes=["security_claim_not_supported"],
                rows=rows,
                contradicting=observations,
                extra_limitations=extra,
            )
        if chosen == CLAIM_REMEDIATION_APPLIED:
            attested = report.remediation.application.state == "independently_attested"
            proposal_ids = [proposal.proposal_id for proposal in report.remediation.proposals]
            if attested:
                return self._finish(
                    **dict(common, claim_id=chosen),
                    decision="inconclusive",
                    sufficiency="insufficient",
                    rationale=(
                        "An operator attestation says remediation was applied. Phase 13 did not verify that "
                        "application. The proposals stay proposals, and the claim is inconclusive."
                    ),
                    reason_codes=["attestation_not_verified"],
                    rows=rows,
                    exclusion_overrides={proposal_id: "attestation_is_not_verification" for proposal_id in proposal_ids},
                    missing=["Phase 13 records application_verified_by_phase13 as false."],
                    extra_limitations=extra,
                )
            return self._finish(
                **dict(common, claim_id=chosen),
                decision="not_supported",
                sufficiency="sufficient",
                rationale=(
                    "The remediation records are proposals. They were not attested as applied, and Phase 13 "
                    "does not apply them. The claim that remediation was applied is not supported."
                ),
                reason_codes=["proposal_not_executed"],
                rows=rows,
                contradicting=proposal_ids,
                extra_limitations=extra,
            )
        confirmed = rule.status
        supports = chosen == CLAIM_OBJECTIVE_OBSERVED and confirmed == "objective_achieved"
        supports = supports or (chosen == CLAIM_OBJECTIVE_NOT_OBSERVED and confirmed == "objective_not_observed")
        if supports:
            narrow = (
                " The supported claim is only that the objective was not observed under the tested conditions."
                if chosen == CLAIM_OBJECTIVE_NOT_OBSERVED
                else " The supported claim is only that the objective was observed under the tested conditions."
            )
            return self._finish(
                **dict(common, claim_id=chosen),
                decision="supported",
                sufficiency="sufficient",
                rationale=(
                    f"The scenario rules and the recorded condition results agree on {confirmed}.{narrow} "
                    "This is not a statement that the application is secure."
                ),
                reason_codes=["rules_confirmed"],
                rows=rows,
                supporting=rule_ids,
                extra_limitations=extra,
            )
        opposite = "objective_achieved" if confirmed == "objective_not_observed" else "objective_not_observed"
        return self._finish(
            **dict(common, claim_id=chosen),
            decision="not_supported",
            sufficiency="sufficient",
            rationale=(
                f"The evidence supports {confirmed}, so the claim of {opposite} is not supported. "
                "That conclusion is limited to the tested conditions and is not a security guarantee."
            ),
            reason_codes=["rules_confirmed", "claim_not_supported_by_evidence"],
            rows=rows,
            contradicting=rule_ids,
            extra_limitations=extra,
        )

    def _confirm(self, report: ValidationReport):
        scenario = report.scenario
        if (
            not scenario.registered
            or not scenario.definition_sha256
            or report.oracle is None
            or report.objective is None
        ):
            return None, "definition_unconfirmed"
        definition = self.registry.get(scenario.scenario_id)
        if definition is None or definition.definition_sha256() != scenario.definition_sha256:
            return None, "definition_unconfirmed"
        expected = [condition.condition_id for condition in definition.oracle.conditions]
        actual = [condition.condition_id for condition in report.oracle.conditions]
        if actual != expected or any(condition.result == "not_evaluated" for condition in report.oracle.conditions):
            return None, "definition_unconfirmed"
        results = {condition.condition_id: condition.result for condition in report.oracle.conditions}
        rule = apply_rules(definition.oracle, results)
        if rule is None or rule.status != report.status or rule.reason_code != report.reason_code:
            return None, "rules_conflict"
        return rule, "rules_confirmed"

    def _finish(
        self,
        *,
        report: ValidationReport,
        handoff,
        provenance: str,
        execution_mode: str | None,
        confidence_ignored: bool,
        claim_id: str,
        decision: str,
        sufficiency: str,
        rationale: str,
        reason_codes: list[str],
        rows: list[tuple[str, str, str]],
        supporting: list[str] | None = None,
        contradicting: list[str] | None = None,
        missing: list[str] | None = None,
        exclusion_overrides: dict[str, str] | None = None,
        extra_limitations: list[str] | None = None,
    ) -> JudgeAssessment:
        codes = list(reason_codes)
        if provenance == "mocked" and "mocked_simulation" not in codes:
            codes.append("mocked_simulation")
        records = _assign(rows, supporting or [], contradicting or [], exclusion_overrides or {})
        limitations = [*_BASE_LIMITATIONS, _PROVENANCE_LIMITATION[provenance], *_sensitive_notes(report)]
        limitations.extend(extra_limitations or [])
        usable = bool(handoff.usable_as_post_remediation_evidence)
        eligible = provenance == "live" and decision in ("supported", "not_supported") and sufficiency == "sufficient"
        return JudgeAssessment(
            assessed_at_utc=_stamp(self.clock),
            incident_id=report.incident_id,
            scenario_id=report.scenario.scenario_id,
            validation_run_id=report.validation_run_id,
            claim_id=claim_id,
            claim=CLAIM_TEXT[claim_id],
            decision=decision,
            rationale=redact_text(rationale),
            reason_codes=codes,
            supporting_evidence_ids=[item.evidence_id for item in records if item.used_as == "supporting"],
            contradicting_evidence_ids=[item.evidence_id for item in records if item.used_as == "contradicting"],
            missing_evidence=[redact_text(item) for item in (missing or [])],
            evidence_sufficiency=sufficiency,
            evidence_index=records,
            upstream_status=report.status,
            upstream_reason_code=report.reason_code,
            upstream_plan_status=report.upstream.plan_result_status,
            handoff_version=handoff.handoff_version,
            usable_as_post_remediation_evidence=usable,
            not_usable_reasons=list(handoff.not_usable_reasons),
            transport_kind=report.execution.transport_kind,
            execution_mode=execution_mode,
            limitations=limitations,
            upstream_limitations=_upstream_limitations(report),
            eligible_for_downstream=eligible,
            evidence_provenance=provenance,
            live_evidence=provenance == "live",
            caller_confidence_ignored=confidence_ignored,
        )

    def _unevaluated(
        self,
        *,
        reason_codes: list[str],
        rationale: str,
        missing: list[str],
        claim_id: str,
        incident_id: str | None = None,
        confidence_ignored: bool = False,
        confidence_rejected: bool = False,
        report: ValidationReport | None = None,
        execution_mode: str | None = None,
    ) -> JudgeAssessment:
        handoff = build_phase14_handoff(report) if report is not None else None
        provenance = _provenance(report, handoff) if report is not None and handoff is not None else "unverified"
        limitations = [*_BASE_LIMITATIONS, _PROVENANCE_LIMITATION[provenance]]
        if report is not None:
            limitations.extend(_sensitive_notes(report))
        return JudgeAssessment(
            assessed_at_utc=_stamp(self.clock),
            incident_id=incident_id,
            scenario_id=report.scenario.scenario_id if report is not None else None,
            validation_run_id=report.validation_run_id if report is not None else None,
            claim_id=claim_id,
            claim=CLAIM_TEXT[claim_id],
            decision="not_evaluated",
            rationale=redact_text(rationale),
            reason_codes=reason_codes,
            missing_evidence=[redact_text(item) for item in missing],
            evidence_sufficiency="not_applicable",
            upstream_status=report.status if report is not None else None,
            upstream_reason_code=report.reason_code if report is not None else None,
            upstream_plan_status=report.upstream.plan_result_status if report is not None else None,
            handoff_version=handoff.handoff_version if handoff is not None else None,
            usable_as_post_remediation_evidence=bool(handoff and handoff.usable_as_post_remediation_evidence),
            not_usable_reasons=list(handoff.not_usable_reasons) if handoff is not None else [],
            transport_kind=report.execution.transport_kind if report is not None else None,
            execution_mode=execution_mode,
            limitations=limitations,
            upstream_limitations=_upstream_limitations(report) if report is not None else [],
            eligible_for_downstream=False,
            evidence_provenance=provenance,
            live_evidence=provenance == "live",
            caller_confidence_ignored=confidence_ignored,
            caller_confidence_rejected=confidence_rejected,
        )


def evaluate_payload(payload: object, **kwargs) -> JudgeAssessment:
    """Assess a saved report object with the default registry and the current clock."""
    return JudgeAgent().evaluate_payload(payload, **kwargs)


def _stamp(clock: Clock) -> str:
    now = clock()
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise JudgeInputError("clock must return a timezone-aware datetime")
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _confidence_rejection(value: object) -> str | None:
    if value is _MISSING:
        return None
    if type(value) not in (int, float) or isinstance(value, bool):
        return "not a number"
    if value != value or value in (float("inf"), float("-inf")):  # NaN or infinity
        return "not finite"
    if value < 0 or value > 1:
        return "out of range"
    return None


def _known_claim(claim_id: str | None) -> str:
    if claim_id in CLAIM_TEXT:
        return claim_id
    return CLAIM_NOT_ASSESSED


def _default_claim(status: str) -> str:
    if status == "objective_achieved":
        return CLAIM_OBJECTIVE_OBSERVED
    if status == "objective_not_observed":
        return CLAIM_OBJECTIVE_NOT_OBSERVED
    return CLAIM_NOT_ASSESSED


def _peek_incident(payload: object) -> str | None:
    if isinstance(payload, dict):
        value = payload.get("incident_id")
        if isinstance(value, str) and _INCIDENT_ID.fullmatch(value):
            return value
    return None


def _cited(value: list[str] | None) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    if any(type(item) is not str or item == "" for item in value):
        return None
    return value


def _provenance(report: ValidationReport, handoff) -> str:
    if handoff.usable_as_post_remediation_evidence and report.execution.transport_kind == "live":
        return "live"
    if report.execution.transport_kind == "fake":
        return "mocked"
    if report.execution.transport_kind == "none" or not report.execution.attempted:
        return "dry_run"
    return "unverified"


def _mode_conflict(report: ValidationReport, mode: str) -> bool:
    if mode == "dry-run":
        return report.execution.attempted or report.execution.transport_kind != "none"
    if mode == "live":
        return report.execution.transport_kind != "live"
    if mode.startswith("live-refused"):
        return report.execution.attempted
    return False


def _observation_ids(report: ValidationReport) -> list[str]:
    return [item.observation_id for item in report.evidence]


def _observations_conflict(report: ValidationReport) -> bool:
    seen: dict[str, str] = {}
    for item in report.evidence:
        role = item.attributes.get("role")
        status = item.attributes.get("http_status")
        if not role or status is None:
            continue
        prior = seen.get(role)
        if prior is not None and prior != status:
            return True
        seen[role] = status
    return False


def _catalog_rows(report: ValidationReport) -> tuple[list[tuple[str, str, str]], str | None]:
    rows: list[tuple[str, str, str]] = []
    seen: dict[str, str] = {}

    def add(evidence_id: str, role: str, source: str) -> str | None:
        prior = seen.get(evidence_id)
        if prior is None:
            seen[evidence_id] = role
            rows.append((evidence_id, role, source))
            return None
        if prior != role:
            return "evidence_id_ambiguous"
        return None

    observation_ids = _observation_ids(report)
    if len(observation_ids) != len(set(observation_ids)):
        return [], "evidence_id_ambiguous"
    for evidence_id in observation_ids:
        problem = add(evidence_id, "validation_observation", "validation_report.evidence")
        if problem:
            return [], problem
    if report.oracle is not None:
        for condition in report.oracle.conditions:
            problem = add(condition.condition_id, "oracle_condition", "validation_report.oracle.conditions")
            if problem:
                return [], problem
    for proposal in report.remediation.proposals:
        problem = add(proposal.proposal_id, "proposal", "validation_report.remediation.proposals")
        if problem:
            return [], problem
        groups = (
            (proposal.observed_evidence_ids, "observed_incident_evidence", "phase12.observed_evidence"),
            (proposal.hypothesis_ids, "hypothesis", "phase12.hypothesis"),
            (proposal.configuration_fact_ids, "configuration", "phase12.configuration"),
            (proposal.reference_ids, "reference", "phase12.reference"),
            (proposal.reference_chunk_ids, "reference", "phase12.reference_chunk"),
        )
        for identifiers, role, source in groups:
            for evidence_id in identifiers:
                problem = add(evidence_id, role, source)
                if problem:
                    return [], problem
    return rows, None


def _assign(
    rows: list[tuple[str, str, str]],
    supporting: list[str],
    contradicting: list[str],
    exclusion_overrides: dict[str, str],
) -> list[EvidenceRecord]:
    support = set(supporting)
    contra = set(contradicting)
    records: list[EvidenceRecord] = []
    for evidence_id, role, source in rows:
        if evidence_id in support and evidence_id not in contra:
            records.append(EvidenceRecord(evidence_id=evidence_id, role=role, source=source, used_as="supporting"))
        elif evidence_id in contra and evidence_id not in support:
            records.append(EvidenceRecord(evidence_id=evidence_id, role=role, source=source, used_as="contradicting"))
        elif role == "oracle_condition":
            records.append(EvidenceRecord(evidence_id=evidence_id, role=role, source=source, used_as="context"))
        else:
            reason = exclusion_overrides.get(evidence_id, _ROLE_EXCLUSION[role])
            records.append(
                EvidenceRecord(
                    evidence_id=evidence_id, role=role, source=source, used_as="excluded", exclusion_reason=reason
                )
            )
    return records


def _sensitive_notes(report: ValidationReport) -> list[str]:
    notes = []
    for item in report.evidence:
        if any(_SENSITIVE_KEY.search(key) for key in item.attributes):
            notes.append(
                f"Observation {item.observation_id} included a sensitive attribute. "
                "The attribute value was not copied into this decision."
            )
    return notes


def _upstream_limitations(report: ValidationReport) -> list[str]:
    cleaned = []
    for item in report.limitations:
        text = redact_text(item).strip()
        if text:
            cleaned.append(text)
    return cleaned
