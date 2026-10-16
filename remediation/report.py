"""
Phase 12 - an incident-level remediation report that records what was actually done.

    analysis/*.json  ->  Phase 11D (FTS5 v1 retrieval, grounded context, validated answer)
                     ->  planner input (relevance filter, configuration facts, sufficiency gate)
                     ->  one remediation plan per usable finding

`run_incident_remediation(incident_dir, store, llm_client)` reuses `rag.report.run_incident_rag` for the
whole first half (nothing is duplicated) and then plans once per context. The same `LLMClient` serves both
steps, so with a live provider each finding costs two calls: one grounded answer, one plan. A caller that
already holds an `IncidentRagReport` can pass it as `rag_report` to avoid repeating the first half.

Without an `llm_client` nothing is planned (`results` is empty), exactly like Phase 11D without a model.

The report carries no timestamps and no secrets, so the same inputs and model responses give the same
report. It is advisory: `execution_performed` is always false.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge.store import KnowledgeStore
from llm.base import LLMClient
from rag.context import PRODUCTION_RETRIEVER_NAME, ContextLimits
from rag.report import IncidentRagReport, run_incident_rag
from remediation.configuration import ConfigurationFactInput, build_configuration_context
from remediation.contract import PLAN_CAVEATS
from remediation.inputs import PlannerInputError, build_planner_input
from remediation.planner import RemediationPlanResult, plan_remediation, redact_credentials, scrub_message
from retrieval.contract import DEFAULT_LIMIT, Retriever
from retrieval.query_builder import load_analysis_artifacts

REMEDIATION_REPORT_SCHEMA_VERSION = "1.0"

FAILED_PLAN_STATUSES = frozenset({"llm_error", "rejected_invalid_response", "rejected_grounding"})

# Phase 11D statuses where the model DID return text that was then refused; only these keep the text.
REJECTED_ANSWER_STATUSES = frozenset({"rejected_citations", "rejected_invalid_response"})
REJECTED_ANSWER_WARNING = (
    "rejected_answer is the model's unvalidated output, kept for diagnosis only. It is NOT accepted "
    "evidence and must not be treated as an answer. It may contain incident details such as request "
    "strings and paths, so keep this report local."
)


@dataclass(frozen=True)
class UpstreamAnswerDiagnostic:
    """
    Why a Phase 11D answer was not accepted, kept so a failure upstream of the planner is not invisible.

    `message` is scrubbed (key-like strings, authorization headers and the provider's `failed_generation`
    body removed, then truncated); `issue_codes` are validation codes only, never model text.
    `planner_model_called` is False when the planner stopped at its sufficiency gate, which is the normal
    case after an upstream failure: the planner's own model was not called.

    `rejected_answer` is the model's own returned text (credentials redacted, nothing else changed), kept
    ONLY when a response was actually returned but rejected (`rejected_citations` or
    `rejected_invalid_response`), so the offending ids can be inspected. It is None for a successful
    answer, for an upstream `llm_error` (there is no answer; a provider error body such as
    `failed_generation` is never treated as one) and when nothing was generated. It can contain
    incident details (request strings, paths): see `REJECTED_ANSWER_WARNING`.
    """

    finding_index: int
    status: str
    message: str
    issue_codes: tuple[str, ...]
    llm: str | None
    planner_model_called: bool
    rejected_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "finding_index": self.finding_index,
            "upstream_phase": "11D",
            "status": self.status,
            "message": self.message,
            "issue_codes": list(self.issue_codes),
            "llm": self.llm,
            "planner_model_called": self.planner_model_called,
            "rejected_answer": self.rejected_answer,
        }
        if self.rejected_answer is not None:
            data["rejected_answer_warning"] = REJECTED_ANSWER_WARNING
        return data


@dataclass(frozen=True)
class IncidentRemediationReport:
    incident_id: str
    retrievers: tuple[str, ...]
    grounded_answer_statuses: tuple[tuple[int, str], ...]   # (finding_index, Phase 11D answer status)
    results: tuple[RemediationPlanResult, ...]               # empty when no LLM client was supplied
    skipped_findings: tuple[tuple[int, str], ...]
    planning_requested: bool
    grounded_answer_diagnostics: tuple[UpstreamAnswerDiagnostic, ...] = ()   # one per Phase 11D answer not accepted

    @property
    def uses_only_production_retriever(self) -> bool:
        return bool(self.retrievers) and all(name == PRODUCTION_RETRIEVER_NAME for name in self.retrievers)

    @property
    def any_failure(self) -> bool:
        """A model call failed or a response (answer or plan) was rejected. Insufficient-evidence is not a failure."""
        answer_failed = any(
            status in {"llm_error", "rejected_invalid_response", "rejected_citations"}
            for _, status in self.grounded_answer_statuses
        )
        return self.planning_requested and (answer_failed or any(r.status in FAILED_PLAN_STATUSES for r in self.results))

    @property
    def nothing_planned_reason(self) -> str | None:
        """Why a requested run produced no plan at all (every finding skipped, or none present); else None."""
        if not self.planning_requested or self.results:
            return None
        if self.skipped_findings:
            count = len(self.skipped_findings)
            return (f"all {count} finding{'s were' if count != 1 else ' was'} skipped before retrieval "
                    "(no informative subject), so no answer and no plan was produced")
        return "the analysis contains no finding to plan, so no answer and no plan was produced"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REMEDIATION_REPORT_SCHEMA_VERSION,
            "incident_id": self.incident_id,
            "retrieval_methods_used": list(self.retrievers),
            "production_retriever": PRODUCTION_RETRIEVER_NAME,
            "uses_only_production_retriever": self.uses_only_production_retriever,
            "planning_requested": self.planning_requested,
            "grounded_answer_statuses": [
                {"finding_index": index, "status": status} for index, status in self.grounded_answer_statuses
            ],
            "skipped_findings": [{"finding_index": i, "reason": r} for i, r in self.skipped_findings],
            "nothing_planned_reason": self.nothing_planned_reason,
            "grounded_answer_diagnostics": [item.to_dict() for item in self.grounded_answer_diagnostics],
            "plans": [result.to_dict() for result in self.results],
            "advisory_only": True,
            "execution_performed": False,
            "ground_truth_used": False,
            "caveats": list(PLAN_CAVEATS),
        }


def run_incident_remediation(
    incident_dir: str | Path,
    store: KnowledgeStore,
    llm_client: LLMClient | None = None,
    *,
    rag_report: IncidentRagReport | None = None,
    extra_configuration: tuple[ConfigurationFactInput, ...] = (),
    retriever: Retriever | None = None,
    limits: ContextLimits | None = None,
    query_limit: int = DEFAULT_LIMIT,
) -> IncidentRemediationReport:
    """Plan once per usable finding of the incident (only when `llm_client` is given)."""
    _root_cause, reconstruction = load_analysis_artifacts(incident_dir)
    rag = rag_report or run_incident_rag(
        incident_dir, store, llm_client, retriever=retriever, limits=limits, query_limit=query_limit
    )
    if isinstance(reconstruction, dict) and reconstruction.get("incident_id") not in (None, rag.incident_id):
        raise PlannerInputError("the RAG report and the reconstruction belong to different incidents")

    results: tuple[RemediationPlanResult, ...] = ()
    if llm_client is not None:
        configuration = build_configuration_context(reconstruction, extra_configuration)
        planned = []
        for position, context in enumerate(rag.contexts):
            answer = rag.results[position] if position < len(rag.results) else None
            planned.append(plan_remediation(build_planner_input(context, answer, configuration), llm_client))
        results = tuple(planned)

    called = {result.finding_index: result.model_called for result in results}
    diagnostics = tuple(
        UpstreamAnswerDiagnostic(
            finding_index=answer.context.finding_index,
            status=answer.status,
            message=scrub_message(answer.message),
            issue_codes=tuple(issue.code for issue in answer.validation.issues) if answer.validation else (),
            llm=answer.llm,
            planner_model_called=called.get(answer.context.finding_index, False),
            rejected_answer=(
                redact_credentials(answer.raw_response)
                if answer.status in REJECTED_ANSWER_STATUSES and isinstance(answer.raw_response, str)
                else None
            ),
        )
        for answer in rag.results
        if answer.status != "answered"
    )

    return IncidentRemediationReport(
        incident_id=rag.incident_id,
        retrievers=rag.retrievers,
        grounded_answer_statuses=tuple(
            (result.context.finding_index, result.status) for result in rag.results
        ),
        results=results,
        skipped_findings=rag.skipped_findings,
        planning_requested=llm_client is not None,
        grounded_answer_diagnostics=diagnostics,
    )
