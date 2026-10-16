"""
Phase 11 (Milestone 11D) - an incident-level report that records what was actually done.

`run_incident_rag(incident_dir, store, llm_client=None)` builds the contexts (production retriever
FTS5 v1 unless the caller passes another one) and, only if an `LLMClient` is given, one grounded
answer per context. The report states the retrieval method that was really used for every
context, so a result can never be read as coming from a different retriever.

The report holds no timestamps and no secrets, so the same inputs and the same model responses
give the same report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge.store import KnowledgeStore
from llm.base import LLMClient
from rag.answer import GroundedAnswerResult, generate_grounded_answer
from rag.context import PRODUCTION_RETRIEVER_NAME, ContextLimits, GroundedContext
from rag.pipeline import build_contexts_for_incident
from retrieval.contract import DEFAULT_LIMIT, Retriever

REPORT_SCHEMA_VERSION = "1.0"

REPORT_CAVEATS = (
    "Retrieved passages are general reference knowledge from a small curated CWE/CAPEC subset; they do "
    "not confirm the incident hypothesis and an absent passage does not refute it.",
    "FTS5 v1 is keyword retrieval with no relevance threshold; it can return passages that do not fit "
    "the incident. Relevance labels in an answer are the model's own judgement.",
    "Citation validation checks that citations exist and sit in the right category; it does not prove "
    "that a statement is entailed by its sources.",
    "No retrieval-quality or answer-quality improvement is claimed or measured by this report.",
)


@dataclass(frozen=True)
class IncidentRagReport:
    incident_id: str
    retrievers: tuple[str, ...]
    contexts: tuple[GroundedContext, ...]
    results: tuple[GroundedAnswerResult, ...]      # empty when no LLM client was supplied
    skipped_findings: tuple[tuple[int, str], ...]  # (finding_index, reason)
    generation_requested: bool

    @property
    def uses_only_production_retriever(self) -> bool:
        return bool(self.retrievers) and all(name == PRODUCTION_RETRIEVER_NAME for name in self.retrievers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "incident_id": self.incident_id,
            "retrieval_methods_used": list(self.retrievers),
            "production_retriever": PRODUCTION_RETRIEVER_NAME,
            "uses_only_production_retriever": self.uses_only_production_retriever,
            "generation_requested": self.generation_requested,
            "skipped_findings": [{"finding_index": i, "reason": r} for i, r in self.skipped_findings],
            "contexts": [context.to_dict() for context in self.contexts],
            "answers": [result.to_dict() for result in self.results],
            "ground_truth_used": False,
            "caveats": list(REPORT_CAVEATS),
        }


def run_incident_rag(
    incident_dir: str | Path,
    store: KnowledgeStore,
    llm_client: LLMClient | None = None,
    *,
    retriever: Retriever | None = None,
    limits: ContextLimits | None = None,
    query_limit: int = DEFAULT_LIMIT,
    max_characters_per_passage: int | None = None,
) -> IncidentRagReport:
    """Contexts for every usable finding and, if `llm_client` is given, a grounded answer for each.

    `max_characters_per_passage` (optional) shortens only reference-passage text in the model prompt.
    """
    built = build_contexts_for_incident(
        incident_dir, store, retriever=retriever, limits=limits, query_limit=query_limit
    )
    results = (
        tuple(
            generate_grounded_answer(c, llm_client, max_characters_per_passage=max_characters_per_passage)
            for c in built.contexts
        )
        if llm_client
        else ()
    )
    return IncidentRagReport(
        incident_id=built.incident_id,
        retrievers=tuple(dict.fromkeys(c.retriever for c in built.contexts)),
        contexts=built.contexts,
        results=results,
        skipped_findings=tuple((s.finding_index, s.reason) for s in built.skipped),
        generation_requested=llm_client is not None,
    )
