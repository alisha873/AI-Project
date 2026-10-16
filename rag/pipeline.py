"""
Phase 11 (Milestone 11D) - from incident analysis to retrieved, citable context.

    analysis/root_cause_analysis.json   (Phase 9)  -+
    analysis/incident_reconstruction.json (Phase 8) -+--> build_retrieval_queries
                                                          |
                                      PRODUCTION RETRIEVER (FTS5 v1)  <- KnowledgeStore
                                                          |
                                                    build_grounded_context

Production retrieval method: `fts5-bm25-v1`
-------------------------------------------
`create_production_retriever(store)` returns `retrieval.fts5.Fts5Retriever`
(keyword search, BM25 ranking), the Milestone 11A baseline. It is the default of
every function here and nothing changes it silently:

- No reciprocal-rank fusion and no semantic (embedding) retrieval is imported by
  this package (a test checks this with an AST scan). In Milestone 11C neither
  beat the FTS5 baselines on the frozen benchmark (PROJECT_STATUS section 3.8),
  so both stay EXPERIMENTAL.
- A caller may pass a different `Retriever` explicitly. The choice is then
  visible in the output: `GroundedContext.retriever` holds its name and
  `is_production_retriever` is False.

Only the two analysis files are read (via `load_analysis_artifacts`). Queries
are built by the existing deterministic query builder, which refuses
answer-key fields; this module never opens `ground_truth/`.

This module does not call a language model. Answer generation is in `rag.answer`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from knowledge.store import KnowledgeStore
from rag.context import (
    PRODUCTION_RETRIEVER_NAME,
    ContextLimits,
    GroundedContext,
    build_grounded_context,
)
from retrieval.contract import DEFAULT_LIMIT, Retriever
from retrieval.fts5 import RETRIEVER_NAME as _FTS5_V1_NAME
from retrieval.fts5 import Fts5Retriever
from retrieval.query_builder import (
    SkippedFinding,
    build_retrieval_queries,
    load_analysis_artifacts,
)

# The production name lives in rag.context (one place); this guards against drift.
assert PRODUCTION_RETRIEVER_NAME == _FTS5_V1_NAME


@dataclass(frozen=True)
class IncidentContexts:
    """All contexts for one incident: one per usable finding, plus findings that were skipped."""

    incident_id: str
    contexts: tuple[GroundedContext, ...]
    skipped: tuple[SkippedFinding, ...]


def create_production_retriever(store: KnowledgeStore) -> Fts5Retriever:
    """The production baseline: FTS5 v1 over an open store."""
    return Fts5Retriever(store)


def build_contexts_from_analysis(
    root_cause: Mapping,
    reconstruction: Mapping | None,
    store: KnowledgeStore,
    *,
    retriever: Retriever | None = None,
    limits: ContextLimits | None = None,
    query_limit: int = DEFAULT_LIMIT,
) -> IncidentContexts:
    """
    Build and run one query per usable finding, then a `GroundedContext` for each.

    `retriever=None` means the production FTS5 v1 retriever. Raises the retrieval
    package's errors (`QueryBuildError`, `InvalidRetrievalQueryError`, ...) and
    `ContextBuildError`; nothing is swallowed.
    """
    active = retriever if retriever is not None else create_production_retriever(store)
    built = build_retrieval_queries(root_cause, reconstruction, limit=query_limit)
    contexts = []
    for incident_query in built.queries:
        result = active.retrieve(incident_query.query)
        contexts.append(
            build_grounded_context(
                incident_query,
                result,
                root_cause=root_cause,
                reconstruction=reconstruction,
                store=store,
                limits=limits,
            )
        )
    return IncidentContexts(built.incident_id, tuple(contexts), built.skipped)


def build_contexts_for_incident(
    incident_dir: str | Path,
    store: KnowledgeStore,
    *,
    retriever: Retriever | None = None,
    limits: ContextLimits | None = None,
    query_limit: int = DEFAULT_LIMIT,
) -> IncidentContexts:
    """`load_analysis_artifacts` (the only two files read) followed by `build_contexts_from_analysis`."""
    root_cause, reconstruction = load_analysis_artifacts(incident_dir)
    return build_contexts_from_analysis(
        root_cause,
        reconstruction,
        store,
        retriever=retriever,
        limits=limits,
        query_limit=query_limit,
    )
