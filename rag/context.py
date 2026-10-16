"""
Phase 11 (Milestone 11D) - the structured, citable context for grounded answers.

A `GroundedContext` is everything an answer generator may use, split into three
kinds of statements that must never be mixed up (PROJECT_SPEC, principle 2):

1. OBSERVED EVIDENCE  (`ObservedEvidence`)
   Normalized events from the incident (for example `nginx-19`) and the Phase 8
   claims that cite them. Citable by `evidence_id`.
2. HYPOTHESES         (`HypothesisItem`)
   Phase 9 findings. Their `status` is copied unchanged. Unless it is exactly
   "confirmed" a finding is an inferred hypothesis and `is_confirmed` is False.
   Citable by `hypothesis_id` (`H1`, `H2`, ...).
3. REFERENCE GUIDANCE (`ReferencePassage`)
   Passages retrieved from the security knowledge base (CWE/CAPEC). They are
   general reference knowledge. They do NOT confirm that the incident's
   hypothesis is correct and an absent passage does not refute it
   (`RESULT_MEANING`). Citable by `reference_id`, which is simply the chunk id
   (for example `cwe:CWE-89#0001`), so a citation can be followed back to the
   store without a lookup table.

What is preserved for every reference passage (nothing is paraphrased)
----------------------------------------------------------------------
document id, chunk id, ordinal, character offsets, the exact chunk text, source,
source id, source version, document type, title, source URL, licence, the
retrieval rank and score, the chunk hash, the document content hash and the
full provenance record (origin kind, artifact path and SHA-256, adapter and
version, retrieval and ingestion timestamps). Before a passage is accepted it is
compared with the stored document: `SourceDocument.contains_chunk` must hold and
the retrieved text must equal the stored text, otherwise building fails loudly.

What is NOT used
----------------
Only `analysis/root_cause_analysis.json` and `analysis/incident_reconstruction.json`
content (already loaded by the caller) and retrieval results. No `ground_truth`
data and no benchmark labels. Input mappings that contain an answer-key field
name are refused (`FORBIDDEN_INPUT_KEYS`), exactly as the query builder does.
Evidence IDs that Phase 9 cites but that the Phase 8 reconstruction does not
describe are listed in `unresolved_evidence_ids` and are NOT citable: no
description, no claim.

Bounded size
------------
`ContextLimits` keeps the prompt small (default: up to 10 documents, 1 chunk each).
Chunk text is never truncated; whole chunks or whole documents are left out and
counted in `omitted_documents` / `omitted_chunks`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from knowledge.store import KnowledgeStore, KnowledgeStoreError
from retrieval.contract import DEFAULT_LIMIT, RESULT_MEANING, RetrievalResult
from retrieval.query_builder import FORBIDDEN_INPUT_KEYS, IncidentQuery, is_placeholder

CONTEXT_SCHEMA_VERSION = "1.0"

# The one retriever Phase 11D treats as the production baseline (see `rag.pipeline`).
PRODUCTION_RETRIEVER_NAME = "fts5-bm25-v1"

# Default: show every document a default query retrieves (the retrieval limit) with its single
# best-matching chunk. Chosen for prompt size (at most ~10 chunks of <= 1200 characters), not
# tuned to any retrieval result; callers may raise or lower it.
DEFAULT_MAX_DOCUMENTS = DEFAULT_LIMIT
DEFAULT_MAX_CHUNKS_PER_DOCUMENT = 1

_REQUEST_KEYS = ("method", "path", "query", "status", "upstream", "upstream_status", "request_id")
_EVENT_TEXT_KEYS = (
    "timestamp",
    "source",
    "event_type",
    "action",
    "source_entity",
    "destination_entity",
    "stage_label",
    "interpretation_status",
)


class RagError(Exception):
    """Base class for errors raised deliberately by the grounded-answer package."""


class ContextBuildError(RagError, ValueError):
    """The incident analysis, retrieval result or stored passage is inconsistent."""


# ============================================================
# VALUE TYPES
# ============================================================

@dataclass(frozen=True)
class ContextLimits:
    max_documents: int = DEFAULT_MAX_DOCUMENTS
    max_chunks_per_document: int = DEFAULT_MAX_CHUNKS_PER_DOCUMENT

    def __post_init__(self) -> None:
        for name in ("max_documents", "max_chunks_per_document"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")


@dataclass(frozen=True)
class ObservedEvidence:
    """One normalized incident event that a Phase 9 finding cites, as Phase 8 described it."""

    evidence_id: str
    event: dict[str, str | None]            # string fields copied from the timeline entry
    request: dict[str, str | None]          # request_context fields (all as text), may be empty
    claim_ids: tuple[str, ...]
    claim_statements: tuple[str, ...]
    claim_limitations: tuple[str, ...]
    basis: str = "observed"


@dataclass(frozen=True)
class HypothesisItem:
    """A Phase 9 finding. `status` is copied verbatim; a hypothesis is not evidence."""

    hypothesis_id: str
    finding_index: int
    status: str
    is_confirmed: bool
    vulnerability_type: str | None
    root_cause: str | None
    affected_component: str | None
    rationale: str | None
    evidence_needed_to_confirm: tuple[str, ...]
    supporting_evidence_ids: tuple[str, ...]   # all IDs the finding cites
    basis: str = "inferred_hypothesis"


@dataclass(frozen=True)
class ReferencePassage:
    """One retrieved chunk with its identity and provenance, copied from the store."""

    reference_id: str          # == chunk_id
    doc_id: str
    chunk_id: str
    ordinal: int
    char_start: int
    char_end: int
    text: str
    document_rank: int
    chunk_score: float
    source: str
    source_id: str
    source_version: str
    doc_type: str
    title: str
    source_url: str | None
    license: str | None
    chunk_sha256: str
    document_content_sha256: str
    provenance: dict[str, str]
    basis: str = "general_reference_knowledge"


@dataclass(frozen=True)
class GroundedContext:
    incident_id: str
    finding_index: int
    retriever: str
    is_production_retriever: bool
    retrieval_status: str                  # "ok" | "no_results"
    possibly_incomplete: bool
    backend_query: str | None
    dropped_terms: tuple[str, ...]
    query_notes: tuple[str, ...]
    query_text: str
    query_phrases: tuple[str, ...]
    query_omitted_fields: tuple[str, ...]
    observed_evidence: tuple[ObservedEvidence, ...]
    unresolved_evidence_ids: tuple[str, ...]
    hypotheses: tuple[HypothesisItem, ...]
    references: tuple[ReferencePassage, ...]
    omitted_documents: int
    omitted_chunks: int
    meaning: str = RESULT_MEANING
    schema_version: str = CONTEXT_SCHEMA_VERSION

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.observed_evidence)

    @property
    def hypothesis_ids(self) -> tuple[str, ...]:
        return tuple(item.hypothesis_id for item in self.hypotheses)

    @property
    def reference_ids(self) -> tuple[str, ...]:
        return tuple(item.reference_id for item in self.references)

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready copy (tuples become lists when serialized)."""
        return asdict(self)


# ============================================================
# INPUT HELPERS
# ============================================================

def _mapping(value: object, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ContextBuildError(f"{where} must be an object, got {type(value).__name__}")
    return value


def _refuse_answer_keys(mapping: Mapping, where: str) -> None:
    found = sorted(FORBIDDEN_INPUT_KEYS.intersection(str(key) for key in mapping))
    if found:
        raise ContextBuildError(
            f"{where} contains evaluation-answer key(s) {found}; grounded context must never "
            "be built from ground truth or relevance labels"
        )


def _text_or_none(value: object, where: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContextBuildError(f"{where} must be a string or null, got {type(value).__name__}")
    return value.strip() or None


def _text_list(value: object, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ContextBuildError(f"{where} must be a list of strings or null")
    if any(not isinstance(item, str) for item in value):
        raise ContextBuildError(f"{where} must contain only strings")
    return tuple(item.strip() for item in value if item.strip())


def _scalar_text(value: object) -> str | None:
    """JSON scalars as text (200 -> '200'); None stays None; containers are not copied."""
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    return str(value)


# ============================================================
# OBSERVED EVIDENCE (Phase 8)
# ============================================================

def _request_fields(raw: object, where: str) -> dict[str, str | None]:
    if raw is None:
        return {}
    request = _mapping(raw, where)
    return {key: _scalar_text(request.get(key)) for key in _REQUEST_KEYS if key in request}


def _resolve_observed_evidence(
    reconstruction: Mapping | None, evidence_ids: tuple[str, ...]
) -> tuple[tuple[ObservedEvidence, ...], tuple[str, ...]]:
    """
    Describe each cited event from the Phase 8 timeline and claims.

    An ID found in neither the timeline nor a claim is `unresolved`: it has no
    description we can show, so it is not citable.
    """
    if reconstruction is None or not evidence_ids:
        return (), tuple(evidence_ids)

    timeline = reconstruction.get("timeline")
    claims = reconstruction.get("claims")
    for name, value in (("timeline", timeline), ("claims", claims)):
        if value is not None and not isinstance(value, (list, tuple)):
            raise ContextBuildError(f"reconstruction.{name} must be a list or null")

    events: dict[str, Mapping] = {}
    for index, raw in enumerate(timeline or ()):
        entry = _mapping(raw, f"reconstruction.timeline[{index}]")
        event_id = _text_or_none(entry.get("event_id"), f"reconstruction.timeline[{index}].event_id")
        if event_id and event_id not in events:
            events[event_id] = entry

    claims_by_event: dict[str, list[Mapping]] = {}
    for index, raw in enumerate(claims or ()):
        claim = _mapping(raw, f"reconstruction.claims[{index}]")
        for event_id in _text_list(claim.get("evidence_ids"), f"reconstruction.claims[{index}].evidence_ids"):
            claims_by_event.setdefault(event_id, []).append(claim)

    resolved: list[ObservedEvidence] = []
    unresolved: list[str] = []
    for evidence_id in evidence_ids:
        entry = events.get(evidence_id)
        related = claims_by_event.get(evidence_id, [])
        if entry is None and not related:
            unresolved.append(evidence_id)
            continue

        event_fields: dict[str, str | None] = {}
        request: dict[str, str | None] = {}
        if entry is not None:
            for key in _EVENT_TEXT_KEYS:
                if key in entry:
                    event_fields[key] = _scalar_text(entry.get(key))
            request = _request_fields(entry.get("request_context"), f"timeline[{evidence_id}].request_context")
        if not request and related:  # fall back to the claim's own copy of the request
            request = _request_fields(related[0].get("request_context"), f"claim[{evidence_id}].request_context")

        ordered = sorted(related, key=lambda claim: str(claim.get("claim_id")))
        claim_ids = tuple(str(c.get("claim_id")) for c in ordered if c.get("claim_id") is not None)
        statements = tuple(s for s in (_text_or_none(c.get("statement"), "claim.statement") for c in ordered) if s)
        limitations = tuple(s for s in (_text_or_none(c.get("limitation"), "claim.limitation") for c in ordered) if s)
        resolved.append(
            ObservedEvidence(
                evidence_id=evidence_id,
                event=event_fields,
                request=request,
                claim_ids=claim_ids,
                claim_statements=statements,
                claim_limitations=limitations,
            )
        )
    return tuple(resolved), tuple(unresolved)


# ============================================================
# HYPOTHESES (Phase 9)
# ============================================================

def _hypothesis_from_finding(
    finding: Mapping, finding_index: int, status: str, supporting: tuple[str, ...]
) -> HypothesisItem:
    where = f"root_cause.findings[{finding_index}]"

    def informative(key: str) -> str | None:
        text = _text_or_none(finding.get(key), f"{where}.{key}")
        return None if is_placeholder(text) else text

    return HypothesisItem(
        hypothesis_id=f"H{finding_index + 1}",
        finding_index=finding_index,
        status=status,
        is_confirmed=status == "confirmed",
        vulnerability_type=informative("vulnerability_type"),
        root_cause=informative("root_cause"),
        affected_component=informative("affected_component"),
        rationale=_text_or_none(finding.get("rationale"), f"{where}.rationale"),
        evidence_needed_to_confirm=_text_list(
            finding.get("evidence_needed_to_confirm"), f"{where}.evidence_needed_to_confirm"
        ),
        supporting_evidence_ids=supporting,
    )


# ============================================================
# REFERENCE PASSAGES (retrieval)
# ============================================================

def _reference_passages(
    result: RetrievalResult, store: KnowledgeStore, limits: ContextLimits
) -> tuple[tuple[ReferencePassage, ...], int, int]:
    """Verified passages for the top documents, plus how many documents/chunks were left out."""
    passages: list[ReferencePassage] = []
    omitted_chunks = 0
    kept_items = result.items[: limits.max_documents]
    omitted_documents = len(result.items) - len(kept_items)
    for item in result.items[limits.max_documents:]:
        omitted_chunks += len(item.chunks)

    for item in kept_items:
        try:
            document = store.get_document(item.doc_id)
            stored_chunks = {chunk.chunk_id: chunk for chunk in store.get_chunks(item.doc_id)}
        except KnowledgeStoreError as exc:
            raise ContextBuildError(f"cannot read {item.doc_id!r} from the knowledge store: {exc}") from exc

        chosen = item.chunks[: limits.max_chunks_per_document]
        omitted_chunks += len(item.chunks) - len(chosen)
        for match in chosen:
            stored = stored_chunks.get(match.chunk_id)
            if stored is None or not document.contains_chunk(stored) or stored.text != match.text:
                raise ContextBuildError(
                    f"retrieved chunk {match.chunk_id!r} does not match the stored document; "
                    "refusing to use unverifiable reference text"
                )
            passages.append(
                ReferencePassage(
                    reference_id=match.chunk_id,
                    doc_id=item.doc_id,
                    chunk_id=match.chunk_id,
                    ordinal=match.ordinal,
                    char_start=match.char_start,
                    char_end=match.char_end,
                    text=match.text,
                    document_rank=item.rank,
                    chunk_score=match.score,
                    source=item.source,
                    source_id=item.source_id,
                    source_version=document.source_version,
                    doc_type=item.doc_type,
                    title=item.title,
                    source_url=item.source_url,
                    license=document.license,
                    chunk_sha256=stored.chunk_sha256,
                    document_content_sha256=document.content_sha256,
                    provenance={k: str(v) for k, v in document.provenance.model_dump().items()},
                )
            )
    return tuple(passages), omitted_documents, omitted_chunks


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_grounded_context(
    incident_query: IncidentQuery,
    result: RetrievalResult,
    *,
    root_cause: Mapping,
    reconstruction: Mapping | None,
    store: KnowledgeStore,
    limits: ContextLimits | None = None,
) -> GroundedContext:
    """
    Combine one incident query, its retrieval result and the Phase 8/9 content
    into a `GroundedContext`. Raises `ContextBuildError` for inconsistent input.

    `result` must be the result of exactly `incident_query.query`. The function
    itself never searches; it only verifies and arranges what was retrieved.
    """
    limits = limits or ContextLimits()
    if not isinstance(incident_query, IncidentQuery):
        raise ContextBuildError("incident_query must be an IncidentQuery")
    if not isinstance(result, RetrievalResult):
        raise ContextBuildError("result must be a RetrievalResult")
    if result.query != incident_query.query:
        raise ContextBuildError("the retrieval result was not produced by this incident query")

    root_cause = _mapping(root_cause, "root_cause")
    _refuse_answer_keys(root_cause, "root_cause")
    if reconstruction is not None:
        reconstruction = _mapping(reconstruction, "reconstruction")
        _refuse_answer_keys(reconstruction, "reconstruction")
        if reconstruction.get("incident_id") != incident_query.incident_id:
            raise ContextBuildError("reconstruction belongs to a different incident")
    if root_cause.get("incident_id") != incident_query.incident_id:
        raise ContextBuildError("root-cause analysis belongs to a different incident")

    findings = root_cause.get("findings")
    index = incident_query.finding_index
    if not isinstance(findings, (list, tuple)) or not 0 <= index < len(findings):
        raise ContextBuildError(f"root_cause.findings[{index}] does not exist")
    finding = _mapping(findings[index], f"root_cause.findings[{index}]")
    _refuse_answer_keys(finding, f"root_cause.findings[{index}]")

    observed, unresolved = _resolve_observed_evidence(reconstruction, incident_query.evidence_ids)
    hypothesis = _hypothesis_from_finding(
        finding, index, incident_query.finding_status, incident_query.evidence_ids
    )
    references, omitted_documents, omitted_chunks = _reference_passages(result, store, limits)

    return GroundedContext(
        incident_id=incident_query.incident_id,
        finding_index=index,
        retriever=result.retriever,
        is_production_retriever=result.retriever == PRODUCTION_RETRIEVER_NAME,
        retrieval_status=result.status,
        possibly_incomplete=result.possibly_incomplete,
        backend_query=result.backend_query,
        dropped_terms=result.dropped_terms,
        query_notes=result.query_notes,
        query_text=incident_query.query.text,
        query_phrases=incident_query.query.phrases,
        query_omitted_fields=incident_query.omitted_fields,
        observed_evidence=observed,
        unresolved_evidence_ids=unresolved,
        hypotheses=(hypothesis,),
        references=references,
        omitted_documents=omitted_documents,
        omitted_chunks=omitted_chunks,
    )
