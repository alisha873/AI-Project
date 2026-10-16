"""
Phase 11 (Milestone 11A) - the retrieval contract.

This module defines WHAT a retriever accepts and returns. It contains no
search logic, no database code and no model code, so a keyword backend (FTS5,
today) and any future semantic or dual-vector backend can sit behind the same
types without the incident-analysis code changing.

Query
-----
`RetrievalQuery` holds
- `text`       natural-language description of what to look up. A semantic
               backend can embed it as-is.
- `phrases`    optional exact multi-word phrases or identifiers (for example
               "sql injection"). A lexical backend treats them as phrase terms;
               a semantic backend may ignore them.
- `limit`      the maximum number of SOURCE DOCUMENTS to return (1..100).
- `doc_types`  optional filter on `SourceDocument.doc_type` (for example only
               "weakness"). Two filtered calls are the simplest way to build
               two retrieval lanes later; see "Open decisions" below.

Result
------
`RetrievalResult.items` is a tuple of `RetrievedItem`, one per DISTINCT source
document, best first. Several matching chunks of one document are folded into
one item (its `chunks`), so a source is never counted twice.

Identity: every item carries `doc_id` (e.g. "cwe:CWE-89"), `source` ("cwe"),
`source_id` ("CWE-89") and, for each matched chunk, `chunk_id` and character
offsets into the document's content, all copied unchanged from the knowledge
store. The full document and its provenance are available from
`KnowledgeStore.get_document(doc_id)`; nothing is duplicated here.

Ordering and limits: items are ranked 1..n in the order the backend produced;
`rank` is the list position (there are no fractional ranks). `score` is only
comparable within ONE result list from ONE retriever (for FTS5 it is the
negated BM25, so higher is better). Ties are broken by the backend and must be
deterministic.

Errors and empty results
------------------------
- A query that cannot be run (empty text, no searchable terms, bad `limit`,
  unknown `doc_type`) raises `InvalidRetrievalQueryError`.
- A failing backend raises `RetrievalBackendError` (the original exception is
  chained as `__cause__`).
- A valid query that matches nothing is NOT an error: it returns a result
  with `status == "no_results"` and `items == ()`.
- `possibly_incomplete` is True when a backend could not look at every
  candidate (for FTS5: the 100-chunk window was full yet fewer than `limit`
  distinct documents were seen). Callers that need completeness must check it.

What a result means
-------------------
A result is REFERENCE KNOWLEDGE that may help assess an analysis. It does not
confirm that an incident's root-cause hypothesis is correct, and a missing
result does not refute it. This is stated on `RESULT_MEANING` so downstream
code can display it.

Open decisions (NOT made here)
------------------------------
PROJECT_SPEC.md names two retrieval paths (weakness/vulnerability information,
and attack techniques/behaviour) but does not say whether they are two
embedding spaces or one space with two filters, how their ranked lists are
merged, or whether CVE/NVD records are required. Milestone 11A implements no
vector retrieval and does not claim a dual-vector design. See
PROJECT_STATUS.md for the recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

DEFAULT_LIMIT = 10
MAX_LIMIT = 100
MAX_QUERY_TEXT_CHARS = 4000
MAX_PHRASES = 16

# The values of `knowledge.schema.DocType`, repeated here so this module stays
# independent of the persistence layer. A test pins them to the schema.
VALID_DOC_TYPES: frozenset[str] = frozenset(
    {"vulnerability", "weakness", "attack_pattern", "guideline", "technique"}
)

# A retriever that judges a query too thin to search with confidence starts one of its
# `RetrievalResult.query_notes` with this prefix (Milestone 11B). The evaluation harness
# treats such a note as "the query was flagged"; it does not change the ranked items.
LOW_INFORMATION_PREFIX = "low-information query:"

RESULT_MEANING = (
    "Retrieved items are reference knowledge from a security corpus. They do not "
    "confirm an incident's root-cause hypothesis, and an absent item does not refute it."
)

ResultStatus = Literal["ok", "no_results"]


# ============================================================
# ERRORS
# ============================================================

class RetrievalError(Exception):
    """Base class for every error raised deliberately by the retrieval package."""


class InvalidRetrievalQueryError(RetrievalError, ValueError):
    """The query cannot be run: empty, nothing searchable, or an invalid option."""


class RetrievalBackendError(RetrievalError):
    """The backend failed (for example the knowledge store raised an error)."""


# ============================================================
# QUERY
# ============================================================

@dataclass(frozen=True)
class RetrievalQuery:
    """A backend-independent retrieval request (see the module docstring)."""

    text: str
    phrases: tuple[str, ...] = ()
    limit: int = DEFAULT_LIMIT
    doc_types: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise InvalidRetrievalQueryError("query text must be a non-empty string")
        if len(self.text) > MAX_QUERY_TEXT_CHARS:
            raise InvalidRetrievalQueryError(f"query text exceeds {MAX_QUERY_TEXT_CHARS} characters")
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or not 1 <= self.limit <= MAX_LIMIT:
            raise InvalidRetrievalQueryError(f"limit must be an integer from 1 to {MAX_LIMIT}")

        phrases = self.phrases
        if isinstance(phrases, (str, bytes)) or not isinstance(phrases, (tuple, list)):
            raise InvalidRetrievalQueryError("phrases must be a tuple of strings")
        if len(phrases) > MAX_PHRASES:
            raise InvalidRetrievalQueryError(f"at most {MAX_PHRASES} phrases are allowed")
        if any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise InvalidRetrievalQueryError("every phrase must be a non-empty string")
        object.__setattr__(self, "phrases", tuple(phrases))

        if self.doc_types is not None:
            doc_types = self.doc_types
            if isinstance(doc_types, (str, bytes)) or not isinstance(doc_types, (tuple, list)) or not doc_types:
                raise InvalidRetrievalQueryError("doc_types must be None or a non-empty tuple of strings")
            unknown = sorted({str(d) for d in doc_types} - VALID_DOC_TYPES)
            if unknown:
                raise InvalidRetrievalQueryError(
                    f"unknown doc_type(s) {unknown}; expected a subset of {sorted(VALID_DOC_TYPES)}"
                )
            object.__setattr__(self, "doc_types", tuple(sorted(set(doc_types))))


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class ChunkMatch:
    """One matched chunk of a retrieved document (identity copied from the store)."""

    chunk_id: str
    ordinal: int
    char_start: int
    char_end: int
    text: str
    score: float


@dataclass(frozen=True)
class RetrievedItem:
    """One DISTINCT source document in a ranking, with its matched chunks (best first)."""

    rank: int
    doc_id: str
    source: str
    source_id: str
    doc_type: str
    title: str
    source_url: str | None
    score: float
    chunks: tuple[ChunkMatch, ...]

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be 1 or greater")
        if not self.chunks:
            raise ValueError("a retrieved item must carry at least one matched chunk")
        if any(chunk.chunk_id.split("#")[0] != self.doc_id for chunk in self.chunks):
            raise ValueError("every matched chunk must belong to the item's doc_id")


@dataclass(frozen=True)
class RetrievalResult:
    """
    The ranked outcome of one query.

    `status` is "ok" when `items` is non-empty and "no_results" when it is empty
    (an empty result is a valid outcome, not an error). `backend_query` is the
    exact expression the backend ran, for audit (None if a backend has no
    textual form). `dropped_terms` lists query terms a backend had to leave out
    to stay within its own limits.

    `query_notes` (added in Milestone 11B, optional, empty by default) is a
    human-readable audit trail of how a backend TRANSFORMED the query before
    running it: for example a Unicode hyphen normalized to "-", a quoted span
    treated as a phrase, or a warning that the query carries little
    information. It never changes what `items` mean; callers may ignore it.
    """

    query: RetrievalQuery
    retriever: str
    items: tuple[RetrievedItem, ...]
    status: ResultStatus
    possibly_incomplete: bool = False
    backend_query: str | None = None
    dropped_terms: tuple[str, ...] = field(default=())
    query_notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if [item.rank for item in self.items] != list(range(1, len(self.items) + 1)):
            raise ValueError("item ranks must be exactly 1..n in order")
        doc_ids = [item.doc_id for item in self.items]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("a result must contain each source document at most once")
        if len(self.items) > self.query.limit:
            raise ValueError("a result must not contain more items than the query limit")
        expected: ResultStatus = "ok" if self.items else "no_results"
        if self.status != expected:
            raise ValueError(f"status must be {expected!r} for {len(self.items)} item(s)")

    @property
    def doc_ids(self) -> tuple[str, ...]:
        """Ranked document IDs, best first."""
        return tuple(item.doc_id for item in self.items)


# ============================================================
# RETRIEVER PROTOCOL
# ============================================================

@runtime_checkable
class Retriever(Protocol):
    """
    Anything that can answer a `RetrievalQuery`. Implemented today by
    `retrieval.fts5.Fts5Retriever`. A future semantic retriever implements the
    same two members; a combining retriever (for example fusing a weakness lane
    and an attack-pattern lane) is also just a `Retriever`, so incident-analysis
    code never needs to know which backend it is talking to.
    """

    name: str

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Return a ranked, source-level result. Raises `RetrievalError` subclasses."""
        ...
