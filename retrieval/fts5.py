"""
Phase 11 (Milestone 11A) - keyword retrieval over `KnowledgeStore.search()`.

`Fts5Retriever` is the first implementation of `retrieval.contract.Retriever`.
It does NOT add storage or an index: it builds a safe FTS5 expression, calls
the existing store, and turns chunk hits into source-level items.

How a query text becomes an FTS5 expression (deterministic, no model)
---------------------------------------------------------------------
`KnowledgeStore.search()` takes raw FTS5 syntax, where "-" and ":" are
operators (PROJECT_STATUS limitation 14). Free text must therefore never be
passed through. `compile_fts_expression` instead:

1. pulls out explicit identifiers such as "CWE-89" or "CAPEC-66" and quotes
   each as a phrase (so the hyphen is data, not an operator);
2. quotes every caller-supplied phrase after lowercasing it to plain words;
3. splits the remaining text into letter/digit words, lowercases them, drops
   one-letter words, number-only words and the words in `STOPWORDS`, and
   removes duplicates (first occurrence wins);
4. joins everything with OR, in that order (identifiers, phrases, words).

OR is used because an AND of twenty words would match nothing. BM25 then
ranks documents that match more, and rarer, terms higher. If the expression
would exceed the store's 1000-character limit, terms are dropped from the END
(lowest priority first) and reported in `RetrievalResult.dropped_terms`.

`STOPWORDS` holds ordinary English function words plus words that the Phase 8/9
analysis text uses as boilerplate ("potential", "hypothesis", "event", ...).
They carry no knowledge-base meaning, and removing them is a lexical-backend
decision, so it lives here and not in the query contract.

Source-level folding
--------------------
The store ranks CHUNKS. Chunks are grouped by `doc_id`; a document's rank and
score come from its best chunk, so one source is never counted twice because
several of its chunks match. Document order = order of best chunk, with the
store's own deterministic tie-break (BM25, then doc_id, then ordinal).

The store is always asked for its maximum window (100 chunks) so there are
enough chunks to fill `limit` documents. If that window is full and still
yields fewer than `limit` documents, `possibly_incomplete` is set.
"""

from __future__ import annotations

import re

from knowledge.store import (
    MAX_QUERY_LENGTH,
    MAX_SEARCH_LIMIT,
    InvalidSearchQueryError,
    KnowledgeStore,
    KnowledgeStoreError,
)
from retrieval.contract import (
    ChunkMatch,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)

RETRIEVER_NAME = "fts5-bm25-v1"

# "CWE-89", "CAPEC-66", "CVE-2021-44228". Matched case-insensitively, stored upper-case.
_IDENTIFIER_RE = re.compile(r"\b(?:CWE|CAPEC|CVE)-[0-9]+(?:-[0-9]+)?\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[^\W_]+")

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its itself
    just may me might more most must my no nor not now of off on once one only or other our
    out over own same she should so some such than that the their them then there these they
    this those through to too under until up upon us very was we were what when where which
    while who whom why will with within without would you your
    potential potentially possible possibly likely probable suspected hypothesis hypotheses
    event events normalized available observed indicator indicators
    """.split()
)


class CompiledFtsQuery:
    """The FTS5 expression for a query, with the terms it contains and the terms it dropped."""

    __slots__ = ("expression", "terms", "dropped")

    def __init__(self, expression: str, terms: tuple[str, ...], dropped: tuple[str, ...]) -> None:
        self.expression = expression
        self.terms = terms
        self.dropped = dropped


def _quote(phrase: str) -> str:
    # Words are alphanumeric by construction, so the quote can never be inside them;
    # doubling is only a defensive FTS5-correct escape.
    return '"' + phrase.replace('"', '""') + '"'


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def compile_fts_expression(query: RetrievalQuery) -> CompiledFtsQuery:
    """
    Turn a `RetrievalQuery` into a safe FTS5 expression (see the module docstring).

    Raises `InvalidRetrievalQueryError` if nothing searchable remains.
    """
    candidates: list[str] = []  # human-readable terms, highest priority first
    seen: set[str] = set()

    def add(term: str) -> None:
        if term not in seen:
            seen.add(term)
            candidates.append(term)

    text = query.text
    for match in _IDENTIFIER_RE.finditer(text):
        add(match.group(0).upper())
    text_without_ids = _IDENTIFIER_RE.sub(" ", text)

    for phrase in query.phrases:
        words = _words(phrase)
        if words:
            add(" ".join(words))

    for word in _words(text_without_ids):
        if len(word) < 2 or word.isdigit() or word in STOPWORDS:
            continue
        add(word)

    if not candidates:
        raise InvalidRetrievalQueryError(
            "the query has no searchable terms (only stop words, numbers or punctuation)"
        )

    kept: list[str] = []
    length = 0
    for term in candidates:
        added = len(_quote(term)) + (len(" OR ") if kept else 0)
        if length + added > MAX_QUERY_LENGTH:
            break  # drop this term and every lower-priority one after it
        kept.append(term)
        length += added
    if not kept:
        raise InvalidRetrievalQueryError(f"the first query term exceeds {MAX_QUERY_LENGTH} characters")
    dropped = tuple(candidates[len(kept):])
    return CompiledFtsQuery(" OR ".join(_quote(t) for t in kept), tuple(kept), dropped)


class Fts5Retriever:
    """
    Keyword retriever over an open `KnowledgeStore` (BM25 ranking, source-level results).

    The retriever never writes to the store and keeps no state besides the
    store reference, so identical queries on an unchanged store return
    identical results.
    """

    name = RETRIEVER_NAME

    def __init__(self, store: KnowledgeStore) -> None:
        if not isinstance(store, KnowledgeStore):
            raise TypeError("Fts5Retriever requires a KnowledgeStore")
        self._store = store

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not isinstance(query, RetrievalQuery):
            raise InvalidRetrievalQueryError("retrieve() requires a RetrievalQuery")
        compiled = compile_fts_expression(query)

        try:
            hits = self._store.search(compiled.expression, limit=MAX_SEARCH_LIMIT)
        except InvalidSearchQueryError as exc:
            raise InvalidRetrievalQueryError(f"the backend rejected the compiled query: {exc}") from exc
        except KnowledgeStoreError as exc:
            raise RetrievalBackendError(f"keyword search failed: {exc}") from exc

        allowed = None if query.doc_types is None else set(query.doc_types)
        order: list[str] = []
        grouped: dict[str, list] = {}
        for hit in hits:  # already ordered best-first by the store
            if allowed is not None and hit.chunk.doc_type not in allowed:
                continue
            doc_id = hit.chunk.doc_id
            if doc_id not in grouped:
                grouped[doc_id] = []
                order.append(doc_id)
            grouped[doc_id].append(hit)

        incomplete = len(hits) >= MAX_SEARCH_LIMIT and len(order) < query.limit
        items = []
        for rank, doc_id in enumerate(order[: query.limit], start=1):
            doc_hits = grouped[doc_id]
            first = doc_hits[0].chunk  # best chunk
            chunks = tuple(
                ChunkMatch(
                    chunk_id=h.chunk.chunk_id,
                    ordinal=h.chunk.ordinal,
                    char_start=h.chunk.char_start,
                    char_end=h.chunk.char_end,
                    text=h.chunk.text,
                    score=-h.bm25,
                )
                for h in doc_hits
            )
            items.append(
                RetrievedItem(
                    rank=rank,
                    doc_id=doc_id,
                    source=first.source,
                    source_id=first.source_id,
                    doc_type=first.doc_type,
                    title=first.title,
                    source_url=first.source_url,
                    score=-doc_hits[0].bm25,
                    chunks=chunks,
                )
            )
        return RetrievalResult(
            query=query,
            retriever=self.name,
            items=tuple(items),
            status="ok" if items else "no_results",
            possibly_incomplete=incomplete,
            backend_query=compiled.expression,
            dropped_terms=compiled.dropped,
        )
