"""
Phase 11 (Milestone 11B) - hardened keyword retrieval, `fts5-bm25-v2`.

`retrieval.fts5` (`fts5-bm25-v1`) is left UNCHANGED so the Milestone 11A
baseline stays reproducible. This module adds a second retriever behind the
same `Retriever` contract, over the same `KnowledgeStore.search()`. It adds no
tables, index or hash and never writes to the store.

What changes compared with v1 (query handling)
----------------------------------------------
1. Unicode normalization. The text is NFKC-normalized (full-width letters and
   digits, ligatures), every dash or hyphen variant (U+2010..U+2015, U+2212,
   U+FE58, U+FE63, U+FF0D) becomes "-", invisible formatting characters are
   removed (U+200B becomes a space). So "CWE<U+2011>79" is the identifier
   "CWE-79". v1 does not recognize it. Each transformation is reported in
   `RetrievalResult.query_notes`.
2. Identifiers. "CWE-89", "cwe-89", "CWE 89" and "CWE_89" (also CAPEC and CVE
   numbers) are read as the identifier and searched as the quoted phrase
   "CWE-89". "CWE89" (no separator) is NOT recognized; it is one unknown word.
3. Quoted spans. Text inside balanced double quotes ("..." or <U+201C>...<U+201D>)
   is searched as an exact phrase, in addition to its words. An unbalanced
   quotation mark is ignored and reported.
4. Hyphenated compounds ("server-side", "dot-dot-slash") are searched as an
   adjacent-word phrase in addition to their words (all-digit compounds such as
   dates are not).
5. Everything that was ignored (stop words, one-letter words, bare numbers) or
   cut to stay within the store's 1000-character query limit is reported
   (`query_notes` and `dropped_terms`); nothing disappears silently.
6. Low-information queries. A query whose ONLY searchable content is one plain
   word (for example "attack") still runs but carries a note starting with
   `LOW_INFORMATION_PREFIX`. A query with no searchable content at all raises
   `InvalidRetrievalQueryError`. Two generic words ("security problem") are NOT
   detected: low information is a deliberately simple lexical rule, not a
   judgement about meaning.

Order of terms, highest priority first (this is also the order in which terms
are cut if the expression is too long): identifiers, caller `phrases`, quoted
spans, hyphenated compounds, words. Terms are OR-ed, so BM25 ranks documents
that match more and rarer terms higher. Nothing is interpolated into SQL
structure: the finished expression contains only double-quoted alphanumeric
phrases and the word OR, and the store binds it as a parameter.

Ranking change (documents, not terms)
-------------------------------------
The store ranks CHUNKS. v1 ranks a document by its single best chunk. v2's
default `aggregation="top2_sum"` ranks a document by the SUM of its two best
chunk scores (a document with one matching chunk uses that one score). The
idea is that two strongly matching passages are more evidence than one. This
was chosen from seven variants tried on the DEVELOPMENT split only, so the
development gain may be partly selection noise; the held-out split is the test.
`aggregation="best_chunk"` reproduces v1's document ranking (with the v2 query
handling) and exists so the two effects can be measured separately. Document
ties are broken by `doc_id`. `RetrievedItem.score` is the aggregated score.

Lane filtering
--------------
`RetrievalQuery.doc_types` is applied to the chunks the store returned. The
store cannot filter by type, so the filter runs AFTER its 100-chunk window.
Documents in the requested lane whose best chunk lies beyond that window cannot
be seen. If the window is full and fewer than `limit` documents were found,
`possibly_incomplete` is set and a note says so; if `limit` documents were
found, they are exactly the top `limit` of that lane. A query for several types
returns one ranking with each document at most once.
"""

from __future__ import annotations

import re
import unicodedata

from knowledge.store import (
    MAX_QUERY_LENGTH,
    MAX_SEARCH_LIMIT,
    InvalidSearchQueryError,
    KnowledgeStore,
    KnowledgeStoreError,
)
from retrieval.contract import (
    LOW_INFORMATION_PREFIX,
    ChunkMatch,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)
from retrieval.fts5 import STOPWORDS, _quote, _words

RETRIEVER_NAME = "fts5-bm25-v2"
AGGREGATIONS = ("top2_sum", "best_chunk")
MAX_QUOTED_SPANS = 16
_NOTE_LIST_CAP = 25

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
_DASH_RE = re.compile(f"[{_DASHES}]")
_QUOTED_RE = re.compile(r'"([^"]*)"|\u201c([^\u201d]*)\u201d')
_IDENTIFIER_RE = re.compile(r"\b(CWE|CAPEC|CVE)[-_ ]([0-9]+(?:-[0-9]+)?)\b", re.IGNORECASE)
_COMPOUND_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)+")


def normalize_query_text(text: str) -> tuple[str, list[str]]:
    """NFKC, dash variants to '-', invisible characters removed. Returns (text, notes)."""
    notes: list[str] = []
    result = unicodedata.normalize("NFKC", text)
    if result != text:
        changed = sum(1 for a, b in zip(text, result) if a != b) or abs(len(text) - len(result))
        notes.append(f"normalized {changed} character(s) with Unicode NFKC")
    dashes = len(_DASH_RE.findall(result))
    if dashes:
        result = _DASH_RE.sub("-", result)
        notes.append(f"normalized {dashes} Unicode dash/hyphen character(s) to '-'")
    cleaned: list[str] = []
    removed = 0
    for char in result:
        if char == "\u200b":
            cleaned.append(" ")
        elif unicodedata.category(char) == "Cf":
            removed += 1
        else:
            cleaned.append(char)
    if removed:
        notes.append(f"removed {removed} invisible formatting character(s)")
    return "".join(cleaned), notes


class CompiledQueryV2:
    """A v2 FTS5 expression with its terms, cut terms, audit notes and low-information flag."""

    __slots__ = ("expression", "terms", "dropped", "notes", "low_information")

    def __init__(self, expression, terms, dropped, notes, low_information):
        self.expression = expression
        self.terms = terms
        self.dropped = dropped
        self.notes = notes
        self.low_information = low_information


def _cap(items: list[str]) -> str:
    shown = ", ".join(items[:_NOTE_LIST_CAP])
    return shown if len(items) <= _NOTE_LIST_CAP else f"{shown} (+{len(items) - _NOTE_LIST_CAP} more)"


def compile_fts_expression_v2(query: RetrievalQuery) -> CompiledQueryV2:
    """
    Turn a `RetrievalQuery` into a safe FTS5 expression (see the module docstring).

    Raises `InvalidRetrievalQueryError` if no searchable term remains.
    """
    text, notes = normalize_query_text(query.text)
    candidates: list[str] = []  # display form, highest priority first
    seen: set[str] = set()
    identifier_found = False
    distinct_words: set[str] = set()

    def add(term: str, *, identifier: bool = False) -> None:
        nonlocal identifier_found
        words = _words(term)
        key = " ".join(words)
        if not words or key in seen:
            return
        seen.add(key)
        candidates.append(term)
        if identifier:
            identifier_found = True
        else:
            distinct_words.update(words)

    for match in _IDENTIFIER_RE.finditer(text):
        canonical = f"{match.group(1).upper()}-{match.group(2)}"
        if match.group(0) != canonical:
            note = f"read {match.group(0)!r} as the identifier {canonical!r}"
            if note not in notes:
                notes.append(note)
        add(canonical, identifier=True)
    rest = _IDENTIFIER_RE.sub(" ", text)

    for phrase in query.phrases:
        phrase_text, phrase_notes = normalize_query_text(phrase)
        notes.extend(n for n in phrase_notes if n not in notes)
        add(" ".join(_words(phrase_text)))

    spans = [(m.group(1) if m.group(1) is not None else m.group(2)) for m in _QUOTED_RE.finditer(rest)]
    for index, span in enumerate(spans):
        if index >= MAX_QUOTED_SPANS:
            notes.append(f"ignored {len(spans) - MAX_QUOTED_SPANS} quoted span(s) beyond the first {MAX_QUOTED_SPANS}")
            break
        if _words(span):
            add(" ".join(_words(span)))
        else:
            notes.append("ignored a quoted span with no letters or digits")
    if re.search(r'["\u201c\u201d]', _QUOTED_RE.sub(" ", rest)):
        notes.append("ignored an unbalanced quotation mark")

    for match in _COMPOUND_RE.finditer(rest):
        parts = _words(match.group(0))
        if len(parts) >= 2 and not all(part.isdigit() for part in parts):
            add(" ".join(parts))

    ignored: list[str] = []
    for word in _words(rest):
        if len(word) < 2 or word.isdigit() or word in STOPWORDS:
            if word not in ignored:
                ignored.append(word)
            continue
        add(word)
    if ignored:
        notes.append(f"ignored low-content word(s): {_cap(ignored)}")

    if not candidates:
        raise InvalidRetrievalQueryError(
            "the query has no searchable terms (only stop words, numbers or punctuation)"
        )

    kept: list[str] = []
    length = 0
    for term in candidates:
        added = len(_quote(term)) + (len(" OR ") if kept else 0)
        if length + added > MAX_QUERY_LENGTH:
            break
        kept.append(term)
        length += added
    if not kept:
        raise InvalidRetrievalQueryError(f"the first query term exceeds {MAX_QUERY_LENGTH} characters")
    dropped = tuple(candidates[len(kept):])
    if dropped:
        notes.append(f"cut {len(dropped)} lowest-priority term(s) to stay within {MAX_QUERY_LENGTH} characters")

    low_information = (not identifier_found) and len(distinct_words) == 1
    if low_information:
        only = next(iter(distinct_words))
        notes.append(
            f"{LOW_INFORMATION_PREFIX} the only searchable word is {only!r}; "
            "a ranking from one word is weak evidence"
        )
    return CompiledQueryV2(" OR ".join(_quote(t) for t in kept), tuple(kept), dropped, tuple(notes), low_information)


class Fts5RetrieverV2:
    """
    Hardened keyword retriever over an open `KnowledgeStore` (BM25, source-level results).

    Stateless apart from the store reference and the chosen `aggregation`, so
    identical queries on an unchanged store return identical results.
    `name` is "fts5-bm25-v2" for the default aggregation and
    "fts5-bm25-v2-bestchunk" for the ablation, so reports always say which ran.
    """

    def __init__(self, store: KnowledgeStore, *, aggregation: str = "top2_sum") -> None:
        if not isinstance(store, KnowledgeStore):
            raise TypeError("Fts5RetrieverV2 requires a KnowledgeStore")
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}")
        self._store = store
        self._aggregation = aggregation

    @property
    def name(self) -> str:
        return RETRIEVER_NAME if self._aggregation == "top2_sum" else f"{RETRIEVER_NAME}-bestchunk"

    def _document_score(self, chunk_scores: list[float]) -> float:
        # chunk_scores arrive best-first from the store, so the first is the maximum.
        return sum(chunk_scores[:2]) if self._aggregation == "top2_sum" else chunk_scores[0]

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not isinstance(query, RetrievalQuery):
            raise InvalidRetrievalQueryError("retrieve() requires a RetrievalQuery")
        compiled = compile_fts_expression_v2(query)

        try:
            hits = self._store.search(compiled.expression, limit=MAX_SEARCH_LIMIT)
        except InvalidSearchQueryError as exc:
            raise InvalidRetrievalQueryError(f"the backend rejected the compiled query: {exc}") from exc
        except KnowledgeStoreError as exc:
            raise RetrievalBackendError(f"keyword search failed: {exc}") from exc

        allowed = None if query.doc_types is None else set(query.doc_types)
        grouped: dict[str, list] = {}
        for hit in hits:  # best chunk first
            if allowed is not None and hit.chunk.doc_type not in allowed:
                continue
            grouped.setdefault(hit.chunk.doc_id, []).append(hit)

        scored = {
            doc_id: self._document_score([-h.bm25 for h in doc_hits]) for doc_id, doc_hits in grouped.items()
        }
        order = sorted(scored, key=lambda doc_id: (-scored[doc_id], doc_id))

        notes = list(compiled.notes)
        incomplete = len(hits) >= MAX_SEARCH_LIMIT and len(order) < query.limit
        if incomplete:
            notes.append(
                f"the store's {MAX_SEARCH_LIMIT}-chunk window was full and only {len(order)} matching "
                f"document(s) were seen, so further matches may exist"
            )

        items = []
        for rank, doc_id in enumerate(order[: query.limit], start=1):
            doc_hits = grouped[doc_id]
            first = doc_hits[0].chunk
            items.append(
                RetrievedItem(
                    rank=rank,
                    doc_id=doc_id,
                    source=first.source,
                    source_id=first.source_id,
                    doc_type=first.doc_type,
                    title=first.title,
                    source_url=first.source_url,
                    score=scored[doc_id],
                    chunks=tuple(
                        ChunkMatch(
                            chunk_id=h.chunk.chunk_id,
                            ordinal=h.chunk.ordinal,
                            char_start=h.chunk.char_start,
                            char_end=h.chunk.char_end,
                            text=h.chunk.text,
                            score=-h.bm25,
                        )
                        for h in doc_hits
                    ),
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
            query_notes=tuple(notes),
        )
