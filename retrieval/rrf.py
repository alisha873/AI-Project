"""
Phase 11 (Milestone 11C) - source-level Reciprocal Rank Fusion (RRF) of the two knowledge lanes.

This implements the specification written in `retrieval.lanes` (Milestone 11B), unchanged:

    rrf(d) = sum over the result lists R in which document d occurs of  1 / (k + rank_R(d))

with `k = RRF_DEFAULT_K = 60` and `rank_R(d)` the 1-based position of d in list R.

- Fusion works on SOURCE DOCUMENTS, not chunks. Each input is already a `RetrievalResult` (one item per
  `doc_id`); `rrf_scores` additionally keeps only the first occurrence of an id inside one list, so a document
  counts at most once per list however many chunks matched.
- Only RANKS are used. Raw lane scores (cosine, BM25) are never added together; they are kept, with their
  lane and rank, in `FusedItem.contributions` for display.
- Deterministic order: higher RRF score; then smaller best lane rank; then the order of the lanes (weakness
  before attack pattern); then `doc_id`. Scores are compared after rounding to 12 decimals so floating-point
  noise cannot reorder mathematically equal documents.
- The limit is applied AFTER fusion. Each lane is asked for `lane_depth` (default 100, the contract maximum)
  results, so the fused scores use the true lane ranks.
- A lane that returns nothing contributes nothing; if all lanes are empty the result is `no_results`.

Honest caveat (specified in 11B and measured in 11C)
----------------------------------------------------
The CWE and CAPEC lanes are DISJOINT: a document is in exactly one lane, so it has exactly one contribution
and its RRF score is just 1/(k + rank). RRF therefore cannot reward a document for being ranked highly by
several lists, and it cannot change the order of documents WITHIN a lane. It only decides how the two lanes
are interleaved. Whether that helps is an empirical question; compare it with the unfiltered retriever.

An RRF score is a rank-based fusion value that is only used to order one list. It is NOT a probability or a
calibrated relevance estimate, and its absolute value carries no meaning (with k = 60 the top document of
any lane scores 1/61 regardless of how good the match is).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from retrieval.contract import (
    MAX_LIMIT,
    InvalidRetrievalQueryError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    Retriever,
)
from retrieval.lanes import LANES, RRF_DEFAULT_K, RRF_MIN_K, Lane, lane_query

SCORE_PRECISION = 12
RRF_SCORE_MEANING = (
    "RRF scores are rank-based fusion values used only to order this list; they are not probabilities "
    "or calibrated relevance estimates."
)


def check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < RRF_MIN_K:
        raise ValueError(f"k must be an integer >= {RRF_MIN_K}")
    return k


def rrf_scores(ranked_lists: Sequence[Sequence[str]], k: int = RRF_DEFAULT_K) -> dict[str, float]:
    """
    The RRF score of every document that occurs in `ranked_lists` (each a best-first list of document ids).

    A document counts at most once per list (the first occurrence defines its rank within that list).
    """
    check_k(k)
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass(frozen=True)
class LaneContribution:
    """One lane's view of a fused document: the lane name, its rank there and its own raw score."""

    lane: str
    lane_rank: int
    lane_score: float


@dataclass(frozen=True)
class FusedItem:
    """A fused document with its per-lane diagnostics (the `item` is what a `RetrievalResult` carries)."""

    item: RetrievedItem
    rrf_score: float
    k: int
    contributions: tuple[LaneContribution, ...]


@dataclass(frozen=True)
class FusedResult:
    """The fused `RetrievalResult` plus per-lane diagnostics."""

    result: RetrievalResult
    fused: tuple[FusedItem, ...]
    lane_results: tuple[tuple[str, RetrievalResult], ...]
    k: int


def fuse_lane_results(
    lane_results: Sequence[tuple[Lane, RetrievalResult]],
    *,
    query: RetrievalQuery,
    name: str,
    k: int = RRF_DEFAULT_K,
) -> FusedResult:
    """Fuse already-ranked, source-level lane results (see the module docstring)."""
    check_k(k)
    names = [lane.name for lane, _ in lane_results]
    if len(set(names)) != len(names):
        raise ValueError("lane names must be unique")

    contributions: dict[str, list[tuple[int, LaneContribution, RetrievedItem]]] = {}
    for lane_index, (lane, result) in enumerate(lane_results):
        seen: set[str] = set()
        for item in result.items:
            if item.doc_id in seen:  # defensive: a source counts once per list
                continue
            seen.add(item.doc_id)
            contributions.setdefault(item.doc_id, []).append(
                (lane_index, LaneContribution(lane.name, item.rank, item.score), item)
            )

    scores = rrf_scores([[item.doc_id for item in result.items] for _, result in lane_results], k)
    ordered = []
    for doc_id, parts in contributions.items():
        best_index, best, best_item = min(parts, key=lambda part: (part[1].lane_rank, part[0]))
        ordered.append((round(scores[doc_id], SCORE_PRECISION), best.lane_rank, best_index, doc_id, parts, best_item))
    ordered.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))

    fused: list[FusedItem] = []
    items: list[RetrievedItem] = []
    for rank, (_, _, _, doc_id, parts, best_item) in enumerate(ordered[: query.limit], start=1):
        item = dataclasses.replace(best_item, rank=rank, score=scores[doc_id])
        items.append(item)
        fused.append(FusedItem(item, scores[doc_id], k, tuple(c for _, c, _ in sorted(parts, key=lambda p: p[0]))))

    notes: list[str] = []
    for _, result in lane_results:
        notes.extend(note for note in result.query_notes if note not in notes)
    notes.append(RRF_SCORE_MEANING)
    counts = ", ".join(f"{lane.name}={len(result.items)}" for lane, result in lane_results)
    result = RetrievalResult(
        query=query,
        retriever=name,
        items=tuple(items),
        status="ok" if items else "no_results",
        possibly_incomplete=any(r.possibly_incomplete for _, r in lane_results),
        backend_query=f"rrf k={k} lanes=[{counts}] limit_applied_after_fusion={query.limit}",
        query_notes=tuple(notes),
    )
    return FusedResult(result, tuple(fused), tuple((lane.name, r) for lane, r in lane_results), k)


class LaneRetriever:
    """Wraps a retriever so every query is restricted to one lane's document types."""

    def __init__(self, retriever: Retriever, lane: Lane) -> None:
        self._retriever = retriever
        self.lane = lane
        self.name = f"{retriever.name}[{lane.name}]"

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return self._retriever.retrieve(lane_query(query, self.lane))


class RrfRetriever:
    """
    A `Retriever` that runs one inner retriever once per lane and fuses the lane rankings with RRF.

    The same inner retriever (and so the same embedding model) serves every lane; a lane is only a
    `doc_types` filter. The caller's `query.doc_types` must be None because the lanes define the filters.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        lanes: Sequence[Lane] = LANES,
        k: int = RRF_DEFAULT_K,
        lane_depth: int = MAX_LIMIT,
    ) -> None:
        self._retriever = retriever
        self._lanes = tuple(lanes)
        if not self._lanes:
            raise ValueError("at least one lane is required")
        self.k = check_k(k)
        if isinstance(lane_depth, bool) or not isinstance(lane_depth, int) or not 1 <= lane_depth <= MAX_LIMIT:
            raise ValueError(f"lane_depth must be an integer from 1 to {MAX_LIMIT}")
        self._lane_depth = lane_depth
        self.name = f"rrf-k{self.k}[{retriever.name}]"

    def retrieve_detailed(self, query: RetrievalQuery) -> FusedResult:
        if not isinstance(query, RetrievalQuery):
            raise InvalidRetrievalQueryError("retrieve() requires a RetrievalQuery")
        if query.doc_types is not None:
            raise InvalidRetrievalQueryError(
                "the fusion retriever defines its own lane filters; query.doc_types must be None"
            )
        lane_results = []
        for lane in self._lanes:
            lane_request = dataclasses.replace(query, doc_types=lane.doc_types, limit=max(query.limit, self._lane_depth))
            lane_results.append((lane, self._retriever.retrieve(lane_request)))
        return fuse_lane_results(lane_results, query=query, name=self.name, k=self.k)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        return self.retrieve_detailed(query).result
