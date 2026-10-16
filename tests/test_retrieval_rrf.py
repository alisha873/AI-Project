"""Tests for source-level Reciprocal Rank Fusion over the CWE and CAPEC lanes (scripted lanes + fake embedder)."""

from __future__ import annotations

import dataclasses

import pytest

from retrieval.contract import (
    InvalidRetrievalQueryError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    ChunkMatch,
    Retriever,
)
from retrieval.embeddings import FakeEmbedder
from retrieval.lanes import ATTACK_PATTERN_LANE, LANES, RRF_DEFAULT_K, WEAKNESS_LANE, Lane
from retrieval.rrf import (
    RRF_SCORE_MEANING,
    LaneRetriever,
    RrfRetriever,
    check_k,
    fuse_lane_results,
    rrf_scores,
)
from retrieval.semantic import SemanticRetriever
from tests.retrieval_helpers import make_store, small_corpus

QUERY = RetrievalQuery(text="anything searchable here", limit=10)


def item(rank: int, doc_id: str, doc_type: str, score: float = 0.5) -> RetrievedItem:
    source = doc_id.split(":")[0]
    return RetrievedItem(
        rank=rank, doc_id=doc_id, source=source, source_id=doc_id.split(":")[1], doc_type=doc_type,
        title=f"title {doc_id}", source_url=None, score=score,
        chunks=(ChunkMatch(chunk_id=f"{doc_id}#0000", ordinal=0, char_start=0, char_end=5, text="text", score=score),),
    )


def lane_result(doc_ids, doc_type, query=QUERY, *, notes=(), incomplete=False) -> RetrievalResult:
    items = tuple(item(rank, doc_id, doc_type, 1.0 / rank) for rank, doc_id in enumerate(doc_ids, start=1))
    return RetrievalResult(
        query=query, retriever="stub", items=items, status="ok" if items else "no_results",
        possibly_incomplete=incomplete, backend_query="stub", query_notes=tuple(notes),
    )


class Scripted:
    """An inner retriever that returns a fixed ranking per requested doc_type and records the requests."""

    name = "scripted"

    def __init__(self, by_type: dict[str, list[str]]):
        self.by_type = by_type
        self.requests: list[RetrievalQuery] = []

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        self.requests.append(query)
        doc_type = query.doc_types[0]
        ids = self.by_type.get(doc_type, [])[: query.limit]
        return lane_result(ids, doc_type, query)


W = ["cwe:CWE-1", "cwe:CWE-2", "cwe:CWE-3"]
P = ["capec:CAPEC-1", "capec:CAPEC-2"]


# --- the formula --------------------------------------------------------------------------------------

def test_rrf_scores_match_hand_computed_values():
    scores = rrf_scores([["a", "b", "c"], ["b", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63)
    assert scores["d"] == pytest.approx(1 / 62)


def test_k_changes_the_scores_and_is_validated():
    assert rrf_scores([["a"]], k=1)["a"] == pytest.approx(1 / 2)
    assert rrf_scores([["a"]], k=10)["a"] == pytest.approx(1 / 11)
    assert RRF_DEFAULT_K == 60 and check_k(60) == 60
    for bad in (0, -5, True, 1.5, "60", None):
        with pytest.raises(ValueError):
            check_k(bad)


def test_a_document_counts_once_per_list_even_if_repeated():
    assert rrf_scores([["a", "a", "b"]], k=60) == {"a": pytest.approx(1 / 61), "b": pytest.approx(1 / 62)}


def test_empty_input_scores_nothing():
    assert rrf_scores([], 60) == {} and rrf_scores([[], []], 60) == {}


# --- fusing two disjoint lanes ------------------------------------------------------------------------------

def test_disjoint_lanes_interleave_by_rank_with_deterministic_ties():
    scripted = Scripted({"weakness": W, "attack_pattern": P})
    fused = RrfRetriever(scripted).retrieve_detailed(QUERY)
    # W1 and P1 tie at 1/61 (weakness lane first); W2 and P2 tie at 1/62; W3 is last at 1/63.
    assert [i.doc_id for i in fused.result.items] == [W[0], P[0], W[1], P[1], W[2]]
    assert [i.score for i in fused.result.items] == pytest.approx([1 / 61, 1 / 61, 1 / 62, 1 / 62, 1 / 63])
    assert [i.rank for i in fused.result.items] == [1, 2, 3, 4, 5]
    assert fused.k == 60


def test_each_disjoint_document_has_exactly_one_contribution_with_its_lane_rank_and_raw_score():
    fused = RrfRetriever(Scripted({"weakness": W, "attack_pattern": P})).retrieve_detailed(QUERY)
    by_doc = {f.item.doc_id: f for f in fused.fused}
    for doc_id, lane, rank in ((W[1], "weakness", 2), (P[0], "attack_pattern", 1)):
        (only,) = by_doc[doc_id].contributions
        assert (only.lane, only.lane_rank) == (lane, rank)
        assert only.lane_score == pytest.approx(1.0 / rank)  # the lane's own raw score, kept for display
        assert by_doc[doc_id].rrf_score == pytest.approx(1 / (60 + rank))


def test_fusion_never_changes_the_order_of_documents_within_one_lane():
    scripted = Scripted({"weakness": W, "attack_pattern": P})
    ids = [i.doc_id for i in RrfRetriever(scripted).retrieve(QUERY).items]
    assert [d for d in ids if d.startswith("cwe:")] == W
    assert [d for d in ids if d.startswith("capec:")] == P


def test_raw_scores_are_ignored_only_ranks_matter():
    class Loud(Scripted):
        def retrieve(self, query):
            result = super().retrieve(query)
            boost = 1000.0 if query.doc_types == ("attack_pattern",) else 0.001
            items = tuple(dataclasses.replace(i, score=boost / i.rank) for i in result.items)
            return dataclasses.replace(result, items=items)

    quiet = [i.doc_id for i in RrfRetriever(Scripted({"weakness": W, "attack_pattern": P})).retrieve(QUERY).items]
    loud = [i.doc_id for i in RrfRetriever(Loud({"weakness": W, "attack_pattern": P})).retrieve(QUERY).items]
    assert quiet == loud


def test_limit_is_applied_after_fusion_and_lanes_are_asked_for_more():
    scripted = Scripted({"weakness": W, "attack_pattern": P})
    limited = RrfRetriever(scripted).retrieve(dataclasses.replace(QUERY, limit=2))
    assert [i.doc_id for i in limited.items] == [W[0], P[0]]
    assert all(request.limit == 100 for request in scripted.requests)  # lane depth, not the caller's limit
    assert [request.doc_types for request in scripted.requests] == [("weakness",), ("attack_pattern",)]
    assert limited.status == "ok"


def test_a_shallow_lane_depth_truncates_each_lane_before_fusion():
    scripted = Scripted({"weakness": W, "attack_pattern": P})
    result = RrfRetriever(scripted, lane_depth=1).retrieve(dataclasses.replace(QUERY, limit=1))
    assert scripted.requests[0].limit == 1
    assert [i.doc_id for i in result.items] == [W[0]]


# --- empty, single and overlapping lanes ----------------------------------------------------------------------

def test_an_empty_lane_contributes_nothing():
    result = RrfRetriever(Scripted({"weakness": W})).retrieve(QUERY)
    assert [i.doc_id for i in result.items] == W
    assert result.items[0].score == pytest.approx(1 / 61)


def test_all_lanes_empty_gives_no_results():
    result = RrfRetriever(Scripted({})).retrieve(QUERY)
    assert result.items == () and result.status == "no_results"


def test_a_document_in_two_lists_gets_both_contributions_and_rises():
    lane_a, lane_b = Lane("a", ("weakness",), "lane a"), Lane("b", ("weakness",), "lane b")
    results = [(lane_a, lane_result(["cwe:X", "cwe:Y"], "weakness")), (lane_b, lane_result(["cwe:Y", "cwe:Z"], "weakness"))]
    fused = fuse_lane_results(results, query=QUERY, name="rrf-test", k=60)
    assert [i.doc_id for i in fused.result.items] == ["cwe:Y", "cwe:X", "cwe:Z"]
    y = fused.fused[0]
    assert y.rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert [(c.lane, c.lane_rank) for c in y.contributions] == [("a", 2), ("b", 1)]


def test_ties_prefer_the_smaller_best_lane_rank_then_lane_order_then_doc_id():
    lane_a, lane_b = Lane("a", ("weakness",), "lane a"), Lane("b", ("weakness",), "lane b")
    # X: rank 1 in A and rank 3 in B -> 1/61 + 1/63 = 0.032266 ; Y: rank 2 in both -> 2/62 = 0.032258.
    # X is higher by a small but real margin, so X comes first although Y is never worse than rank 2.
    results = [
        (lane_a, lane_result(["cwe:X", "cwe:Y"], "weakness")),
        (lane_b, lane_result(["cwe:Q", "cwe:Y", "cwe:X"], "weakness")),
    ]
    fused = fuse_lane_results(results, query=QUERY, name="t", k=60)
    assert [i.doc_id for i in fused.result.items] == ["cwe:X", "cwe:Y", "cwe:Q"]
    assert [i.score for i in fused.result.items] == pytest.approx([1 / 61 + 1 / 63, 2 / 62, 1 / 61])
    # exact tie (1/61 each, both best-rank 1): lane order decides, not doc_id ("cwe:B" is in the earlier lane)
    tie = fuse_lane_results(
        [(lane_a, lane_result(["cwe:B"], "weakness")), (lane_b, lane_result(["cwe:A"], "weakness"))],
        query=QUERY, name="t", k=60,
    )
    assert [i.doc_id for i in tie.result.items] == ["cwe:B", "cwe:A"]


def test_duplicate_lane_names_are_rejected():
    lane = Lane("dup", ("weakness",), "x")
    with pytest.raises(ValueError, match="unique"):
        fuse_lane_results([(lane, lane_result(["cwe:A"], "weakness")), (lane, lane_result(["cwe:B"], "weakness"))],
                          query=QUERY, name="t")


def test_a_repeated_document_inside_one_lane_counts_once():
    lane = Lane("a", ("weakness",), "lane a")
    # RetrievalResult itself forbids repeated documents, so the repeat guard is exercised through rrf_scores,
    # which is what fuse_lane_results uses for the numbers.
    assert rrf_scores([["cwe:A", "cwe:A", "cwe:B"]])["cwe:A"] == pytest.approx(1 / 61)
    single = fuse_lane_results([(lane, lane_result(["cwe:A"], "weakness"))], query=QUERY, name="t")
    assert [i.doc_id for i in single.result.items] == ["cwe:A"]
    with pytest.raises(ValueError, match="at most once"):
        lane_result(["cwe:A", "cwe:A"], "weakness")


# --- result metadata and contract --------------------------------------------------------------------------------

def test_the_result_documents_what_it_is_and_is_not():
    fused = RrfRetriever(Scripted({"weakness": W, "attack_pattern": P}), k=60).retrieve(QUERY)
    assert fused.retriever == "rrf-k60[scripted]"
    assert RRF_SCORE_MEANING in fused.query_notes
    assert "rrf k=60" in fused.backend_query and "limit_applied_after_fusion=10" in fused.backend_query
    assert "weakness=3" in fused.backend_query and "attack_pattern=2" in fused.backend_query


def test_lane_notes_are_carried_through_without_duplicates():
    class Noisy(Scripted):
        def retrieve(self, query):
            result = super().retrieve(query)
            return dataclasses.replace(result, query_notes=("low-information query: weak",))

    notes = RrfRetriever(Noisy({"weakness": W, "attack_pattern": P})).retrieve(QUERY).query_notes
    assert notes.count("low-information query: weak") == 1


def test_possibly_incomplete_propagates():
    class Partial(Scripted):
        def retrieve(self, query):
            return dataclasses.replace(super().retrieve(query), possibly_incomplete=query.doc_types == ("weakness",))

    assert RrfRetriever(Partial({"weakness": W})).retrieve(QUERY).possibly_incomplete


def test_construction_and_query_validation():
    scripted = Scripted({})
    with pytest.raises(ValueError):
        RrfRetriever(scripted, lanes=())
    with pytest.raises(ValueError):
        RrfRetriever(scripted, k=0)
    for depth in (0, 101, True, 2.5):
        with pytest.raises(ValueError):
            RrfRetriever(scripted, lane_depth=depth)
    retriever = RrfRetriever(scripted)
    with pytest.raises(InvalidRetrievalQueryError, match="doc_types"):
        retriever.retrieve(dataclasses.replace(QUERY, doc_types=("weakness",)))
    with pytest.raises(InvalidRetrievalQueryError):
        retriever.retrieve("text")
    assert isinstance(retriever, Retriever)


def test_the_default_lanes_are_the_two_specified_ones():
    assert LANES == (WEAKNESS_LANE, ATTACK_PATTERN_LANE)


# --- lane wrapper ---------------------------------------------------------------------------------------------

def test_a_lane_retriever_restricts_every_query_to_its_lane_and_is_named_after_it():
    scripted = Scripted({"weakness": W, "attack_pattern": P})
    wrapped = LaneRetriever(scripted, ATTACK_PATTERN_LANE)
    result = wrapped.retrieve(QUERY)
    assert wrapped.name == "scripted[attack_pattern]"
    assert [i.doc_id for i in result.items] == P
    assert scripted.requests[-1].doc_types == ("attack_pattern",)


# --- with the real semantic retriever (fake embedder) -------------------------------------------------------------

def test_fusion_over_the_semantic_retriever_is_deterministic_and_returns_each_document_once():
    with make_store(*small_corpus()) as store:
        semantic = SemanticRetriever(store, FakeEmbedder(dimension=256))
        rrf = RrfRetriever(semantic)
        query = RetrievalQuery(text="alpha attacker database request", limit=10)
        first, second = rrf.retrieve(query), rrf.retrieve(query)
        plain = semantic.retrieve(query)
    assert first == second
    ids = [i.doc_id for i in first.items]
    assert len(ids) == len(set(ids)) == 4
    assert sorted(ids) == sorted(i.doc_id for i in plain.items)
    # within each lane the fused order equals the semantic lane order
    for prefix in ("cwe:", "capec:"):
        assert [d for d in ids if d.startswith(prefix)] == [d for d in (i.doc_id for i in plain.items) if d.startswith(prefix)]
    assert first.items[0].score == pytest.approx(1 / 61)
