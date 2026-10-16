"""Tests for the semantic retriever (fake embedder; synthetic documents; no model and no network)."""

from __future__ import annotations

import numpy as np
import pytest

from retrieval.contract import (
    LOW_INFORMATION_PREFIX,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalQuery,
    Retriever,
)
from retrieval.embeddings import EmbeddingError, FakeEmbedder
from retrieval.semantic import AGGREGATION, SemanticRetriever, load_chunks
from retrieval.vector_cache import VectorCache
from tests.retrieval_helpers import make_doc, make_store, small_corpus

DIM = 256  # wide enough that hash collisions between the few test words are very unlikely


def embedder(**kwargs) -> FakeEmbedder:
    return FakeEmbedder(dimension=DIM, **kwargs)


@pytest.fixture
def store():
    with make_store(*small_corpus()) as opened:
        yield opened


def multi_chunk_doc(source_id: str, paragraphs: list[str], **kwargs):
    """A document long enough to be split into several chunks (the chunker works on 1200-character windows)."""
    return make_doc(source_id, f"Long {source_id}", "\n\n".join(paragraphs), **kwargs)


def filler(word: str, count: int = 60) -> str:
    return " ".join([word] * count)


# --- basic ranking and scores -----------------------------------------------------------------------

def test_it_satisfies_the_retriever_protocol_and_names_the_model(store):
    retriever = SemanticRetriever(store, embedder())
    assert isinstance(retriever, Retriever)
    assert retriever.name.startswith("semantic-") and "@" in retriever.name
    assert retriever.document_count == 4 and retriever.chunk_count == len(load_chunks(store))


def test_the_most_similar_document_ranks_first_with_cosine_scores(store):
    fake = embedder()
    retriever = SemanticRetriever(store, fake)
    result = retriever.retrieve(RetrievalQuery(text="alpha alpha database query request"))
    assert result.status == "ok" and result.items[0].doc_id == "cwe:CWE-1"
    assert [item.rank for item in result.items] == list(range(1, len(result.items) + 1))
    scores = [item.score for item in result.items]
    assert scores == sorted(scores, reverse=True)
    # the score is exactly the dot product of the unit query vector and the unit chunk vector
    chunks = {c.doc_id: c for c in load_chunks(store)}
    query_vector = fake.embed_queries(["alpha alpha database query request"])[0].astype(np.float64)
    chunk_vector = fake.embed_passages([chunks["cwe:CWE-1"].text])[0].astype(np.float64)
    assert result.items[0].score == pytest.approx(float(query_vector @ chunk_vector), abs=1e-6)
    assert all(-1.0 - 1e-6 <= s <= 1.0 + 1e-6 for s in scores)


def test_every_document_is_returned_once_up_to_the_limit(store):
    retriever = SemanticRetriever(store, embedder())
    everything = retriever.retrieve(RetrievalQuery(text="alpha database request", limit=10))
    assert sorted(item.doc_id for item in everything.items) == sorted(store.list_doc_ids())
    limited = retriever.retrieve(RetrievalQuery(text="alpha database request", limit=2))
    assert [i.doc_id for i in limited.items] == [i.doc_id for i in everything.items[:2]]
    assert not limited.possibly_incomplete


def test_repeated_queries_give_identical_results(store):
    retriever = SemanticRetriever(store, embedder())
    query = RetrievalQuery(text="gamma attacker internal service")
    assert retriever.retrieve(query) == retriever.retrieve(query)
    assert retriever.retrieve(query) == SemanticRetriever(store, embedder()).retrieve(query)


def test_a_semantic_retriever_finds_documents_that_share_no_query_keyword_only_if_vectors_say_so(store):
    # With the hash-bag fake the vectors are keyword-driven, so this only checks the plumbing:
    # a query with no overlap still returns a full, ordered, finite ranking (it does not return "no results").
    result = SemanticRetriever(store, embedder()).retrieve(RetrievalQuery(text="zebra yacht xylophone"))
    assert result.status == "ok" and len(result.items) == 4
    assert all(np.isfinite(item.score) for item in result.items)


# --- filters -----------------------------------------------------------------------------------------

def test_doc_type_filter_restricts_results(store):
    retriever = SemanticRetriever(store, embedder())
    weaknesses = retriever.retrieve(RetrievalQuery(text="alpha database request", doc_types=("weakness",)))
    patterns = retriever.retrieve(RetrievalQuery(text="alpha database request", doc_types=("attack_pattern",)))
    assert {i.doc_type for i in weaknesses.items} == {"weakness"} and len(weaknesses.items) == 2
    assert {i.doc_type for i in patterns.items} == {"attack_pattern"} and len(patterns.items) == 2
    assert "doc_types=weakness" in weaknesses.backend_query


def test_a_filtered_ranking_equals_the_unfiltered_ranking_restricted_to_that_type(store):
    retriever = SemanticRetriever(store, embedder())
    query = "alpha attacker database request"
    full = retriever.retrieve(RetrievalQuery(text=query))
    only = retriever.retrieve(RetrievalQuery(text=query, doc_types=("attack_pattern",)))
    assert [i.doc_id for i in only.items] == [i.doc_id for i in full.items if i.doc_type == "attack_pattern"]


def test_a_filter_with_no_matching_documents_gives_no_results():
    with make_store(*small_corpus()[:2]) as only_weaknesses:
        result = SemanticRetriever(only_weaknesses, embedder()).retrieve(
            RetrievalQuery(text="alpha database request", doc_types=("attack_pattern",))
        )
    assert result.status == "no_results" and result.items == ()


def test_an_empty_store_gives_no_results():
    with make_store() as empty:
        retriever = SemanticRetriever(empty, embedder())
        assert retriever.retrieve(RetrievalQuery(text="alpha database")).status == "no_results"
        assert retriever.chunk_count == 0


# --- chunk-to-document folding ------------------------------------------------------------------------

def test_a_document_with_several_chunks_appears_once_scored_by_its_best_chunk():
    fake = embedder()
    docs = [
        multi_chunk_doc("CWE-10", [filler("apple"), filler("banana"), filler("cherry"), filler("target phrase words")]),
        make_doc("CWE-11", "Other", "completely unrelated gamma delta epsilon"),
    ]
    with make_store(*docs) as store:
        chunks = [c for c in load_chunks(store) if c.doc_id == "cwe:CWE-10"]
        assert len(chunks) >= 3, "the test document must span several chunks"
        result = SemanticRetriever(store, fake, max_chunks_per_item=2).retrieve(
            RetrievalQuery(text="target phrase words")
        )
    long_item = next(i for i in result.items if i.doc_id == "cwe:CWE-10")
    assert [i.doc_id for i in result.items].count("cwe:CWE-10") == 1
    assert result.items[0].doc_id == "cwe:CWE-10"
    assert long_item.score == max(m.score for m in long_item.chunks) == long_item.chunks[0].score
    assert len(long_item.chunks) == 2  # capped by max_chunks_per_item
    assert [m.score for m in long_item.chunks] == sorted((m.score for m in long_item.chunks), reverse=True)
    assert "target phrase words" in long_item.chunks[0].text


def test_scores_are_never_summed_across_chunks():
    # Many mediocre chunks must not beat one excellent chunk: score == best single chunk, so <= 1.
    docs = [multi_chunk_doc("CWE-20", [filler("alpha target")] * 8), make_doc("CWE-21", "Exact", "alpha target")]
    with make_store(*docs) as store:
        result = SemanticRetriever(store, embedder()).retrieve(RetrievalQuery(text="alpha target"))
    assert all(item.score <= 1.0 + 1e-6 for item in result.items)


def test_chunk_matches_carry_exact_text_and_offsets():
    docs = [multi_chunk_doc("CWE-30", [filler("one"), filler("two"), filler("three")])]
    with make_store(*docs) as store:
        stored = {c.chunk_id: c for c in load_chunks(store)}
        result = SemanticRetriever(store, embedder()).retrieve(RetrievalQuery(text="two two two"))
    for match in result.items[0].chunks:
        chunk = stored[match.chunk_id]
        assert (match.text, match.char_start, match.char_end, match.ordinal) == (
            chunk.text, chunk.char_start, chunk.char_end, chunk.ordinal
        )


def test_identical_documents_tie_and_are_ordered_by_doc_id():
    docs = [make_doc("CWE-42", "B", "same words here"), make_doc("CWE-41", "A", "same words here"),
            make_doc("CWE-43", "C", "same words here")]
    with make_store(*docs) as store:
        result = SemanticRetriever(store, embedder()).retrieve(RetrievalQuery(text="same words here"))
    assert [i.doc_id for i in result.items] == ["cwe:CWE-41", "cwe:CWE-42", "cwe:CWE-43"]
    assert len({i.score for i in result.items}) == 1


# --- provenance, audit string, notes ---------------------------------------------------------------------

def test_items_carry_source_provenance_and_the_audit_string_names_the_model(store):
    fake = embedder(query_instruction="Represent: ")
    result = SemanticRetriever(store, fake).retrieve(RetrievalQuery(text="alpha database request"))
    top = result.items[0]
    assert (top.source, top.source_id, top.title) == ("cwe", "CWE-1", "Alpha Weakness")
    assert top.source_url == "https://example.invalid/CWE-1"
    assert result.retriever == f"semantic-{fake.config.short_name}"
    for fragment in (fake.config.model_id, fake.config.model_revision, fake.config.fingerprint()[:16],
                     f"aggregation={AGGREGATION}", "cosine", "'Represent: '"):
        assert fragment in result.backend_query


def test_the_query_instruction_reaches_the_model_for_queries_only(store):
    fake = embedder(query_instruction="Represent: ")
    SemanticRetriever(store, fake).retrieve(RetrievalQuery(text="  alpha database  "))
    assert fake.received_queries == ["Represent: alpha database"]
    assert all(not p.startswith("Represent:") for p in fake.received_passages)


def test_a_one_word_query_is_flagged_and_an_empty_one_is_refused(store):
    retriever = SemanticRetriever(store, embedder())
    assert any(n.startswith(LOW_INFORMATION_PREFIX) for n in retriever.retrieve(RetrievalQuery(text="alpha")).query_notes)
    assert not retriever.retrieve(RetrievalQuery(text="alpha database request")).query_notes
    for text in ("the of and", "!!!"):
        with pytest.raises(InvalidRetrievalQueryError):
            retriever.retrieve(RetrievalQuery(text=text))


def test_phrases_are_ignored_with_a_note(store):
    retriever = SemanticRetriever(store, embedder())
    plain = retriever.retrieve(RetrievalQuery(text="alpha database request"))
    with_phrases = retriever.retrieve(RetrievalQuery(text="alpha database request", phrases=("gamma attacker",)))
    assert [i.doc_id for i in with_phrases.items] == [i.doc_id for i in plain.items]
    assert any("phrases" in n for n in with_phrases.query_notes)


# --- cache integration ---------------------------------------------------------------------------------------

def test_cached_and_uncached_retrievers_return_identical_results(store, tmp_path):
    cache = VectorCache(tmp_path / "cache")
    query = RetrievalQuery(text="gamma attacker internal service")
    uncached = SemanticRetriever(store, embedder()).retrieve(query)
    first = SemanticRetriever(store, embedder(), cache=cache)
    second = SemanticRetriever(store, embedder(), cache=cache)
    assert first.cache_report.created == first.chunk_count and second.cache_report.reused == second.chunk_count
    assert first.retrieve(query).items == uncached.items == second.retrieve(query).items


def test_rebuild_cache_recomputes_every_vector(store, tmp_path):
    cache = VectorCache(tmp_path / "cache")
    SemanticRetriever(store, embedder(), cache=cache)
    rebuilt = SemanticRetriever(store, embedder(), cache=cache, rebuild_cache=True)
    assert rebuilt.cache_report.created == rebuilt.chunk_count and rebuilt.cache_report.reused == 0


# --- errors -------------------------------------------------------------------------------------------------------

def test_construction_and_call_validation(store):
    with pytest.raises(TypeError):
        SemanticRetriever(object(), embedder())
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            SemanticRetriever(store, embedder(), max_chunks_per_item=bad)
    with pytest.raises(InvalidRetrievalQueryError):
        SemanticRetriever(store, embedder()).retrieve("alpha")


def test_embedding_failures_become_backend_errors(store):
    class Failing(FakeEmbedder):
        def embed_passages(self, texts):
            raise EmbeddingError("passages exploded")

    with pytest.raises(RetrievalBackendError, match="passages exploded"):
        SemanticRetriever(store, Failing(dimension=DIM))

    class QueryFailing(FakeEmbedder):
        def embed_queries(self, texts):
            raise EmbeddingError("query exploded")

    retriever = SemanticRetriever(store, QueryFailing(dimension=DIM))
    with pytest.raises(RetrievalBackendError, match="query exploded"):
        retriever.retrieve(RetrievalQuery(text="alpha database request"))
