"""
Optional test of the real approved model (BAAI/bge-small-en-v1.5). It is SKIPPED, with a stated reason, when the
verified model directory is not on this machine; every other 11C test runs without a model and without a network.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from retrieval.embeddings_st import (
    BGE_DIMENSION,
    BGE_QUERY_INSTRUCTION,
    ModelFilesError,
    SentenceTransformerEmbedder,
    default_model_dir,
    verify_model_directory,
)
from retrieval.semantic import SemanticRetriever
from retrieval.contract import RetrievalQuery
from retrieval.vector_cache import VectorCache
from tests.retrieval_helpers import make_store, small_corpus


def _model_available() -> tuple[bool, str]:
    try:
        verify_model_directory(default_model_dir())
    except ModelFilesError as exc:
        return False, f"the verified model directory is not available ({exc}); run scripts/download_embedding_model.py"
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False, "sentence-transformers is not installed (requirements-embeddings.txt)"
    return True, ""


AVAILABLE, REASON = _model_available()
pytestmark = pytest.mark.skipif(not AVAILABLE, reason=REASON)


@pytest.fixture(scope="module")
def blocked_network_attempts():
    """Block and record every network connection attempt for the whole module."""
    attempts: list[str] = []
    saved = (socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection)

    def blocker(name):
        def blocked(*args, **kwargs):
            attempts.append(name)
            raise OSError("network access is blocked by the offline test")
        return blocked

    socket.socket.connect, socket.socket.connect_ex = blocker("connect"), blocker("connect_ex")
    socket.getaddrinfo, socket.create_connection = blocker("getaddrinfo"), blocker("create_connection")
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo, socket.create_connection = saved


@pytest.fixture(scope="module")
def embedder(blocked_network_attempts):
    return SentenceTransformerEmbedder()


def test_the_loaded_model_matches_its_pinned_configuration(embedder):
    config = embedder.config
    assert (config.dimension, config.pooling, config.max_seq_length, config.similarity) == (BGE_DIMENSION, "cls", 512, "cosine")
    assert config.query_instruction == BGE_QUERY_INSTRUCTION and config.passage_prefix == ""
    assert embedder.runtime_versions["sentence-transformers"] == "6.1.0"


def test_vectors_are_finite_unit_length_and_semantically_ordered(embedder):
    passages = embedder.embed_passages([
        "SQL injection occurs when untrusted input is concatenated into a database query.",
        "A path traversal weakness lets an attacker read files outside the intended directory.",
        "Bananas are rich in potassium.",
    ])
    query = embedder.embed_queries(["how do attackers manipulate database queries through user input"])
    assert passages.shape == (3, BGE_DIMENSION) and passages.dtype == np.float32 and np.isfinite(passages).all()
    assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-4)
    similarity = passages @ query[0]
    assert similarity[0] > similarity[1] > similarity[2]


def test_repeated_embedding_is_stable_to_float_precision(embedder):
    text = ["A cross-site scripting weakness lets script run in another user's browser."]
    assert np.allclose(embedder.embed_passages(text), embedder.embed_passages(text), atol=1e-6)


def test_no_chunk_of_the_test_corpus_exceeds_the_model_length_limit(embedder):
    with make_store(*small_corpus()) as store:
        texts = [c.text for doc_id in store.list_doc_ids() for c in store.get_chunks(doc_id)]
    assert max(embedder.token_lengths(texts)) < embedder.config.max_seq_length


def test_semantic_retrieval_with_the_real_model_and_a_cache_round_trip(embedder, tmp_path):
    cache = VectorCache(tmp_path / "cache")
    with make_store(*small_corpus()) as store:
        first = SemanticRetriever(store, embedder, cache=cache)
        second = SemanticRetriever(store, embedder, cache=cache)
        query = RetrievalQuery(text="attacker sends a crafted request to the database", limit=4)
        assert first.cache_report.created == first.chunk_count and second.cache_report.reused == second.chunk_count
        assert first.retrieve(query).items == second.retrieve(query).items
        assert len(first.retrieve(query).items) == 4


def test_inference_made_no_network_attempt(embedder, blocked_network_attempts):
    embedder.embed_queries(["a final inference after everything else"])
    assert blocked_network_attempts == []
