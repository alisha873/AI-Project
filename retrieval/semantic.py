"""
Phase 11 (Milestone 11C) - semantic (embedding) retrieval behind the existing contract.

`SemanticRetriever` implements `retrieval.contract.Retriever`: it takes a `RetrievalQuery` and returns a
`RetrievalResult` of `RetrievedItem`s, exactly like the two FTS5 retrievers. No contract field was
added or changed.

How a question is answered
--------------------------
1. The query gets the same input checks as `fts5-bm25-v2`: text with no searchable word (only stop
   words, numbers or punctuation) raises `InvalidRetrievalQueryError`; a query whose only searchable
   content is one plain word runs but carries a `LOW_INFORMATION_PREFIX` note. (The check is the
   v2 keyword compiler's lexical rule, reused unchanged, so both backends flag the same queries.)
2. The query text is embedded as a QUERY (for the approved model: with its recommended instruction).
   The `phrases` field is ignored (the text is embedded whole); a note says so.
3. Only chunks whose stored `doc_type` is in `query.doc_types` are eligible (None = all). The filter is
   applied to the real stored chunks BEFORE ranking, so unlike FTS5 there is no window that can hide a match.
4. Every eligible chunk gets a cosine score: its stored passage vector dotted with the query vector (both
   are unit length, so the dot product IS the cosine). Computed in float64 for reproducibility.
5. Chunks are ranked by (score descending, `doc_id`, ordinal) - the last two make ties deterministic.
6. Chunks are folded into source documents. Rule (documented, fixed): a document's score is its BEST chunk's
   score. Chunk scores are never summed, because a long document would win merely by having more chunks.
   Document order is (best score descending, `doc_id`). One `RetrievedItem` per document, with its best
   `max_chunks_per_item` chunks (default 3) as `ChunkMatch`es with chunk id, character span, text and cosine,
   so the evidence behind a rank is never lost.

What the scores mean
--------------------
A cosine score is only meaningful for ORDERING one result list from one model. For `bge-small-en-v1.5` the
model card says typical scores sit in roughly [0.6, 1], so an absolute value is not a probability or a
"relevant / not relevant" decision. A dense retriever always returns its nearest neighbours; it does not
abstain on its own (out-of-corpus questions still return documents; see the evaluation reports).

Snapshot: the retriever reads all chunks from the store once, when it is built (and obtains their vectors
from the cache or the embedder). Build a new retriever after the store changes. The store is never written.
"""

from __future__ import annotations

import numpy as np

from knowledge.schema import Chunk
from knowledge.store import KnowledgeStore, KnowledgeStoreError
from retrieval.contract import (
    LOW_INFORMATION_PREFIX,
    ChunkMatch,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)
from retrieval.embeddings import Embedder, EmbeddingError
from retrieval.fts5_v2 import compile_fts_expression_v2
from retrieval.vector_cache import CacheError, CacheReport, VectorCache

AGGREGATION = "best_chunk"
DEFAULT_MAX_CHUNKS_PER_ITEM = 3


def load_chunks(store: KnowledgeStore) -> list[Chunk]:
    """Every stored chunk, validated by the store, ordered by (doc_id, ordinal)."""
    try:
        chunks: list[Chunk] = []
        for doc_id in sorted(store.list_doc_ids()):
            chunks.extend(store.get_chunks(doc_id))
    except KnowledgeStoreError as exc:
        raise RetrievalBackendError(f"could not read chunks from the knowledge store: {exc}") from exc
    return chunks


class SemanticRetriever:
    """Embedding retriever over an open `KnowledgeStore` (see the module docstring)."""

    def __init__(
        self,
        store: KnowledgeStore,
        embedder: Embedder,
        *,
        cache: VectorCache | None = None,
        rebuild_cache: bool = False,
        max_chunks_per_item: int = DEFAULT_MAX_CHUNKS_PER_ITEM,
    ) -> None:
        if not isinstance(store, KnowledgeStore):
            raise TypeError("SemanticRetriever requires a KnowledgeStore")
        if isinstance(max_chunks_per_item, bool) or not isinstance(max_chunks_per_item, int) or max_chunks_per_item < 1:
            raise ValueError("max_chunks_per_item must be a positive integer")
        self._embedder = embedder
        self._max_chunks = max_chunks_per_item
        self._chunks = load_chunks(store)
        self._doc_ids = sorted({chunk.doc_id for chunk in self._chunks})
        self.cache_report: CacheReport | None = None
        try:
            if cache is not None:
                matrix, self.cache_report = cache.sync(self._chunks, embedder, rebuild=rebuild_cache)
            elif self._chunks:
                matrix = embedder.embed_passages([chunk.text for chunk in self._chunks])
            else:
                matrix = np.zeros((0, embedder.config.dimension), dtype=np.float32)
        except (EmbeddingError, CacheError) as exc:
            raise RetrievalBackendError(f"could not obtain chunk embeddings: {exc}") from exc
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self._types = [chunk.doc_type for chunk in self._chunks]

    @property
    def name(self) -> str:
        return f"semantic-{self._embedder.config.short_name}"

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len(self._doc_ids)

    def _audit_string(self, query: RetrievalQuery) -> str:
        config = self._embedder.config
        types = "all" if query.doc_types is None else ",".join(query.doc_types)
        return (
            f"semantic model={config.model_id}@{config.model_revision} "
            f"embedding_fingerprint={config.fingerprint()[:16]} similarity={config.similarity} "
            f"query_instruction={config.query_instruction!r} passage_prefix={config.passage_prefix!r} "
            f"aggregation={AGGREGATION} max_chunks_per_item={self._max_chunks} doc_types={types}"
        )

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        if not isinstance(query, RetrievalQuery):
            raise InvalidRetrievalQueryError("retrieve() requires a RetrievalQuery")
        compiled = compile_fts_expression_v2(query)  # same "is there anything to search" rule as fts5-bm25-v2
        notes = [note for note in compiled.notes if note.startswith(LOW_INFORMATION_PREFIX)]
        if query.phrases:
            notes.append("the phrases field is ignored by the semantic backend; the text is embedded as a whole")

        try:
            query_vector = np.asarray(self._embedder.embed_queries([query.text])[0], dtype=np.float64)
        except EmbeddingError as exc:
            raise RetrievalBackendError(f"could not embed the query: {exc}") from exc

        allowed = None if query.doc_types is None else set(query.doc_types)
        eligible = [i for i, doc_type in enumerate(self._types) if allowed is None or doc_type in allowed]
        scores = (self._matrix[eligible] @ query_vector).tolist() if eligible else []
        order = sorted(
            range(len(eligible)),
            key=lambda j: (-scores[j], self._chunks[eligible[j]].doc_id, self._chunks[eligible[j]].ordinal),
        )

        per_doc: dict[str, list[tuple[float, Chunk]]] = {}
        for j in order:  # best chunk first, so the first chunk of each document is its best
            chunk = self._chunks[eligible[j]]
            per_doc.setdefault(chunk.doc_id, []).append((scores[j], chunk))

        items = []
        for rank, (doc_id, matches) in enumerate(list(per_doc.items())[: query.limit], start=1):
            best_score, best = matches[0]
            items.append(
                RetrievedItem(
                    rank=rank,
                    doc_id=doc_id,
                    source=best.source,
                    source_id=best.source_id,
                    doc_type=best.doc_type,
                    title=best.title,
                    source_url=best.source_url,
                    score=best_score,
                    chunks=tuple(
                        ChunkMatch(
                            chunk_id=chunk.chunk_id, ordinal=chunk.ordinal, char_start=chunk.char_start,
                            char_end=chunk.char_end, text=chunk.text, score=score,
                        )
                        for score, chunk in matches[: self._max_chunks]
                    ),
                )
            )
        return RetrievalResult(
            query=query,
            retriever=self.name,
            items=tuple(items),
            status="ok" if items else "no_results",
            possibly_incomplete=False,
            backend_query=self._audit_string(query),
            query_notes=tuple(notes),
        )
