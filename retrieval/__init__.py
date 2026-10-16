"""
Phase 11 (Milestone 11A) - retrieval contract, keyword backend, query builder
and evaluation harness.

- `retrieval.contract`      typed query/result/error types and the `Retriever` protocol.
- `retrieval.fts5`          `Fts5Retriever`: keyword retrieval over `KnowledgeStore.search()`.
- `retrieval.query_builder` deterministic queries from Phase 8/9 analysis output.
- `retrieval.evaluation`    Recall@k / MRR harness and the fixed benchmark loader.

Milestone 11B adds:

- `retrieval.fts5_v2`       `Fts5RetrieverV2`: hardened keyword retrieval (the 11A retriever is unchanged).
- `retrieval.lanes`         the weakness / attack-pattern lanes and the written dual-lane specification.
- `retrieval.benchmark_v2`, `retrieval.evaluation_v2`  the versioned benchmark loader and the grouped harness.

No embeddings, vector index, fusion, reranking or generation exist in this package.
"""

from retrieval.contract import (
    ChunkMatch,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    Retriever,
)
from retrieval.fts5 import Fts5Retriever
from retrieval.fts5_v2 import Fts5RetrieverV2
from retrieval.lanes import ATTACK_PATTERN_LANE, LANES, WEAKNESS_LANE, Lane

__all__ = [
    "ATTACK_PATTERN_LANE",
    "ChunkMatch",
    "Fts5Retriever",
    "Fts5RetrieverV2",
    "LANES",
    "Lane",
    "WEAKNESS_LANE",
    "InvalidRetrievalQueryError",
    "RetrievalBackendError",
    "RetrievalError",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedItem",
    "Retriever",
]
