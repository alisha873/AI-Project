"""
Phase 11 (Milestone 11D) - grounded answers over retrieved security knowledge.

- `rag.context`   structured, citable context (observed evidence, hypotheses, reference passages).
- `rag.pipeline`  incident analysis -> queries -> production retriever (FTS5 v1) -> context.

Further modules (prompt, citation validation, answer generation) are added step by step.
The production retriever is `fts5-bm25-v1`; semantic retrieval and RRF are experimental
and are not used by this package.
"""
