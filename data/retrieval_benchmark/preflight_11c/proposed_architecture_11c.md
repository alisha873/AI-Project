# Proposed architecture for Milestone 11C (design only, **not implemented**)

Nothing here exists in code. It is a proposal for you to approve, change or reject. It was written after reading `retrieval/contract.py`, `retrieval/lanes.py`, `retrieval/evaluation*.py`, `knowledge/schema.py`, `knowledge/store.py` and the 11B architecture lock.

## 1. The idea in plain words

Today a question is answered by matching **words** (SQLite FTS5 with BM25). 11C adds a second way: turn the question and every text chunk into **meaning vectors** with a small embedding model and rank chunks by closeness. We then check, on a fresh frozen question set, whether that is better, worse or no different than the word-matching baseline. We do **not** assume it is better.

```
question --(prefix)--> embedding model --> query vector --+
                                                          |  cosine
247 chunks --(prefix)--> embedding model --> chunk vectors-+--> chunk scores
                                                          |
                         fold chunks to source documents (best chunk wins; chunk provenance kept)
                                                          |
              per lane: weakness list | attack-pattern list --> RRF (k=60) --> final ranking
```

## 2. Shared model, two lanes

- **One** embedding model instance is shared by both lanes. The lanes differ only by the filter `doc_type` (the existing `lanes.py` already defines `WEAKNESS_LANE` and `ATTACK_PATTERN_LANE` over the `weakness` / `attack_pattern` document types). No schema change is needed; the data already carries `doc_type`.
- The semantic retriever supports `RetrievalQuery.doc_types`, so a lane is simply "run the same retriever with this filter", the same way FTS5 lanes work.

## 3. Semantic retrieval at chunk level, folded to documents

1. Embed each chunk's stored text (default: the chunk text exactly as stored). A variant "document title + chunk text" is a development-set experiment, not the default.
2. Score = cosine between L2-normalised vectors (a dot product).
3. **Fold to source:** the document's score is its best chunk (default). Alternatives (for example sum of the best two, as in FTS5 v2) are development-only ablations, fixed before the final run.
4. Return the existing `RetrievalResult` unchanged: one `RetrievedItem` per document, with `ChunkMatch` entries (chunk id, ordinal, character span, text, score) so **provenance is retained** exactly as for FTS5.

**Contract compatibility: no change needed.** Existing fields already carry what a semantic retriever must say: `retriever` (name such as `semantic-bge-small-en-v1.5@5c38ec7`), `backend_query` (an audit string describing the prefix, model, revision and metric), `query_notes` (for example "the phrases field is ignored by this backend" and the existing low-information warning), `possibly_incomplete` (False; the whole corpus is scored). If a later need arises (for example exposing which lane produced a fused item), it should be added as an optional, defaulted field, as `query_notes` was in 11B, so old callers keep working.

## 4. Fusion with RRF

The specification already lives in `retrieval/lanes.py` and is kept: `score(d) = sum over lanes of 1 / (60 + rank_in_lane(d))`, deterministic ties (higher raw score, smaller best rank, lane order, then `doc_id`).

**Disjoint-lane caveat (important):** the weakness and attack-pattern lanes contain different documents. Each document appears in exactly one list, so its RRF score is just `1/(60+rank)`. RRF therefore **cannot improve the order within a lane**; it only decides how the two lanes are interleaved (rank 1 of each lane first, then rank 2, and so on, with ties broken by raw score). A fusion that really combines evidence about the same document would need two *different rankers over the same lane* (for example BM25 + embeddings); that is a possible later step and is **not** part of this proposal.

## 5. Where do the vectors live? Three options

| Option | How it works | Pros | Cons |
|---|---|---|---|
| **A. Recompute in memory every run** | encode all 247 chunks at start-up | simplest; nothing to invalidate | pays the model load and encode time on every run and every test; harder to compare runs byte for byte |
| **B. Fingerprinted derived cache (proposed)** | after encoding, save a float32 matrix plus a small JSON manifest next to the corpus; reuse it only if the manifest's fingerprint matches | simple (one `.npy` and one `.json`), reproducible, no new native dependency, easy to delete and rebuild; `allow_pickle=False` so loading cannot run code | we must get the invalidation rules right |
| **C. Dedicated vector store** (FAISS, Chroma, sqlite vector extension, ...) | index with approximate or exact search | scales to large corpora | 247 vectors fit in 0.36 MiB; extra native dependency and moving parts for no benefit at this size |

**Proposed: Option B** (and Option A remains available as the automatic fallback when the cache is missing or stale). For scale: 247 chunks x 384 dimensions x 4 bytes = **379,392 bytes**.

Do **not** put vectors into the source `knowledge` tables. The corpus is source-derived and fingerprinted; vectors are *derived* data and must stay separable and disposable.

### Cache invalidation keys (all stored in the manifest; any mismatch = rebuild)
1. **Chunk text:** SHA-256 of each chunk's text.
2. **Chunk identity:** `chunk_id`, character span, ordinal; plus the chunker version and parameters (`char-structure-v1`, 1200 / 120).
3. **Corpus fingerprint** (document-level cross-check).
4. **Model id and exact commit hash**, and the SHA-256 of `model.safetensors`.
5. **Tokenisation and preprocessing:** tokenizer files' hashes, maximum length, truncation, lower-casing, the exact prefix strings and any title/header injection.
6. **Dimension and dtype.**
7. **Pooling, normalisation and similarity** setting.
8. **Library versions** (`sentence-transformers`, `transformers`, `torch`) are *recorded*. Because the e5 card itself notes that different library versions can cause tiny numerical differences, a version mismatch triggers a warning and, for the final evaluation, a rebuild.

Queries are **never** cached across runs; they are cheap and may depend on the instruction setting.

## 6. Planned code layout (proposal only)

| New file | Purpose |
|---|---|
| `retrieval/embedding.py` | `Embedder` interface; `SentenceTransformerEmbedder` (local, offline, pinned revision, safetensors only); deterministic `FakeEmbedder` for tests |
| `retrieval/vector_cache.py` | build, validate, load the derived cache described above |
| `retrieval/semantic.py` | `SemanticRetriever` implementing the existing `Retriever` protocol |
| `retrieval/fusion.py` | RRF combiner (a `Retriever` over lane retrievers) following the `lanes.py` specification |
| `retrieval/benchmark_v3.py` | strict loader for the v3 format (mirrors `benchmark_v2.py`; keeps v2's loader untouched) |
| `scripts/run_retrieval_semantic_comparison.py` | runs S1-S5 on development sets; refuses the v3 final set unless a **freeze record** (pre-registration note, benchmark SHA-256, model commit, config hash, code fingerprint) is supplied |
| `tests/test_*` | contract conformance with the fake embedder, cache invalidation, prefix handling, RRF ties, no-network guarantees |

Existing modules (`fts5.py`, `fts5_v2.py`, `evaluation*.py`, benchmarks v1/v2) stay unchanged.

## 7. Evaluation plan (summary; full rules in `evaluation_protocol_11c.md`)
- Systems: FTS5 v1, FTS5 v2, semantic over both lanes, lane-specific semantic, RRF of lane-specific lists.
- Same frozen questions; Recall@1/3/5 and MRR at source level; per lane and aggregate; strict labels primary, lenient as sensitivity.
- Each report records: corpus fingerprint, benchmark version and SHA-256, model id and commit, embedding configuration, implementation (code) fingerprint, package versions, encode time and peak memory.
- Design choices (candidate model, bge query instruction on or off, aggregation rule, title injection, any threshold) are made on development data only, then frozen.

## 8. Behaviour cases and abstention
A dense retriever always returns *something*, and the cards say similarity scores occupy a narrow band, so a raw cosine cut-off is fragile. Keep the existing lexical low-information gate (`query_builder`) in front, and treat out-of-corpus questions as **reported observations** (top scores and what was returned) unless a rule is chosen on development behaviour cases and written down before the final run.

## 9. Testing without network
All unit tests use the fake embedder (deterministic vectors derived from a hash of the text), so the base suite never needs the model packages or internet. One optional integration test is skipped unless a local model directory and an environment variable are present.

## 10. Caveats that must travel with every result
1. 23 documents and 247 chunks cannot establish generalisation to the full CWE/CAPEC catalogues, or to real incident evidence.
2. v3 labels are AI-authored, single-pass, **not blind and not independently reviewed**.
3. Semantic similarity is not evidence of attack causality, and retrieval is reference knowledge, not incident evidence.
4. Neither a good nor a bad result here says anything about the pen-testing or GNN components.

## 11. Not in this proposal
RAG answer generation, hybrid (same-lane BM25 + embedding) fusion, re-ranking with a cross-encoder, fine-tuning, GPU use, a vector database, any change to the stored corpus or to the dashboard.
