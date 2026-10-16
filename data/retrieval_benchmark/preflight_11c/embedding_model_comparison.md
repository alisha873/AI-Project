# Embedding model comparison for Milestone 11C (preflight, nothing installed or run)

Review date: **2026-10-16** (all web facts below were read on that date; Hugging Face repositories can change, which is why revisions are pinned by commit hash).
Status: **research only.** No model was downloaded, no embedding computed, no package installed. Nothing here shows that any model is better on AutoShield's data.

## 1. What the model has to do

`EMBEDDING_MODEL_REQUIREMENTS` in the project spec asks for: local execution, a clear licence, a dimension and maximum input length that suit a 1200-character chunk, reproducibility through pinning, CPU-only operation, no fine-tuning, and evaluation against the frozen FTS5 baseline.

A plain-language picture: an embedding model turns a chunk of text into a list of numbers (a vector). Chunks and questions with similar meaning should get vectors that point in similar directions. We then rank chunks by the angle between the question vector and each chunk vector (cosine similarity).

## 2. Candidates and verified facts

All "verified" items come from the publisher's Hugging Face repository (model card, API metadata, or raw config files at the pinned commit). Items marked **estimated** or **not verified** are explicitly not measured.

| Property | `sentence-transformers/all-MiniLM-L6-v2` | `BAAI/bge-small-en-v1.5` | `intfloat/e5-small-v2` |
|---|---|---|---|
| Publisher namespace / model card | `sentence-transformers` organisation on Hugging Face. [card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `BAAI` organisation on Hugging Face. [card](https://huggingface.co/BAAI/bge-small-en-v1.5) | `intfloat` namespace on Hugging Face (paper authors per the card: Wang, Yang, Huang, Jiao, Yang, Jiang, Majumder, Wei). [card](https://huggingface.co/intfloat/e5-small-v2); paper [arXiv:2212.03533](https://arxiv.org/pdf/2212.03533.pdf) (cited by the card). Affiliations were not checked. |
| Commit to pin (verified) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (last modified 2026-06-01) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` (last modified 2024-02-22) | `ffb93f3bd4047442299a41ebb6fa998a38507c52` (last modified 2025-02-17) |
| Licence (card metadata) | Apache-2.0 | MIT | MIT |
| Parameters (safetensors header) | 22,713,728 (22.7 M) | 33,360,512 (33.4 M) | 33,360,512 (33.4 M) |
| Vector dimension | 384 | 384 | 384 |
| Max input length | **256 word pieces** (`sentence_bert_config.json`; card: longer input is truncated; training used 128) | **512** (`sentence_bert_config.json`) | **512** (`sentence_bert_config.json`; card: "truncated to at most 512 tokens") |
| Pooling | mean (card code) | **CLS token** (`1_Pooling/config.json`) | mean (`1_Pooling/config.json`) |
| Normalisation | L2-normalised (card code; `modules.json` has a Normalize step) | L2-normalised (`modules.json` has a Normalize step) | card examples normalise explicitly (`normalize_embeddings=True`); its `modules.json` was not fetched, so we would normalise ourselves |
| Query / passage prefixes | none | optional query instruction `Represent this sentence for searching relevant passages: `; **no** instruction on passages; card says v1.5 works without it, with reduced retrieval ability | **required**: `query: ` for questions and `passage: ` for chunks; card says performance degrades without them |
| Similarity | cosine (= dot product after normalisation) | cosine; card warns scores sit roughly in [0.6, 1], so an absolute score is not a similarity judgement | cosine; card says scores sit roughly in 0.7-1.0 and only relative order matters |
| Language | English | English | English only (card limitation) |
| Weights file (verified) | `model.safetensors` 90,868,376 B (86.7 MiB), sha256 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` | `model.safetensors` 133,466,304 B (127.3 MiB), sha256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad` | `model.safetensors` 133,466,304 B (127.3 MiB), sha256 `45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1` |
| Other formats in the repo | `pytorch_model.bin` (pickle), `tf_model.h5`, `rust_model.ot`, ONNX (FP32 `onnx/model.onnx` 90,405,214 B plus O1-O4 and int8 variants), OpenVINO | `pytorch_model.bin` (pickle), ONNX FP32 `onnx/model.onnx` 133,093,490 B (sha256 `828e1496...`) | `pytorch_model.bin` (pickle), `tf_model.h5`, ONNX (`model.onnx` 133,093,468 B, `onnx/model_O4.onnx`, int8), OpenVINO |
| Disk (weights only) | about 87 MiB (safetensors) | about 127 MiB | about 127 MiB |
| RAM at runtime | **estimated** about 0.1 GiB of float32 weights plus library overhead; not measured | **estimated** about 0.15 GiB of float32 weights plus overhead; not measured | same as bge-small; not measured |
| CPU feasibility | Verified only that the model is small; **speed not measured**. 247 chunks is a tiny workload | same | same |
| Popularity (context only) | ~216 M downloads | ~59 M downloads | ~0.88 M downloads |
| Gated / needs token | no | no | no |

Notes on the table:
- "Parameters" come from the safetensors metadata (`I64: 512` is a small position-id buffer, `F32` is the real weights).
- Memory use and speed are **estimates, not measurements**. They will be measured in 11C itself (record peak memory and encode time on the 247 chunks).
- The e5 card states 12 layers and an embedding size of 384. A layer count for bge-small was not separately read.
- Training-data licences were not assessed. The model licence is not a statement about every training source (MiniLM's card lists about 1.17 billion training pairs from many public datasets).

## 3. Does a 1200-character chunk fit?

We chunk with `max_chars=1200`, `overlap_chars=120`, giving 247 chunks from 23 documents. Measured in memory today (no tokenizer, no model): a **lower bound** on token counts uses a regular expression that counts words and punctuation marks, +2 for the special start/end tokens. Real WordPiece tokenisation splits rare technical words further, so real counts are higher.

| Statistic over 247 chunks | Value |
|---|---|
| characters, min / median / mean / max | 129 / 961 / 886.7 / 1200 |
| token lower bound, min / median / mean / max | 30 / 182 / 173.7 / 296 |
| chunks whose lower bound already exceeds 256 | **7 (2.8 %)** (certain truncation for MiniLM) |
| chunks whose lower bound exceeds 512 | 0 |
| chunks that would exceed 256 if real counts are 1.2x / 1.3x / 1.4x the lower bound (an **assumption**, not measured) | 74 (30 %) / 106 (43 %) / 122 (49 %) |

Reading: with a 256-token limit, MiniLM would quietly ignore the end of a noticeable share of chunks, and the missing text often holds the facets (mitigations, consequences) that questions ask about. The two 512-token models are very unlikely to truncate any chunk (maximum lower bound 296; even 1.4x is about 415). The exact figures must be re-measured with the real tokenizer in 11C before any claim is made.

## 4. Technical suitability versus demonstrated quality

- **Technical suitability** (what this document establishes): licence, size, input length, offline use, pinning, prefix rules.
- **Demonstrated quality on AutoShield**: **none yet.** Publishers' benchmark numbers (MTEB and others) measure other corpora and are deliberately not quoted here. They do not tell us how a model behaves on 23 CWE/CAPEC records. Only the planned, pre-registered evaluation can do that.

## 5. Recommendation (awaiting your approval)

**Preferred: `BAAI/bge-small-en-v1.5` at commit `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.**
Why: 512-token input fits every measured chunk; retrieval-oriented model with a documented optional query instruction; MIT licence; safetensors weights available; no prefix required on passages (less to get wrong than e5); 384-dimension vectors.

**Fallback: `intfloat/e5-small-v2` at commit `ffb93f3bd4047442299a41ebb6fa998a38507c52`.**
Why: same size class, 512-token input, MIT licence, safetensors available; it is a different model family with a different training recipe, so if bge-small behaves badly on this corpus the fallback is not likely to fail for the same reason. Cost: it **requires** `query: ` / `passage: ` prefixes, so a unit test must prove the prefixes are applied.

**Not recommended as the primary model: `sentence-transformers/all-MiniLM-L6-v2`.** It is the smallest and the most widely used, with no prefix rules and the simplest behaviour, but its 256-token limit truncates at least 2.8 % of our chunks for certain and plausibly 30-49 %. It could still be run as an optional third comparison, but only with that caveat stated beside its numbers.

This recommendation is a **technical-suitability judgement**, not a claim that bge-small will beat FTS5, nor that it will beat the others.

## 6. Open uncertainties (not verified)

- Real token counts with each model's own tokenizer (needs the tokenizer files; no download was done).
- Encode speed and peak memory on this Windows machine.
- Whether a query instruction helps bge on this corpus (the plan answers this with development data only, see the protocol).
- Training-data provenance and licensing of each model.
- e5's `modules.json` (the normalisation step) was not read.
