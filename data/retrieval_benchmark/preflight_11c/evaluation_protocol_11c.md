# Evaluation protocol for Milestone 11C (frozen before any 11C code exists)

Status: **proposed, to be followed when 11C is approved.** Nothing in this protocol has been run. No retriever or model has touched the v3 questions.

## 1. The three kinds of data, and what each may be used for

| Set | Contents | Role in 11C | Status |
|---|---|---|---|
| **v1** | 22 cases (`retrieval_benchmark_v1.json`, unchanged) | development | results already observed in earlier phases |
| **v2 development** | 33 cases (22 carried from v1 plus 11 new, including 4 behaviour cases) | development | results observed |
| **v2 held-out** | 24 cases (20 scored plus 4 behaviour) | **development extension, "observed"**. It may inform model/configuration choices, but must **never again be described as untouched** | **spent**: it was run in an earlier phase |
| **v3 final held-out** (`benchmark_11c.json`) | 30 scored cases + 6 behaviour cases | the **only** final test of 11C | **frozen, never run** |

Rules:
1. **All design decisions are made on development data only** (v1, v2 development, and, with the "observed" label, v2 held-out): which candidate model, query instruction yes/no for bge, chunk aggregation rule, tokenizer/truncation settings, normalisation, any score threshold.
2. The v3 scored cases are run **once**, after the implementation, model commit and configuration are frozen (section 3).
3. Behaviour cases (low-information and out-of-corpus questions) are reported **separately** and never mixed into Recall or MRR.
4. v1 and v2 are not edited. Their historical results stay valid as historical results.

## 2. The frozen v3 benchmark

| Field | Value |
|---|---|
| File | `data/retrieval_benchmark/preflight_11c/benchmark_11c.json` |
| Format / id | `autoshield.retrieval_benchmark.v3` / `mitre-subset-retrieval-benchmark-v3-final-heldout` |
| **SHA-256** | `31e26a84813783c235cfa899b02d42a1d5eec006e691f2a1825f21b14c144d10` |
| Corpus fingerprint | `813692f1fce5a5039a9e29a3ac3bb63cc1bb44b0ce5464d6be13d30cf71a2af5` (CWE 4.20 + CAPEC 3.9, 23 documents, recomputed from the pinned archives; identical to the v2 value) |
| Scored cases | 30 (CWE-only 16, CAPEC-only 10, both lanes 4; 23 single-label, 7 multi-label) |
| Behaviour cases | 6 (2 low-information, 4 out-of-corpus) |
| Evidence | 40 verbatim source quotes, each verified present in the stated record and section; 30 exclusivity checks over the full corpus |
| Verifier | `verify_benchmark_11c.py` (stdlib plus the repo's read-only corpus builder; runs no retriever, writes nothing). Re-run it before the final run: it must print the same SHA-256 and "ALL CHECKS PASSED" |

### How labels were made (and what that does and does not give you)
- Written by the AI assistant from the **source text of the records only**, in a single pass, **before any retrieval was run on these questions.** Each relevant label cites a verbatim quote; rejected near-misses are listed with reasons.
- **Not** blind, **not** independently annotated, **not** human-reviewed; no second reviewer exists. Errors made by the author may be correlated with errors of other AI reviewers.
- The author had seen v1/v2 retrieval results in earlier phases; it had seen **no** result for any v3 question.
- Strict labels follow the facet rule; **7 cases are flagged as close calls** and **8 carry lenient extras** (other-lane records the sources themselves declare related). The strict set is primary; a "strict plus lenient" label set is a sensitivity analysis only.
- **Audit trail:** three label changes were made while authoring, after mechanical checks, **before the freeze and before any retrieval** (RB3-F13 and RB3-F20 gained a paraphrased-advice record; RB3-F26's check was scoped). They are recorded in the file under `label_audit`.
- **Coverage gap:** 22 of the 23 records appear as relevant in at least one scored case; **CAPEC-242 does not**. Question styles: 25 descriptive, 3 identifier, 2 short.
- Residual risk: exclusivity checks use exact phrases, so another record might state the same advice in different words (that is how two of the three changes above were found). Other such misses may remain.

### The 7 disputed v2 labels
Unchanged: RB-002, RB-014, RB-015, RB-020, RB2-H02, RB2-H04, RB2-H09. No human-review packet was prepared and none is required. v2 results continue to be reported with both the adjudicated and the strict "agreed by author and every reviewer" label sets, as the v2 evaluator already does.

### If a labelling defect is found after retrieval has been observed
Do **not** edit the file. Document the defect (case id, what is wrong, evidence), keep the reported results as they are, and create a new benchmark version (v4) for any re-test. A defect found **before** the final run (for example by re-running the verifier) must still be handled by a new version, not a silent edit, because the SHA-256 above is already public in this project's records.

## 3. Freeze checklist (all items must be true before the v3 run)

1. Implementation code committed or fingerprinted (the comparison script already prints a code fingerprint; the report must include it).
2. Model id **and commit hash** fixed; `model.safetensors` SHA-256 verified.
3. Embedding configuration fixed: prefixes, max length, truncation, pooling, normalisation, dtype, chunker version and parameters (`char-structure-v1`, 1200/120).
4. Aggregation (chunk -> document) and fusion parameters fixed (RRF k = 60, deterministic tie-breaks).
5. Any behaviour-case rule or score threshold fixed using development behaviour cases only.
6. Package versions pinned and recorded (`pip freeze`).
7. Corpus fingerprint equals the value in section 2; benchmark SHA-256 equals the value in section 2.
8. A short **pre-registration note** is written down (systems, metrics, label sets, decision rules) **before** the run.
9. Full test suite passes.

## 4. Systems to compare (same frozen questions, same corpus)

| Id | System | Notes |
|---|---|---|
| S1 | FTS5 v1 (`fts5-bm25-v1`) | historical baseline |
| S2 | FTS5 v2 (`fts5-bm25-v2`, top-2-chunk aggregation) | current lexical retriever |
| S3 | semantic, **both lanes together** | one ranking over all 23 documents |
| S4 | semantic, **lane-specific** | the same model, run once for weaknesses and once for attack patterns; evaluated per lane |
| S5 | **RRF** of the two lane lists (k = 60) | same lanes as S4, fused |

Semantic retrieval is chunk-level: every chunk gets a cosine score; chunks are **folded to source documents** (document score = best chunk by default), keeping the chunk provenance in the result.

Pre-registered caveat for S5: the two lanes contain **disjoint** documents, so each document is in exactly one list. RRF then cannot reorder documents **within** a lane; it only decides how the two lanes interleave. A fusion that can actually add information (lexical + semantic over the same lane) is a **different** system and is **out of scope** for 11C unless separately approved.

## 5. Metrics and reporting

- **Recall@1, @3, @5** and **MRR**, at the **source-document level** (duplicate chunks of one record are folded first), exactly as the existing evaluators define them. Diagnostic Recall@10 is **not** a selection criterion (23 documents make it nearly meaningless).
- Reported **per lane** (CWE-only cases, CAPEC-only cases, mixed cases), **per category**, and **in aggregate**, with the random-ranking comparator.
- **Counts first:** with 30 questions, report "number of questions answered correctly at rank <= k" next to percentages. A change of one or two questions is within the noise of such a set.
- Compare systems **question by question** (which questions each one gets right). **No significance claims**, no confidence-interval claims presented as proof. Any interval that is shown must be labelled exploratory.
- Two label sets: **strict (primary)** and **strict plus lenient (sensitivity)**.
- Behaviour cases reported separately with the same "handled / not handled" definitions as the v2 evaluator. A dense retriever always returns nearest neighbours, and the model cards say scores sit in a narrow band (bge about 0.6-1, e5 about 0.7-1), so **no abstention claim is allowed unless a rule chosen on development data** is stated in advance.
- Required in every report: corpus fingerprint; benchmark id and SHA-256; model id and commit; safetensors SHA-256; embedding configuration (prefixes, max length, pooling, normalisation); chunker version; implementation (code) fingerprint; package versions; machine description; wall-clock encode time and peak memory.
- Mandatory wording in every report: *"a 23-document, hand-curated corpus and 30 AI-labelled questions; results do not show generalisation, and are not independently reviewed."*

## 6. Run policy

- The final run on v3 happens **once**. If it fails for a technical reason that produced no results (for example a crash), the cause is documented and the run may be repeated; once any v3 result has been **seen**, the system is not changed and re-run to look for a better score.
- A change made after seeing v3 results makes v3 "spent" for that variant; further testing requires a new benchmark version.
- Reporting must include systems that **lose** to the lexical baseline. No selective reporting.

## 7. What this protocol does not claim
- That semantic retrieval will beat FTS5 (or the reverse).
- That 30 or 57 questions on 23 documents say anything about a full CWE/CAPEC corpus or about incident evidence.
- That the labels are correct beyond the evidence quoted from the sources.
