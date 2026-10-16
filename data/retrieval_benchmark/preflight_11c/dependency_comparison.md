# Dependency comparison for Milestone 11C (preflight, nothing installed)

Review date: **2026-10-16**. Sources: PyPI project metadata (JSON), the sentence-transformers installation and efficiency pages, the `huggingface_hub` environment-variable reference, and each model's Hugging Face repository (see `embedding_model_comparison.md`). No package was installed, upgraded or removed; `requirements.txt` and the virtual environment are untouched.

## 1. Current environment (verified by inspection, read-only)

| Item | Value |
|---|---|
| Python | 3.12.0 (pyenv, `.venv`) |
| `torch` | **2.14.1+cpu** already installed (CPU build, `torch.version.cuda` is None, CUDA unavailable, 16 threads). `torch` folder is about 502 MB |
| `numpy` | installed |
| Absent today | `sentence_transformers`, `transformers`, `tokenizers`, `onnxruntime`, `safetensors`, `huggingface_hub`, `sklearn`, `scipy`, `fastembed`, `optimum` |

`requirements.txt` lists `torch` without a version. The 1323-test suite does not need any of the absent packages.

## 2. Verified package facts (PyPI metadata, reviewed 2026-10-16)

| Package | Latest release seen | Python | Declared dependencies (runtime, no extras) |
|---|---|---|---|
| `sentence-transformers` | 6.1.0 (2026-09-18), Apache-2.0 | >=3.10 | `transformers>=5.0.0,<6.0.0`, `tokenizers>=0.19`, `huggingface-hub>=1.3.0,<2.0.0`, `torch>=2.2`, `numpy>=1.24.0`, `scikit-learn>=1.1.0`, `scipy>=1.0.0`, `typing_extensions>=4.10.0`, `tqdm>=4.0.0` |
| `huggingface-hub` | 2.2.0 (2026-10-08), Apache-2.0 | >=3.10 | `click`, `filelock`, `fsspec`, `hf-xet` (on x86_64/AMD64/arm64), `httpx2`, `packaging`, `pyyaml`, `tqdm`, `typing-extensions` |
| `onnxruntime` | 1.31.0, MIT | **>=3.11** | `flatbuffers`, `numpy>=1.21.6`, `packaging`, `protobuf>=4.25.8` |
| `tokenizers` | 0.23.3 | >=3.10 | `huggingface-hub>=0.16.4,<3.0` |
| `safetensors` | 0.8.0 | >=3.10 | none listed in the metadata we read |
| `fastembed` | 0.9.0 (2026-10-07), Apache-2.0 | >=3.10 | `huggingface-hub>=0.20,<2.0`, `loguru`, `mmh3`, `numpy`, `onnxruntime` (version ranges per Python), `pillow`, `py-rust-stemmers`, `requests`, `tokenizers`, `tqdm` |
| `transformers` | **not verified** | - | The registry request for its metadata was blocked by the tool's review, so it was not retried. What we do know: sentence-transformers 6.1.0 requires `transformers>=5.0,<6`, and its installation page recommends "Python 3.10+, PyTorch 2.2+, and transformers v5.0+". `transformers`' own dependencies and size are **not verified** |

Observations that matter for pinning:
1. **Version ceiling clash.** `huggingface-hub` has reached 2.x, but sentence-transformers 6.1.0 and fastembed 0.9.0 both require `<2.0`. An unpinned install would silently settle on a 1.x release. This is exactly the kind of drift that exact pins prevent.
2. **Fast churn.** sentence-transformers went 5.7.0 (2026-08-06) -> 6.0.0 (2026-08-18) -> 6.0.1 -> 6.1.0 (2026-09-18) and now requires `transformers` 5.x. Upgrades can change behaviour, so a frozen evaluation must record exact versions.
3. **Python floor.** `onnxruntime` 1.31.0 needs Python 3.11+; the project's 3.12.0 satisfies it.
4. **Wheel sizes were not verified.** The registry JSON for the binary packages was truncated by the fetch tool, and I did not retry. Any size statement for `scikit-learn`, `scipy`, `onnxruntime`, `tokenizers` or `transformers` would be a guess, so none is made.

## 3. The options

### Option A - `sentence-transformers` on the existing CPU PyTorch  (recommended)
- **New packages to install:** `sentence-transformers`, `transformers`, `tokenizers`, `huggingface-hub`, `scikit-learn`, `scipy`, `safetensors` (pulled in by transformers; not verified directly) plus small helpers. `torch` and `numpy` are already present and satisfy the minimums (`torch>=2.2`, `numpy>=1.24`).
- **Complexity:** low. The model repositories are already sentence-transformers repositories (modules for transformer, pooling, normalisation), so the library applies the right pooling and normalisation from the repo's own config files. Fewer places for us to make a numerical mistake.
- **Windows CPU:** supported; the installation page documents plain `pip install`. Windows caveat from the Hugging Face docs: without developer mode, the cache cannot use symlinks and may duplicate large files (`HF_HUB_DISABLE_SYMLINKS_WARNING=1` hides the warning, and downloading to a plain local directory avoids the issue).
- **Offline:** load from a local directory; and `HF_HUB_OFFLINE=1` stops all Hub HTTP calls (documented). Telemetry can be switched off with `HF_HUB_DISABLE_TELEMETRY=1` (or `DO_NOT_TRACK`).
- **Reproducibility:** pin the model commit, verify the SHA-256 of `model.safetensors`, pin package versions, record them in every evaluation report.
- **Testability without network:** put a small `Embedder` interface in front of the library and test everything else with a deterministic fake embedder. The real model is covered by one optional test that is skipped unless a local model directory exists.
- **Contract compatibility:** the library is hidden behind our own retriever class that returns the existing `RetrievalResult`; no change to the contract is forced.
- **Maintenance cost (two students):** moderate. One mainstream library, but a large dependency tree that moves quickly (see observation 2).

### Option B - sentence-transformers with its ONNX backend (`sentence-transformers[onnx]`)
- Verified: the `onnx` extra adds `optimum-onnx[onnxruntime]`; the efficiency page says the ONNX backend uses an ONNX file in the repo if present and **otherwise converts the model**. It still needs torch, transformers and everything in Option A, **so it is not a smaller install.** It adds `onnxruntime` and `optimum-onnx`.
- Benefit is CPU speed, which is irrelevant for 247 chunks and a few dozen queries. Cost is another layer of moving parts and a possible silent export step. **Not recommended now.**

### Option C - direct `onnxruntime` + `tokenizers` + `numpy`
- Smallest **runtime** set (no torch at inference, no transformers), and the ONNX file for bge-small (`onnx/model.onnx`, 133,093,490 B, sha256 `828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35`) is verified to exist in the pinned repo.
- But **we** would have to write and test tokenisation settings (truncation to 512, padding), attention-mask handling, **CLS pooling** (bge) or mean pooling (e5), and L2 normalisation. The ONNX file is a different artifact from the safetensors weights, so equivalence of its output to the PyTorch model would need its own check; we cannot assume it.
- The smaller install does **not** remove the need to download the model once, and `torch` is already installed here anyway, so the saving is mostly theoretical in this project.
- **Not recommended** unless a deployment target without PyTorch appears.

### Option D - `fastembed`
- Verified: ONNX-based, Apache-2.0, depends on `onnxruntime`, `tokenizers`, `huggingface-hub<2.0`, `loguru`, `mmh3`, `pillow`, `py-rust-stemmers`, `requests`, `numpy`. It has its own supported-model list and download logic.
- **Not verified:** which of our candidates it supports, which file/revision it fetches, and whether we can pin a commit. Handing model selection and download to a wrapper weakens our pinning story. **Not recommended.**

## 4. Model-file safety

Each candidate repository contains a pickle-format `pytorch_model.bin` **and** `model.safetensors` (verified in the file lists). Python pickle files can execute arbitrary code when loaded; safetensors files hold only tensors. Rules for 11C:
1. Download **only** the safetensors weights and the small config/tokenizer files (use `allow_patterns` in `snapshot_download`); never download or load `pytorch_model.bin`, `tf_model.h5`, `rust_model.ot` or ONNX/OpenVINO files.
2. Pass the exact commit hash as `revision` (a branch name such as `main` can move).
3. Check `model.safetensors` against the SHA-256 recorded in `embedding_model_comparison.md` before first use; record the hash in the evaluation report.
4. Load with `local_files_only` / `HF_HUB_OFFLINE=1`. Request safetensors loading explicitly (for example `use_safetensors=True` through the model kwargs); the exact keyword must be confirmed by a test when the packages are installed.
5. Keep the model directory **outside git** (about 90-130 MB); commit only its pinned id, commit hash and file hashes.

## 5. Recommendation (awaiting your approval)

**Option A, with exact pins.**

| Item | Proposal |
|---|---|
| Package | `sentence-transformers==6.1.0` (the version reviewed today) |
| Resolution | In the approved 11C step, first run `pip install --dry-run` against a **new, separate** file `requirements-embeddings.txt` (not `requirements.txt`) to see the full resolved set; then pin **every** resolved package with `==` (and hashes via `pip-compile --generate-hashes` or `pip install --require-hashes`), including `transformers`, `tokenizers`, `huggingface-hub` (will be 1.x because of the `<2.0` cap), `safetensors`, `scikit-learn`, `scipy` |
| Torch | keep the installed `torch 2.14.1+cpu`; record it; do not let the resolver replace it with a CUDA build |
| Optionality | the embedding stack is an **optional** dependency set; the base suite must keep passing without it (fake embedder in tests) |
| Evidence | every evaluation report records `pip freeze` output, the model commit and the safetensors SHA-256 |
| Environment | `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` for evaluation runs, after the one-time approved download |

**Honest cost:** this adds roughly seven new packages including two large compiled scientific ones; their sizes were not verified here, so the real install size is **unknown until the dry-run**. If that size is unacceptable, Option C is the fallback design.
