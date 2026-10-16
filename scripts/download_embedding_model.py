"""
One-time, explicit download of the approved embedding model (Phase 11, Milestone 11C).

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.download_embedding_model

This is the ONLY step in 11C that uses the network. Evaluation and inference never do (they load
the verified local copy with `local_files_only=True`).

What it does, in order:
1. Asks Hugging Face for the file metadata of the EXACT pinned commit (not `main`) and checks that
   the commit is the pinned one and that `model.safetensors` has the pinned full SHA-256.
2. Downloads only the safetensors weights and the small tokenizer/config files at that commit
   (never `pytorch_model.bin`, ONNX or any other format) into `--model-dir`
   (default `data/models/bge-small-en-v1.5-<revision>`, or `$AUTOSHIELD_MODEL_DIR`).
3. Verifies the downloaded directory with `retrieval.embeddings_st.verify_model_directory`
   (weights SHA-256, small-file git blob ids, no unapproved files).

The model directory is ignored by git. Telemetry is disabled for this process.
Exit status: 0 verified; 1 any failure; 2 command-line misuse.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from retrieval.embeddings import EmbeddingError  # noqa: E402
from retrieval.embeddings_st import (  # noqa: E402
    ALLOWED_DOWNLOAD_PATTERNS,
    BGE_MODEL_ID,
    BGE_REVISION,
    BGE_WEIGHTS_SHA256,
    WEIGHTS_FILE,
    default_model_dir,
    verify_model_directory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.download_embedding_model",
        description="Download and verify the pinned BGE embedding model (the only 11C step that uses the network).",
    )
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="target directory (default: $AUTOSHIELD_MODEL_DIR or data/models/bge-small-en-v1.5-<rev>)")
    parser.add_argument("--verify-only", action="store_true",
                        help="do not download; only verify an existing directory (no network)")
    return parser


def _check_remote_metadata() -> None:
    from huggingface_hub import HfApi

    info = HfApi().model_info(BGE_MODEL_ID, revision=BGE_REVISION, files_metadata=True)
    if info.sha != BGE_REVISION:
        raise EmbeddingError(f"Hugging Face resolved {BGE_REVISION} to {info.sha}")
    weights = [s for s in info.siblings if s.rfilename == WEIGHTS_FILE]
    if len(weights) != 1 or weights[0].lfs is None or weights[0].lfs.sha256 != BGE_WEIGHTS_SHA256:
        raise EmbeddingError("the remote SHA-256 of model.safetensors differs from the pinned value")
    print(f"remote metadata OK: {BGE_MODEL_ID} @ {info.sha}; {WEIGHTS_FILE} sha256 {BGE_WEIGHTS_SHA256}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target = args.model_dir if args.model_dir is not None else default_model_dir()
    try:
        if not args.verify_only:
            from huggingface_hub import snapshot_download

            _check_remote_metadata()
            print(f"downloading {len(ALLOWED_DOWNLOAD_PATTERNS)} approved file(s) to {target} ...")
            snapshot_download(
                repo_id=BGE_MODEL_ID,
                revision=BGE_REVISION,
                allow_patterns=list(ALLOWED_DOWNLOAD_PATTERNS),
                local_dir=str(target),
            )
        files = verify_model_directory(target)
    except (EmbeddingError, OSError, ImportError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # network and hub errors come in many types; report, never hide
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"verified model directory: {files.model_dir}")
    print(f"  {WEIGHTS_FILE} sha256 {files.weights_sha256} (equals the pinned value)")
    for name, digest in files.file_digests:
        print(f"  {name:36s} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
