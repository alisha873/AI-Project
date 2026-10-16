"""
Phase 11 (Milestone 11C) - the approved real embedding model, run locally.

Approved model (decided before this module was written; see PROJECT_STATUS.md):

    BAAI/bge-small-en-v1.5   at the exact commit   5c38ec7c405ec4b44b94cc5a9bb96e735b38267a

`model.safetensors` must have this full SHA-256, which was read from Hugging Face's own
file metadata for that exact commit and matches the digest recorded in the preflight:

    3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad

Safety rules enforced here (each has a test):
1. The model directory is checked BEFORE anything is loaded: the weights' SHA-256 must equal the
   pinned value, every small config/tokenizer file must match its pinned git blob id, and no
   pickle-capable or otherwise unapproved weight file (`pytorch_model.bin`, `*.pt`, `*.pkl`, ONNX, ...)
   may be present at all.
2. Weights are loaded as safetensors only (`use_safetensors=True`), with `trust_remote_code=False`
   and `local_files_only=True`: inference never contacts the network. (Downloading is a separate,
   explicit step in `scripts/download_embedding_model.py`.)
3. The loaded model is INTERROGATED, not trusted: its embedding dimension, pooling mode and maximum
   sequence length are read from the loaded object and must equal the expected values, and a probe
   encode must return a finite vector of the expected dimension.

The model directory lives outside version control (default `data/models/...`, ignored by git) and is
configurable with the `AUTOSHIELD_MODEL_DIR` environment variable or `--model-dir` in the scripts.

BGE query instruction: the model card recommends the instruction below for short queries that search
long passages, and no instruction for passages. AutoShield's questions are short and its chunks are
long, so the instruction is on by default. Whether it helps on this corpus is only an observation
from development data; the default was fixed before any 11C result was seen.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from retrieval.embeddings import BaseEmbedder, EmbeddingConfig, EmbeddingError

REPO_ROOT = Path(__file__).resolve().parents[1]

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_WEIGHTS_SHA256 = "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
BGE_DIMENSION = 384
BGE_MAX_SEQ_LENGTH = 512
BGE_POOLING = "cls"

WEIGHTS_FILE = "model.safetensors"

# Small files at the pinned commit and their git blob ids (SHA-1 of "blob <size>\0<bytes>"), taken
# from the Hugging Face file metadata of that commit. They pin the tokenizer and configuration.
BGE_SMALL_FILES: Mapping[str, str] = {
    "1_Pooling/config.json": "1da070f0993788c8851a652170ff013aa54dd519",
    "config.json": "3992bf890728a92476c700e6e657fd696eb94160",
    "config_sentence_transformers.json": "dcb0c0d97d09b930d13600b1a773ddb27e441aab",
    "modules.json": "952a9b81c0bfd99800fabf352f69c7ccd46c5e43",
    "sentence_bert_config.json": "ea85692bff64b0d1917833c31ddbca8ab10f5455",
    "special_tokens_map.json": "a8b3208c2884c4efb86e49300fdd3dc877220cdf",
    "tokenizer.json": "688882a79f44442ddc1f60d70334a7ff5df0fb47",
    "tokenizer_config.json": "37fca74771bc76a8e01178ce3a6055a0995f8093",
    "vocab.txt": "fb140275c155a9c7c5a3b3e0e77a9e839594a938",
}
ALLOWED_DOWNLOAD_PATTERNS: tuple[str, ...] = (WEIGHTS_FILE, *BGE_SMALL_FILES)

# Any of these under the model directory means "do not load": pickle can run code, and the other
# formats are not the approved safetensors weights.
FORBIDDEN_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".h5", ".msgpack", ".onnx", ".ot")

MODEL_DIR_ENV = "AUTOSHIELD_MODEL_DIR"
RUNTIME_PACKAGES = (
    "sentence-transformers", "transformers", "tokenizers", "torch", "numpy", "safetensors", "huggingface-hub",
)


class ModelFilesError(EmbeddingError):
    """The local model directory is missing, incomplete, altered, or contains unapproved files."""


class EmbeddingDependencyError(EmbeddingError):
    """The optional embedding packages are not installed (see requirements-embeddings.txt)."""


def default_model_dir() -> Path:
    """`$AUTOSHIELD_MODEL_DIR` if set, else `data/models/bge-small-en-v1.5-<revision[:12]>` in the repository."""
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override)
    return REPO_ROOT / "data" / "models" / f"bge-small-en-v1.5-{BGE_REVISION[:12]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The git object id of a file's bytes, so small files can be checked against Hugging Face's metadata."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()  # noqa: S324 - git object id, not security


@dataclass(frozen=True)
class ModelFiles:
    """The result of a successful `verify_model_directory`."""

    model_dir: Path
    weights_sha256: str
    file_digests: tuple[tuple[str, str], ...]  # (relative name, SHA-256), weights included, sorted


def verify_model_directory(model_dir: str | Path) -> ModelFiles:
    """
    Verify a local model directory against the pinned values (see the module docstring).

    Raises `ModelFilesError` naming the first problem; nothing is loaded or executed.
    """
    root = Path(model_dir)
    if not root.is_dir():
        raise ModelFilesError(f"model directory {root} does not exist (run scripts/download_embedding_model.py)")
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ModelFilesError(
                f"unapproved or pickle-capable file {path.relative_to(root).as_posix()!r} in the model directory; "
                "only safetensors weights and small JSON/text files are accepted"
            )
    weights = root / WEIGHTS_FILE
    if not weights.is_file():
        raise ModelFilesError(f"{WEIGHTS_FILE} is missing from {root}")
    weights_digest = sha256_file(weights)
    if weights_digest != BGE_WEIGHTS_SHA256:
        raise ModelFilesError(
            f"{WEIGHTS_FILE} has SHA-256 {weights_digest}, expected {BGE_WEIGHTS_SHA256}; "
            "the file is not the pinned model"
        )
    digests = [(WEIGHTS_FILE, weights_digest)]
    for name, blob_id in sorted(BGE_SMALL_FILES.items()):
        path = root / name
        if not path.is_file():
            raise ModelFilesError(f"required file {name!r} is missing from {root}")
        if git_blob_sha1(path) != blob_id:
            raise ModelFilesError(f"{name!r} does not match the pinned revision (git blob id mismatch)")
        digests.append((name, sha256_file(path)))
    return ModelFiles(root, weights_digest, tuple(sorted(digests)))


def runtime_versions() -> dict[str, str]:
    """Installed versions of the packages that can change the last digits of a vector."""
    versions: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def bge_config(files: ModelFiles, *, query_instruction: str = BGE_QUERY_INSTRUCTION) -> EmbeddingConfig:
    """The pinned `EmbeddingConfig` for the approved model (the expected, to-be-verified values)."""
    return EmbeddingConfig(
        model_id=BGE_MODEL_ID,
        model_revision=BGE_REVISION,
        dimension=BGE_DIMENSION,
        query_instruction=query_instruction,
        passage_prefix="",
        max_seq_length=BGE_MAX_SEQ_LENGTH,
        pooling=BGE_POOLING,
        normalize=True,
        similarity="cosine",
        weights_sha256=files.weights_sha256,
        file_digests=files.file_digests,
    )


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    The approved BGE model, loaded from a verified local directory through sentence-transformers.

    Construction verifies the files, loads on CPU, and checks the loaded model against the expected
    configuration (see the module docstring). It never uses the network.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        query_instruction: str = BGE_QUERY_INSTRUCTION,
        batch_size: int = 16,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise EmbeddingError("batch_size must be a positive integer")
        files = verify_model_directory(default_model_dir() if model_dir is None else model_dir)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingDependencyError(
                "sentence-transformers is not installed; install requirements-embeddings.txt"
            ) from exc
        try:
            model = SentenceTransformer(
                str(files.model_dir),
                device="cpu",
                trust_remote_code=False,
                local_files_only=True,
                model_kwargs={"use_safetensors": True},
            )
        except Exception as exc:  # the library raises many types; none may be silently ignored
            raise EmbeddingError(f"could not load the model from {files.model_dir}: {type(exc).__name__}: {exc}") from exc
        model.eval()  # inference mode: no dropout, so repeated runs give identical vectors
        self._model = model
        self._batch_size = batch_size
        self._check_loaded_model()
        super().__init__(bge_config(files, query_instruction=query_instruction))
        probe = self.embed_passages(["probe sentence for dimension and normalisation check"])
        if probe.shape != (1, BGE_DIMENSION):
            raise EmbeddingError(f"probe embedding has shape {probe.shape}")  # pragma: no cover - validated already

    def _check_loaded_model(self) -> None:
        model = self._model
        # Newer sentence-transformers renamed this method; prefer the new name, fall back to the old.
        get_dimension = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension != BGE_DIMENSION:
            raise EmbeddingError(f"the loaded model reports dimension {dimension}, expected {BGE_DIMENSION}")
        if model.max_seq_length != BGE_MAX_SEQ_LENGTH:
            raise EmbeddingError(
                f"the loaded model reports max_seq_length {model.max_seq_length}, expected {BGE_MAX_SEQ_LENGTH}"
            )
        # Pooling is a module of the sentence-transformers pipeline; in 6.x its mode is `pooling_mode`.
        modes = [
            str(getattr(module, "pooling_mode"))
            for module in model._modules.values()
            if type(module).__name__ == "Pooling" and hasattr(module, "pooling_mode")
        ]
        if modes != [BGE_POOLING]:
            raise EmbeddingError(f"the loaded model reports pooling {modes}, expected ['{BGE_POOLING}']")

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        return runtime_versions()

    def _encode(self, prepared: list[str]) -> object:
        try:
            return self._model.encode(
                prepared,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(f"encoding failed: {type(exc).__name__}: {exc}") from exc

    def token_lengths(self, texts: Sequence[str], *, as_passages: bool = True) -> list[int]:
        """Token counts (special tokens included, no truncation) with the model's own tokenizer."""
        prepared = [(self.config.passage_prefix if as_passages else self.config.query_instruction) + t for t in texts]
        if not prepared:
            return []
        encoded = self._model.tokenizer(prepared, add_special_tokens=True, truncation=False)
        return [len(ids) for ids in encoded["input_ids"]]
