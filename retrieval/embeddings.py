"""
Phase 11 (Milestone 11C) - the embedding interface.

An embedding model turns a piece of text into a list of numbers (a vector), so
that texts with a similar meaning get vectors that point in similar directions.
This module defines the small, typed interface the rest of 11C uses, so a
real model and a test double are interchangeable:

    embedder.embed_passages(["chunk text", ...])  -> float32 array (n, dimension)
    embedder.embed_queries(["question", ...])     -> float32 array (n, dimension)
    embedder.config                               -> the pinned `EmbeddingConfig`

Passages and queries are different "modes". For the approved model
(`BAAI/bge-small-en-v1.5`) the model card recommends putting an instruction in
front of a SHORT QUERY that is searching LONG PASSAGES, and nothing in front of
the passages. `EmbeddingConfig` records both strings; `prepare_queries` and
`prepare_passages` are the only places that apply them, so a passage can never
be embedded as if it were a query by accident.

What is checked (and what is not assumed)
-----------------------------------------
Every returned matrix passes `validate_embeddings`: it must be a 2-D array of
real numbers with one row per input and exactly `config.dimension` columns, no
NaN, no infinity, and (when `config.normalize`) every row must have unit length.
A real model's dimension, pooling and maximum length are READ from the loaded
model and compared with the expected configuration before the model is used;
nothing is taken from the model card on trust (see `retrieval.embeddings_st`).

Fingerprints
------------
`EmbeddingConfig.fingerprint()` is the SHA-256 of a canonical JSON description
of everything that changes the numbers an embedder produces: model id, exact
revision, file digests (weights and tokenizer/config files), dimension,
prefixes, maximum length, pooling, normalisation, similarity and dtype. It uses
no Python `hash()`, so it is the same on every run and machine. Package
versions are NOT part of it; they are recorded separately (`runtime_versions`)
because they may change the last digits of a vector without changing its
meaning. The vector cache compares both.

Model identity versus search configuration
------------------------------------------
Model identity (id, revision, weights) says WHICH network produces vectors. The
rest of the fingerprint says HOW text is turned into network input (prefixes,
length, pooling, normalisation) and how vectors are compared (similarity). A
change to either changes what a stored vector means, so either one invalidates
cached vectors. Choices that only affect how results are folded or fused (the
chunk-to-document rule, RRF's k) do not change any vector and are recorded in
the retriever's audit string instead.

`FakeEmbedder` is a deterministic, download-free stand-in used only by tests. It
hashes words into a few dimensions, so texts that share words get similar
vectors. It says nothing about real model quality and must never appear in a
reported evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from retrieval.contract import RetrievalError

EMBEDDING_CONFIG_FORMAT = "autoshield.embedding_config.v1"
SIMILARITIES = ("cosine", "dot")
POOLINGS = ("cls", "mean", "none")
UNIT_NORM_TOLERANCE = 1e-4
_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_WORD_RE = re.compile(r"[a-z0-9]+")


# ============================================================
# ERRORS
# ============================================================

class EmbeddingError(RetrievalError):
    """An embedding step failed (bad configuration, model files, or model output)."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """The texts given to an embedder are not usable (not strings, or blank)."""


class EmbeddingOutputError(EmbeddingError):
    """An embedder returned a malformed matrix (wrong shape, NaN/inf, not unit length)."""


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Everything that decides which numbers an embedder produces (see the module docstring).

    `file_digests` is a sorted tuple of (relative file name, SHA-256) for the weights and
    the tokenizer / model configuration files, so a changed tokenizer or config file changes
    the fingerprint even when the model id and revision string are unchanged.
    """

    model_id: str
    model_revision: str
    dimension: int
    query_instruction: str
    passage_prefix: str
    max_seq_length: int
    pooling: str
    normalize: bool
    similarity: str
    dtype: str = "float32"
    weights_sha256: str | None = None
    file_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise EmbeddingError(f"{name} must be a non-empty string")
        for name in ("dimension", "max_seq_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise EmbeddingError(f"{name} must be a positive integer")
        for name in ("query_instruction", "passage_prefix"):
            if not isinstance(getattr(self, name), str):
                raise EmbeddingError(f"{name} must be a string (use '' for none)")
        if self.pooling not in POOLINGS:
            raise EmbeddingError(f"pooling must be one of {POOLINGS}")
        if self.similarity not in SIMILARITIES:
            raise EmbeddingError(f"similarity must be one of {SIMILARITIES}")
        if not isinstance(self.normalize, bool):
            raise EmbeddingError("normalize must be a boolean")
        if self.similarity == "cosine" and not self.normalize:
            raise EmbeddingError("cosine similarity as a dot product needs normalised vectors (normalize=True)")
        if self.dtype != "float32":
            raise EmbeddingError("only float32 vectors are supported")
        if self.weights_sha256 is not None and not _SHA256_RE.fullmatch(self.weights_sha256):
            raise EmbeddingError("weights_sha256 must be a 64-character lower-case hex digest")
        digests = tuple(self.file_digests)
        if any(not _SHA256_RE.fullmatch(digest) for _, digest in digests):
            raise EmbeddingError("every file digest must be a 64-character lower-case hex SHA-256")
        object.__setattr__(self, "file_digests", tuple(sorted(digests)))

    def identity(self) -> dict[str, object]:
        """The canonical, JSON-safe description that the fingerprint is computed from."""
        return {
            "format": EMBEDDING_CONFIG_FORMAT,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "weights_sha256": self.weights_sha256,
            "file_digests": [[name, digest] for name, digest in self.file_digests],
            "dimension": self.dimension,
            "query_instruction": self.query_instruction,
            "passage_prefix": self.passage_prefix,
            "max_seq_length": self.max_seq_length,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "similarity": self.similarity,
            "dtype": self.dtype,
        }

    def fingerprint(self) -> str:
        """Deterministic SHA-256 of `identity()` (canonical JSON; no Python hash())."""
        return canonical_sha256(self.identity())

    @property
    def short_name(self) -> str:
        """A readable name for reports: model name, short revision, and whether an instruction is used."""
        base = self.model_id.rsplit("/", 1)[-1]
        suffix = "" if self.query_instruction else "+noinstr"
        return f"{base}@{self.model_revision[:7]}{suffix}"


def canonical_sha256(value: object) -> str:
    """SHA-256 of the canonical JSON form of `value` (sorted keys, no spaces, ASCII)."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# INPUT / OUTPUT VALIDATION
# ============================================================

def _checked_texts(texts: Sequence[str], what: str) -> list[str]:
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise EmbeddingInputError(f"{what} must be a list of strings, not a single string")
    result: list[str] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError(f"{what}[{index}] must be a non-blank string")
        result.append(text)
    return result


def prepare_passages(config: EmbeddingConfig, texts: Sequence[str]) -> list[str]:
    """The exact strings given to the model for passages: `passage_prefix` + the text, unchanged."""
    return [config.passage_prefix + text for text in _checked_texts(texts, "passages")]


def prepare_queries(config: EmbeddingConfig, texts: Sequence[str]) -> list[str]:
    """The exact strings given to the model for queries: `query_instruction` + the text, whitespace-trimmed."""
    return [config.query_instruction + text.strip() for text in _checked_texts(texts, "queries")]


def validate_embeddings(
    matrix: object, *, rows: int, config: EmbeddingConfig, normalize_first: bool = False
) -> np.ndarray:
    """
    Check a model output and return it as a float32 (rows, dimension) array.

    Rejects: non-numeric or object arrays, a wrong rank, the wrong number of rows or columns, NaN,
    infinity, an all-zero row, and (when `config.normalize`) any row that is not unit length.
    With `normalize_first=True` rows are L2-normalised before the unit-length check.
    """
    try:
        array = np.asarray(matrix)
    except Exception as exc:  # a ragged list, for example
        raise EmbeddingOutputError(f"the embedder output is not a numeric matrix: {exc}") from exc
    if array.dtype.kind not in "fiu":
        raise EmbeddingOutputError(f"the embedder output must be numeric, got dtype {array.dtype}")
    if array.ndim != 2:
        raise EmbeddingOutputError(f"the embedder output must be 2-dimensional, got {array.ndim} dimension(s)")
    if array.shape != (rows, config.dimension):
        raise EmbeddingOutputError(
            f"the embedder output has shape {array.shape}, expected {(rows, config.dimension)}"
        )
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise EmbeddingOutputError("the embedder output contains NaN or infinite values")
    norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if rows and (norms < 1e-12).any():
        raise EmbeddingOutputError("the embedder output contains an all-zero vector")
    if normalize_first and rows:
        array = (array / norms[:, None].astype(np.float32)).astype(np.float32)
        norms = np.linalg.norm(array.astype(np.float64), axis=1)
    if config.normalize and rows and not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE):
        raise EmbeddingOutputError(
            f"the configuration says vectors are L2-normalised but row norms range from "
            f"{norms.min():.6f} to {norms.max():.6f}"
        )
    return np.ascontiguousarray(array)


# ============================================================
# INTERFACE AND SHARED BASE
# ============================================================

@runtime_checkable
class Embedder(Protocol):
    """Anything that can embed passages and queries with a known, pinned configuration."""

    config: EmbeddingConfig

    @property
    def runtime_versions(self) -> Mapping[str, str]: ...

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


class BaseEmbedder:
    """
    Shared behaviour: apply the prefixes, call `_encode`, validate the output.

    Subclasses implement `_encode(prepared_texts)` returning a raw matrix. Empty input returns an
    empty (0, dimension) float32 array without calling the model.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        if not isinstance(config, EmbeddingConfig):
            raise TypeError("an embedder needs an EmbeddingConfig")
        self.config = config

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        return {}

    def _encode(self, prepared: list[str]) -> object:  # pragma: no cover - interface
        raise NotImplementedError

    def _run(self, prepared: list[str]) -> np.ndarray:
        if not prepared:
            return np.zeros((0, self.config.dimension), dtype=np.float32)
        raw = self._encode(prepared)
        return validate_embeddings(raw, rows=len(prepared), config=self.config, normalize_first=self.config.normalize)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(prepare_passages(self.config, texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._run(prepare_queries(self.config, texts))


# ============================================================
# TEST DOUBLE
# ============================================================

FAKE_MODEL_ID = "autoshield-test/fake-hash-embedder"
FAKE_REVISION = "0" * 40


class FakeEmbedder(BaseEmbedder):
    """
    Deterministic, download-free embedder for tests (see the module docstring).

    Each lower-case word is hashed (SHA-256) to one of `dimension` slots with a +1 or -1 sign;
    the vector is the sum, L2-normalised. `received_passages` / `received_queries` record the exact
    strings that reached the "model", so tests can prove the prefixes were applied. `transform`
    lets a test corrupt the raw output to exercise the validation.
    """

    def __init__(
        self,
        *,
        dimension: int = 32,
        query_instruction: str = "",
        passage_prefix: str = "",
        revision: str = FAKE_REVISION,
        model_id: str = FAKE_MODEL_ID,
        transform: Callable[[np.ndarray], object] | None = None,
    ) -> None:
        super().__init__(
            EmbeddingConfig(
                model_id=model_id,
                model_revision=revision,
                dimension=dimension,
                query_instruction=query_instruction,
                passage_prefix=passage_prefix,
                max_seq_length=512,
                pooling="mean",
                normalize=True,
                similarity="cosine",
            )
        )
        self._transform = transform
        self.received_passages: list[str] = []
        self.received_queries: list[str] = []
        self.encode_calls = 0

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        return {"fake-embedder": "1"}

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.config.dimension, dtype=np.float64)
        words = _WORD_RE.findall(text.lower())
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.config.dimension
            vector[slot] += 1.0 if digest[4] & 1 else -1.0
        if not vector.any():
            vector[0] = 1.0
        return vector / math.sqrt(float(vector @ vector))

    def _encode(self, prepared: list[str]) -> object:
        self.encode_calls += 1
        matrix = np.stack([self._vector(text) for text in prepared]).astype(np.float32)
        return matrix if self._transform is None else self._transform(matrix)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        prepared = prepare_passages(self.config, texts)
        self.received_passages.extend(prepared)
        return self._run(prepared)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prepared = prepare_queries(self.config, texts)
        self.received_queries.extend(prepared)
        return self._run(prepared)
