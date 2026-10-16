"""
Phase 11 (Milestone 11C) - a fingerprinted, rebuildable cache of chunk embeddings.

The validated documents and chunks in the SQLite `KnowledgeStore` stay the single source of truth.
Vectors are DERIVED data: they can always be recomputed, so they live in a separate directory (default
`data/embedding_cache/`, ignored by git) and are never written into the knowledge tables. There is no
vector database: at 247 chunks x 384 dimensions x 4 bytes the whole index is 379,392 bytes.

Files (a directory per cache)
-----------------------------
    manifest.json            validated JSON: the commit point, and what the vectors mean
    vectors-<hash16>.npy     one float32 matrix, one row per chunk, in manifest order

`.npy` is read with `allow_pickle=False`, and its SHA-256 is checked against the manifest BEFORE it is
parsed, so a corrupt or swapped file is rejected and nothing executable is ever loaded (no pickle
anywhere in this module).

What identifies a cached vector (the entry key)
-----------------------------------------------
    SHA-256 of canonical JSON of { embedding fingerprint, chunk id, SHA-256 of the exact chunk text,
                                   chunker version, mode "passage" }

where the embedding fingerprint (`EmbeddingConfig.fingerprint()`) already covers model id, exact revision,
weights and tokenizer-file digests, dimension, passage prefix, query instruction, maximum length,
pooling, normalisation, similarity and dtype. Changing the chunk text, the chunk's identity or chunker
version, or ANY of those settings changes the key, so the stale vector is not reused. Python's `hash()` is
never used. Package versions are stored too; a different set of versions makes every vector stale (a
rebuild of this tiny corpus is cheap, and the last digits of a vector can depend on them).

Four outcomes are reported separately (`CacheReport`)
-----------------------------------------------------
    reused       a vector taken from the cache because its key matches the current chunk
    created      a vector computed now (new chunk, or nothing reusable)
    invalidated  a cached vector that passed every integrity check but no longer matches: the chunk text or
                 identity changed, the chunk is gone, or the embedding configuration / package versions changed
    rejected     cached vectors thrown away because the cache FAILED an integrity check (corrupt JSON or
                 numbers, wrong shape, NaN, hash mismatch, unknown format, a manifest that disagrees with itself)

Atomic update
-------------
A new `vectors-<hash>.npy` is written to a temporary name, flushed to disk and renamed into place; only
then is `manifest.json` replaced (also by rename). The manifest replacement is the single commit point.
If the process stops earlier, the old manifest and old vectors file are untouched and still valid; leftover
temporary files are ignored and removed by the next successful write. If a reader somehow sees a manifest
that disagrees with its vectors file, it is rejected (hash mismatch) and rebuilt, never half-used.

Rebuild path: `VectorCache.sync(..., rebuild=True)` ignores existing contents; `VectorCache.clear()` deletes the
cache files; or simply delete the directory.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from knowledge.schema import Chunk, sha256_hex
from retrieval.contract import RetrievalError
from retrieval.embeddings import UNIT_NORM_TOLERANCE, Embedder, EmbeddingConfig, canonical_sha256

CACHE_FORMAT = "autoshield.embedding_cache.v1"
MANIFEST_NAME = "manifest.json"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "embedding_cache"

_VECTOR_FILE_RE = re.compile(r"vectors-[0-9a-f]{16}\.npy")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = {
    "cache_format", "embedding_fingerprint", "embedding_config", "runtime_versions", "dimension", "dtype",
    "normalized", "similarity", "vector_file", "vectors_sha256", "entries", "index_fingerprint",
}
_ENTRY_KEYS = {"chunk_id", "doc_id", "ordinal", "chunk_sha256", "chunker_version", "entry_key"}

# Indirection so tests can simulate a crash between the two renames.
_replace = os.replace


class CacheError(RetrievalError):
    """The embedding cache could not be used or written."""


class CacheCorruptError(CacheError):
    """A cache file failed an integrity check."""


# ============================================================
# TYPES
# ============================================================

@dataclass(frozen=True)
class CacheReport:
    """What `VectorCache.sync` did. All counts are numbers of chunk vectors."""

    total: int
    reused: int
    created: int
    invalidated: int
    rejected: int
    status_before: str  # "missing", "valid", "stale", "corrupt" or "rebuild_requested"
    reasons: tuple[str, ...]
    written: bool
    index_fingerprint: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total, "reused": self.reused, "created": self.created,
            "invalidated": self.invalidated, "rejected": self.rejected,
            "status_before": self.status_before, "reasons": list(self.reasons),
            "written": self.written, "index_fingerprint": self.index_fingerprint,
        }


@dataclass(frozen=True)
class CacheEntry:
    chunk_id: str
    doc_id: str
    ordinal: int
    chunk_sha256: str
    chunker_version: str
    entry_key: str


@dataclass(frozen=True)
class CacheRead:
    """The outcome of reading a cache directory (no side effects)."""

    status: str  # "missing", "valid", "stale" or "corrupt"
    reason: str
    entries: tuple[CacheEntry, ...] = ()
    matrix: np.ndarray | None = None
    index_fingerprint: str | None = None


# ============================================================
# KEYS AND FINGERPRINTS
# ============================================================

def entry_key(embedding_fingerprint: str, chunk_id: str, chunk_sha256: str, chunker_version: str) -> str:
    """The deterministic key of one cached passage vector (see the module docstring)."""
    return canonical_sha256({
        "embedding_fingerprint": embedding_fingerprint,
        "chunk_id": chunk_id,
        "chunk_sha256": chunk_sha256,
        "chunker_version": chunker_version,
        "mode": "passage",
    })


def index_fingerprint(embedding_fingerprint: str, entry_keys: Sequence[str]) -> str:
    """Identifies the exact set (and order) of vectors in a cache."""
    return canonical_sha256({"embedding_fingerprint": embedding_fingerprint, "entry_keys": list(entry_keys)})


def _entries_for(chunks: Sequence[Chunk], embedding_fp: str) -> list[CacheEntry]:
    entries = []
    for chunk in chunks:
        digest = sha256_hex(chunk.text)  # recomputed from the text, never taken on trust
        entries.append(CacheEntry(
            chunk.chunk_id, chunk.doc_id, chunk.ordinal, digest, chunk.chunker_version,
            entry_key(embedding_fp, chunk.chunk_id, digest, chunk.chunker_version),
        ))
    return entries


# ============================================================
# READING AND VALIDATING
# ============================================================

def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> object:
    raise ValueError(f"non-finite JSON constant {name}")


def _parse_manifest(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise CacheCorruptError(f"manifest is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != _MANIFEST_KEYS:
        raise CacheCorruptError("manifest has missing or unknown keys")
    if data["cache_format"] != CACHE_FORMAT:
        raise CacheCorruptError(f"unknown cache format {data['cache_format']!r}")
    for key in ("embedding_fingerprint", "vectors_sha256", "index_fingerprint"):
        if not isinstance(data[key], str) or not _SHA256_RE.fullmatch(data[key]):
            raise CacheCorruptError(f"manifest {key} is not a SHA-256 digest")
    if not isinstance(data["vector_file"], str) or not _VECTOR_FILE_RE.fullmatch(data["vector_file"]):
        raise CacheCorruptError("manifest vector_file is not an allowed file name")
    if isinstance(data["dimension"], bool) or not isinstance(data["dimension"], int) or data["dimension"] < 1:
        raise CacheCorruptError("manifest dimension must be a positive integer")
    if data["dtype"] != "float32" or not isinstance(data["normalized"], bool):
        raise CacheCorruptError("manifest dtype must be float32 and normalized a boolean")
    if not isinstance(data["runtime_versions"], dict) or not isinstance(data["embedding_config"], dict):
        raise CacheCorruptError("manifest runtime_versions and embedding_config must be objects")
    if not isinstance(data["entries"], list):
        raise CacheCorruptError("manifest entries must be a list")
    return data


def _parse_entries(manifest: dict[str, Any]) -> tuple[CacheEntry, ...]:
    entries: list[CacheEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(manifest["entries"]):
        if not isinstance(raw, dict) or set(raw) != _ENTRY_KEYS:
            raise CacheCorruptError(f"entry {index} has missing or unknown keys")
        if any(not isinstance(raw[k], str) or not raw[k] for k in _ENTRY_KEYS - {"ordinal"}):
            raise CacheCorruptError(f"entry {index} has a non-string field")
        if isinstance(raw["ordinal"], bool) or not isinstance(raw["ordinal"], int) or raw["ordinal"] < 0:
            raise CacheCorruptError(f"entry {index} has an invalid ordinal")
        if raw["chunk_id"] in seen:
            raise CacheCorruptError(f"duplicate chunk_id {raw['chunk_id']!r} in the manifest")
        seen.add(raw["chunk_id"])
        expected = entry_key(manifest["embedding_fingerprint"], raw["chunk_id"], raw["chunk_sha256"], raw["chunker_version"])
        if raw["entry_key"] != expected:
            raise CacheCorruptError(f"entry {index} key does not match its fields (manifest altered)")
        entries.append(CacheEntry(raw["chunk_id"], raw["doc_id"], raw["ordinal"], raw["chunk_sha256"],
                                  raw["chunker_version"], raw["entry_key"]))
    expected_index = index_fingerprint(manifest["embedding_fingerprint"], [e.entry_key for e in entries])
    if manifest["index_fingerprint"] != expected_index:
        raise CacheCorruptError("manifest index_fingerprint does not match its entries")
    return tuple(entries)


def _load_matrix(directory: Path, manifest: dict[str, Any], rows: int) -> np.ndarray:
    path = directory / manifest["vector_file"]
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CacheCorruptError(f"cannot read the vectors file: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != manifest["vectors_sha256"]:
        raise CacheCorruptError("vectors file SHA-256 does not match the manifest")
    try:
        matrix = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, OSError, EOFError) as exc:
        raise CacheCorruptError(f"vectors file is not a plain numeric .npy array: {exc}") from exc
    if not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32 or matrix.ndim != 2:
        raise CacheCorruptError("vectors must be a 2-dimensional float32 array")
    if matrix.shape != (rows, manifest["dimension"]):
        raise CacheCorruptError(f"vectors have shape {matrix.shape}, manifest says {(rows, manifest['dimension'])}")
    if not np.isfinite(matrix).all():
        raise CacheCorruptError("vectors contain NaN or infinite values")
    if manifest["normalized"] and rows:
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=UNIT_NORM_TOLERANCE):
            raise CacheCorruptError("the manifest says vectors are unit length but some rows are not")
    return matrix


def _config_differences(manifest: dict[str, Any], config: EmbeddingConfig) -> list[str]:
    current = config.identity()
    stored = manifest["embedding_config"]
    return sorted(key for key in set(current) | set(stored) if current.get(key) != stored.get(key))


# ============================================================
# THE CACHE
# ============================================================

class VectorCache:
    """A cache directory for the passage vectors of one embedder configuration (see the module docstring)."""

    def __init__(self, directory: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.directory = Path(directory)

    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    def read(self, config: EmbeddingConfig, runtime_versions: Mapping[str, str]) -> CacheRead:
        """
        Inspect the cache without changing anything.

        "missing": no manifest. "corrupt": a file failed an integrity check. "stale": intact, but made with a
        different embedding configuration or package versions, so no vector may be reused. "valid": intact and
        made with this configuration (individual chunks may still have changed; `sync` decides per chunk).
        """
        if not self.manifest_path.is_file():
            return CacheRead("missing", "no cache manifest")
        try:
            manifest = _parse_manifest(self.manifest_path.read_bytes())
            entries = _parse_entries(manifest)
            matrix = _load_matrix(self.directory, manifest, len(entries))
        except CacheCorruptError as exc:
            return CacheRead("corrupt", str(exc))
        except OSError as exc:
            return CacheRead("corrupt", f"cannot read the manifest: {exc}")
        fingerprint = manifest["index_fingerprint"]
        differences = _config_differences(manifest, config)
        if manifest["embedding_fingerprint"] != config.fingerprint() or differences:
            return CacheRead("stale", f"embedding configuration changed ({', '.join(differences) or 'fingerprint'})",
                             entries, matrix, fingerprint)
        if dict(manifest["runtime_versions"]) != dict(runtime_versions):
            return CacheRead("stale", "package versions changed since the cache was built", entries, matrix, fingerprint)
        return CacheRead("valid", "ok", entries, matrix, fingerprint)

    def sync(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        rebuild: bool = False,
        write: bool = True,
    ) -> tuple[np.ndarray, CacheReport]:
        """
        Return one passage vector per chunk (rows in `chunks` order) and a report of what happened.

        Vectors whose entry key matches are reused; the rest are computed with `embedder.embed_passages` in one
        call. When anything changed, the cache is rewritten atomically (unless `write=False`).
        """
        chunk_list = list(chunks)
        ids = [chunk.chunk_id for chunk in chunk_list]
        if len(set(ids)) != len(ids):
            raise CacheError("chunk ids must be unique")
        config = embedder.config
        dimension = config.dimension
        if not chunk_list:
            return np.zeros((0, dimension), dtype=np.float32), CacheReport(0, 0, 0, 0, 0, "missing", ("no chunks",), False, None)

        wanted = _entries_for(chunk_list, config.fingerprint())
        versions = dict(embedder.runtime_versions)
        reasons: list[str] = []
        old_by_key: dict[str, np.ndarray] = {}
        invalidated = rejected = 0
        if rebuild:
            status = "rebuild_requested"
            reasons.append("rebuild requested; existing cache ignored")
        else:
            current = self.read(config, versions)
            status = current.status
            if status not in ("missing", "valid"):
                reasons.append(current.reason)
            if status == "corrupt":
                rejected = self._count_manifest_entries()
            elif status == "stale":
                invalidated = len(current.entries)
            elif status == "valid" and current.matrix is not None:
                for index, entry in enumerate(current.entries):
                    old_by_key[entry.entry_key] = current.matrix[index]
                wanted_keys = {entry.entry_key for entry in wanted}
                invalidated = sum(1 for entry in current.entries if entry.entry_key not in wanted_keys)
                if invalidated:
                    reasons.append(f"{invalidated} cached vector(s) no longer match the current chunks")

        missing = [i for i, entry in enumerate(wanted) if entry.entry_key not in old_by_key]
        matrix = np.zeros((len(chunk_list), dimension), dtype=np.float32)
        for i, entry in enumerate(wanted):
            if entry.entry_key in old_by_key:
                matrix[i] = old_by_key[entry.entry_key]
        if missing:
            fresh = embedder.embed_passages([chunk_list[i].text for i in missing])
            if fresh.shape != (len(missing), dimension):
                raise CacheError(f"the embedder returned shape {fresh.shape}, expected {(len(missing), dimension)}")
            for row, i in enumerate(missing):
                matrix[i] = fresh[row]

        reused = len(chunk_list) - len(missing)
        fingerprint = index_fingerprint(config.fingerprint(), [e.entry_key for e in wanted])
        needs_write = bool(missing) or invalidated > 0 or status in ("corrupt", "stale", "rebuild_requested")
        written = False
        if needs_write and write:
            self._write(wanted, matrix, config, versions, fingerprint)
            written = True
        return matrix, CacheReport(len(chunk_list), reused, len(missing), invalidated, rejected, status,
                                   tuple(reasons), written, fingerprint)

    def _count_manifest_entries(self) -> int:
        """Entries listed by a corrupt cache's manifest, if its JSON is readable (else 0)."""
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            entries = data.get("entries") if isinstance(data, dict) else None
            return len(entries) if isinstance(entries, list) else 0
        except (OSError, ValueError):
            return 0

    def _write(self, entries: Sequence[CacheEntry], matrix: np.ndarray, config: EmbeddingConfig,
               versions: Mapping[str, str], fingerprint: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, np.ascontiguousarray(matrix, dtype=np.float32), allow_pickle=False)
        payload = buffer.getvalue()
        vectors_sha = hashlib.sha256(payload).hexdigest()
        vector_name = f"vectors-{vectors_sha[:16]}.npy"
        manifest = {
            "cache_format": CACHE_FORMAT,
            "embedding_fingerprint": config.fingerprint(),
            "embedding_config": config.identity(),
            "runtime_versions": dict(versions),
            "dimension": config.dimension,
            "dtype": "float32",
            "normalized": config.normalize,
            "similarity": config.similarity,
            "vector_file": vector_name,
            "vectors_sha256": vectors_sha,
            "entries": [
                {"chunk_id": e.chunk_id, "doc_id": e.doc_id, "ordinal": e.ordinal, "chunk_sha256": e.chunk_sha256,
                 "chunker_version": e.chunker_version, "entry_key": e.entry_key}
                for e in entries
            ],
            "index_fingerprint": fingerprint,
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        self._atomic_write(self.directory / vector_name, payload)
        self._atomic_write(self.manifest_path, manifest_bytes)  # the commit point
        self._remove_leftovers(keep=vector_name)

    def _atomic_write(self, target: Path, data: bytes) -> None:
        temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            with open(temp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            _replace(temp, target)
        except BaseException:
            try:
                temp.unlink()
            except OSError:
                pass
            raise

    def _remove_leftovers(self, *, keep: str) -> None:
        for path in self.directory.iterdir():
            old_vectors = _VECTOR_FILE_RE.fullmatch(path.name) and path.name != keep
            if old_vectors or (path.name.startswith(".") and ".tmp-" in path.name):
                try:
                    path.unlink()
                except OSError:
                    pass  # best effort; a leftover file is ignored by readers

    def clear(self) -> int:
        """Delete the cache files (manifest, vectors, temporaries). Returns how many files were removed."""
        removed = 0
        if self.directory.is_dir():
            for path in self.directory.iterdir():
                if path.name == MANIFEST_NAME or _VECTOR_FILE_RE.fullmatch(path.name) or ".tmp-" in path.name:
                    path.unlink()
                    removed += 1
        return removed
