"""
Phase 10 — Security Knowledge Base: SQLite persistence and keyword search.

`KnowledgeStore` persists validated `SourceDocument` and `Chunk` objects in
one SQLite file (or in memory) and offers FTS5 keyword search over chunk
text. It uses only the standard library `sqlite3` module and the existing
Pydantic models. It does no downloading, embedding, vector search or RAG.

Source of truth vs derived data
-------------------------------
- `documents` and `chunks` are the source of truth. Each row keeps the full
  model as JSON (`document_json`, `chunk_json`) so it can be re-validated
  when loaded, plus a few indexed identity/hash columns that are
  cross-checked against that JSON on every load.
- `chunks_fts` (FTS5) is DERIVED data: the chunk title and text, indexed for
  keyword search. It is always written in the same transaction as `chunks`.
- The complete original document is always kept. Chunks are retrieval
  units only: the chunker may skip whitespace-only windows, so this store
  never assumes that concatenating chunks reproduces the document.

Tables
------
- `store_meta`  : key/value compatibility markers (see `_EXPECTED_META`).
- `documents`   : one row per document, including the per-document
                  ingestion state (chunker version, max_chars,
                  overlap_chars, normalization version, chunk count and a
                  digest of all chunks).
- `chunks`      : one row per chunk, unique on (doc_id, ordinal).
- `chunks_fts`  : FTS5 index; its rowid equals `chunks.row_id`.

Transactions
------------
Every write (`ingest_document`, `delete_document`, schema creation) runs in
exactly one explicit `BEGIN IMMEDIATE ... COMMIT` block. Any exception rolls
the whole block back, so document row, chunks, ingestion state and FTS rows
change together or not at all. Reads that touch several tables run in one
read transaction so they see a consistent snapshot. The connection is in
SQLite autocommit mode (`isolation_level=None`): Python never begins or
commits a transaction implicitly.

Versioning policy (no migration framework)
------------------------------------------
A database whose recorded markers differ from this module's expectations
(store schema version, normalization version, document schema version,
document-hash version) is REJECTED with `IncompatibleStoreError`. The
remedy is to rebuild the store from the original sources.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from knowledge.chunking import (
    CHUNKER_VERSION,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    _validate_settings,
    chunk_document,
)
from knowledge.schema import (
    DOCUMENT_HASH_VERSION,
    SCHEMA_VERSION,
    Chunk,
    SourceDocument,
    sha256_hex,
)

# Layout of the SQLite tables below. Bump when the layout changes.
STORE_SCHEMA_VERSION = "1"

# MANUALLY MAINTAINED compatibility marker. `knowledge.schema` exposes no
# version for `normalize_text`, so the store keeps its own. Bump this value
# whenever the behaviour of `normalize_text` changes: stored hashes and
# offsets were computed under the old behaviour. (A test pins a fingerprint
# of `normalize_text` so such a change does not go unnoticed.)
NORMALIZATION_VERSION = "normalize-text-v1"

DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100
MAX_QUERY_LENGTH = 1000

IN_MEMORY = ":memory:"

# Markers recorded in `store_meta` when a database is created and verified
# every time it is opened.
_EXPECTED_META: dict[str, str] = {
    "store_schema_version": STORE_SCHEMA_VERSION,
    "normalization_version": NORMALIZATION_VERSION,
    "document_schema_version": SCHEMA_VERSION,
    "document_hash_version": str(DOCUMENT_HASH_VERSION),
}

_EXPECTED_TABLES = ("store_meta", "documents", "chunks", "chunks_fts")

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE store_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE documents (
        doc_id                TEXT PRIMARY KEY,
        source                TEXT NOT NULL,
        source_id             TEXT NOT NULL,
        content_sha256        TEXT NOT NULL,
        document_sha256       TEXT NOT NULL,
        document_json         TEXT NOT NULL,
        normalization_version TEXT NOT NULL,
        chunker_version       TEXT NOT NULL,
        max_chars             INTEGER NOT NULL,
        overlap_chars         INTEGER NOT NULL,
        chunk_count           INTEGER NOT NULL CHECK (chunk_count > 0),
        chunks_sha256         TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE chunks (
        row_id       INTEGER PRIMARY KEY,
        chunk_id     TEXT NOT NULL UNIQUE,
        doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
        ordinal      INTEGER NOT NULL CHECK (ordinal >= 0),
        chunk_sha256 TEXT NOT NULL,
        chunk_json   TEXT NOT NULL,
        UNIQUE (doc_id, ordinal)
    )
    """,
    "CREATE VIRTUAL TABLE chunks_fts USING fts5(title, text, tokenize='unicode61')",
)


# ============================================================
# EXCEPTIONS
# ============================================================

class KnowledgeStoreError(Exception):
    """Base class for every error raised deliberately by this module."""


class FTS5UnavailableError(KnowledgeStoreError):
    """The SQLite library lacks FTS5, which keyword search requires."""


class IncompatibleStoreError(KnowledgeStoreError):
    """The database is not a store this code can use (wrong version or foreign file)."""


class StoreCorruptionError(KnowledgeStoreError):
    """Persisted data is malformed or inconsistent with itself."""


class DocumentNotFoundError(KnowledgeStoreError):
    """No document with the requested ID is stored."""


class InvalidIngestionError(KnowledgeStoreError, ValueError):
    """The document, chunks or chunking configuration passed to ingestion are invalid."""


class InvalidSearchQueryError(KnowledgeStoreError, ValueError):
    """The search query or limit is invalid, including malformed FTS5 syntax."""


# ============================================================
# PUBLIC VALUE TYPES
# ============================================================

@dataclass(frozen=True)
class ChunkingConfig:
    """
    The chunking settings a document was (or will be) chunked with.

    Defaults mirror `knowledge.chunking`. The settings are validated with
    the chunker's own rules. The store records them per document so a later
    change of `max_chars`, `overlap_chars` or `chunker_version` is detected.
    """

    max_chars: int = DEFAULT_MAX_CHARS
    overlap_chars: int = DEFAULT_OVERLAP_CHARS
    chunker_version: str = CHUNKER_VERSION

    def __post_init__(self) -> None:
        _validate_settings(self.max_chars, self.overlap_chars, self.chunker_version)


@dataclass(frozen=True)
class IngestResult:
    """
    What `ingest_document` did.

    - `action`: "created" (new document), "unchanged" (nothing written) or
      "updated". For "created" every flag below is False.
    - `content_changed`: normalized content hash differs from the stored one.
    - `metadata_changed`: `document_sha256` differs although `content_sha256`
      is the same (title, version, URL, license, references, ... changed).
    - `provenance_changed`: only the provenance record differs (it is
      excluded from `document_sha256`); the stored provenance is replaced.
    - `config_changed`: chunker version, max_chars, overlap_chars or the
      normalization version differs from the stored ingestion state.
    - `chunks_replaced`: the stored chunks were deleted and rewritten.
    """

    doc_id: str
    action: Literal["created", "unchanged", "updated"]
    chunk_count: int
    content_changed: bool = False
    metadata_changed: bool = False
    provenance_changed: bool = False
    config_changed: bool = False
    chunks_replaced: bool = False


@dataclass(frozen=True)
class SearchHit:
    """
    One keyword-search result: a validated stored chunk and its FTS5 score.

    `bm25` is SQLite's BM25 value: LOWER (more negative) means MORE relevant.
    Results are ordered by it ascending, then by (doc_id, ordinal).
    """

    chunk: Chunk
    bm25: float


# ============================================================
# INTERNAL HELPERS
# ============================================================

def check_fts5_available(connection: sqlite3.Connection) -> bool:
    """
    True if this SQLite build can create FTS5 tables.

    Probes with a throw-away TEMP table instead of trusting compile options.
    `KnowledgeStore` calls this (looked up at call time) when it opens, so
    tests can replace it to simulate a build without FTS5.
    """
    try:
        connection.execute("CREATE VIRTUAL TABLE temp._autoshield_fts5_probe USING fts5(x)")
        connection.execute("DROP TABLE temp._autoshield_fts5_probe")
    except sqlite3.OperationalError:
        return False
    return True


def _canonical_json(model: BaseModel) -> str:
    """Deterministic, pure-ASCII JSON of a model (sorted keys, no whitespace)."""
    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _chunks_digest(chunks: Sequence[Chunk]) -> str:
    """SHA-256 over every chunk in ordinal order (covers text, offsets, metadata)."""
    return sha256_hex("\n".join(_canonical_json(chunk) for chunk in chunks))


def _roundtrip_document(document: SourceDocument) -> str:
    """Serialize `document` and prove the JSON re-validates to an equal model."""
    text = _canonical_json(document)
    if SourceDocument.model_validate_json(text) != document:
        raise ValueError("document does not survive a validation round trip")
    return text


def _roundtrip_chunk(chunk: Chunk) -> str:
    """Serialize `chunk` and prove the JSON re-validates to an equal model."""
    text = _canonical_json(chunk)
    if Chunk.model_validate_json(text) != chunk:
        raise ValueError("chunk does not survive a validation round trip")
    return text


def _is_fts_query_error(exc: sqlite3.OperationalError) -> bool:
    """True if `exc` is SQLite complaining about the MATCH expression itself."""
    message = str(exc).lower()
    return (
        message.startswith("fts5")
        or "syntax error" in message
        or "no such column" in message
        or "unterminated" in message
        or "unknown special query" in message
    )


# ============================================================
# KNOWLEDGE STORE
# ============================================================

class KnowledgeStore:
    """
    SQLite-backed store for security-knowledge documents, chunks and FTS5 search.

    Usage::

        with KnowledgeStore("knowledge.db") as store:
            store.ingest_document(document)           # chunks it with defaults
            store.search('"SQL injection"')

    `path` is required (there is deliberately no implicit default or
    environment lookup): a file path, or ":memory:" for a private in-memory
    database. A missing file (and its parent directories) is created.

    Error contract: lookups of a missing document raise
    `DocumentNotFoundError`; `has_document` and `delete_document` report
    absence with a bool instead. Every SQLite failure is re-raised as
    `KnowledgeStoreError` with the original exception as `__cause__`.
    A store instance is not thread-safe (standard `sqlite3` behaviour).
    """

    def __init__(self, path: str | Path) -> None:
        if isinstance(path, Path):
            path = str(path)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty file path or ':memory:'")
        self._path = path
        if path != IN_MEMORY:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        try:
            conn = sqlite3.connect(path, isolation_level=None)  # explicit transactions only
        except sqlite3.Error as exc:
            raise KnowledgeStoreError(f"cannot open SQLite database {path!r}: {exc}") from exc
        conn.row_factory = sqlite3.Row  # internal only; rows never leave this class
        self._conn: sqlite3.Connection | None = conn
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            if not check_fts5_available(conn):
                raise FTS5UnavailableError(
                    "this Python's SQLite build does not support FTS5, which the knowledge "
                    "store requires for keyword search. Use a Python/SQLite build with "
                    "FTS5 enabled (the standard python.org builds include it)."
                )
            self._initialize_or_verify()
        except sqlite3.Error as exc:
            self.close()
            raise KnowledgeStoreError(f"cannot initialize knowledge store {path!r}: {exc}") from exc
        except BaseException:
            self.close()
            raise

    # ----- lifecycle -----

    @property
    def path(self) -> str:
        """The database path as given (":memory:" for an in-memory store)."""
        return self._path

    @property
    def closed(self) -> bool:
        return self._conn is None

    def close(self) -> None:
        """Close the connection. Safe to call more than once."""
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # ----- internals: connection, transactions, errors -----

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise KnowledgeStoreError("the knowledge store is closed")
        return self._conn

    @contextmanager
    def _transaction(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        """
        One explicit transaction: COMMIT on success, ROLLBACK on any error.

        Writers use BEGIN IMMEDIATE (take the write lock up front); readers
        use a deferred BEGIN so several SELECTs share one snapshot. Nesting
        is refused, so a caller can never commit part of an outer unit.
        """
        conn = self._connection()
        if conn.in_transaction:
            raise KnowledgeStoreError("nested transactions are not supported")
        conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    @contextmanager
    def _db_errors(self, action: str) -> Iterator[None]:
        """Re-raise `sqlite3.Error` as `KnowledgeStoreError`, keeping the cause."""
        try:
            yield
        except sqlite3.Error as exc:
            raise KnowledgeStoreError(f"SQLite error while {action}: {exc}") from exc

    def _initialize_or_verify(self) -> None:
        """Create the schema in an empty database, or verify an existing store."""
        with self._transaction(write=True) as conn:
            names = {
                row["name"]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            if not names:
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                conn.executemany(
                    "INSERT INTO store_meta (key, value) VALUES (?, ?)",
                    sorted(_EXPECTED_META.items()),
                )
                return
            if "store_meta" not in names:
                raise IncompatibleStoreError(
                    f"{self._path!r} is an SQLite database but not an AutoShield "
                    "knowledge store (no 'store_meta' table); refusing to modify it."
                )
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM store_meta")}
            for key, expected in _EXPECTED_META.items():
                if meta.get(key) != expected:
                    raise IncompatibleStoreError(
                        f"{self._path!r} was created with {key}={meta.get(key)!r} but this "
                        f"code requires {expected!r}. There is no migration: rebuild the "
                        "store from the original sources."
                    )
            missing = [table for table in _EXPECTED_TABLES if table not in names]
            if missing:
                raise StoreCorruptionError(f"{self._path!r} is missing tables: {missing}")

    # ----- ingestion -----

    def ingest_document(
        self,
        document: SourceDocument,
        chunks: Sequence[Chunk] | None = None,
        config: ChunkingConfig | None = None,
    ) -> IngestResult:
        """
        Store or update `document` and its chunks atomically and idempotently.

        Args:
            document: a validated `SourceDocument`.
            chunks: the chunks to store, in ordinal order. If None they are
                produced by `chunk_document(document, ...)` using `config`.
            config: the chunking settings `chunks` were made with (defaults
                to `ChunkingConfig()`). The store cannot recover these from
                the chunks, so they are the caller's assertion; it checks
                what it can (chunker version, chunk length <= max_chars).

        Change detection, against the stored row for the same `doc_id`:
            - identical document JSON, chunks and config -> "unchanged",
              nothing is written;
            - `content_sha256` / `document_sha256` differences are reported
              separately (content vs metadata) and the document is updated;
            - if the chunks differ in ANY way (content, copied metadata
              such as the title, chunker version, ...) all old chunks and
              their FTS rows are deleted and the new ones inserted;
            - a different `max_chars` / `overlap_chars` / chunker version /
              normalization version is recorded in `config_changed`.
        Everything happens in one transaction. If anything fails, the
        previously committed state is untouched, and a failed first
        ingestion leaves nothing behind.

        Raises:
            InvalidIngestionError: invalid inputs (nothing was modified).
            KnowledgeStoreError: a database failure (state rolled back).
        """
        config = config if config is not None else ChunkingConfig()
        if not isinstance(config, ChunkingConfig):
            raise InvalidIngestionError("config must be a ChunkingConfig")
        document_json = self._validate_document(document)
        chunk_list = self._prepare_chunks(document, chunks, config)
        chunk_jsons = [self._validate_chunk(document, chunk, config) for chunk in chunk_list]
        self._validate_chunk_sequence(chunk_list)
        digest = _chunks_digest(chunk_list)

        with self._db_errors(f"ingesting {document.doc_id!r}"):
            with self._transaction(write=True) as conn:
                old = conn.execute(
                    "SELECT * FROM documents WHERE doc_id = ?", (document.doc_id,)
                ).fetchone()
                if old is None:
                    self._insert_document_row(conn, document, document_json, config, len(chunk_list), digest)
                    self._insert_chunks(conn, chunk_list, chunk_jsons)
                    return IngestResult(document.doc_id, "created", len(chunk_list))

                content_changed = old["content_sha256"] != document.content_sha256
                document_changed = old["document_sha256"] != document.document_sha256
                config_changed = (
                    old["chunker_version"],
                    old["max_chars"],
                    old["overlap_chars"],
                    old["normalization_version"],
                ) != (config.chunker_version, config.max_chars, config.overlap_chars, NORMALIZATION_VERSION)
                chunks_changed = old["chunks_sha256"] != digest or old["chunk_count"] != len(chunk_list)
                provenance_changed = not document_changed and old["document_json"] != document_json

                if not (document_changed or provenance_changed or config_changed or chunks_changed):
                    return IngestResult(document.doc_id, "unchanged", len(chunk_list))

                self._update_document_row(conn, document, document_json, config, len(chunk_list), digest)
                if chunks_changed:
                    self._delete_chunks(conn, document.doc_id)
                    self._insert_chunks(conn, chunk_list, chunk_jsons)
                return IngestResult(
                    document.doc_id,
                    "updated",
                    len(chunk_list),
                    content_changed=content_changed,
                    metadata_changed=document_changed and not content_changed,
                    provenance_changed=provenance_changed,
                    config_changed=config_changed,
                    chunks_replaced=chunks_changed,
                )

    # ----- ingestion: input validation (no database access) -----

    @staticmethod
    def _validate_document(document: SourceDocument) -> str:
        if not isinstance(document, SourceDocument):
            raise InvalidIngestionError(
                f"document must be a SourceDocument, got {type(document).__name__}"
            )
        try:
            return _roundtrip_document(document)
        except ValueError as exc:  # pydantic.ValidationError is a ValueError
            raise InvalidIngestionError(
                f"document {document.doc_id!r} is not a valid, self-consistent SourceDocument: {exc}"
            ) from exc

    @staticmethod
    def _prepare_chunks(
        document: SourceDocument, chunks: Sequence[Chunk] | None, config: ChunkingConfig
    ) -> list[Chunk]:
        if chunks is None:
            return chunk_document(
                document,
                max_chars=config.max_chars,
                overlap_chars=config.overlap_chars,
                chunker_version=config.chunker_version,
            )
        if isinstance(chunks, (str, bytes)) or not isinstance(chunks, Sequence):
            raise InvalidIngestionError("chunks must be a sequence of Chunk objects")
        return list(chunks)

    @staticmethod
    def _validate_chunk(document: SourceDocument, chunk: Chunk, config: ChunkingConfig) -> str:
        if not isinstance(chunk, Chunk):
            raise InvalidIngestionError(f"chunks must be Chunk objects, got {type(chunk).__name__}")
        try:
            chunk_json = _roundtrip_chunk(chunk)
        except ValueError as exc:
            raise InvalidIngestionError(f"chunk {chunk.chunk_id!r} is not a valid Chunk: {exc}") from exc
        if not document.contains_chunk(chunk):
            raise InvalidIngestionError(
                f"chunk {chunk.chunk_id!r} is not consistent with document {document.doc_id!r} "
                "(wrong document, stale content hash, changed metadata or bad offsets)"
            )
        if chunk.chunker_version != config.chunker_version:
            raise InvalidIngestionError(
                f"chunk {chunk.chunk_id!r} has chunker_version {chunk.chunker_version!r}, "
                f"but the config says {config.chunker_version!r}"
            )
        if len(chunk.text) > config.max_chars:
            raise InvalidIngestionError(
                f"chunk {chunk.chunk_id!r} has {len(chunk.text)} characters, "
                f"more than config.max_chars={config.max_chars}"
            )
        return chunk_json

    @staticmethod
    def _validate_chunk_sequence(chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise InvalidIngestionError(
                "a document must have at least one chunk (non-blank content always yields one)"
            )
        ordinals = [chunk.ordinal for chunk in chunks]
        if ordinals != list(range(len(chunks))):
            raise InvalidIngestionError(
                f"chunk ordinals must be exactly 0..{len(chunks) - 1} in order, got {ordinals}"
            )

    # ----- ingestion: SQL (always called inside a write transaction) -----

    @staticmethod
    def _document_columns(
        document: SourceDocument,
        document_json: str,
        config: ChunkingConfig,
        chunk_count: int,
        digest: str,
    ) -> tuple:
        return (
            document.source,
            document.source_id,
            document.content_sha256,
            document.document_sha256,
            document_json,
            NORMALIZATION_VERSION,
            config.chunker_version,
            config.max_chars,
            config.overlap_chars,
            chunk_count,
            digest,
        )

    def _insert_document_row(self, conn, document, document_json, config, chunk_count, digest) -> None:
        conn.execute(
            "INSERT INTO documents (source, source_id, content_sha256, document_sha256, "
            "document_json, normalization_version, chunker_version, max_chars, overlap_chars, "
            "chunk_count, chunks_sha256, doc_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*self._document_columns(document, document_json, config, chunk_count, digest), document.doc_id),
        )

    def _update_document_row(self, conn, document, document_json, config, chunk_count, digest) -> None:
        conn.execute(
            "UPDATE documents SET source = ?, source_id = ?, content_sha256 = ?, "
            "document_sha256 = ?, document_json = ?, normalization_version = ?, "
            "chunker_version = ?, max_chars = ?, overlap_chars = ?, chunk_count = ?, "
            "chunks_sha256 = ? WHERE doc_id = ?",
            (*self._document_columns(document, document_json, config, chunk_count, digest), document.doc_id),
        )

    @staticmethod
    def _insert_chunks(conn, chunks: Sequence[Chunk], chunk_jsons: Sequence[str]) -> None:
        """Insert each chunk row and its FTS row (rowid = chunks.row_id)."""
        for chunk, chunk_json in zip(chunks, chunk_jsons):
            cursor = conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, ordinal, chunk_sha256, chunk_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (chunk.chunk_id, chunk.doc_id, chunk.ordinal, chunk.chunk_sha256, chunk_json),
            )
            conn.execute(
                "INSERT INTO chunks_fts (rowid, title, text) VALUES (?, ?, ?)",
                (cursor.lastrowid, chunk.title, chunk.text),
            )

    @staticmethod
    def _delete_chunks(conn, doc_id: str) -> None:
        """Delete a document's FTS rows first (no cascade reaches them), then its chunks."""
        row_ids = [
            row["row_id"]
            for row in conn.execute("SELECT row_id FROM chunks WHERE doc_id = ?", (doc_id,))
        ]
        conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(row_id,) for row_id in row_ids])
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    # ----- retrieval -----

    def has_document(self, doc_id: str) -> bool:
        """True if a document with this ID is stored (no content validation)."""
        with self._db_errors("checking for a document"):
            row = self._connection().execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return row is not None

    def list_doc_ids(self) -> list[str]:
        """All stored document IDs, sorted."""
        with self._db_errors("listing documents"):
            rows = self._connection().execute("SELECT doc_id FROM documents ORDER BY doc_id").fetchall()
        return [row["doc_id"] for row in rows]

    def count_documents(self) -> int:
        with self._db_errors("counting documents"):
            return self._connection().execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def count_chunks(self) -> int:
        with self._db_errors("counting chunks"):
            return self._connection().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def get_document(self, doc_id: str) -> SourceDocument:
        """
        Load a document through full Pydantic validation.

        Raises `DocumentNotFoundError` if it is not stored and
        `StoreCorruptionError` if the stored record is malformed or
        disagrees with its own identity/hash columns.
        """
        with self._db_errors(f"reading document {doc_id!r}"):
            with self._transaction(write=False) as conn:
                document, _row = self._load_document(conn, doc_id)
        return document

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """
        Load all chunks of a document, ordered by numeric ordinal.

        Each chunk is validated by Pydantic and checked against its document
        with `SourceDocument.contains_chunk`; ordinals, count, chunker
        version and the stored chunk digest must also agree. Anything else
        raises `StoreCorruptionError`. Missing document: `DocumentNotFoundError`.
        """
        with self._db_errors(f"reading chunks of {doc_id!r}"):
            with self._transaction(write=False) as conn:
                document, row = self._load_document(conn, doc_id)
                chunk_rows = conn.execute(
                    "SELECT row_id, chunk_id, doc_id, ordinal, chunk_sha256, chunk_json "
                    "FROM chunks WHERE doc_id = ? ORDER BY ordinal",
                    (doc_id,),
                ).fetchall()
        chunks = [self._parse_chunk(chunk_row, document) for chunk_row in chunk_rows]
        if len(chunks) != row["chunk_count"] or [c.ordinal for c in chunks] != list(range(len(chunks))):
            raise StoreCorruptionError(
                f"document {doc_id!r} should have {row['chunk_count']} chunks with ordinals "
                f"0..n-1 but has ordinals {[c.ordinal for c in chunks]}"
            )
        if any(c.chunker_version != row["chunker_version"] for c in chunks):
            raise StoreCorruptionError(f"chunks of {doc_id!r} disagree with the stored chunker version")
        if _chunks_digest(chunks) != row["chunks_sha256"]:
            raise StoreCorruptionError(f"chunks of {doc_id!r} do not match the stored chunk digest")
        return chunks

    def _load_document(self, conn: sqlite3.Connection, doc_id: str) -> tuple[SourceDocument, sqlite3.Row]:
        row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            raise DocumentNotFoundError(f"no document with doc_id {doc_id!r} is stored")
        try:
            document = SourceDocument.model_validate_json(row["document_json"])
        except ValueError as exc:
            raise StoreCorruptionError(f"stored document {doc_id!r} fails validation: {exc}") from exc
        mismatches = [
            column
            for column, value in (
                ("doc_id", document.doc_id),
                ("source", document.source),
                ("source_id", document.source_id),
                ("content_sha256", document.content_sha256),
                ("document_sha256", document.document_sha256),
                ("normalization_version", NORMALIZATION_VERSION),
            )
            if row[column] != value
        ]
        if mismatches:
            raise StoreCorruptionError(
                f"stored document {doc_id!r}: columns {mismatches} disagree with its JSON"
            )
        return document, row

    @staticmethod
    def _parse_chunk(row: sqlite3.Row, document: SourceDocument) -> Chunk:
        """Validate one stored chunk row and check it against its document."""
        try:
            chunk = Chunk.model_validate_json(row["chunk_json"])
        except ValueError as exc:
            raise StoreCorruptionError(f"stored chunk {row['chunk_id']!r} fails validation: {exc}") from exc
        if (chunk.chunk_id, chunk.doc_id, chunk.ordinal, chunk.chunk_sha256) != (
            row["chunk_id"],
            row["doc_id"],
            row["ordinal"],
            row["chunk_sha256"],
        ):
            raise StoreCorruptionError(f"stored chunk {row['chunk_id']!r}: columns disagree with its JSON")
        if not document.contains_chunk(chunk):
            raise StoreCorruptionError(
                f"stored chunk {chunk.chunk_id!r} is not consistent with its document "
                f"{document.doc_id!r} (stale or tampered data)"
            )
        return chunk

    # ----- deletion -----

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document, its chunks and their FTS rows in one transaction.

        Returns True if the document existed, False if it was not stored
        (so deleting twice is harmless).
        """
        with self._db_errors(f"deleting document {doc_id!r}"):
            with self._transaction(write=True) as conn:
                if conn.execute("SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone() is None:
                    return False
                self._delete_chunks(conn, doc_id)
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                return True

    # ----- keyword search -----

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchHit]:
        """
        Keyword search over chunk titles and text using SQLite FTS5.

        `query` uses FTS5 query syntax (words, "quoted phrases", AND/OR/NOT,
        prefix*). Note that "-" and ":" are syntax: write "CWE-89" as a
        quoted phrase. The query is passed as a bound parameter, so it can
        never change the SQL statement. Malformed syntax raises
        `InvalidSearchQueryError`; there is no fallback search.

        Hits are ordered by BM25 relevance (best first), ties broken by
        (doc_id, ordinal), and at most `limit` (1..MAX_SEARCH_LIMIT) are
        returned. Every hit chunk is validated like `get_chunks` does.
        This is lexical matching only; it says nothing about semantic
        similarity or about how authoritative a match is.
        """
        if not isinstance(query, str) or not query.strip():
            raise InvalidSearchQueryError("query must be a non-empty string")
        if len(query) > MAX_QUERY_LENGTH:
            raise InvalidSearchQueryError(f"query exceeds {MAX_QUERY_LENGTH} characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise InvalidSearchQueryError(f"limit must be an integer from 1 to {MAX_SEARCH_LIMIT}")

        with self._db_errors("searching"):
            with self._transaction(write=False) as conn:
                try:
                    rows = conn.execute(
                        "SELECT c.row_id, c.chunk_id, c.doc_id, c.ordinal, c.chunk_sha256, "
                        "c.chunk_json, bm25(chunks_fts) AS score "
                        "FROM chunks_fts JOIN chunks AS c ON c.row_id = chunks_fts.rowid "
                        "WHERE chunks_fts MATCH ? "
                        "ORDER BY score, c.doc_id, c.ordinal LIMIT ?",
                        (query, limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    if _is_fts_query_error(exc):
                        raise InvalidSearchQueryError(f"invalid FTS5 query {query!r}: {exc}") from exc
                    raise
                documents: dict[str, SourceDocument] = {}
                for row in rows:
                    if row["doc_id"] not in documents:
                        try:
                            documents[row["doc_id"]] = self._load_document(conn, row["doc_id"])[0]
                        except DocumentNotFoundError as exc:
                            raise StoreCorruptionError(
                                f"chunk {row['chunk_id']!r} belongs to a document that is not stored"
                            ) from exc
        return [
            SearchHit(chunk=self._parse_chunk(row, documents[row["doc_id"]]), bm25=row["score"])
            for row in rows
        ]
