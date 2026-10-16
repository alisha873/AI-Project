"""
Phase 10 — Knowledge ingestion: orchestration.

`ingest_documents(documents, store, config=None)` connects the existing
Milestone 1-3 components:

    validated SourceDocument objects
        -> knowledge.chunking.chunk_document   (canonical chunker)
        -> KnowledgeStore.ingest_document      (validation, change detection,
                                                transaction, FTS5 index)

It does not know how documents were acquired (a local JSON file, a future
CWE/CAPEC adapter, a test) and it owns no storage logic: no tables, hashes,
chunk IDs, FTS operations or transactions. Change detection stays entirely in
the store; the summary only reports what the store's `IngestResult`s say.

Order of work
-------------
1. PRE-FLIGHT, before the first write. Everything that can be checked
   without touching the database is checked up front, so malformed input
   cannot cause a partial batch:
   - every item must be a `SourceDocument` that survives re-validation
     through the schema (a document built with `model_construct` that
     violates the schema is refused, with its index in the input);
   - duplicate `doc_id` values are resolved by the policy below;
   - every document is chunked with the one `ChunkingConfig`.
2. WRITE. Documents are handed to the store one at a time.

Duplicate policy
----------------
Two items with the same `doc_id` in one input are either
- fully identical (`==`: every field including provenance) -> the second is
  dropped and its ID is listed in `IngestionSummary.duplicates_skipped`; or
- different in ANY way -> `DuplicateDocumentError` is raised before any
  write. Nothing is merged and no record "wins" by input order.

Processing order
----------------
Documents are written in ascending `doc_id` order, not input order, so the
same set of documents always produces the same writes, the same summary
order and, after a failure, the same set of already-committed documents.

Atomicity (limited, on purpose)
-------------------------------
The store has no batch transaction, so EACH DOCUMENT is its own transaction.
If the store fails on document N, documents before N stay committed and
document N is rolled back. `BatchIngestionError` then reports the failing ID
and an `IngestionSummary` of what was committed. Re-running the same
ingestion is safe: finished documents come back "unchanged" and the rest are
retried. The batch as a whole is NOT atomic.

Chunks are retrieval units. The full document is always stored by the
store; this module never relies on chunks reproducing the document.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from knowledge.chunking import chunk_document
from knowledge.schema import SourceDocument
from knowledge.store import ChunkingConfig, IngestResult, KnowledgeStore, KnowledgeStoreError


# ============================================================
# EXCEPTIONS
# ============================================================

class IngestionError(Exception):
    """Base class for errors raised by this module."""


class InvalidDocumentInputError(IngestionError, ValueError):
    """The input is not an iterable of valid `SourceDocument` objects."""


class DuplicateDocumentError(IngestionError, ValueError):
    """Two different documents in one input share a `doc_id`. Raised before any write."""


# ============================================================
# RESULT
# ============================================================

@dataclass(frozen=True)
class IngestionSummary:
    """
    Outcome of one `ingest_documents` call.

    - `config`: the `ChunkingConfig` used for the chunker AND the store.
    - `results`: the store's `IngestResult` for each ingested document, in
      ascending `doc_id` order. Their `action` is "created", "updated" or
      "unchanged"; the content/metadata/provenance/config flags are the
      store's own and are not recomputed here.
    - `duplicates_skipped`: `doc_id` of each fully identical repeat that was
      dropped before ingestion (one entry per dropped repeat).
    """

    config: ChunkingConfig
    results: tuple[IngestResult, ...] = ()
    duplicates_skipped: tuple[str, ...] = ()

    @property
    def total(self) -> int:
        """Number of documents handed to the store."""
        return len(self.results)

    @property
    def created(self) -> tuple[str, ...]:
        """`doc_id`s that were new to the store."""
        return self._ids("created")

    @property
    def updated(self) -> tuple[str, ...]:
        """`doc_id`s whose stored document, chunks or configuration changed."""
        return self._ids("updated")

    @property
    def unchanged(self) -> tuple[str, ...]:
        """`doc_id`s that needed no write at all."""
        return self._ids("unchanged")

    def _ids(self, action: str) -> tuple[str, ...]:
        return tuple(result.doc_id for result in self.results if result.action == action)


class BatchIngestionError(IngestionError):
    """
    The store failed on one document in the middle of a batch.

    Attributes:
        failed_doc_id: the document whose ingestion failed (rolled back).
        completed: summary of the documents committed BEFORE the failure.
    The original `KnowledgeStoreError` is the `__cause__`.
    """

    def __init__(self, message: str, *, failed_doc_id: str, completed: IngestionSummary) -> None:
        super().__init__(message)
        self.failed_doc_id = failed_doc_id
        self.completed = completed


# ============================================================
# PRE-FLIGHT
# ============================================================

def _materialize(documents: Iterable[SourceDocument]) -> list[SourceDocument]:
    if isinstance(documents, (str, bytes, dict)):
        raise InvalidDocumentInputError(
            f"documents must be an iterable of SourceDocument objects, got {type(documents).__name__}"
        )
    try:
        return list(documents)
    except TypeError as exc:
        raise InvalidDocumentInputError(
            f"documents must be an iterable of SourceDocument objects, got {type(documents).__name__}"
        ) from exc


def _check_document(index: int, document: object) -> None:
    """Require a SourceDocument that re-validates, unchanged, through its own schema."""
    if not isinstance(document, SourceDocument):
        raise InvalidDocumentInputError(
            f"documents[{index}] must be a SourceDocument, got {type(document).__name__}"
        )
    try:
        revalidated = SourceDocument.model_validate(document.model_dump(mode="json"))
    except (ValidationError, TypeError, ValueError) as exc:
        raise InvalidDocumentInputError(
            f"documents[{index}] ({getattr(document, 'doc_id', '?')!r}) is not a valid "
            f"SourceDocument: {exc}"
        ) from exc
    if revalidated != document:
        raise InvalidDocumentInputError(
            f"documents[{index}] ({document.doc_id!r}) does not match its own re-validated form "
            "(it was probably built without validation)"
        )


def _resolve_duplicates(
    documents: list[SourceDocument],
) -> tuple[dict[str, SourceDocument], list[str]]:
    """Return (unique documents by doc_id, skipped identical repeats); raise on conflicts."""
    unique: dict[str, SourceDocument] = {}
    first_index: dict[str, int] = {}
    skipped: list[str] = []
    conflicts: dict[str, list[int]] = {}

    for index, document in enumerate(documents):
        doc_id = document.doc_id
        if doc_id not in unique:
            unique[doc_id] = document
            first_index[doc_id] = index
        elif unique[doc_id] == document:
            skipped.append(doc_id)
        else:
            conflicts.setdefault(doc_id, [first_index[doc_id]]).append(index)

    if conflicts:
        detail = "; ".join(
            f"{doc_id!r} at documents{indices}" for doc_id, indices in sorted(conflicts.items())
        )
        raise DuplicateDocumentError(
            f"conflicting records share a doc_id (nothing was ingested): {detail}"
        )
    return unique, skipped


# ============================================================
# PUBLIC API
# ============================================================

def ingest_documents(
    documents: Iterable[SourceDocument],
    store: KnowledgeStore,
    config: ChunkingConfig | None = None,
) -> IngestionSummary:
    """
    Chunk validated documents canonically and persist them in `store`.

    Args:
        documents: any iterable (list, generator, ...) of `SourceDocument`.
            It is consumed once and may be empty.
        store: an open `KnowledgeStore`.
        config: chunking settings; `ChunkingConfig()` (the chunker's
            defaults) if omitted. The same object configures the chunker
            and is passed to the store, so they cannot disagree.

    Returns:
        An `IngestionSummary` reflecting the store's actual outcomes. Empty
        input returns an empty summary and writes nothing.

    Raises:
        TypeError: `store` or `config` has the wrong type.
        InvalidDocumentInputError: an item is not a valid `SourceDocument`
            (raised before any write).
        DuplicateDocumentError: conflicting duplicates (before any write).
        IngestionError: the store is closed.
        BatchIngestionError: the store failed mid-batch; earlier documents
            stay committed (see the module docstring). Re-running is safe.
    """
    if not isinstance(store, KnowledgeStore):
        raise TypeError(f"store must be a KnowledgeStore, got {type(store).__name__}")
    if config is None:
        config = ChunkingConfig()
    elif not isinstance(config, ChunkingConfig):
        raise TypeError(f"config must be a ChunkingConfig, got {type(config).__name__}")
    if store.closed:
        raise IngestionError("the knowledge store is closed")

    # --- pre-flight: nothing below touches the database ---
    items = _materialize(documents)
    for index, item in enumerate(items):
        _check_document(index, item)
    unique, skipped = _resolve_duplicates(items)
    prepared = [
        (
            document,
            chunk_document(
                document,
                max_chars=config.max_chars,
                overlap_chars=config.overlap_chars,
                chunker_version=config.chunker_version,
            ),
        )
        for _doc_id, document in sorted(unique.items())
    ]

    # --- writes: one store transaction per document ---
    results: list[IngestResult] = []
    for document, chunks in prepared:
        try:
            results.append(store.ingest_document(document, chunks, config))
        except KnowledgeStoreError as exc:
            raise BatchIngestionError(
                f"ingesting {document.doc_id!r} failed after {len(results)} of "
                f"{len(prepared)} document(s) were committed: {exc}",
                failed_doc_id=document.doc_id,
                completed=IngestionSummary(config, tuple(results), tuple(skipped)),
            ) from exc
    return IngestionSummary(config, tuple(results), tuple(skipped))
