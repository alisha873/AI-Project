"""
Phase 10 — Knowledge ingestion.

- `load_documents(path)`: local JSON file -> validated `SourceDocument`s.
- `ingest_documents(documents, store, config=None)`: validated documents ->
  canonical chunks -> `KnowledgeStore`.

The two are independent: the pipeline accepts documents from any source.
"""

from ingestion.local_json import FORMAT_ID, LocalJsonError, load_documents
from ingestion.pipeline import (
    BatchIngestionError,
    DuplicateDocumentError,
    IngestionError,
    IngestionSummary,
    InvalidDocumentInputError,
    ingest_documents,
)

__all__ = [
    "FORMAT_ID",
    "BatchIngestionError",
    "DuplicateDocumentError",
    "IngestionError",
    "IngestionSummary",
    "InvalidDocumentInputError",
    "LocalJsonError",
    "ingest_documents",
    "load_documents",
]
