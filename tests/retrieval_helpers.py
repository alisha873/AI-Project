"""
Shared helpers for the Phase 11 retrieval tests (not a test module).

Every document built here is SYNTHETIC test text (the content is invented and
only meant to make keyword matches predictable). It uses the real schema,
chunker and store; it is not security knowledge and is never loaded into a real
corpus. Identifiers such as "CWE-1" are made up for the test.
"""

from __future__ import annotations

from pathlib import Path

from knowledge.schema import ProvenanceRecord, SourceDocument
from knowledge.store import KnowledgeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"
BENCHMARK_PATH = REPO_ROOT / "data" / "retrieval_benchmark" / "retrieval_benchmark_v1.json"
RAW_DIR = REPO_ROOT / "data" / "knowledge_sources" / "raw"
REAL_CWE_ZIP = RAW_DIR / "cwec_v4.20.xml.zip"
REAL_CAPEC_ZIP = RAW_DIR / "capec_v3.9.zip"
INCIDENT_001 = REPO_ROOT / "incident_001"

ARCHIVE_SKIP_REASON = (
    "the pinned official archives are not in data/knowledge_sources/raw/ "
    "(download them from the official URLs in the manifest); real-corpus retrieval is NOT tested"
)


def archives_present() -> bool:
    return REAL_CWE_ZIP.is_file() and REAL_CAPEC_ZIP.is_file()


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        origin_kind="bundled_fixture",
        artifact_path="fixtures/retrieval_tests.json",
        artifact_sha256="0" * 64,
        adapter="retrieval-tests",
        adapter_version="1",
        retrieved_at_utc="2026-10-16T00:00:00Z",
        ingested_at_utc="2026-10-16T00:00:00Z",
    )


def make_doc(
    source_id: str,
    title: str,
    content: str,
    *,
    source: str = "cwe",
    doc_type: str = "weakness",
) -> SourceDocument:
    return SourceDocument(
        source=source,
        source_id=source_id,
        source_version="test-1",
        doc_type=doc_type,
        title=title,
        content=content,
        source_url=f"https://example.invalid/{source_id}",
        provenance=_provenance(),
    )


def make_store(*documents: SourceDocument) -> KnowledgeStore:
    """An in-memory store holding `documents` (caller closes it, usually via `with`)."""
    store = KnowledgeStore(":memory:")
    for document in documents:
        store.ingest_document(document)
    return store


def small_corpus() -> list[SourceDocument]:
    """Four synthetic documents with clearly separated vocabulary."""
    return [
        make_doc("CWE-1", "Alpha Weakness", "alpha alpha database query built from request text"),
        make_doc("CWE-2", "Beta Weakness", "beta beta filesystem path escapes the allowed folder"),
        make_doc(
            "CAPEC-1", "Alpha Attack", "alpha attacker sends crafted request to the database",
            source="capec", doc_type="attack_pattern",
        ),
        make_doc(
            "CAPEC-2", "Gamma Attack", "gamma attacker reaches an internal service",
            source="capec", doc_type="attack_pattern",
        ),
    ]
