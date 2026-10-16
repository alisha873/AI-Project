"""
Phase 10 — Milestone 4B: the pinned MITRE corpus (manifest + orchestration).

    data/knowledge_sources/manifest.json   (pins versions, checksums, selection)
        -> load_manifest()                  strict validation of the manifest
        -> build_corpus_documents()         verify + parse BOTH archives (CWE, CAPEC)
        -> ingest_corpus()                  existing `ingest_documents` + `KnowledgeStore`

Everything runs on local files; nothing here uses the network.

All-or-nothing preparation: BOTH archives are verified and parsed into
validated documents BEFORE the first write, so a corrupt CAPEC archive cannot
leave a CWE-only corpus behind. After that the existing pipeline writes one
store transaction per document (the batch is not atomic; see
`ingestion.pipeline`). Rerunning is always safe: finished documents come back
"unchanged" and a failed document is retried.

The manifest also records the adapter ("parser") version it was verified
with. If an adapter changes (`ADAPTER_VERSION`), the manifest must be updated
deliberately, so changed extraction logic cannot slip in unnoticed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ingestion import mitre_capec, mitre_cwe
from ingestion.mitre_xml import (
    OFFICIAL_RECORD_URLS,
    ArtifactSpec,
    MitreSourceError,
    SelectedRecord,
)
from ingestion.pipeline import IngestionSummary, ingest_documents
from knowledge.schema import SourceDocument
from knowledge.store import ChunkingConfig, KnowledgeStore

MANIFEST_FORMAT = "autoshield.mitre_corpus_manifest.v1"

_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ADAPTER_VERSIONS = {"cwe": mitre_cwe.ADAPTER_VERSION, "capec": mitre_capec.ADAPTER_VERSION}


class ManifestError(MitreSourceError):
    """The manifest is unreadable, malformed, inconsistent or out of date with the code."""


# ============================================================
# MANIFEST
# ============================================================

@dataclass(frozen=True)
class SourceEntry:
    """One source (CWE or CAPEC) as pinned in the manifest."""

    key: str
    title: str
    spec: ArtifactSpec
    official_url: str
    terms_url: str
    selection: tuple[SelectedRecord, ...]


@dataclass(frozen=True)
class CorpusManifest:
    path: Path
    description: str
    corpus_build_utc: str
    cwe: SourceEntry
    capec: SourceEntry


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate key {key!r} in the manifest")
        result[key] = value
    return result


def _object(value: object, where: str, required: set[str], optional: set[str] = frozenset()) -> dict:
    """An object with exactly the required keys (plus any optional ones); nothing else."""
    if not isinstance(value, dict):
        raise ManifestError(f"{where} must be an object")
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise ManifestError(f"{where} is missing key(s) {missing}")
    if unknown:
        raise ManifestError(f"{where} has unknown key(s) {unknown}")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{where} must be a non-empty string")
    return value


def _timestamp(value: object, where: str) -> str:
    text = _text(value, where)
    if not _TIMESTAMP_RE.fullmatch(text):
        raise ManifestError(f"{where} must look like 2026-01-31T12:00:00Z, got {text!r}")
    return text


def _source_entry(key: str, raw: object, build_utc: str) -> SourceEntry:
    where = f"sources.{key}"
    data = _object(
        raw,
        where,
        {"title", "version", "catalog_date", "parser_version", "official_pages", "archive",
         "record_url_pattern", "license", "selected"},
    )
    pages = _object(data["official_pages"], f"{where}.official_pages", {"downloads", "archive", "terms_of_use"})
    archive = _object(
        data["archive"],
        f"{where}.archive",
        {"filename", "official_url", "sha256", "xml_member", "retrieved_at_utc"},
    )

    if _text(data["parser_version"], f"{where}.parser_version") != _ADAPTER_VERSIONS[key]:
        raise ManifestError(
            f"{where}.parser_version is {data['parser_version']!r} but the adapter in the code is "
            f"{_ADAPTER_VERSIONS[key]!r}; re-verify the extraction, then update the manifest"
        )
    if data["record_url_pattern"] != OFFICIAL_RECORD_URLS[key]:
        raise ManifestError(f"{where}.record_url_pattern differs from the pattern used by the adapter")
    sha = _text(archive["sha256"], f"{where}.archive.sha256")
    if not _SHA256_RE.fullmatch(sha):
        raise ManifestError(f"{where}.archive.sha256 must be 64 lowercase hexadecimal characters")
    filename = _text(archive["filename"], f"{where}.archive.filename")
    if "/" in filename or "\\" in filename:
        raise ManifestError(f"{where}.archive.filename must be a bare file name")

    selected = data["selected"]
    if not isinstance(selected, list) or not selected:
        raise ManifestError(f"{where}.selected must be a non-empty list")
    selection: list[SelectedRecord] = []
    for index, entry in enumerate(selected):
        item = _object(entry, f"{where}.selected[{index}]", {"id", "name"}, {"status"})
        try:
            selection.append(
                SelectedRecord(
                    _text(item["id"], f"{where}.selected[{index}].id"),
                    _text(item["name"], f"{where}.selected[{index}].name"),
                    None if "status" not in item else _text(item["status"], f"{where}.selected[{index}].status"),
                )
            )
        except MitreSourceError as exc:
            raise ManifestError(f"{where}.selected[{index}]: {exc}") from exc
    ids = [item.record_id for item in selection]
    if len(set(ids)) != len(ids):
        raise ManifestError(f"{where}.selected repeats an ID")

    try:
        spec = ArtifactSpec(
            source_version=_text(data["version"], f"{where}.version"),
            catalog_date=_text(data["catalog_date"], f"{where}.catalog_date"),
            archive_sha256=sha,
            xml_member=_text(archive["xml_member"], f"{where}.archive.xml_member"),
            artifact_filename=filename,
            license=_text(data["license"], f"{where}.license"),
            retrieved_at_utc=_timestamp(archive["retrieved_at_utc"], f"{where}.archive.retrieved_at_utc"),
            ingested_at_utc=build_utc,
        )
    except MitreSourceError as exc:
        raise ManifestError(f"{where}: {exc}") from exc
    return SourceEntry(
        key=key,
        title=_text(data["title"], f"{where}.title"),
        spec=spec,
        official_url=_text(archive["official_url"], f"{where}.archive.official_url"),
        terms_url=_text(pages["terms_of_use"], f"{where}.official_pages.terms_of_use"),
        selection=tuple(selection),
    )


def load_manifest(path: str | Path) -> CorpusManifest:
    """
    Read and strictly validate the manifest. Unknown or missing keys, duplicate
    keys, malformed checksums or timestamps, a manifest that claims to be the
    complete catalogue, or a parser version that disagrees with the code all
    raise `ManifestError`.
    """
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {manifest_path}: {exc}") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest {manifest_path} is not valid JSON: {exc}") from exc

    top = _object(
        raw,
        "manifest",
        {"manifest_format", "description", "is_complete_catalogue", "corpus_build_utc", "sources"},
        {"corpus_build_utc_note"},
    )
    if top["manifest_format"] != MANIFEST_FORMAT:
        raise ManifestError(f"manifest_format must be {MANIFEST_FORMAT!r}, got {top['manifest_format']!r}")
    if top["is_complete_catalogue"] is not False:
        raise ManifestError("is_complete_catalogue must be false: this manifest pins a bounded subset")
    build_utc = _timestamp(top["corpus_build_utc"], "corpus_build_utc")
    sources = _object(top["sources"], "sources", {"cwe", "capec"})
    return CorpusManifest(
        path=manifest_path,
        description=_text(top["description"], "description"),
        corpus_build_utc=build_utc,
        cwe=_source_entry("cwe", sources["cwe"], build_utc),
        capec=_source_entry("capec", sources["capec"], build_utc),
    )


# ============================================================
# BUILD AND INGEST
# ============================================================

def build_corpus_documents(
    manifest: CorpusManifest, *, cwe_archive: str | Path, capec_archive: str | Path
) -> list[SourceDocument]:
    """
    Verify and parse both pinned archives; return the curated documents (CWE
    first, then CAPEC, each in manifest order). Any failure in either source
    raises before anything is returned, so no partial corpus exists.
    """
    cwe_docs = mitre_cwe.parse_cwe_archive(cwe_archive, manifest.cwe.spec, manifest.cwe.selection)
    capec_docs = mitre_capec.parse_capec_archive(capec_archive, manifest.capec.spec, manifest.capec.selection)
    return cwe_docs + capec_docs


@dataclass(frozen=True)
class CorpusReport:
    """Outcome of a complete, successful corpus build."""

    cwe_version: str
    capec_version: str
    selected_cwe: int
    selected_capec: int
    summary: IngestionSummary

    @property
    def ingested(self) -> int:
        return self.summary.total


def ingest_corpus_documents(
    manifest: CorpusManifest,
    documents: list[SourceDocument],
    *,
    store: KnowledgeStore,
    config: ChunkingConfig | None = None,
) -> CorpusReport:
    """
    Ingest documents already produced by `build_corpus_documents` through the
    existing pipeline with one `ChunkingConfig`. Callers that need to open a
    database only AFTER the archives are verified (the CLI) use this directly:
    a failed verification then never creates or touches a database file.
    Raises on any failure (including `BatchIngestionError` from the pipeline);
    a report is returned only for a complete build.
    """
    summary = ingest_documents(documents, store, config)
    expected = len(manifest.cwe.selection) + len(manifest.capec.selection)
    if summary.total != expected:
        raise MitreSourceError(
            f"ingested {summary.total} document(s) but the manifest selects {expected}; "
            "the corpus build is incomplete"
        )
    return CorpusReport(
        cwe_version=manifest.cwe.spec.source_version,
        capec_version=manifest.capec.spec.source_version,
        selected_cwe=len(manifest.cwe.selection),
        selected_capec=len(manifest.capec.selection),
        summary=summary,
    )


def ingest_corpus(
    manifest: CorpusManifest,
    *,
    cwe_archive: str | Path,
    capec_archive: str | Path,
    store: KnowledgeStore,
    config: ChunkingConfig | None = None,
) -> CorpusReport:
    """Verify and parse both archives, then ingest (`build_corpus_documents` + `ingest_corpus_documents`)."""
    documents = build_corpus_documents(manifest, cwe_archive=cwe_archive, capec_archive=capec_archive)
    return ingest_corpus_documents(manifest, documents, store=store, config=config)
