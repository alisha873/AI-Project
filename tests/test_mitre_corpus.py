"""
Tests for `ingestion.mitre_corpus` and `scripts.ingest_mitre_corpus`.

Three tiers, kept apart on purpose:
1. MANIFEST tests use the committed `data/knowledge_sources/manifest.json`.
2. EXTRACT tests (always run, no network): the verbatim official extracts from
   `tests/fixtures/mitre/` go through the real adapters, the real pipeline, the
   real canonical chunker and a real SQLite store. Small, but a genuine source structure.
3. REAL-ARCHIVE tests (`needs_real_archives`): the pinned official ZIPs in
   `data/knowledge_sources/raw/` (gitignored, downloaded by hand). They are SKIPPED
   with an explicit reason when those files are absent, so a green run without the
   archives does NOT demonstrate the real-corpus integration.

The keyword-retrieval checks are a sanity check that distinctive source words reach
the expected records through FTS5. They say nothing about semantic relevance,
forensic reasoning or remediation quality.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

from ingestion import mitre_capec, mitre_cwe
from ingestion.mitre_corpus import (
    CorpusManifest,
    ManifestError,
    build_corpus_documents,
    ingest_corpus,
    load_manifest,
)
from ingestion.mitre_xml import (
    OFFICIAL_RECORD_URLS,
    ArchiveError,
    ChecksumMismatchError,
    MitreSourceError,
    SelectedRecord,
    SourceVersionError,
)
from knowledge.chunking import CHUNKER_VERSION, chunk_document
from knowledge.store import ChunkingConfig, KnowledgeStore
from scripts.ingest_mitre_corpus import main as cli_main
from tests.mitre_helpers import (
    FIXTURE_DIR,
    Archive,
    capec_extract_archive,
    cwe_archive,
    cwe_extract_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"
NOTICE_PATH = REPO_ROOT / "data" / "knowledge_sources" / "NOTICE.md"
RAW_DIR = REPO_ROOT / "data" / "knowledge_sources" / "raw"
REAL_CWE_ZIP = RAW_DIR / "cwec_v4.20.xml.zip"
REAL_CAPEC_ZIP = RAW_DIR / "capec_v3.9.zip"

needs_real_archives = pytest.mark.skipif(
    not (REAL_CWE_ZIP.is_file() and REAL_CAPEC_ZIP.is_file()),
    reason="the pinned official archives are not in data/knowledge_sources/raw/ "
    "(download them from the official URLs in the manifest); real-corpus integration is NOT tested",
)

EXPECTED_CWE_IDS = ["20", "22", "23", "74", "77", "78", "79", "89", "94", "287", "306", "862", "918"]
EXPECTED_CAPEC_IDS = ["7", "66", "88", "108", "115", "126", "137", "242", "248", "664"]


# ============================================================
# MANIFEST (committed file)
# ============================================================

@pytest.fixture(scope="module")
def manifest() -> CorpusManifest:
    """The committed manifest, loaded (and strictly validated) once per module."""
    return load_manifest(MANIFEST_PATH)


class TestCommittedManifest:
    def test_pins_the_intended_releases(self, manifest):
        assert manifest.cwe.spec.source_version == "4.20"
        assert manifest.cwe.spec.catalog_date == "2026-04-30"
        assert manifest.capec.spec.source_version == "3.9"
        assert manifest.capec.spec.catalog_date == "2023-01-24"

    def test_selection_is_the_bounded_curated_subset(self, manifest):
        assert [s.record_id for s in manifest.cwe.selection] == EXPECTED_CWE_IDS
        assert [s.record_id for s in manifest.capec.selection] == EXPECTED_CAPEC_IDS
        assert 10 <= len(manifest.cwe.selection) <= 15
        assert 8 <= len(manifest.capec.selection) <= 12
        assert all(s.name and s.status for s in manifest.cwe.selection + manifest.capec.selection)

    def test_archives_are_pinned_by_real_looking_checksums_and_official_urls(self, manifest):
        for entry, host in ((manifest.cwe, "https://cwe.mitre.org/"), (manifest.capec, "https://capec.mitre.org/")):
            assert re.fullmatch(r"[0-9a-f]{64}", entry.spec.archive_sha256)
            assert entry.spec.archive_sha256 != "0" * 64
            assert entry.official_url.startswith(host)
            assert "latest" not in entry.official_url  # versioned archive, not a mutable alias
            assert entry.terms_url.startswith(host) and entry.terms_url.endswith("/about/termsofuse.html")
        assert manifest.cwe.spec.artifact_filename == "cwec_v4.20.xml.zip"
        assert manifest.capec.spec.artifact_filename == "capec_v3.9.zip"

    def test_manifest_says_it_is_a_subset_and_uses_the_build_time_for_ingestion(self, manifest):
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        assert data["is_complete_catalogue"] is False
        assert "BOUNDED" in data["description"]
        assert manifest.cwe.spec.ingested_at_utc == manifest.capec.spec.ingested_at_utc == manifest.corpus_build_utc

    def test_licence_text_carries_the_mitre_copyright_designation_and_terms(self, manifest):
        for entry in (manifest.cwe, manifest.capec):
            assert "Copyright \u00a9" in entry.spec.license
            assert "The MITRE Corporation" in entry.spec.license
            assert entry.terms_url in entry.spec.license
            assert "NOTICE.md" in entry.spec.license

    def test_notice_file_exists_and_covers_both_sources(self):
        notice = NOTICE_PATH.read_text(encoding="utf-8")
        for needle in (
            "CWE", "CAPEC", "The MITRE Corporation", "https://cwe.mitre.org/about/termsofuse.html",
            "https://capec.mitre.org/about/termsofuse.html", "reproduce MITRE",
        ):
            assert needle in notice

    def test_parser_versions_match_the_code(self):
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sources"]
        assert data["cwe"]["parser_version"] == mitre_cwe.ADAPTER_VERSION
        assert data["capec"]["parser_version"] == mitre_capec.ADAPTER_VERSION
        assert data["cwe"]["record_url_pattern"] == OFFICIAL_RECORD_URLS["cwe"]


def mutated_manifest(tmp_path: Path, mutate) -> Path:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestManifestStrictness:
    def test_unchanged_copy_loads(self, tmp_path):
        assert load_manifest(mutated_manifest(tmp_path, lambda d: None)).cwe.key == "cwe"

    @pytest.mark.parametrize(
        "mutate, message",
        [
            (lambda d: d.update(extra=1), "unknown key"),
            (lambda d: d.pop("sources"), "missing key"),
            (lambda d: d["sources"]["cwe"].update(surprise=True), "unknown key"),
            (lambda d: d["sources"]["cwe"]["archive"].pop("sha256"), "missing key"),
            (lambda d: d.update(manifest_format="other.v9"), "manifest_format"),
            (lambda d: d.update(is_complete_catalogue=True), "bounded subset"),
            (lambda d: d.update(corpus_build_utc="yesterday"), "corpus_build_utc"),
            (lambda d: d["sources"]["cwe"]["archive"].update(sha256="ABC"), "sha256"),
            (lambda d: d["sources"]["cwe"]["archive"].update(sha256="0" * 63), "sha256"),
            (lambda d: d["sources"]["capec"]["archive"].update(filename="raw/x.zip"), "bare file name"),
            (lambda d: d["sources"]["cwe"]["archive"].update(retrieved_at_utc="2026-10-15"), "retrieved_at_utc"),
            (lambda d: d["sources"]["cwe"].update(parser_version="mitre-cwe-0"), "parser_version"),
            (lambda d: d["sources"]["capec"].update(record_url_pattern="https://example.org/{id}"), "record_url_pattern"),
            (lambda d: d["sources"]["cwe"].update(selected=[]), "non-empty list"),
            (lambda d: d["sources"]["cwe"]["selected"].append(dict(d["sources"]["cwe"]["selected"][0])), "repeats"),
            (lambda d: d["sources"]["cwe"]["selected"][0].update(id="CWE-20"), "digits"),
            (lambda d: d["sources"]["cwe"]["selected"][0].pop("name"), "missing key"),
            (lambda d: d["sources"]["cwe"].update(license=""), "license"),
        ],
    )
    def test_invalid_manifests_are_rejected(self, tmp_path, mutate, message):
        with pytest.raises(ManifestError, match=message):
            load_manifest(mutated_manifest(tmp_path, mutate))

    def test_duplicate_json_keys_are_rejected(self, tmp_path):
        path = tmp_path / "dup.json"
        path.write_text('{"manifest_format": "a", "manifest_format": "b"}', encoding="utf-8")
        with pytest.raises(ManifestError, match="duplicate key"):
            load_manifest(path)

    def test_invalid_json_and_missing_file_are_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ManifestError, match="not valid JSON"):
            load_manifest(bad)
        with pytest.raises(ManifestError, match="cannot read manifest"):
            load_manifest(tmp_path / "missing.json")

    def test_manifest_errors_are_mitre_source_errors(self, tmp_path):
        with pytest.raises(MitreSourceError):
            load_manifest(tmp_path / "missing.json")


# ============================================================
# EXTRACT TIER (verbatim official extracts, no network)
# ============================================================

EXTRACT_CWE = (
    SelectedRecord("918", "Server-Side Request Forgery (SSRF)", "Incomplete"),
    SelectedRecord("1187", "DEPRECATED: Use of Uninitialized Resource", "Deprecated"),
)
EXTRACT_CAPEC = (
    SelectedRecord("664", "Server Side Request Forgery", "Stable"),
    SelectedRecord("137", "Parameter Injection", "Stable"),
    SelectedRecord("106", "DEPRECATED: XSS through Log Files", "Deprecated"),
    SelectedRecord("242", "Code Injection", "Stable"),
)
EXTRACT_DOC_IDS = [
    "cwe:CWE-918", "cwe:CWE-1187",
    "capec:CAPEC-664", "capec:CAPEC-137", "capec:CAPEC-106", "capec:CAPEC-242",
]


def _source_json(key: str, archive: Archive, selection, *, version: str, date: str, parser: str) -> dict:
    host = f"https://{key}.mitre.org"
    return {
        "title": f"{key.upper()} extract (test)",
        "version": version,
        "catalog_date": date,
        "parser_version": parser,
        "official_pages": {
            "downloads": f"{host}/data/downloads.html",
            "archive": f"{host}/data/archive.html",
            "terms_of_use": f"{host}/about/termsofuse.html",
        },
        "archive": {
            "filename": archive.spec.artifact_filename,
            "official_url": f"{host}/test/{archive.spec.artifact_filename}",
            "sha256": archive.sha256,
            "xml_member": archive.spec.xml_member,
            "retrieved_at_utc": archive.spec.retrieved_at_utc,
        },
        "record_url_pattern": OFFICIAL_RECORD_URLS[key],
        "license": archive.spec.license,
        "selected": [
            {k: v for k, v in (("id", s.record_id), ("name", s.name), ("status", s.status)) if v is not None}
            for s in selection
        ],
    }


def write_extract_manifest(
    directory: Path, cwe: Archive, capec: Archive, *, cwe_version: str = "4.20"
) -> Path:
    data = {
        "manifest_format": "autoshield.mitre_corpus_manifest.v1",
        "description": "TEST manifest for verbatim extracts (a bounded subset)",
        "is_complete_catalogue": False,
        "corpus_build_utc": "2026-10-15T22:53:30Z",
        "sources": {
            "cwe": _source_json("cwe", cwe, EXTRACT_CWE, version=cwe_version, date="2026-04-30", parser=mitre_cwe.ADAPTER_VERSION),
            "capec": _source_json("capec", capec, EXTRACT_CAPEC, version="3.9", date="2023-01-24", parser=mitre_capec.ADAPTER_VERSION),
        },
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class Extract:
    """An extract-based corpus on disk: archives, manifest file and loaded manifest."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.cwe = cwe_extract_archive(directory)
        self.capec = capec_extract_archive(directory)
        self.manifest_path = write_extract_manifest(directory, self.cwe, self.capec)
        self.manifest = load_manifest(self.manifest_path)

    def ingest(self, store: KnowledgeStore, **kwargs):
        return ingest_corpus(
            self.manifest, cwe_archive=self.cwe.path, capec_archive=self.capec.path, store=store, **kwargs
        )


@pytest.fixture()
def extract(tmp_path) -> Extract:
    return Extract(tmp_path / "corpus")


def doc_ids_for(store: KnowledgeStore, query: str) -> set[str]:
    return {hit.chunk.doc_id for hit in store.search(query, limit=100)}


class TestExtractEndToEnd:
    def test_build_returns_validated_documents_cwe_first_then_capec(self, extract):
        docs = build_corpus_documents(extract.manifest, cwe_archive=extract.cwe.path, capec_archive=extract.capec.path)
        assert [d.doc_id for d in docs] == EXTRACT_DOC_IDS
        assert all(d.provenance.artifact_sha256 in (extract.cwe.sha256, extract.capec.sha256) for d in docs)

    def test_ingestion_persists_documents_and_real_canonical_chunks(self, extract):
        with KnowledgeStore(":memory:") as store:
            report = extract.ingest(store)
            assert report.ingested == 6
            assert (report.selected_cwe, report.selected_capec) == (2, 4)
            assert (report.cwe_version, report.capec_version) == ("4.20", "3.9")
            assert sorted(report.summary.created) == sorted(EXTRACT_DOC_IDS)
            assert store.count_documents() == 6

            docs = build_corpus_documents(extract.manifest, cwe_archive=extract.cwe.path, capec_archive=extract.capec.path)
            for doc in docs:
                stored_chunks = store.get_chunks(doc.doc_id)
                assert list(stored_chunks) == list(chunk_document(doc))  # the canonical chunker, default config
                assert all(c.chunker_version == CHUNKER_VERSION for c in stored_chunks)
            assert len(store.get_chunks("cwe:CWE-918")) > 1  # real record text exceeds one chunk
            assert store.get_document("cwe:CWE-918") == docs[0]

    def test_one_chunking_config_reaches_chunker_and_store(self, extract):
        config = ChunkingConfig(max_chars=500, overlap_chars=50)
        with KnowledgeStore(":memory:") as store:
            extract.ingest(store, config=config)
            chunks = [c for doc_id in store.list_doc_ids() for c in store.get_chunks(doc_id)]
            assert chunks and all(len(c.text) <= 500 for c in chunks)
            assert len(store.get_chunks("cwe:CWE-918")) > len(chunk_document(store.get_document("cwe:CWE-918")))

    def test_distinctive_source_terms_are_searchable(self, extract):
        with KnowledgeStore(":memory:") as store:
            extract.ingest(store)
            assert doc_ids_for(store, "XSPA") == {"cwe:CWE-918"}
            assert doc_ids_for(store, '"cross site port attack"') == {"cwe:CWE-918"}
            assert doc_ids_for(store, '"server side request forgery"') >= {"cwe:CWE-918", "capec:CAPEC-664"}
            assert doc_ids_for(store, '"parameter injection"') >= {"capec:CAPEC-137"}
            assert doc_ids_for(store, '"deprecated because"') == {"cwe:CWE-1187"}

    def test_relationships_survive_the_persistence_path(self, extract):
        with KnowledgeStore(":memory:") as store:
            extract.ingest(store)
            capec = store.get_document("capec:CAPEC-664")
            cwe = store.get_document("cwe:CWE-918")
            weakness_links = {r.target_doc_id for r in capec.references if r.relation == "RelatedWeakness"}
            pattern_links = {r.target_doc_id for r in cwe.references if r.relation == "RelatedAttackPattern"}
            assert weakness_links == {"cwe:CWE-918", "cwe:CWE-20"}
            assert pattern_links == {"capec:CAPEC-664"}
            assert store.has_document("cwe:CWE-918") and store.has_document("capec:CAPEC-664")  # resolvable
            # Retained although the target is not part of this corpus:
            assert "cwe:CWE-441" in {r.target_doc_id for r in cwe.references}
            assert not store.has_document("cwe:CWE-441")

    def test_provenance_and_attribution_are_persisted(self, extract):
        with KnowledgeStore(":memory:") as store:
            extract.ingest(store)
            doc = store.get_document("capec:CAPEC-664")
            assert doc.provenance.artifact_sha256 == extract.capec.sha256
            assert doc.license == extract.capec.spec.license
            assert doc.source_url == "https://capec.mitre.org/data/definitions/664.html"

    def test_repeated_build_causes_no_writes(self, extract):
        database = extract.directory / "kb.sqlite"
        with KnowledgeStore(database) as store:
            first = extract.ingest(store)
        assert len(first.summary.created) == 6
        digest = hashlib.sha256(database.read_bytes()).hexdigest()
        with KnowledgeStore(database) as store:
            second = extract.ingest(store)
        assert second.summary.created == () and second.summary.updated == ()
        assert sorted(second.summary.unchanged) == sorted(EXTRACT_DOC_IDS)
        assert not any(
            r.content_changed or r.metadata_changed or r.provenance_changed or r.config_changed or r.chunks_replaced
            for r in second.summary.results
        )
        assert hashlib.sha256(database.read_bytes()).hexdigest() == digest  # database file untouched

    def test_changed_source_content_is_detected_not_silently_accepted(self, extract, tmp_path):
        with KnowledgeStore(":memory:") as store:
            extract.ingest(store)
            revised_xml = (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_bytes().replace(
                b"Cross Site Port Attack", b"Cross Site Port Attack REVISED"
            )
            new_dir = tmp_path / "revised"
            new_dir.mkdir()
            new_cwe = cwe_archive(new_dir, revised_xml)
            assert new_cwe.sha256 != extract.cwe.sha256
            manifest = load_manifest(write_extract_manifest(new_dir, new_cwe, extract.capec))
            report = ingest_corpus(manifest, cwe_archive=new_cwe.path, capec_archive=extract.capec.path, store=store)

            results = {r.doc_id: r for r in report.summary.results}
            assert results["cwe:CWE-918"].action == "updated" and results["cwe:CWE-918"].content_changed
            # same text, but its artifact checksum changed: only the provenance differs
            assert results["cwe:CWE-1187"].action == "updated" and results["cwe:CWE-1187"].provenance_changed
            assert not results["cwe:CWE-1187"].content_changed
            assert all(results[d].action == "unchanged" for d in EXTRACT_DOC_IDS[2:])
            assert doc_ids_for(store, "REVISED") == {"cwe:CWE-918"}

    def test_a_new_source_version_is_detected_as_a_metadata_change(self, extract, tmp_path):
        with KnowledgeStore(":memory:") as store:
            extract.ingest(store)
            xml = (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_bytes().replace(b'Version="4.20"', b'Version="4.21"', 1)
            new_dir = tmp_path / "v421"
            new_dir.mkdir()
            new_cwe = cwe_archive(new_dir, xml, version="4.21", filename="cwec_v4.21.xml.zip", member="cwec_v4.21.xml")
            manifest = load_manifest(write_extract_manifest(new_dir, new_cwe, extract.capec, cwe_version="4.21"))
            report = ingest_corpus(manifest, cwe_archive=new_cwe.path, capec_archive=extract.capec.path, store=store)
            results = {r.doc_id: r for r in report.summary.results}
            assert results["cwe:CWE-918"].action == "updated" and results["cwe:CWE-918"].metadata_changed
            assert not results["cwe:CWE-918"].content_changed
            assert store.get_document("cwe:CWE-918").source_version == "4.21"

    def test_archive_declaring_another_version_than_pinned_fails(self, extract, tmp_path):
        xml = (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_bytes().replace(b'Version="4.20"', b'Version="4.21"', 1)
        new_dir = tmp_path / "other"
        new_dir.mkdir()
        other = cwe_archive(new_dir, xml)
        # The manifest pins 4.20 but this archive (checksum re-pinned to be consistent) declares 4.21.
        manifest = load_manifest(write_extract_manifest(new_dir, other, extract.capec))
        with KnowledgeStore(":memory:") as store:
            with pytest.raises(SourceVersionError, match="4.21"):
                ingest_corpus(manifest, cwe_archive=other.path, capec_archive=extract.capec.path, store=store)
            assert store.count_documents() == 0

    def test_a_bad_second_source_prevents_any_write_even_though_the_first_is_fine(self, extract):
        corrupt = extract.directory / "capec_v3.9.zip"
        corrupt.write_bytes(corrupt.read_bytes() + b"tampered")
        with KnowledgeStore(":memory:") as store:
            with pytest.raises(ChecksumMismatchError):
                extract.ingest(store)
            assert store.count_documents() == 0

    def test_missing_archive_fails_clearly_and_writes_nothing(self, extract):
        with KnowledgeStore(":memory:") as store:
            with pytest.raises(ArchiveError, match="not found"):
                ingest_corpus(extract.manifest, cwe_archive=extract.cwe.path,
                              capec_archive=extract.directory / "gone.zip", store=store)
            assert store.count_documents() == 0

    def test_a_curated_record_missing_from_the_archive_fails_the_whole_build(self, extract, tmp_path):
        # Archive whose CWE-1187 record is gone (checksum re-pinned): the build must not "succeed" with 5 of 6.
        text = (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_text(encoding="utf-8")
        text = re.sub(r'<Weakness ID="1187"[\s\S]*?</Weakness>', "", text)
        new_dir = tmp_path / "missing"
        new_dir.mkdir()
        short = cwe_archive(new_dir, text.encode("utf-8"))
        manifest = load_manifest(write_extract_manifest(new_dir, short, extract.capec))
        with KnowledgeStore(":memory:") as store:
            with pytest.raises(MitreSourceError, match="1187 is missing"):
                ingest_corpus(manifest, cwe_archive=short.path, capec_archive=extract.capec.path, store=store)
            assert store.count_documents() == 0


# ============================================================
# COMMAND LINE (extract tier)
# ============================================================

def run_cli(extract: Extract, database: Path, *extra: str) -> int:
    return cli_main([
        "--cwe-archive", str(extract.cwe.path),
        "--capec-archive", str(extract.capec.path),
        "--database", str(database),
        "--manifest", str(extract.manifest_path),
        *extra,
    ])


def raw_sql(database: Path, statement: str) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


class TestCommandLine:
    def test_success_reports_versions_counts_and_exits_zero(self, extract, capsys):
        database = extract.directory / "out" / "kb.sqlite"
        assert run_cli(extract, database) == 0
        out = capsys.readouterr().out
        assert "Result: OK - complete corpus build" in out
        assert "CWE version 4.20, CAPEC version 3.9" in out
        assert "selected 6 (2 CWE + 4 CAPEC); ingested 6" in out
        assert "created 6, updated 0, unchanged 0" in out
        assert "errors: none" in out
        with KnowledgeStore(database) as store:
            assert store.count_documents() == 6

    def test_rerun_reports_everything_unchanged(self, extract, capsys):
        database = extract.directory / "kb.sqlite"
        assert run_cli(extract, database) == 0
        capsys.readouterr()
        assert run_cli(extract, database) == 0
        assert "created 0, updated 0, unchanged 6" in capsys.readouterr().out

    def test_the_store_is_closed_cleanly(self, extract):
        database = extract.directory / "kb.sqlite"
        assert run_cli(extract, database) == 0
        database.unlink()  # impossible on Windows if the SQLite handle were still open

    def test_missing_manifest_fails_with_a_nonzero_status(self, extract, capsys):
        database = extract.directory / "kb.sqlite"
        extract.manifest_path.unlink()
        assert run_cli(extract, database) == 1
        assert "manifest error" in capsys.readouterr().err
        assert not database.exists()

    def test_missing_archive_fails_and_never_claims_success(self, extract, capsys):
        database = extract.directory / "kb.sqlite"
        extract.capec.path.unlink()
        assert run_cli(extract, database) == 1
        captured = capsys.readouterr()
        assert "FAILED" in captured.err and "not found" in captured.err
        assert "Result: OK" not in captured.out
        assert not database.exists()  # verification failed before the database was opened

    def test_checksum_mismatch_fails_with_an_actionable_message(self, extract, capsys):
        database = extract.directory / "kb.sqlite"
        extract.cwe.path.write_bytes(extract.cwe.path.read_bytes() + b"x")
        assert run_cli(extract, database) == 1
        captured = capsys.readouterr()
        assert "checksum mismatch" in captured.err and "update the manifest" in captured.err
        assert "Result: OK" not in captured.out
        assert not database.exists()  # verification failed before the database was opened

    def test_a_failed_verification_leaves_an_existing_database_byte_identical(self, extract, capsys):
        database = extract.directory / "kb.sqlite"
        assert run_cli(extract, database) == 0
        capsys.readouterr()
        before = hashlib.sha256(database.read_bytes()).hexdigest()
        extract.cwe.path.write_bytes(extract.cwe.path.read_bytes() + b"x")
        assert run_cli(extract, database) == 1
        assert "Result: OK" not in capsys.readouterr().out
        assert hashlib.sha256(database.read_bytes()).hexdigest() == before

    def test_missing_arguments_are_a_usage_error(self):
        with pytest.raises(SystemExit) as info:
            cli_main(["--database", "x.sqlite"])
        assert info.value.code == 2

    def test_a_midway_store_failure_is_reported_as_incomplete_and_rerun_is_safe(self, extract, capsys):
        database = extract.directory / "kb.sqlite"
        KnowledgeStore(database).close()  # create the schema first
        raw_sql(
            database,
            "CREATE TRIGGER boom BEFORE INSERT ON chunks WHEN NEW.doc_id = 'capec:CAPEC-664' "
            "BEGIN SELECT RAISE(ABORT, 'injected failure'); END",
        )
        assert run_cli(extract, database) == 1
        captured = capsys.readouterr()
        assert "INCOMPLETE" in captured.err and "capec:CAPEC-664" in captured.err
        assert "created 3" in captured.err and "rerunning the same command is safe" in captured.err
        assert "Result: OK" not in captured.out
        with KnowledgeStore(database) as store:
            assert store.count_documents() == 3  # committed before the failure, none after

        raw_sql(database, "DROP TRIGGER boom")
        assert run_cli(extract, database) == 0
        assert "created 3, updated 0, unchanged 3" in capsys.readouterr().out
        with KnowledgeStore(database) as store:
            assert store.count_documents() == 6


# ============================================================
# REAL PINNED ARCHIVES (skipped, with a reason, when not downloaded)
# ============================================================

# Keyword-retrieval SANITY CHECK. Each entry: a plain-language question, the FTS5 query a
# caller would derive from distinctive source wording, and records that must be among the hits.
# It shows lexical reachability only; it does not evaluate semantic relevance, forensic
# reasoning or remediation quality.
RETRIEVAL_CHECKS = [
    ("Which weakness is building SQL commands from untrusted input?", '"SQL injection"', {"cwe:CWE-89", "capec:CAPEC-66"}),
    ("How does blind SQL injection work?", '"blind SQL injection"', {"capec:CAPEC-7"}),
    ("Which attack runs OS commands through SQL injection?", '"command line execution"', {"capec:CAPEC-108"}),
    ("How can paths escape a restricted directory?", '"path traversal"', {"cwe:CWE-22", "cwe:CWE-23", "capec:CAPEC-126"}),
    ("How is a server made to send attacker-chosen requests?", '"server-side request forgery"', {"cwe:CWE-918", "capec:CAPEC-664"}),
    ("How are shell commands run through application input?", '"OS command injection"', {"cwe:CWE-78", "capec:CAPEC-88"}),
    ("How is arbitrary code injected into an interpreter?", '"code injection"', {"cwe:CWE-94", "capec:CAPEC-242"}),
    ("Which weakness is a critical function reachable without authentication?", '"missing authentication"', {"cwe:CWE-306"}),
    ("Which weakness covers weak or wrong authentication?", '"improper authentication"', {"cwe:CWE-287"}),
    ("Which attack pattern bypasses authentication?", '"authentication bypass"', {"capec:CAPEC-115"}),
    ("Which weakness is a missing authorization check?", '"missing authorization"', {"cwe:CWE-862"}),
    ("Which weakness lets script run in a victim's browser?", '"cross-site scripting"', {"cwe:CWE-79"}),
    ("Which mitigation separates data from SQL code?", '"parameterized queries"', {"cwe:CWE-89", "capec:CAPEC-66"}),
    ("How is a request parameter manipulated to alter behaviour?", '"parameter injection"', {"capec:CAPEC-137"}),
    ("Which class covers missing validation of input?", '"improper input validation"', {"cwe:CWE-20"}),
    ("Which attack pattern is generic command injection?", '"command injection"', {"cwe:CWE-77", "capec:CAPEC-248"}),
]


@pytest.fixture(scope="module")
def documents(manifest):
    """The 23 curated documents built from the real archives (only used by skippable tests)."""
    return build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)


@needs_real_archives
class TestRealPinnedArchives:
    def test_actual_archive_bytes_match_the_pinned_checksums(self, manifest):
        assert hashlib.sha256(REAL_CWE_ZIP.read_bytes()).hexdigest() == manifest.cwe.spec.archive_sha256
        assert hashlib.sha256(REAL_CAPEC_ZIP.read_bytes()).hexdigest() == manifest.capec.spec.archive_sha256

    def test_every_selected_id_exists_with_the_pinned_name_and_status(self, manifest, documents):
        by_id = {d.doc_id: d for d in documents}
        assert len(documents) == len(by_id) == 23
        for entry, source in ((manifest.cwe, "cwe"), (manifest.capec, "capec")):
            for selected in entry.selection:
                doc = by_id[f"{source}:{source.upper()}-{selected.record_id}"]
                assert doc.title == selected.name
                assert f"Status: {selected.status}" in doc.content
                assert doc.source_version == entry.spec.source_version

    def test_no_selected_record_is_deprecated(self, manifest):
        assert all(s.status != "Deprecated" for s in manifest.cwe.selection + manifest.capec.selection)

    def test_documents_carry_attribution_urls_and_real_provenance(self, manifest, documents):
        for doc in documents:
            entry = manifest.cwe if doc.source == "cwe" else manifest.capec
            assert doc.license == entry.spec.license
            assert doc.source_url == OFFICIAL_RECORD_URLS[doc.source].format(id=doc.source_id.split("-")[1])
            assert doc.provenance.artifact_sha256 == entry.spec.archive_sha256
            assert doc.provenance.artifact_path == entry.spec.artifact_filename
            assert doc.source_modified and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", doc.source_modified)

    def test_explicit_cross_references_are_preserved(self, documents):
        refs = {d.doc_id: {(r.relation, r.target_doc_id) for r in d.references} for d in documents}
        assert ("RelatedAttackPattern", "capec:CAPEC-66") in refs["cwe:CWE-89"]
        assert ("RelatedWeakness", "cwe:CWE-89") in refs["capec:CAPEC-66"]
        assert ("RelatedAttackPattern", "capec:CAPEC-664") in refs["cwe:CWE-918"]
        assert ("RelatedWeakness", "cwe:CWE-918") in refs["capec:CAPEC-664"]
        assert ("RelatedWeakness", "cwe:CWE-88") in refs["capec:CAPEC-137"]  # CWE-88 is not in the corpus
        assert "cwe:CWE-88" not in refs

    def test_stable_identity_across_repeated_builds(self, manifest, documents):
        again = build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
        assert again == documents

    def test_ingestion_is_idempotent_and_writes_nothing_the_second_time(self, manifest, tmp_path):
        database = tmp_path / "real.sqlite"
        with KnowledgeStore(database) as store:
            first = ingest_corpus(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP, store=store)
        assert len(first.summary.created) == 23
        digest = hashlib.sha256(database.read_bytes()).hexdigest()
        with KnowledgeStore(database) as store:
            second = ingest_corpus(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP, store=store)
            assert len(second.summary.unchanged) == 23 and not second.summary.created and not second.summary.updated
            assert store.count_documents() == 23
        assert hashlib.sha256(database.read_bytes()).hexdigest() == digest

    def test_keyword_retrieval_sanity_check(self, manifest):
        """Distinctive source wording reaches the expected records through FTS5 (lexical only)."""
        with KnowledgeStore(":memory:") as store:
            ingest_corpus(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP, store=store)
            failures = []
            for question, query, expected in RETRIEVAL_CHECKS:
                found = doc_ids_for(store, query)
                if not expected <= found:
                    failures.append((question, query, sorted(expected - found)))
            assert not failures, failures

    def test_command_line_on_the_real_archives(self, tmp_path, capsys):
        database = tmp_path / "cli.sqlite"
        args = [
            "--cwe-archive", str(REAL_CWE_ZIP), "--capec-archive", str(REAL_CAPEC_ZIP),
            "--database", str(database), "--manifest", str(MANIFEST_PATH),
        ]
        assert cli_main(args) == 0
        assert "selected 23 (13 CWE + 10 CAPEC); ingested 23" in capsys.readouterr().out
        assert cli_main(args) == 0
        assert "created 0, updated 0, unchanged 23" in capsys.readouterr().out

    def test_a_single_flipped_byte_is_caught_by_the_checksum(self, manifest, tmp_path):
        data = bytearray(REAL_CWE_ZIP.read_bytes())
        data[len(data) // 2] ^= 0x01
        damaged = tmp_path / "cwec_v4.20.xml.zip"
        damaged.write_bytes(bytes(data))
        with pytest.raises(ChecksumMismatchError):
            build_corpus_documents(manifest, cwe_archive=damaged, capec_archive=REAL_CAPEC_ZIP)

    def test_a_different_pinned_version_than_the_archive_declares_is_refused(self, manifest):
        wrong = dataclasses.replace(
            manifest,
            cwe=dataclasses.replace(
                manifest.cwe, spec=dataclasses.replace(manifest.cwe.spec, source_version="4.19")
            ),
        )
        with pytest.raises(SourceVersionError, match="4.20.*4.19"):
            build_corpus_documents(wrong, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
