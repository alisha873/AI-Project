import ast
import dataclasses
import json
import sqlite3
from pathlib import Path

import pytest

import ingestion.pipeline as pipeline_module
from ingestion import (
    BatchIngestionError,
    DuplicateDocumentError,
    IngestionError,
    IngestionSummary,
    InvalidDocumentInputError,
    ingest_documents,
    load_documents,
)
from knowledge.chunking import CHUNKER_VERSION, chunk_document
from knowledge.schema import ProvenanceRecord, SourceDocument, make_chunk_id
from knowledge.store import ChunkingConfig, IngestResult, KnowledgeStore, KnowledgeStoreError

FIXTURE = Path(__file__).parent / "fixtures" / "knowledge_ingestion" / "synthetic_documents.json"
SMALL = ChunkingConfig(max_chars=200, overlap_chars=20)

CWE_ID = "cwe:SYNTHETIC-0001"
CAPEC_ID = "capec:SYNTHETIC-0002"
OWASP_ID = "owasp:SYNTHETIC-0003"
SORTED_IDS = [CAPEC_ID, CWE_ID, OWASP_ID]  # ascending doc_id: the pipeline's processing order


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def fixture_docs() -> list[SourceDocument]:
    return load_documents(FIXTURE)


def variant_docs(tmp_path: Path, mutate) -> list[SourceDocument]:
    """The fixture with `mutate(records)` applied, loaded through the real loader."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutate({record["source"] + ":" + record["source_id"]: record for record in data["documents"]})
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_documents(path)


def by_id(documents):
    return {document.doc_id: document for document in documents}


def canonical_chunks(document: SourceDocument, config: ChunkingConfig = SMALL):
    return chunk_document(
        document,
        max_chars=config.max_chars,
        overlap_chars=config.overlap_chars,
        chunker_version=config.chunker_version,
    )


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "kb.db"


@pytest.fixture
def store(db_path):
    with KnowledgeStore(db_path) as opened:
        yield opened


def raw_rows(path, sql, params=()):
    connection = sqlite3.connect(path)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def raw_exec(path, *statements):
    connection = sqlite3.connect(path)
    try:
        for statement in statements:
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def table_counts(path):
    return {
        table: raw_rows(path, f"SELECT COUNT(*) FROM {table}")[0][0]
        for table in ("documents", "chunks", "chunks_fts")
    }


def install_write_audit(path):
    """Triggers that log every insert/update/delete of documents/chunks (detects rewrites)."""
    raw_exec(
        path,
        "CREATE TABLE audit (event TEXT)",
        *[
            f"CREATE TRIGGER audit_{table}_{event.lower()} AFTER {event} ON {table} "
            f"BEGIN INSERT INTO audit VALUES ('{table}:{event}'); END"
            for table in ("documents", "chunks")
            for event in ("INSERT", "UPDATE", "DELETE")
        ],
    )


def audit_events(path):
    return [row[0] for row in raw_rows(path, "SELECT event FROM audit ORDER BY rowid")]


def add_failure_trigger(path, doc_id):
    raw_exec(
        path,
        f"CREATE TRIGGER boom BEFORE INSERT ON chunks WHEN NEW.doc_id = '{doc_id}' "
        "BEGIN SELECT RAISE(ABORT, 'injected failure'); END",
    )


def assert_store_empty(store):
    assert store.count_documents() == 0
    assert store.count_chunks() == 0


# ------------------------------------------------------------
# end to end: local JSON -> chunks -> SQLite -> search
# ------------------------------------------------------------

def test_fixture_is_ingested_end_to_end_into_a_fresh_store(store):
    documents = load_documents(FIXTURE)

    summary = ingest_documents(documents, store, SMALL)

    assert isinstance(summary, IngestionSummary)
    assert summary.total == 3
    assert summary.created == tuple(SORTED_IDS)
    assert summary.updated == () and summary.unchanged == () and summary.duplicates_skipped == ()
    assert summary.config == SMALL
    assert store.list_doc_ids() == SORTED_IDS


def test_stored_documents_equal_the_loaded_documents_with_provenance_and_references(store):
    documents = by_id(fixture_docs())
    ingest_documents(documents.values(), store, SMALL)

    for doc_id, document in documents.items():
        stored = store.get_document(doc_id)
        assert stored == document
        assert stored.content == document.content  # the complete source text is kept
        assert stored.provenance == document.provenance
        assert stored.references == document.references
        assert stored.document_sha256 == document.document_sha256
    assert store.get_document(CAPEC_ID).references[0].target_doc_id == CWE_ID


def test_persisted_chunks_are_the_canonical_chunks_with_valid_identities(store):
    documents = by_id(fixture_docs())
    ingest_documents(documents.values(), store, SMALL)

    for doc_id, document in documents.items():
        chunks = store.get_chunks(doc_id)
        assert chunks == canonical_chunks(document)
        assert len(chunks) > 1
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk.chunk_id == make_chunk_id(doc_id, chunk.ordinal)
            assert chunk.doc_id == doc_id
            assert document.contains_chunk(chunk)
            assert (chunk.source, chunk.source_id, chunk.doc_type) == (
                document.source,
                document.source_id,
                document.doc_type,
            )
            assert chunk.title == document.title
            assert chunk.doc_content_sha256 == document.content_sha256
            assert chunk.chunker_version == SMALL.chunker_version
            assert len(chunk.text) <= SMALL.max_chars
    assert store.count_chunks() == sum(len(canonical_chunks(d)) for d in documents.values())


def test_distinctive_terms_are_searchable_after_ingestion(store):
    ingest_documents(fixture_docs(), store, SMALL)

    for term, expected_doc in (("quokkaflux", CWE_ID), ("marmotvolt", CAPEC_ID), ("lemurgrid", OWASP_ID)):
        hits = store.search(term, limit=100)
        assert hits, term
        assert {hit.chunk.doc_id for hit in hits} == {expected_doc}
        assert all(term in hit.chunk.text or term in hit.chunk.title.lower() for hit in hits)
    assert store.search("nonexistentterm") == []


def test_data_survives_closing_and_reopening_the_database(db_path):
    documents = by_id(fixture_docs())
    with KnowledgeStore(db_path) as first:
        ingest_documents(documents.values(), first, SMALL)

    with KnowledgeStore(db_path) as reopened:
        assert reopened.list_doc_ids() == SORTED_IDS
        for doc_id, document in documents.items():
            assert reopened.get_document(doc_id) == document
            assert reopened.get_chunks(doc_id) == canonical_chunks(document)
        assert reopened.search("marmotvolt")


def test_the_default_configuration_is_the_chunkers_defaults(store, monkeypatch):
    seen = []
    real = pipeline_module.chunk_document
    monkeypatch.setattr(pipeline_module, "chunk_document", lambda d, **kw: (seen.append(kw), real(d, **kw))[1])

    summary = ingest_documents(fixture_docs(), store)

    assert summary.config == ChunkingConfig()
    assert seen and all(
        kw == {"max_chars": 1200, "overlap_chars": 120, "chunker_version": CHUNKER_VERSION} for kw in seen
    )
    assert all(len(store.get_chunks(doc_id)) == 1 for doc_id in SORTED_IDS)  # short documents


# ------------------------------------------------------------
# the same configuration reaches the chunker and the store
# ------------------------------------------------------------

def test_one_config_drives_both_the_chunker_and_the_store(store, db_path, monkeypatch):
    config = ChunkingConfig(max_chars=150, overlap_chars=15, chunker_version="test-chunker-v9")
    chunker_calls, store_calls = [], []
    real_chunker = pipeline_module.chunk_document
    real_ingest = store.ingest_document

    def chunker_spy(document, **kwargs):
        chunker_calls.append(kwargs)
        return real_chunker(document, **kwargs)

    def store_spy(document, chunks=None, config=None):
        store_calls.append((document, chunks, config))
        return real_ingest(document, chunks, config)

    monkeypatch.setattr(pipeline_module, "chunk_document", chunker_spy)
    monkeypatch.setattr(store, "ingest_document", store_spy)

    ingest_documents(fixture_docs(), store, config)

    expected_kwargs = {"max_chars": 150, "overlap_chars": 15, "chunker_version": "test-chunker-v9"}
    assert chunker_calls == [expected_kwargs] * 3
    assert len(store_calls) == 3
    for document, chunks, passed_config in store_calls:
        assert passed_config is config
        assert chunks == chunk_document(document, **expected_kwargs)  # canonical chunker output
        assert {chunk.chunker_version for chunk in chunks} == {"test-chunker-v9"}
    assert sorted(raw_rows(db_path, "SELECT chunker_version, max_chars, overlap_chars FROM documents")) == [
        ("test-chunker-v9", 150, 15)
    ] * 3


def test_changing_the_configuration_regenerates_chunks_and_a_repeat_is_unchanged(store):
    documents = fixture_docs()
    ingest_documents(documents, store, SMALL)
    other = ChunkingConfig(max_chars=100, overlap_chars=10)

    changed = ingest_documents(documents, store, other)

    assert changed.updated == tuple(SORTED_IDS)
    assert all(r.config_changed and r.chunks_replaced for r in changed.results)
    assert not any(r.content_changed or r.metadata_changed for r in changed.results)
    for document in documents:
        assert store.get_chunks(document.doc_id) == canonical_chunks(document, other)
    assert store.count_chunks() == sum(len(canonical_chunks(d, other)) for d in documents)

    repeated = ingest_documents(documents, store, other)
    assert repeated.unchanged == tuple(SORTED_IDS) and repeated.created == () and repeated.updated == ()


def test_a_new_chunker_version_alone_is_detected(store):
    documents = fixture_docs()
    ingest_documents(documents, store, SMALL)
    newer = ChunkingConfig(SMALL.max_chars, SMALL.overlap_chars, "char-structure-v2")

    summary = ingest_documents(documents, store, newer)

    assert summary.updated == tuple(SORTED_IDS)
    assert {c.chunker_version for d in documents for c in store.get_chunks(d.doc_id)} == {"char-structure-v2"}


# ------------------------------------------------------------
# idempotency and updates
# ------------------------------------------------------------

def test_re_ingesting_the_same_documents_writes_nothing(store, db_path):
    ingest_documents(fixture_docs(), store, SMALL)
    install_write_audit(db_path)
    counts = table_counts(db_path)

    for _ in range(3):
        summary = ingest_documents(fixture_docs(), store, SMALL)  # freshly loaded each time
        assert summary.unchanged == tuple(SORTED_IDS)
        assert summary.created == () and summary.updated == ()
        assert all(r.action == "unchanged" for r in summary.results)

    assert audit_events(db_path) == []  # no row inserted, updated or deleted
    assert table_counts(db_path) == counts


def test_a_content_change_is_applied_and_stale_chunks_disappear(store, db_path, tmp_path):
    ingest_documents(fixture_docs(), store, SMALL)
    assert store.search("quokkaflux")
    # The title is indexed too, so the old term must be replaced there as well.
    changed = variant_docs(
        tmp_path,
        lambda records: records[CWE_ID].update(
            content=records[CWE_ID]["content"].replace("quokkaflux", "wombatpulse"),
            title=records[CWE_ID]["title"].replace("quokkaflux", "wombatpulse"),
        ),
    )

    summary = ingest_documents(changed, store, SMALL)

    assert summary.updated == (CWE_ID,)
    assert summary.unchanged == (CAPEC_ID, OWASP_ID)
    result = {r.doc_id: r for r in summary.results}[CWE_ID]
    assert result.content_changed and result.chunks_replaced
    assert store.get_document(CWE_ID) == by_id(changed)[CWE_ID]
    assert store.get_chunks(CWE_ID) == canonical_chunks(by_id(changed)[CWE_ID])
    assert store.search("quokkaflux") == []  # stale text is no longer searchable
    assert {hit.chunk.doc_id for hit in store.search("wombatpulse", limit=100)} == {CWE_ID}
    assert table_counts(db_path)["chunks"] == table_counts(db_path)["chunks_fts"]


def test_a_metadata_change_with_identical_content_is_applied(store, tmp_path):
    original = by_id(fixture_docs())
    ingest_documents(original.values(), store, SMALL)
    changed = variant_docs(
        tmp_path, lambda records: records[CAPEC_ID].update(title="[SYNTHETIC TEST DATA] Renamed attack pattern")
    )
    assert by_id(changed)[CAPEC_ID].content_sha256 == original[CAPEC_ID].content_sha256

    summary = ingest_documents(changed, store, SMALL)

    assert summary.updated == (CAPEC_ID,)
    result = summary.results[0]
    assert result.metadata_changed and not result.content_changed
    assert result.chunks_replaced  # chunks copy the title
    assert store.get_document(CAPEC_ID).title.endswith("Renamed attack pattern")
    assert {c.title for c in store.get_chunks(CAPEC_ID)} == {by_id(changed)[CAPEC_ID].title}
    assert store.search("Renamed")


def test_metadata_that_chunks_do_not_copy_updates_only_the_document(store, db_path, tmp_path):
    ingest_documents(fixture_docs(), store, SMALL)
    install_write_audit(db_path)
    changed = variant_docs(tmp_path, lambda records: records[OWASP_ID].update(license="synthetic-test-data (relicensed)"))

    summary = ingest_documents(changed, store, SMALL)

    assert summary.updated == (OWASP_ID,)
    assert summary.results[-1].metadata_changed and not summary.results[-1].chunks_replaced
    assert store.get_document(OWASP_ID).license == "synthetic-test-data (relicensed)"
    assert audit_events(db_path) == ["documents:UPDATE"]


def test_a_provenance_only_change_is_reported_as_such(store, tmp_path):
    ingest_documents(fixture_docs(), store, SMALL)
    changed = variant_docs(
        tmp_path,
        lambda records: records[CWE_ID]["provenance"].update(ingested_at_utc="2026-10-17T09:00:00Z"),
    )

    summary = ingest_documents(changed, store, SMALL)

    assert summary.updated == (CWE_ID,)
    result = summary.results[1]
    assert result.provenance_changed and not (result.content_changed or result.metadata_changed)
    assert store.get_document(CWE_ID).provenance.ingested_at_utc == "2026-10-17T09:00:00Z"


def test_created_updated_and_unchanged_are_reported_together(store, tmp_path):
    first = fixture_docs()
    ingest_documents([d for d in first if d.doc_id in (CAPEC_ID, CWE_ID)], store, SMALL)
    second = variant_docs(
        tmp_path,
        lambda records: records[CAPEC_ID].update(
            content=records[CAPEC_ID]["content"].replace("marmotvolt", "ibexamp")
        ),
    )

    summary = ingest_documents(second, store, SMALL)

    assert summary.created == (OWASP_ID,)
    assert summary.updated == (CAPEC_ID,)
    assert summary.unchanged == (CWE_ID,)
    assert [r.doc_id for r in summary.results] == SORTED_IDS
    assert summary.total == 3
    assert all(isinstance(r, IngestResult) for r in summary.results)


def test_a_summary_only_reports_what_the_store_returned(store, monkeypatch):
    fake = {
        doc_id: IngestResult(doc_id, action, 1)
        for doc_id, action in zip(SORTED_IDS, ("updated", "created", "unchanged"))
    }
    monkeypatch.setattr(store, "ingest_document", lambda document, chunks=None, config=None: fake[document.doc_id])

    summary = ingest_documents(fixture_docs(), store, SMALL)

    assert (summary.updated, summary.created, summary.unchanged) == ((CAPEC_ID,), (CWE_ID,), (OWASP_ID,))
    assert summary.results == tuple(fake[i] for i in SORTED_IDS)


def test_the_summary_is_immutable():
    summary = IngestionSummary(SMALL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.results = ()
    assert (summary.total, summary.created, summary.updated, summary.unchanged) == (0, (), (), ())


# ------------------------------------------------------------
# duplicate identities
# ------------------------------------------------------------

def conflicting_pair(tmp_path):
    original = fixture_docs()
    different = variant_docs(tmp_path, lambda records: records[CWE_ID].update(title="[SYNTHETIC TEST DATA] Other"))
    return original, by_id(different)[CWE_ID]


def test_conflicting_duplicates_are_rejected_before_any_write(store, tmp_path):
    original, conflicting = conflicting_pair(tmp_path)

    with pytest.raises(DuplicateDocumentError) as caught:
        ingest_documents([*original, conflicting], store, SMALL)  # conflict is the LAST item

    message = str(caught.value)
    assert CWE_ID in message and "[0, 3]" in message  # fixture order: cwe is first, conflict is fourth
    assert "nothing was ingested" in message
    assert_store_empty(store)  # the valid documents before it were not written either


def test_conflict_detection_does_not_depend_on_input_order(store, tmp_path):
    original, conflicting = conflicting_pair(tmp_path)

    for ordering in ([conflicting, *original], [*original, conflicting], [original[1], conflicting, *original]):
        with pytest.raises(DuplicateDocumentError, match=CWE_ID):
            ingest_documents(ordering, store, SMALL)
    assert_store_empty(store)


def test_a_conflict_with_data_already_stored_is_not_a_duplicate_just_an_update(store, tmp_path):
    original, conflicting = conflicting_pair(tmp_path)
    ingest_documents(original, store, SMALL)

    summary = ingest_documents([conflicting], store, SMALL)  # same doc_id across runs is normal

    assert summary.updated == (CWE_ID,)


def test_provenance_only_differences_between_duplicates_still_conflict(store):
    document = fixture_docs()[0]
    provenance = ProvenanceRecord(**{**document.provenance.model_dump(), "ingested_at_utc": "2026-10-18T00:00:00Z"})
    twin = SourceDocument(**{**{n: getattr(document, n) for n in SourceDocument.model_fields}, "provenance": provenance})
    assert twin.document_sha256 == document.document_sha256

    with pytest.raises(DuplicateDocumentError):
        ingest_documents([document, twin], store, SMALL)
    assert_store_empty(store)


def test_identical_duplicates_are_collapsed_and_reported(store):
    documents = fixture_docs()
    repeated = [documents[0], *fixture_docs(), documents[2]]  # equal records, separately built

    summary = ingest_documents(repeated, store, SMALL)

    assert summary.total == 3
    assert summary.created == tuple(SORTED_IDS)
    assert sorted(summary.duplicates_skipped) == sorted([CWE_ID, OWASP_ID])
    assert store.count_documents() == 3


def test_processing_order_is_by_doc_id_whatever_the_input_order(tmp_path):
    forward, backward = fixture_docs(), list(reversed(fixture_docs()))
    with KnowledgeStore(tmp_path / "a.db") as first, KnowledgeStore(tmp_path / "b.db") as second:
        summary_a = ingest_documents(forward, first, SMALL)
        summary_b = ingest_documents(backward, second, SMALL)

        assert [r.doc_id for r in summary_a.results] == SORTED_IDS
        assert summary_a == summary_b
        for doc_id in SORTED_IDS:
            assert first.get_document(doc_id) == second.get_document(doc_id)
            assert first.get_chunks(doc_id) == second.get_chunks(doc_id)


# ------------------------------------------------------------
# partial batch failure
# ------------------------------------------------------------

def test_a_failure_on_a_later_document_keeps_earlier_documents_committed(store, db_path):
    add_failure_trigger(db_path, CWE_ID)  # processing order: capec (ok), cwe (fails), owasp (not reached)

    with pytest.raises(BatchIngestionError, match="injected failure") as caught:
        ingest_documents(fixture_docs(), store, SMALL)

    error = caught.value
    assert error.failed_doc_id == CWE_ID
    assert error.completed.created == (CAPEC_ID,)
    assert error.completed.total == 1
    assert isinstance(error.__cause__, KnowledgeStoreError)
    assert "1 of 3" in str(error)
    assert store.list_doc_ids() == [CAPEC_ID]  # committed before the failure
    assert not store.has_document(CWE_ID) and not store.has_document(OWASP_ID)
    assert store.get_chunks(CAPEC_ID) == canonical_chunks(by_id(fixture_docs())[CAPEC_ID])
    counts = table_counts(db_path)
    assert counts["chunks"] == counts["chunks_fts"] == len(canonical_chunks(by_id(fixture_docs())[CAPEC_ID]))


def test_rerunning_after_a_partial_failure_completes_the_batch(store, db_path):
    add_failure_trigger(db_path, CWE_ID)
    with pytest.raises(BatchIngestionError):
        ingest_documents(fixture_docs(), store, SMALL)
    raw_exec(db_path, "DROP TRIGGER boom")

    summary = ingest_documents(fixture_docs(), store, SMALL)

    assert summary.unchanged == (CAPEC_ID,)  # finished earlier, nothing rewritten
    assert summary.created == (CWE_ID, OWASP_ID)
    assert store.list_doc_ids() == SORTED_IDS
    assert ingest_documents(fixture_docs(), store, SMALL).unchanged == tuple(SORTED_IDS)


def test_a_failed_update_keeps_the_previous_version_of_that_document(store, db_path, tmp_path):
    originals = by_id(fixture_docs())
    ingest_documents(originals.values(), store, SMALL)

    def edit_all(records):
        for record in records.values():
            record["title"] += " (edited)"

    edited = variant_docs(tmp_path, edit_all)
    add_failure_trigger(db_path, OWASP_ID)  # last in processing order

    with pytest.raises(BatchIngestionError) as caught:
        ingest_documents(edited, store, SMALL)

    assert caught.value.completed.updated == (CAPEC_ID, CWE_ID)
    assert store.get_document(CAPEC_ID).title.endswith("(edited)")  # committed
    assert store.get_document(CWE_ID).title.endswith("(edited)")  # committed
    assert store.get_document(OWASP_ID) == originals[OWASP_ID]  # rolled back to the old version
    assert store.get_chunks(OWASP_ID) == canonical_chunks(originals[OWASP_ID])


# ------------------------------------------------------------
# empty and invalid input
# ------------------------------------------------------------

@pytest.mark.parametrize("empty", [[], (), iter(()), (d for d in [])], ids=["list", "tuple", "iterator", "generator"])
def test_empty_input_returns_an_empty_summary_and_writes_nothing(store, db_path, empty):
    install_write_audit(db_path)

    summary = ingest_documents(empty, store, SMALL)

    assert summary == IngestionSummary(SMALL)
    assert summary.total == 0
    assert audit_events(db_path) == []
    assert_store_empty(store)


def test_a_generator_of_documents_is_accepted(store):
    summary = ingest_documents((d for d in fixture_docs()), store, SMALL)
    assert summary.created == tuple(SORTED_IDS)


@pytest.mark.parametrize("bad", [None, 42, 3.5, "documents", b"documents", {"doc_id": "x"}])
def test_a_non_iterable_or_text_input_is_rejected(store, bad):
    with pytest.raises(InvalidDocumentInputError, match="iterable"):
        ingest_documents(bad, store, SMALL)
    assert_store_empty(store)


@pytest.mark.parametrize("bad_item", [{"source": "cwe"}, "cwe:CWE-89", None, 7, ("a", "b")])
def test_items_that_are_not_documents_are_rejected_with_their_index(store, bad_item):
    documents = fixture_docs()

    with pytest.raises(InvalidDocumentInputError, match=r"documents\[2\] must be a SourceDocument"):
        ingest_documents([documents[0], documents[1], bad_item], store, SMALL)
    assert_store_empty(store)  # the valid documents before it were not written


def test_a_single_document_is_not_accepted_where_a_collection_is_expected(store):
    with pytest.raises(InvalidDocumentInputError, match=r"documents\[0\] must be a SourceDocument"):
        ingest_documents(fixture_docs()[0], store, SMALL)
    assert_store_empty(store)


def test_a_document_built_without_validation_is_rejected_before_any_write(store):
    good = fixture_docs()
    fields = {name: getattr(good[2], name) for name in SourceDocument.model_fields}
    broken = SourceDocument.model_construct(**{**fields, "content": "  not normalized \r\n"})

    with pytest.raises(InvalidDocumentInputError, match=r"documents\[3\]"):
        ingest_documents([*good, broken], store, SMALL)
    assert_store_empty(store)


def test_wrong_argument_types_and_a_closed_store_are_rejected(db_path):
    documents = fixture_docs()
    with KnowledgeStore(db_path) as opened:
        with pytest.raises(TypeError, match="store"):
            ingest_documents(documents, "kb.db", SMALL)
        with pytest.raises(TypeError, match="store"):
            ingest_documents(documents, None)
        with pytest.raises(TypeError, match="config"):
            ingest_documents(documents, opened, {"max_chars": 100})
    with pytest.raises(IngestionError, match="closed"):
        ingest_documents(documents, opened, SMALL)
    with pytest.raises(IngestionError, match="closed"):
        ingest_documents([], opened, SMALL)


def test_an_invalid_chunking_config_cannot_even_be_created():
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars=50, overlap_chars=50)


# ------------------------------------------------------------
# boundaries
# ------------------------------------------------------------

def test_the_pipeline_owns_no_storage_logic_and_does_not_know_the_loader():
    tree = ast.parse(Path(pipeline_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert "sqlite3" not in imported
    assert "ingestion.local_json" not in imported
    assert imported <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "pydantic",
        "knowledge.chunking",
        "knowledge.schema",
        "knowledge.store",
    }
