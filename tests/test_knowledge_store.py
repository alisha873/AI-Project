import hashlib
import sqlite3

import pytest

import knowledge.store as store_module
from knowledge.chunking import CHUNKER_VERSION, chunk_document
from knowledge.schema import (
    Chunk,
    ExternalReference,
    ProvenanceRecord,
    SourceDocument,
    normalize_text,
)
from knowledge.store import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    NORMALIZATION_VERSION,
    ChunkingConfig,
    DocumentNotFoundError,
    FTS5UnavailableError,
    IncompatibleStoreError,
    IngestResult,
    InvalidIngestionError,
    InvalidSearchQueryError,
    KnowledgeStore,
    KnowledgeStoreError,
    SearchHit,
    StoreCorruptionError,
)

SMALL = ChunkingConfig(max_chars=120, overlap_chars=20)


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def make_provenance(**overrides) -> ProvenanceRecord:
    values = {
        "origin_kind": "bundled_fixture",
        "artifact_path": "tests/fixtures/knowledge/cwe_sample.jsonl",
        "artifact_sha256": "a" * 64,
        "adapter": "normalized_jsonl",
        "adapter_version": "0.1",
        "retrieved_at_utc": "2026-10-16T00:00:00Z",
        "ingested_at_utc": "2026-10-16T01:00:00Z",
    }
    values.update(overrides)
    return ProvenanceRecord(**values)


def make_document(content: str | None = None, **overrides) -> SourceDocument:
    values = {
        "source": "cwe",
        "source_id": "CWE-89",
        "source_version": "fixture-1",
        "doc_type": "weakness",
        "title": "SQL Injection",
        "content": content if content is not None else long_content("alpha"),
        "source_url": "https://cwe.mitre.org/data/definitions/89.html",
        "license": "CC-BY-4.0",
        "references": (
            ExternalReference(relation="ChildOf", target_source="cwe", target_source_id="CWE-74"),
        ),
        "provenance": make_provenance(),
    }
    values.update(overrides)
    return SourceDocument(**values)


def long_content(marker: str, paragraphs: int = 12) -> str:
    """Multi-paragraph text (several chunks at 120 chars) containing `marker`."""
    return "\n\n".join(
        f"Paragraph {index} of the {marker} entry: filler words describe the weakness.\n"
        f"second line {index} with more filler text"
        for index in range(paragraphs)
    )


def ingest(store: KnowledgeStore, document: SourceDocument, config: ChunkingConfig = SMALL):
    return store.ingest_document(document, config=config)


def expected_chunks(document: SourceDocument, config: ChunkingConfig = SMALL) -> list[Chunk]:
    return chunk_document(
        document,
        max_chars=config.max_chars,
        overlap_chars=config.overlap_chars,
        chunker_version=config.chunker_version,
    )


@pytest.fixture
def store():
    with KnowledgeStore(":memory:") as opened:
        yield opened


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "knowledge.db"


@pytest.fixture
def file_store(db_path):
    with KnowledgeStore(db_path) as opened:
        yield opened


def raw_rows(db_path, sql, params=()):
    """Query the database file through an independent connection."""
    connection = sqlite3.connect(db_path)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


def raw_exec(db_path, *statements):
    """Run statements (e.g. to corrupt the data or add a trigger) on an independent connection."""
    connection = sqlite3.connect(db_path)
    try:
        for statement in statements:
            connection.execute(*statement) if isinstance(statement, tuple) else connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def table_counts(db_path):
    return {
        table: raw_rows(db_path, f"SELECT COUNT(*) FROM {table}")[0][0]
        for table in ("documents", "chunks", "chunks_fts")
    }


# ------------------------------------------------------------
# initialization and lifecycle
# ------------------------------------------------------------

def test_creates_a_file_backed_database_and_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "knowledge.db"
    assert not path.exists()

    with KnowledgeStore(path) as opened:
        assert opened.path == str(path)
        assert opened.count_documents() == 0

    assert path.is_file()
    tables = {row[0] for row in raw_rows(path, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"store_meta", "documents", "chunks", "chunks_fts"} <= tables


def test_accepts_str_and_path_for_the_database_location(tmp_path):
    with KnowledgeStore(str(tmp_path / "a.db")):
        pass
    with KnowledgeStore(tmp_path / "b.db"):
        pass
    assert (tmp_path / "a.db").is_file() and (tmp_path / "b.db").is_file()


@pytest.mark.parametrize("bad_path", ["", "   ", None, 42])
def test_rejects_an_invalid_path_instead_of_using_a_default(bad_path):
    with pytest.raises(ValueError):
        KnowledgeStore(bad_path)


def test_in_memory_store_works_and_instances_are_isolated():
    document = make_document()
    with KnowledgeStore(":memory:") as first, KnowledgeStore(":memory:") as second:
        ingest(first, document)
        assert first.has_document(document.doc_id)
        assert not second.has_document(document.doc_id)


def test_reopening_a_file_database_returns_the_stored_data(db_path):
    document = make_document()
    with KnowledgeStore(db_path) as first:
        ingest(first, document)

    with KnowledgeStore(db_path) as reopened:
        assert reopened.list_doc_ids() == [document.doc_id]
        assert reopened.get_document(document.doc_id) == document
        assert reopened.get_chunks(document.doc_id) == expected_chunks(document)
        assert reopened.search("alpha")


def test_reopening_does_not_duplicate_the_schema_or_metadata(db_path):
    with KnowledgeStore(db_path):
        pass
    with KnowledgeStore(db_path):
        pass
    meta = dict(raw_rows(db_path, "SELECT key, value FROM store_meta"))
    assert meta["normalization_version"] == NORMALIZATION_VERSION
    assert meta["store_schema_version"] == "1"
    assert len(meta) == 4


def test_context_manager_closes_the_connection():
    with KnowledgeStore(":memory:") as opened:
        assert not opened.closed
    assert opened.closed
    with pytest.raises(KnowledgeStoreError, match="closed"):
        opened.count_documents()
    with pytest.raises(KnowledgeStoreError, match="closed"):
        opened.search("anything")


def test_close_is_idempotent(db_path):
    opened = KnowledgeStore(db_path)
    opened.close()
    opened.close()
    assert opened.closed


def test_file_is_released_after_close(db_path):
    opened = KnowledgeStore(db_path)
    ingest(opened, make_document())
    opened.close()
    db_path.unlink()  # fails on Windows if the file were still open
    assert not db_path.exists()


def test_fts5_unavailable_fails_clearly_and_leaves_no_partial_store(db_path, monkeypatch):
    monkeypatch.setattr(store_module, "check_fts5_available", lambda connection: False)

    with pytest.raises(FTS5UnavailableError, match="FTS5"):
        KnowledgeStore(db_path)

    tables = raw_rows(db_path, "SELECT name FROM sqlite_master") if db_path.exists() else []
    assert tables == []


def test_fts5_unavailable_is_also_detected_when_opening_an_existing_store(db_path, monkeypatch):
    with KnowledgeStore(db_path):
        pass
    monkeypatch.setattr(store_module, "check_fts5_available", lambda connection: False)
    with pytest.raises(FTS5UnavailableError):
        KnowledgeStore(db_path)


def test_real_fts5_capability_check_succeeds_on_this_machine():
    connection = sqlite3.connect(":memory:")
    try:
        assert store_module.check_fts5_available(connection) is True
        names = connection.execute("SELECT name FROM temp.sqlite_master").fetchall()
        assert names == []  # the probe table was dropped again
    finally:
        connection.close()


@pytest.mark.parametrize(
    "key, value",
    [
        ("store_schema_version", "999"),
        ("normalization_version", "normalize-text-v0"),
        ("document_schema_version", "0.1"),
        ("document_hash_version", "999"),
    ],
)
def test_incompatible_version_markers_are_rejected(db_path, key, value):
    with KnowledgeStore(db_path):
        pass
    raw_exec(db_path, ("UPDATE store_meta SET value = ? WHERE key = ?", (value, key)))

    with pytest.raises(IncompatibleStoreError, match=key):
        KnowledgeStore(db_path)


def test_a_foreign_sqlite_database_is_rejected_and_left_untouched(db_path):
    raw_exec(db_path, "CREATE TABLE unrelated (x INTEGER)", "INSERT INTO unrelated VALUES (1)")

    with pytest.raises(IncompatibleStoreError, match="not an AutoShield"):
        KnowledgeStore(db_path)

    assert [row[0] for row in raw_rows(db_path, "SELECT name FROM sqlite_master")] == ["unrelated"]
    assert raw_rows(db_path, "SELECT x FROM unrelated") == [(1,)]


def test_a_store_with_a_missing_table_is_reported_as_corrupt(db_path):
    with KnowledgeStore(db_path):
        pass
    raw_exec(db_path, "DROP TABLE chunks_fts")
    with pytest.raises(StoreCorruptionError, match="chunks_fts"):
        KnowledgeStore(db_path)


def test_a_non_database_file_gives_a_store_error(db_path):
    db_path.write_bytes(b"this is definitely not an sqlite database" * 20)
    with pytest.raises(KnowledgeStoreError):
        KnowledgeStore(db_path)


def test_normalize_text_behaviour_still_matches_the_recorded_normalization_version():
    """
    Guard for the MANUALLY maintained NORMALIZATION_VERSION. If this fails,
    `normalize_text` changed behaviour: bump NORMALIZATION_VERSION in
    knowledge/store.py, then update the fingerprint below.
    """
    samples = [
        "a\r\nb\rc\x0bd\x0ce\x85f\u2028g\u2029h",
        "\ufeff\n\n  indented first line\n\n\n\nsecond",
        "trailing   \nspaces\t\nhard break  \nnext",
        "```\ncode  \n\n\n\nkept\n```\n\n\n\nafter",
        "~~~\nunclosed fence   \n\n\n",
        "e\u0301 \u212b nfc \x00\x1b control\x1f unit",
    ]
    fingerprint = hashlib.sha256("\x00".join(normalize_text(s) for s in samples).encode()).hexdigest()
    assert NORMALIZATION_VERSION == "normalize-text-v1"
    assert fingerprint == "6f3e25b95e449149eff84b7f6beb11883cd1ce8658773aaa6899a6ab0feb85de"


# ------------------------------------------------------------
# document and chunk persistence
# ------------------------------------------------------------

def test_a_stored_document_round_trips_with_all_fields(store):
    document = make_document()
    ingest(store, document)

    loaded = store.get_document(document.doc_id)

    assert loaded == document
    assert isinstance(loaded, SourceDocument)
    assert loaded.doc_id == "cwe:CWE-89"
    assert loaded.content == document.content  # complete original content is kept
    assert loaded.references == document.references
    assert loaded.provenance == document.provenance
    assert loaded.content_sha256 == document.content_sha256
    assert loaded.document_sha256 == document.document_sha256


def test_stored_chunks_round_trip_with_every_field(store):
    document = make_document()
    original = expected_chunks(document)
    assert len(original) > 3
    ingest(store, document)

    loaded = store.get_chunks(document.doc_id)

    assert loaded == original
    for stored, source in zip(loaded, original):
        assert isinstance(stored, Chunk)
        assert stored.model_dump() == source.model_dump()
        assert document.contains_chunk(stored)
        assert stored.chunker_version == SMALL.chunker_version


def test_chunks_carry_document_metadata_and_section(store):
    document = make_document()
    ingest(store, document)
    chunk = store.get_chunks(document.doc_id)[0]

    assert (chunk.source, chunk.source_id, chunk.doc_type) == ("cwe", "CWE-89", "weakness")
    assert chunk.title == document.title
    assert chunk.source_url == document.source_url
    assert chunk.doc_content_sha256 == document.content_sha256
    assert chunk.section is None


def test_a_chunk_with_a_section_keeps_it(store):
    document = make_document()
    first = expected_chunks(document)[0]
    sectioned = Chunk(**{**first.model_dump(), "section": "Description"})
    rest = expected_chunks(document)[1:]

    store.ingest_document(document, [sectioned, *rest], SMALL)

    assert store.get_chunks(document.doc_id)[0].section == "Description"


def test_chunks_are_returned_in_numeric_ordinal_order_past_ordinal_9999(store):
    # Ordinals 0..10004: chunk IDs "#9999" and "#10000" sort wrongly as text.
    document = make_document("x" * 10005)
    config = ChunkingConfig(max_chars=1, overlap_chars=0)

    store.ingest_document(document, config=config)

    chunks = store.get_chunks(document.doc_id)
    assert [chunk.ordinal for chunk in chunks] == list(range(10005))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert sorted(chunk_ids) != chunk_ids  # proves text order would have been wrong


def test_loading_uses_pydantic_validation_and_never_model_construct(store, monkeypatch):
    document = make_document()
    ingest(store, document)

    def forbidden(*args, **kwargs):
        raise AssertionError("model_construct must not be used to load persisted data")

    monkeypatch.setattr(SourceDocument, "model_construct", forbidden)
    monkeypatch.setattr(Chunk, "model_construct", forbidden)

    assert store.get_document(document.doc_id) == document
    assert store.get_chunks(document.doc_id) == expected_chunks(document)
    assert store.search("alpha")


def test_has_document_and_listing(store):
    first = make_document()
    second = make_document(long_content("beta"), source_id="CWE-79", title="Cross-site Scripting")
    ingest(store, second)
    ingest(store, first)

    assert store.has_document("cwe:CWE-89")
    assert not store.has_document("cwe:CWE-1")
    assert store.list_doc_ids() == ["cwe:CWE-79", "cwe:CWE-89"]
    assert store.count_documents() == 2
    assert store.count_chunks() == len(expected_chunks(first)) + len(expected_chunks(second))


def test_missing_documents_raise_a_clear_error(store):
    with pytest.raises(DocumentNotFoundError, match="cwe:CWE-404"):
        store.get_document("cwe:CWE-404")
    with pytest.raises(DocumentNotFoundError):
        store.get_chunks("cwe:CWE-404")
    assert store.has_document("cwe:CWE-404") is False


def test_ingest_with_omitted_chunks_uses_the_deterministic_chunker(store):
    document = make_document()
    result = store.ingest_document(document, config=SMALL)

    assert result.action == "created"
    assert store.get_chunks(document.doc_id) == expected_chunks(document)


def test_default_config_uses_the_chunker_defaults(store):
    document = make_document("word " * 600 + "end")

    store.ingest_document(document)

    chunks = store.get_chunks(document.doc_id)
    assert chunks == chunk_document(document)
    assert {chunk.chunker_version for chunk in chunks} == {CHUNKER_VERSION}
    row = store._connection().execute("SELECT max_chars, overlap_chars FROM documents").fetchone()
    assert (row["max_chars"], row["overlap_chars"]) == (1200, 120)


def test_the_complete_document_is_kept_even_when_chunks_do_not_cover_it(store):
    # A long whitespace run inside a fence: the chunker skips whitespace-only
    # windows, so chunks are retrieval units, not a lossless reconstruction.
    content = "```\nstart\n" + " " * 300 + "\nend\n```"
    document = make_document(content)
    config = ChunkingConfig(max_chars=50, overlap_chars=0)

    store.ingest_document(document, config=config)

    chunks = store.get_chunks(document.doc_id)
    covered = {i for chunk in chunks for i in range(chunk.char_start, chunk.char_end)}
    assert covered != set(range(len(document.content)))  # chunks leave a gap
    assert store.get_document(document.doc_id).content == document.content  # document is complete


# ------------------------------------------------------------
# input validation: nothing is modified when ingestion is refused
# ------------------------------------------------------------

def assert_empty(store):
    assert store.count_documents() == 0
    assert store.count_chunks() == 0
    assert store.search("alpha") == []


def test_rejects_something_that_is_not_a_document(store):
    with pytest.raises(InvalidIngestionError, match="SourceDocument"):
        store.ingest_document({"doc_id": "cwe:CWE-89"})
    assert_empty(store)


def test_rejects_a_document_built_without_validation(store):
    good = make_document()
    fields = {name: getattr(good, name) for name in SourceDocument.model_fields}
    broken = SourceDocument.model_construct(**{**fields, "content": "  not normalized  \r\n"})

    with pytest.raises(InvalidIngestionError, match="not a valid"):
        store.ingest_document(broken, config=SMALL)
    assert_empty(store)


def test_rejects_chunks_that_are_not_chunk_objects(store):
    document = make_document()
    with pytest.raises(InvalidIngestionError, match="Chunk objects"):
        store.ingest_document(document, [expected_chunks(document)[0].model_dump()], SMALL)
    with pytest.raises(InvalidIngestionError, match="sequence"):
        store.ingest_document(document, "chunks", SMALL)
    assert_empty(store)


def test_rejects_an_empty_chunk_list(store):
    with pytest.raises(InvalidIngestionError, match="at least one chunk"):
        store.ingest_document(make_document(), [], SMALL)
    assert_empty(store)


def test_rejects_chunks_of_another_document(store):
    document = make_document()
    other = make_document(long_content("beta"), source_id="CWE-79")

    with pytest.raises(InvalidIngestionError, match="not consistent"):
        store.ingest_document(document, expected_chunks(other), SMALL)
    assert_empty(store)


def test_rejects_stale_chunks_from_older_content(store):
    old = make_document(long_content("alpha"))
    new = make_document(long_content("omega"))

    with pytest.raises(InvalidIngestionError, match="not consistent"):
        store.ingest_document(new, expected_chunks(old), SMALL)
    assert_empty(store)


def test_rejects_chunks_with_a_missing_or_reordered_ordinal(store):
    document = make_document()
    chunks = expected_chunks(document)

    with pytest.raises(InvalidIngestionError, match="ordinals"):
        store.ingest_document(document, chunks[1:], SMALL)
    with pytest.raises(InvalidIngestionError, match="ordinals"):
        store.ingest_document(document, [chunks[1], chunks[0], *chunks[2:]], SMALL)
    assert_empty(store)


def test_rejects_chunks_whose_version_or_size_contradicts_the_config(store):
    document = make_document()
    chunks = expected_chunks(document)

    with pytest.raises(InvalidIngestionError, match="chunker_version"):
        store.ingest_document(document, chunks, ChunkingConfig(120, 20, "another-chunker-v2"))
    with pytest.raises(InvalidIngestionError, match="max_chars"):
        store.ingest_document(document, chunks, ChunkingConfig(max_chars=40, overlap_chars=5))
    assert_empty(store)


def test_rejects_a_chunk_that_was_built_without_validation(store):
    document = make_document()
    chunks = expected_chunks(document)
    fields = {name: getattr(chunks[0], name) for name in Chunk.model_fields}
    forged = Chunk.model_construct(**{**fields, "text": "forged text"})

    with pytest.raises(InvalidIngestionError):
        store.ingest_document(document, [forged, *chunks[1:]], SMALL)
    assert_empty(store)


def test_chunking_config_validates_like_the_chunker():
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars=10, overlap_chars=10)
    with pytest.raises(ValueError):
        ChunkingConfig(max_chars=0, overlap_chars=0)
    with pytest.raises(ValueError):
        ChunkingConfig(chunker_version="  ")
    with KnowledgeStore(":memory:") as opened:
        with pytest.raises(InvalidIngestionError, match="ChunkingConfig"):
            opened.ingest_document(make_document(), config="not a config")


def test_a_refused_update_keeps_the_previous_version(store):
    old = make_document(long_content("alpha"))
    ingest(store, old)
    new = make_document(long_content("omega"))

    with pytest.raises(InvalidIngestionError):
        store.ingest_document(new, expected_chunks(old), SMALL)  # stale chunks

    assert store.get_document(old.doc_id) == old
    assert store.get_chunks(old.doc_id) == expected_chunks(old)


# ------------------------------------------------------------
# corrupt or inconsistent persisted records are never silently accepted
# ------------------------------------------------------------

@pytest.fixture
def populated(db_path):
    """A file-backed store holding one document; yields (store, document, db_path)."""
    document = make_document()
    with KnowledgeStore(db_path) as opened:
        ingest(opened, document)
        yield opened, document, db_path


def test_malformed_document_json_is_reported_as_corruption(populated):
    opened, document, path = populated
    raw_exec(path, ("UPDATE documents SET document_json = ?", ("{not json",)))
    with pytest.raises(StoreCorruptionError, match="fails validation"):
        opened.get_document(document.doc_id)
    with pytest.raises(StoreCorruptionError):
        opened.get_chunks(document.doc_id)


def test_tampered_document_content_fails_its_hash_checks(populated):
    opened, document, path = populated
    text = raw_rows(path, "SELECT document_json FROM documents")[0][0]
    assert "alpha" in text
    raw_exec(path, ("UPDATE documents SET document_json = ?", (text.replace("alpha", "gamma"),)))
    with pytest.raises(StoreCorruptionError, match="fails validation"):
        opened.get_document(document.doc_id)


@pytest.mark.parametrize("column", ["content_sha256", "document_sha256", "source_id", "normalization_version"])
def test_document_columns_that_disagree_with_the_json_are_corruption(populated, column):
    opened, document, path = populated
    raw_exec(path, (f"UPDATE documents SET {column} = ?", ("0" * 64,)))
    with pytest.raises(StoreCorruptionError, match="disagree"):
        opened.get_document(document.doc_id)


def test_malformed_chunk_json_is_reported_as_corruption(populated):
    opened, document, path = populated
    raw_exec(path, ("UPDATE chunks SET chunk_json = ? WHERE ordinal = 1", ("[]",)))
    with pytest.raises(StoreCorruptionError, match="fails validation"):
        opened.get_chunks(document.doc_id)


def test_tampered_chunk_text_fails_chunk_validation(populated):
    opened, document, path = populated
    text = raw_rows(path, "SELECT chunk_json FROM chunks WHERE ordinal = 0")[0][0]
    raw_exec(path, ("UPDATE chunks SET chunk_json = ? WHERE ordinal = 0", (text.replace("Paragraph", "Paragrapx", 1),)))
    with pytest.raises(StoreCorruptionError, match="fails validation"):
        opened.get_chunks(document.doc_id)


def test_chunk_columns_that_disagree_with_the_json_are_corruption(populated):
    opened, document, path = populated
    raw_exec(path, ("UPDATE chunks SET chunk_sha256 = ? WHERE ordinal = 0", ("f" * 64,)))
    with pytest.raises(StoreCorruptionError, match="disagree"):
        opened.get_chunks(document.doc_id)


def test_a_valid_but_replaced_chunk_is_caught_by_the_chunk_digest(populated):
    opened, document, path = populated
    # A perfectly valid chunk of the same document (shifted by one character),
    # with matching columns: only the stored digest can reveal the swap.
    original = expected_chunks(document)[0]
    shifted = Chunk(
        doc_id=original.doc_id, ordinal=0, text=document.content[1 : original.char_end],
        char_start=1, char_end=original.char_end, doc_content_sha256=document.content_sha256,
        chunker_version=original.chunker_version, source=original.source, source_id=original.source_id,
        doc_type=original.doc_type, title=original.title, source_url=original.source_url,
    )
    assert document.contains_chunk(shifted)
    raw_exec(
        path,
        (
            "UPDATE chunks SET chunk_json = ?, chunk_sha256 = ? WHERE ordinal = 0",
            (store_module._canonical_json(shifted), shifted.chunk_sha256),
        ),
    )
    with pytest.raises(StoreCorruptionError, match="digest"):
        opened.get_chunks(document.doc_id)


def test_a_missing_chunk_row_is_corruption(populated):
    opened, document, path = populated
    raw_exec(path, "DELETE FROM chunks WHERE ordinal = 1")
    with pytest.raises(StoreCorruptionError, match="should have"):
        opened.get_chunks(document.doc_id)


def test_search_never_returns_a_corrupt_chunk(populated):
    opened, document, path = populated
    raw_exec(path, ("UPDATE chunks SET chunk_json = ? WHERE ordinal = 0", ("{}",)))
    with pytest.raises(StoreCorruptionError):
        opened.search("alpha", limit=MAX_SEARCH_LIMIT)


def test_search_reports_chunks_whose_document_row_vanished(populated):
    opened, document, path = populated
    raw_exec(path, "DELETE FROM documents")  # raw connection: foreign keys are off
    with pytest.raises(StoreCorruptionError, match="not stored"):
        opened.search("alpha")


# ------------------------------------------------------------
# idempotency and change detection
# ------------------------------------------------------------

@pytest.fixture
def audited(db_path):
    """
    A file-backed store plus an audit table filled by triggers on every
    insert, update or delete of documents/chunks (detects unnecessary rewrites).
    """
    with KnowledgeStore(db_path) as opened:
        raw_exec(
            db_path,
            "CREATE TABLE audit (event TEXT)",
            *[
                f"CREATE TRIGGER audit_{table}_{event.lower()} AFTER {event} ON {table} "
                f"BEGIN INSERT INTO audit VALUES ('{table}:{event}'); END"
                for table in ("documents", "chunks")
                for event in ("INSERT", "UPDATE", "DELETE")
            ],
        )
        yield opened, db_path


def audit_events(path):
    return [row[0] for row in raw_rows(path, "SELECT event FROM audit ORDER BY rowid")]


def test_first_ingestion_reports_created(store):
    document = make_document()

    result = ingest(store, document)

    assert isinstance(result, IngestResult)
    assert result == IngestResult(document.doc_id, "created", len(expected_chunks(document)))


def test_reingesting_an_unchanged_document_is_idempotent_and_writes_nothing(audited):
    opened, path = audited
    document = make_document()
    ingest(opened, document)
    counts = table_counts(path)
    stored_chunks = opened.get_chunks(document.doc_id)
    raw_exec(path, "DELETE FROM audit")

    for _ in range(3):
        result = ingest(opened, make_document())  # an equal but separately built document

        assert result.action == "unchanged"
        assert not (result.content_changed or result.metadata_changed or result.provenance_changed)
        assert not (result.config_changed or result.chunks_replaced)

    assert audit_events(path) == []  # no row was inserted, updated or deleted
    assert table_counts(path) == counts
    assert opened.get_chunks(document.doc_id) == stored_chunks
    assert opened.get_document(document.doc_id) == document


def test_identities_are_stable_across_ingestions(store):
    document = make_document()
    ingest(store, document)
    first_ids = [chunk.chunk_id for chunk in store.get_chunks(document.doc_id)]

    ingest(store, make_document(long_content("omega")))  # same source/source_id -> same doc_id

    assert store.list_doc_ids() == [document.doc_id]
    assert [c.chunk_id for c in store.get_chunks(document.doc_id)][:3] == first_ids[:3]
    assert all(c.chunk_id.startswith(f"{document.doc_id}#") for c in store.get_chunks(document.doc_id))


def test_content_change_replaces_every_stale_chunk(store):
    old = make_document(long_content("alpha", paragraphs=12))
    new = make_document(long_content("omega", paragraphs=4))
    ingest(store, old)
    assert [hit.chunk.doc_id for hit in store.search("alpha", MAX_SEARCH_LIMIT)]

    result = ingest(store, new)

    assert result.action == "updated"
    assert result.content_changed and result.chunks_replaced
    assert not result.metadata_changed
    assert store.get_document(new.doc_id) == new
    assert store.get_document(new.doc_id).content_sha256 != old.content_sha256
    chunks = store.get_chunks(new.doc_id)
    assert chunks == expected_chunks(new)
    assert store.count_chunks() == len(chunks) < len(expected_chunks(old))  # none left over
    assert all(chunk.doc_content_sha256 == new.content_sha256 for chunk in chunks)
    assert store.search("alpha") == []
    assert store.search("omega")


def test_metadata_change_with_identical_content_is_detected(store):
    old = make_document()
    new = make_document(title="SQL Injection (Renamed)")
    assert old.content_sha256 == new.content_sha256
    assert old.document_sha256 != new.document_sha256
    ingest(store, old)

    result = ingest(store, new)

    assert result.action == "updated"
    assert result.metadata_changed and not result.content_changed
    assert result.chunks_replaced  # chunks copy the title, so they must follow it
    assert store.get_document(new.doc_id).title == "SQL Injection (Renamed)"
    assert {chunk.title for chunk in store.get_chunks(new.doc_id)} == {"SQL Injection (Renamed)"}
    assert all(new.contains_chunk(chunk) for chunk in store.get_chunks(new.doc_id))
    assert store.search("Renamed")
    assert store.search("Injection") != []  # title words are indexed...
    assert all(hit.chunk.title.endswith("(Renamed)") for hit in store.search("Injection", MAX_SEARCH_LIMIT))


def test_source_url_change_is_propagated_to_chunks(store):
    ingest(store, make_document())
    new = make_document(source_url="https://example.org/cwe-89")

    result = ingest(store, new)

    assert result.metadata_changed and result.chunks_replaced
    assert {c.source_url for c in store.get_chunks(new.doc_id)} == {"https://example.org/cwe-89"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"license": "MIT"},
        {"source_version": "fixture-2"},
        {"source_modified": "2026-10-01"},
        {"references": ()},
    ],
)
def test_metadata_that_chunks_do_not_copy_updates_only_the_document(audited, overrides):
    opened, path = audited
    ingest(opened, make_document())
    raw_exec(path, "DELETE FROM audit")
    new = make_document(**overrides)

    result = ingest(opened, new)

    assert result.action == "updated"
    assert result.metadata_changed and not result.content_changed
    assert not result.chunks_replaced
    assert opened.get_document(new.doc_id) == new
    assert audit_events(path) == ["documents:UPDATE"]  # chunks were not rewritten
    assert all(new.contains_chunk(chunk) for chunk in opened.get_chunks(new.doc_id))


def test_provenance_only_change_updates_the_provenance(audited):
    opened, path = audited
    ingest(opened, make_document())
    raw_exec(path, "DELETE FROM audit")
    new = make_document(provenance=make_provenance(ingested_at_utc="2026-10-17T09:00:00Z"))
    assert new.document_sha256 == make_document().document_sha256  # provenance is not hashed

    result = ingest(opened, new)

    assert result.action == "updated"
    assert result.provenance_changed
    assert not (result.content_changed or result.metadata_changed or result.chunks_replaced)
    assert opened.get_document(new.doc_id).provenance.ingested_at_utc == "2026-10-17T09:00:00Z"
    assert audit_events(path) == ["documents:UPDATE"]


@pytest.mark.parametrize(
    "new_config",
    [
        ChunkingConfig(max_chars=120, overlap_chars=20, chunker_version="char-structure-v2"),
        ChunkingConfig(max_chars=200, overlap_chars=20),
        ChunkingConfig(max_chars=120, overlap_chars=40),
    ],
    ids=["chunker_version", "max_chars", "overlap_chars"],
)
def test_changed_chunking_configuration_regenerates_the_chunks(store, new_config):
    document = make_document()
    ingest(store, document)
    before = store.get_chunks(document.doc_id)

    result = ingest(store, document, new_config)

    assert result.action == "updated"
    assert result.config_changed and result.chunks_replaced
    assert not (result.content_changed or result.metadata_changed)
    after = store.get_chunks(document.doc_id)
    assert after == expected_chunks(document, new_config)
    assert after != before
    assert store.count_chunks() == len(after)
    assert {chunk.chunker_version for chunk in after} == {new_config.chunker_version}
    row = store._connection().execute(
        "SELECT chunker_version, max_chars, overlap_chars FROM documents"
    ).fetchone()
    assert tuple(row) == (new_config.chunker_version, new_config.max_chars, new_config.overlap_chars)


def test_a_config_change_that_yields_identical_chunks_is_still_recorded(audited):
    opened, path = audited
    document = make_document("A short entry.")
    ingest(opened, document, ChunkingConfig(max_chars=100, overlap_chars=10))
    raw_exec(path, "DELETE FROM audit")

    result = ingest(opened, document, ChunkingConfig(max_chars=200, overlap_chars=10))

    assert result.action == "updated"
    assert result.config_changed and not result.chunks_replaced
    assert audit_events(path) == ["documents:UPDATE"]
    row = raw_rows(path, "SELECT max_chars FROM documents")[0]
    assert row == (200,)


def test_switching_the_configuration_back_and_forth_leaves_no_leftovers(store):
    document = make_document()
    for config in (SMALL, ChunkingConfig(200, 20), SMALL, ChunkingConfig(60, 10), SMALL):
        ingest(store, document, config)
        assert store.get_chunks(document.doc_id) == expected_chunks(document, config)
        assert store.count_chunks() == len(expected_chunks(document, config))
        assert store.count_documents() == 1


# ------------------------------------------------------------
# atomicity: failures roll everything back together
# ------------------------------------------------------------

def add_failure_trigger(path, name, event, table, condition="1"):
    """Make SQLite itself abort the matching statement (a real mid-transaction failure)."""
    raw_exec(
        path,
        f"CREATE TRIGGER {name} BEFORE {event} ON {table} WHEN {condition} "
        f"BEGIN SELECT RAISE(ABORT, 'injected failure'); END",
    )


def test_failed_first_ingestion_leaves_no_partial_data(file_store, db_path):
    document = make_document()
    # Fails on the third chunk, after the document row, two chunks and two FTS rows exist.
    add_failure_trigger(db_path, "boom", "INSERT", "chunks", "NEW.ordinal = 2")

    with pytest.raises(KnowledgeStoreError, match="injected failure") as caught:
        ingest(file_store, document)

    assert isinstance(caught.value.__cause__, sqlite3.Error)
    assert table_counts(db_path) == {"documents": 0, "chunks": 0, "chunks_fts": 0}
    assert not file_store.has_document(document.doc_id)
    assert file_store.search("alpha") == []


def test_the_store_is_usable_again_after_a_failed_ingestion(file_store, db_path):
    document = make_document()
    add_failure_trigger(db_path, "boom", "INSERT", "chunks", "NEW.ordinal = 2")
    with pytest.raises(KnowledgeStoreError):
        ingest(file_store, document)

    raw_exec(db_path, "DROP TRIGGER boom")
    result = ingest(file_store, document)

    assert result.action == "created"
    assert file_store.get_chunks(document.doc_id) == expected_chunks(document)
    assert table_counts(db_path)["chunks_fts"] == table_counts(db_path)["chunks"]


def test_failed_replacement_keeps_the_previous_committed_version(file_store, db_path):
    old = make_document(long_content("alpha"))
    ingest(file_store, old)
    before = table_counts(db_path)
    add_failure_trigger(db_path, "boom", "INSERT", "chunks", "NEW.ordinal = 2")

    with pytest.raises(KnowledgeStoreError, match="injected failure"):
        ingest(file_store, make_document(long_content("omega")))

    assert table_counts(db_path) == before
    assert file_store.get_document(old.doc_id) == old
    assert file_store.get_chunks(old.doc_id) == expected_chunks(old)
    assert file_store.search("alpha")  # the old FTS rows are still there ...
    assert file_store.search("omega") == []  # ... and none of the new ones leaked in


def test_failed_metadata_replacement_rolls_back_document_chunks_and_index(file_store, db_path):
    old = make_document()
    ingest(file_store, old)
    add_failure_trigger(db_path, "boom", "INSERT", "chunks", "NEW.ordinal = 1")

    with pytest.raises(KnowledgeStoreError):
        ingest(file_store, make_document(title="Different Title"))

    assert file_store.get_document(old.doc_id).title == "SQL Injection"
    assert {c.title for c in file_store.get_chunks(old.doc_id)} == {"SQL Injection"}
    assert file_store.search("Different") == []


def test_failed_document_row_update_rolls_back_the_replaced_chunks(file_store, db_path):
    old = make_document()
    ingest(file_store, old)
    add_failure_trigger(db_path, "boom", "UPDATE", "documents")

    with pytest.raises(KnowledgeStoreError, match="injected failure"):
        ingest(file_store, make_document(long_content("omega")))

    assert file_store.get_document(old.doc_id) == old
    assert file_store.get_chunks(old.doc_id) == expected_chunks(old)


def test_failed_config_only_update_keeps_the_stored_configuration(file_store, db_path):
    document = make_document("A short entry.")
    ingest(file_store, document, ChunkingConfig(max_chars=100, overlap_chars=10))
    add_failure_trigger(db_path, "boom", "UPDATE", "documents")

    with pytest.raises(KnowledgeStoreError):
        ingest(file_store, document, ChunkingConfig(max_chars=200, overlap_chars=10))

    assert raw_rows(db_path, "SELECT max_chars, overlap_chars FROM documents") == [(100, 10)]


def test_failed_deletion_rolls_back_chunks_and_index_rows(file_store, db_path):
    document = make_document()
    ingest(file_store, document)
    before = table_counts(db_path)
    add_failure_trigger(db_path, "boom", "DELETE", "documents")

    with pytest.raises(KnowledgeStoreError, match="injected failure"):
        file_store.delete_document(document.doc_id)

    assert table_counts(db_path) == before
    assert file_store.get_chunks(document.doc_id) == expected_chunks(document)
    assert file_store.search("alpha")


def test_other_documents_are_unaffected_by_a_failed_ingestion(file_store, db_path):
    other = make_document(long_content("beta"), source_id="CWE-79")
    ingest(file_store, other)
    add_failure_trigger(db_path, "boom", "INSERT", "chunks", "NEW.doc_id = 'cwe:CWE-89' AND NEW.ordinal = 1")

    with pytest.raises(KnowledgeStoreError):
        ingest(file_store, make_document())

    assert file_store.list_doc_ids() == [other.doc_id]
    assert file_store.get_chunks(other.doc_id) == expected_chunks(other)


# ------------------------------------------------------------
# deletion
# ------------------------------------------------------------

def test_deleting_a_document_removes_its_chunks_and_index_entries(file_store, db_path):
    keep = make_document(long_content("beta"), source_id="CWE-79")
    drop = make_document()
    ingest(file_store, keep)
    ingest(file_store, drop)
    assert file_store.search("alpha")

    assert file_store.delete_document(drop.doc_id) is True

    assert not file_store.has_document(drop.doc_id)
    assert file_store.list_doc_ids() == [keep.doc_id]
    assert raw_rows(db_path, "SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (drop.doc_id,)) == [(0,)]
    assert table_counts(db_path)["chunks_fts"] == len(expected_chunks(keep))
    assert file_store.search("alpha") == []
    assert file_store.search("beta")
    with pytest.raises(DocumentNotFoundError):
        file_store.get_document(drop.doc_id)
    with pytest.raises(DocumentNotFoundError):
        file_store.get_chunks(drop.doc_id)


def test_repeated_and_unknown_deletions_return_false_without_side_effects(store):
    document = make_document()
    ingest(store, document)

    assert store.delete_document(document.doc_id) is True
    assert store.delete_document(document.doc_id) is False
    assert store.delete_document("cwe:CWE-404") is False
    assert store.count_documents() == store.count_chunks() == 0


def test_a_deleted_document_can_be_ingested_again(store):
    document = make_document()
    ingest(store, document)
    store.delete_document(document.doc_id)

    assert ingest(store, document).action == "created"
    assert store.get_chunks(document.doc_id) == expected_chunks(document)


# ------------------------------------------------------------
# keyword search
# ------------------------------------------------------------

def test_search_finds_chunks_with_a_distinctive_term(store):
    alpha = make_document(long_content("alpha"))
    beta = make_document(long_content("zebrafish"), source_id="CWE-79", title="Cross-site Scripting")
    ingest(store, alpha)
    ingest(store, beta)

    hits = store.search("zebrafish", limit=MAX_SEARCH_LIMIT)

    assert hits
    assert all(isinstance(hit, SearchHit) and isinstance(hit.chunk, Chunk) for hit in hits)
    assert {hit.chunk.doc_id for hit in hits} == {beta.doc_id}
    assert all("zebrafish" in hit.chunk.text for hit in hits)
    stored = store.get_chunks(beta.doc_id)
    assert all(hit.chunk in stored for hit in hits)  # the exact persisted, validated chunk
    assert all(beta.contains_chunk(hit.chunk) for hit in hits)


def test_search_matches_the_document_title(store):
    ingest(store, make_document(title="Improper Neutralization of Special Elements"))

    hits = store.search("Neutralization", limit=MAX_SEARCH_LIMIT)

    assert len(hits) == len(expected_chunks(make_document()))


def test_search_without_matches_or_on_an_empty_store_returns_an_empty_list(store):
    assert store.search("anything") == []
    ingest(store, make_document())
    assert store.search("nonexistentterm") == []


def test_search_ordering_is_deterministic_with_a_tie_breaker(store):
    # Identical text and title -> identical BM25 scores; (doc_id, ordinal) must decide.
    for source_id in ("CWE-2", "CWE-10", "CWE-1"):
        ingest(store, make_document("tiebreaker needle once", source_id=source_id))
    ingest(store, make_document("\n\n".join(["tiebreaker needle"] * 3), source_id="CWE-3"))

    first = store.search("needle", limit=MAX_SEARCH_LIMIT)
    second = store.search("needle", limit=MAX_SEARCH_LIMIT)

    assert [(h.chunk.chunk_id, h.bm25) for h in first] == [(h.chunk.chunk_id, h.bm25) for h in second]
    scores = [hit.bm25 for hit in first]
    assert scores == sorted(scores)  # best (lowest) BM25 first
    keys = [(hit.bm25, hit.chunk.doc_id, hit.chunk.ordinal) for hit in first]
    assert keys == sorted(keys)  # equal scores are ordered by doc_id, then numeric ordinal
    identical = [hit for hit in first if hit.chunk.doc_id in {"cwe:CWE-1", "cwe:CWE-2", "cwe:CWE-10"}]
    assert len({hit.bm25 for hit in identical}) == 1  # genuinely tied scores
    assert [hit.chunk.doc_id for hit in identical] == ["cwe:CWE-1", "cwe:CWE-10", "cwe:CWE-2"]


def test_ties_within_one_document_follow_ordinal_order(store):
    document = make_document("\n\n".join(["repeated needle phrase"] * 12))
    ingest(store, document, ChunkingConfig(max_chars=24, overlap_chars=0))

    hits = store.search("needle", limit=MAX_SEARCH_LIMIT)

    ordinals = [hit.chunk.ordinal for hit in hits]
    assert len(ordinals) > 10
    assert ordinals == sorted(ordinals)  # numeric, so 2 comes before 10


def test_search_limit_bounds_the_result_count(store):
    ingest(store, make_document("\n\n".join(["repeated needle phrase"] * 12)), ChunkingConfig(24, 0))

    assert len(store.search("needle", limit=3)) == 3
    assert len(store.search("needle", limit=1)) == 1
    assert len(store.search("needle")) == DEFAULT_SEARCH_LIMIT
    assert len(store.search("needle", limit=MAX_SEARCH_LIMIT)) == 12
    assert [h.chunk.chunk_id for h in store.search("needle", limit=3)] == [
        h.chunk.chunk_id for h in store.search("needle", limit=MAX_SEARCH_LIMIT)[:3]
    ]


@pytest.mark.parametrize("limit", [0, -1, MAX_SEARCH_LIMIT + 1, True, False, 2.5, "5", None])
def test_invalid_limits_are_rejected(store, limit):
    with pytest.raises(InvalidSearchQueryError, match="limit"):
        store.search("needle", limit=limit)


@pytest.mark.parametrize("query", ["", "   ", None, 42, b"needle", "x" * 1001])
def test_invalid_queries_are_rejected(store, query):
    with pytest.raises(InvalidSearchQueryError):
        store.search(query)


def test_replaced_chunks_are_no_longer_searchable(store):
    ingest(store, make_document(long_content("alpha")))
    assert store.search("alpha")

    ingest(store, make_document(long_content("omega")))

    assert store.search("alpha") == []
    assert {hit.chunk.doc_id for hit in store.search("omega", MAX_SEARCH_LIMIT)} == {"cwe:CWE-89"}


def test_index_row_count_matches_chunk_count_through_every_operation(file_store, db_path):
    document = make_document()

    def in_sync():
        counts = table_counts(db_path)
        return counts["chunks"] == counts["chunks_fts"]

    ingest(file_store, document)
    assert in_sync()
    ingest(file_store, make_document(long_content("omega", paragraphs=3)))
    assert in_sync()
    ingest(file_store, make_document(long_content("omega", paragraphs=3)), ChunkingConfig(60, 5))
    assert in_sync()
    file_store.delete_document(document.doc_id)
    assert in_sync() and table_counts(db_path)["chunks_fts"] == 0


@pytest.mark.parametrize("query", ['"unterminated', "AND", "(", "alpha AND", "nosuchcolumn:alpha"])
def test_malformed_fts_queries_raise_a_clear_error(store, query):
    ingest(store, make_document())

    with pytest.raises(InvalidSearchQueryError, match="invalid FTS5 query") as caught:
        store.search(query)

    assert isinstance(caught.value.__cause__, sqlite3.OperationalError)
    assert store.search("alpha")  # the store is still healthy afterwards


def test_hyphenated_identifiers_must_be_quoted_phrases(store):
    ingest(store, make_document("Reference identifier CWE-89 appears here."))

    assert store.search('"CWE-89"')
    with pytest.raises(InvalidSearchQueryError):
        store.search("CWE-89")  # '-' is FTS5 syntax; the error is explicit, not a silent fallback


def test_fts5_phrase_and_prefix_syntax_is_supported(store):
    ingest(store, make_document("Improper neutralization of special elements in a command."))

    assert store.search('"special elements"')
    assert store.search("neutral*")
    assert store.search("improper AND command")
    assert store.search("improper NOT command") == []


@pytest.mark.parametrize(
    "query",
    [
        "x'; DROP TABLE documents; --",
        "\" ; DROP TABLE chunks; --",
        "alpha' OR '1'='1",
        "alpha); DELETE FROM documents; --",
    ],
)
def test_search_input_cannot_alter_sql_structure(store, query):
    document = make_document()
    ingest(store, document)

    try:
        store.search(query)
    except InvalidSearchQueryError:
        pass  # a syntax error is an acceptable outcome; an executed statement is not

    assert store.has_document(document.doc_id)
    assert store.get_chunks(document.doc_id) == expected_chunks(document)


def test_quoted_sql_text_is_just_a_phrase_to_search_for(store):
    ingest(store, make_document())

    assert store.search('"; DROP TABLE documents; --"') == []
    assert store.count_documents() == 1


def test_ids_with_sql_metacharacters_are_bound_not_interpolated(store):
    ingest(store, make_document())
    hostile = "cwe:CWE-89' OR '1'='1"

    assert store.has_document(hostile) is False
    assert store.delete_document(hostile) is False
    with pytest.raises(DocumentNotFoundError):
        store.get_document(hostile)
    assert store.count_documents() == 1
