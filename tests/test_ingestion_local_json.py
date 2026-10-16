import ast
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import ingestion.local_json as local_json_module
from ingestion.local_json import FORMAT_ID, LocalJsonError, load_documents
from knowledge.schema import SourceDocument, content_sha256, make_doc_id

FIXTURE = Path(__file__).parent / "fixtures" / "knowledge_ingestion" / "synthetic_documents.json"


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def fixture_data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def write_json(tmp_path: Path, data, name: str = "docs.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def envelope(*records) -> dict:
    return {"format": FORMAT_ID, "documents": list(records)}


def record(**overrides) -> dict:
    """A minimal valid, clearly synthetic record."""
    base = copy.deepcopy(fixture_data()["documents"][0])
    base.update(overrides)
    return base


# ------------------------------------------------------------
# valid input
# ------------------------------------------------------------

def test_loads_the_fixture_as_validated_source_documents():
    documents = load_documents(FIXTURE)

    assert len(documents) == 3
    assert all(isinstance(document, SourceDocument) for document in documents)
    assert [d.doc_id for d in documents] == [
        "cwe:SYNTHETIC-0001",
        "capec:SYNTHETIC-0002",
        "owasp:SYNTHETIC-0003",
    ]  # file order is preserved


def test_accepts_str_and_path_and_loading_is_repeatable():
    assert load_documents(str(FIXTURE)) == load_documents(FIXTURE) == load_documents(FIXTURE)


def test_metadata_and_content_are_preserved_exactly():
    raw = fixture_data()["documents"]
    first, second, third = load_documents(FIXTURE)

    assert (first.source, first.source_id, first.doc_type) == ("cwe", "SYNTHETIC-0001", "weakness")
    assert first.source_version == "synthetic-fixture-v1"
    assert first.title == raw[0]["title"]
    assert first.content == raw[0]["content"]  # the fixture is already normalized
    assert first.license == raw[0]["license"]
    assert first.source_url is None and first.source_modified is None  # absent stays absent
    assert first.references == ()
    assert second.references[0].relation == "CanExploit"
    assert second.references[0].target_doc_id == first.doc_id
    assert third.doc_type == "guideline"
    assert first.provenance.origin_kind == "bundled_fixture"
    assert first.provenance.artifact_path == raw[0]["provenance"]["artifact_path"]
    assert first.provenance.retrieved_at_utc == "2026-10-16T00:00:00Z"
    assert first.provenance.adapter == "local_json"


def test_derived_fields_are_computed_by_the_schema_not_the_loader():
    for document in load_documents(FIXTURE):
        assert document.doc_id == make_doc_id(document.source, document.source_id)
        assert document.content_sha256 == content_sha256(document.content)
        assert len(document.document_sha256) == 64


def test_supplied_derived_fields_are_accepted_when_correct(tmp_path):
    document = load_documents(FIXTURE)[0]
    path = write_json(tmp_path, envelope(document.model_dump(mode="json")))

    assert load_documents(path) == [document]


def test_a_dumped_document_round_trips_through_a_file(tmp_path):
    documents = load_documents(FIXTURE)
    path = write_json(tmp_path, envelope(*[d.model_dump(mode="json") for d in documents]))

    assert load_documents(path) == documents


def test_a_byte_order_mark_is_tolerated(tmp_path):
    path = tmp_path / "bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(fixture_data()).encode("utf-8"))

    assert len(load_documents(path)) == 3


def test_content_is_normalized_by_the_schema(tmp_path):
    path = write_json(tmp_path, envelope(record(content="line one \r\nline two\r\n\r\n\r\n")))

    assert load_documents(path)[0].content == "line one\nline two"


def test_fixture_is_clearly_labelled_synthetic():
    data = fixture_data()
    assert "SYNTHETIC TEST DATA" in data["description"]
    for document in load_documents(FIXTURE):
        assert document.source_id.startswith("SYNTHETIC-")
        assert "SYNTHETIC TEST DATA" in document.title
        assert document.content.startswith("SYNTHETIC TEST DATA")
        assert document.source_version.startswith("synthetic")
        assert "synthetic" in document.license
        assert document.source_url is None  # no link to a genuine record


# ------------------------------------------------------------
# empty input
# ------------------------------------------------------------

def test_an_empty_documents_list_is_valid_and_returns_an_empty_list(tmp_path):
    assert load_documents(write_json(tmp_path, envelope())) == []


# ------------------------------------------------------------
# unreadable files
# ------------------------------------------------------------

def test_a_missing_file_is_reported_with_its_path(tmp_path):
    path = tmp_path / "missing.json"
    with pytest.raises(LocalJsonError, match="missing.json") as caught:
        load_documents(path)
    assert isinstance(caught.value.__cause__, OSError)


def test_a_directory_is_not_a_readable_file(tmp_path):
    with pytest.raises(LocalJsonError, match="cannot read") as caught:
        load_documents(tmp_path)
    assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize("bad", [None, 42, b"docs.json", ["docs.json"]])
def test_a_path_of_the_wrong_type_is_rejected(bad):
    with pytest.raises(TypeError, match="path"):
        load_documents(bad)


def test_invalid_utf8_is_reported(tmp_path):
    path = tmp_path / "latin1.json"
    path.write_bytes(b'{"format": "\xff\xfe"}')
    with pytest.raises(LocalJsonError, match="UTF-8") as caught:
        load_documents(path)
    assert isinstance(caught.value.__cause__, UnicodeDecodeError)


# ------------------------------------------------------------
# malformed JSON and unsupported structure
# ------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "{not json", '{"format": ', "[1, 2"])
def test_malformed_json_is_reported_with_path_and_position(tmp_path, text):
    path = tmp_path / "broken.json"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(LocalJsonError, match="broken.json: malformed JSON") as caught:
        load_documents(path)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_duplicate_json_keys_are_rejected_instead_of_silently_merged(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(
        '{"format": "' + FORMAT_ID + '", "documents": [], "documents": []}', encoding="utf-8"
    )
    with pytest.raises(LocalJsonError, match="duplicate JSON key 'documents'"):
        load_documents(path)


def test_duplicate_keys_inside_a_record_are_rejected(tmp_path):
    text = json.dumps(envelope(record())).replace('"source_id"', '"title": "x", "source_id"', 1)
    path = tmp_path / "dup_record.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(LocalJsonError, match="duplicate JSON key 'title'"):
        load_documents(path)


@pytest.mark.parametrize("top_level", [[], [record()], "documents", 7, None, True])
def test_unsupported_top_level_structures_are_rejected(tmp_path, top_level):
    with pytest.raises(LocalJsonError, match="unsupported top-level JSON"):
        load_documents(write_json(tmp_path, top_level))


@pytest.mark.parametrize("fmt", [None, "", "autoshield.local_documents.v2", 1])
def test_a_missing_or_wrong_format_marker_is_rejected(tmp_path, fmt):
    data = envelope(record())
    if fmt is None:
        del data["format"]
    else:
        data["format"] = fmt
    with pytest.raises(LocalJsonError, match="'format' must be"):
        load_documents(write_json(tmp_path, data))


def test_unknown_top_level_keys_are_rejected(tmp_path):
    data = envelope(record())
    data["document"] = []  # typo for "documents"
    with pytest.raises(LocalJsonError, match="unknown top-level keys.*document"):
        load_documents(write_json(tmp_path, data))


def test_a_missing_or_non_list_documents_value_is_rejected(tmp_path):
    with pytest.raises(LocalJsonError, match="'documents' must be a list"):
        load_documents(write_json(tmp_path, {"format": FORMAT_ID}))
    with pytest.raises(LocalJsonError, match="'documents' must be a list"):
        load_documents(write_json(tmp_path, {"format": FORMAT_ID, "documents": record()}))


def test_a_non_string_description_is_rejected(tmp_path):
    data = envelope(record())
    data["description"] = 5
    with pytest.raises(LocalJsonError, match="description"):
        load_documents(write_json(tmp_path, data))


@pytest.mark.parametrize("bad_record", ["text", 5, None, [record()]])
def test_records_must_be_objects(tmp_path, bad_record):
    path = write_json(tmp_path, envelope(record(), bad_record))
    with pytest.raises(LocalJsonError, match=r"documents\[1\] must be a JSON object"):
        load_documents(path)


# ------------------------------------------------------------
# invalid document records
# ------------------------------------------------------------

def test_an_invalid_record_names_the_path_and_record_index(tmp_path):
    good = record(source_id="SYNTHETIC-0001")
    bad = record(source_id="SYNTHETIC-0002", source_url="ftp://example.invalid/x")
    path = write_json(tmp_path, envelope(good, bad), name="records.json")

    with pytest.raises(LocalJsonError) as caught:
        load_documents(path)

    message = str(caught.value)
    assert "records.json" in message
    assert "documents[1]" in message
    assert "source_url" in message
    assert isinstance(caught.value.__cause__, ValidationError)


def test_nothing_is_returned_when_any_record_is_invalid(tmp_path):
    path = write_json(tmp_path, envelope(record(source_id="SYNTHETIC-0001"), record(title="  ")))
    with pytest.raises(LocalJsonError, match=r"documents\[1\]"):
        load_documents(path)


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"source": "not-a-source"}, "source"),
        ({"doc_type": "poem"}, "doc_type"),
        ({"source_id": "has space"}, "source_id"),
        ({"source_id": "has#hash"}, "source_id"),
        ({"content": "   \n  "}, "content"),
        ({"source_modified": "yesterday"}, "source_modified"),
        ({"license": "  "}, "license"),
        ({"unexpected_field": 1}, "unexpected_field"),
        ({"references": [{"relation": "ChildOf"}]}, "references"),
        ({"provenance": {"origin_kind": "bundled_fixture"}}, "provenance"),
    ],
)
def test_schema_violations_are_reported_for_the_offending_field(tmp_path, overrides, field):
    with pytest.raises(LocalJsonError, match=field):
        load_documents(write_json(tmp_path, envelope(record(**overrides))))


@pytest.mark.parametrize("missing", ["source", "source_id", "source_version", "doc_type", "title", "content", "provenance"])
def test_required_fields_are_never_invented(tmp_path, missing):
    bad = record()
    del bad[missing]
    with pytest.raises(LocalJsonError, match=missing):
        load_documents(write_json(tmp_path, envelope(bad)))


def test_wrong_supplied_hashes_are_rejected(tmp_path):
    bad_content_hash = record(content_sha256="0" * 64)
    with pytest.raises(LocalJsonError, match="content_sha256"):
        load_documents(write_json(tmp_path, envelope(bad_content_hash)))
    bad_doc_hash = record(document_sha256="0" * 64)
    with pytest.raises(LocalJsonError, match="document_sha256"):
        load_documents(write_json(tmp_path, envelope(bad_doc_hash)))


def test_a_supplied_doc_id_must_match_source_and_source_id(tmp_path):
    with pytest.raises(LocalJsonError, match="doc_id"):
        load_documents(write_json(tmp_path, envelope(record(doc_id="cwe:SYNTHETIC-9999"))))


def test_validation_errors_do_not_dump_the_document_content(tmp_path):
    long_content = "secret-marker " * 2000
    path = write_json(tmp_path, envelope(record(content=long_content, doc_type="poem")))
    with pytest.raises(LocalJsonError) as caught:
        load_documents(path)
    assert "secret-marker" not in str(caught.value)


# ------------------------------------------------------------
# duplicate identities
# ------------------------------------------------------------

def test_a_repeated_doc_id_is_an_error_with_both_indices(tmp_path):
    a = record(source_id="SYNTHETIC-0001")
    b = record(source_id="SYNTHETIC-0002")
    path = write_json(tmp_path, envelope(a, b, copy.deepcopy(a)))

    with pytest.raises(LocalJsonError) as caught:
        load_documents(path)

    message = str(caught.value)
    assert "documents[2]" in message and "documents[0]" in message
    assert "cwe:SYNTHETIC-0001" in message


def test_conflicting_records_with_one_doc_id_are_an_error(tmp_path):
    first = record(title="First title")
    second = record(title="Second title")
    with pytest.raises(LocalJsonError, match="repeats doc_id"):
        load_documents(write_json(tmp_path, envelope(first, second)))


# ------------------------------------------------------------
# boundaries: the adapter does not touch storage, chunking or the pipeline
# ------------------------------------------------------------

def test_the_loader_does_not_import_storage_chunking_or_the_pipeline():
    tree = ast.parse(Path(local_json_module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")

    assert imported == {"__future__", "json", "pathlib", "pydantic", "knowledge.schema"}
