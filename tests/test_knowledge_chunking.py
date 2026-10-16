import hashlib

import pytest
from pydantic import ValidationError

from knowledge.chunking import (
    CHUNKER_VERSION,
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_document,
)
from knowledge.schema import Chunk, ProvenanceRecord, SourceDocument, make_chunk_id


def make_document(content: str, **overrides) -> SourceDocument:
    values = {
        "source": "cwe",
        "source_id": "CWE-89",
        "source_version": "fixture-1",
        "doc_type": "weakness",
        "title": "SQL Injection",
        "content": content,
        "source_url": "https://cwe.mitre.org/data/definitions/89.html",
        "provenance": ProvenanceRecord(
            origin_kind="bundled_fixture",
            artifact_path="tests/fixtures/knowledge/cwe_sample.jsonl",
            artifact_sha256="a" * 64,
            adapter="normalized_jsonl",
            adapter_version="0.1",
            retrieved_at_utc="2026-10-16T00:00:00Z",
            ingested_at_utc="2026-10-16T01:00:00Z",
        ),
    }
    values.update(overrides)
    return SourceDocument(**values)


def spans(chunks):
    return [(chunk.char_start, chunk.char_end) for chunk in chunks]


def texts(chunks):
    return [chunk.text for chunk in chunks]


def long_content() -> str:
    """Deterministic multi-paragraph text that normalization leaves unchanged."""
    paragraphs = []
    for index in range(30):
        words = " ".join(f"word{j}" for j in range(index % 7 + 3))
        paragraphs.append(f"Paragraph {index}: {words}\nsecond line {'x' * (index * 3)}")
    return "\n\n".join(paragraphs)


FENCED_CONTENT = "```\naaaa\n\n\n\nbb cc dd ee ff\n```\n\ntail text after the fence"

DOCUMENTS = {
    "multi_paragraph": long_content(),
    "unbroken": "x" * 500,
    "fenced": FENCED_CONTENT,
    "words": " ".join(f"token{i}" for i in range(120)),
}

SETTINGS = [(1, 0), (2, 1), (5, 0), (5, 4), (17, 3), (50, 10), (200, 50), (1200, 120)]


def covered_indices(chunks):
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.char_start, chunk.char_end))
    return covered


# ------------------------------------------------------------
# defaults and determinism
# ------------------------------------------------------------

def test_defaults_are_explicit_and_consistent():
    assert DEFAULT_MAX_CHARS == 1200
    assert DEFAULT_OVERLAP_CHARS == 120
    assert 0 <= DEFAULT_OVERLAP_CHARS < DEFAULT_MAX_CHARS
    assert CHUNKER_VERSION == "char-structure-v1"


def test_default_settings_are_applied():
    document = make_document("word " * 600 + "end")  # 3003 characters

    chunks = chunk_document(document)

    assert len(chunks) > 1
    assert {chunk.chunker_version for chunk in chunks} == {CHUNKER_VERSION}
    assert max(len(chunk.text) for chunk in chunks) <= DEFAULT_MAX_CHARS
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.char_end - later.char_start == DEFAULT_OVERLAP_CHARS


def test_repeated_calls_produce_identical_chunks_ids_and_hashes():
    document = make_document(long_content())

    first = chunk_document(document, max_chars=120, overlap_chars=20)
    second = chunk_document(document, max_chars=120, overlap_chars=20)
    rebuilt = chunk_document(make_document(long_content()), max_chars=120, overlap_chars=20)

    assert len(first) > 5
    assert first == second == rebuilt
    assert [c.chunk_id for c in first] == [c.chunk_id for c in rebuilt]
    assert [c.chunk_sha256 for c in first] == [c.chunk_sha256 for c in rebuilt]
    assert [c.model_dump() for c in first] == [c.model_dump() for c in rebuilt]


def test_returned_list_is_new_on_each_call_and_holds_validated_chunks():
    document = make_document(long_content())

    first = chunk_document(document, max_chars=100, overlap_chars=0)
    first.clear()
    second = chunk_document(document, max_chars=100, overlap_chars=0)

    assert second
    assert all(isinstance(chunk, Chunk) for chunk in second)
    for chunk in second:  # validated models survive a validation round trip
        assert Chunk.model_validate(chunk.model_dump()) == chunk


# ------------------------------------------------------------
# natural-boundary preferences
# ------------------------------------------------------------

def test_paragraph_boundary_is_preferred_over_a_later_line_or_space():
    # Window [0, 16) holds a paragraph break at 5-6, a line break at 12 and
    # spaces; the paragraph break must win although it is earlier.
    document = make_document("aa bb\n\ncc dd\nee ff gg")

    chunks = chunk_document(document, max_chars=16, overlap_chars=0)

    assert texts(chunks) == ["aa bb\n\n", "cc dd\nee ff gg"]
    assert spans(chunks) == [(0, 7), (7, 21)]


def test_the_last_paragraph_boundary_in_the_window_is_used():
    document = make_document("aa\n\nbb\n\ncc dd ee ff gg hh")

    chunks = chunk_document(document, max_chars=11, overlap_chars=0)

    # Window [0, 11) = "aa\n\nbb\n\ncc d": the SECOND paragraph break ends at 8.
    assert chunks[0].text == "aa\n\nbb\n\n"


def test_line_boundary_is_preferred_over_a_later_space():
    # No paragraph break. Window [0, 14) = "aa bb cc\ndd e": the line break
    # ends at 9, a later space at 11 must not be used.
    document = make_document("aa bb cc\ndd ee ff gg")

    chunks = chunk_document(document, max_chars=14, overlap_chars=0)

    assert texts(chunks) == ["aa bb cc\n", "dd ee ff gg"]
    assert spans(chunks) == [(0, 9), (9, 20)]


def test_whitespace_boundary_is_used_when_there_is_no_line_break():
    document = make_document("aaaa bbbb cccc dddd")

    chunks = chunk_document(document, max_chars=12, overlap_chars=0)

    # Window [0, 12) = "aaaa bbbb cc": cut after the last space, not inside "cccc".
    assert texts(chunks) == ["aaaa bbbb ", "cccc dddd"]


def test_boundary_that_would_leave_the_window_unusable_falls_back_to_the_next_kind():
    # The only paragraph/line breaks end at offset 4. With overlap 5 a chunk
    # ending there would not move forward (4 <= 5), so they are unusable and
    # the chunker hard-cuts instead of stepping back or shrinking the overlap.
    document = make_document("aa\n\nbbbbbbbbbbbbbb")

    chunks = chunk_document(document, max_chars=8, overlap_chars=5)

    assert spans(chunks) == [(0, 8), (3, 11), (6, 14), (9, 17), (12, 18)]

    # With overlap 3 the same break IS usable (4 > 3), so it is chosen.
    assert chunk_document(document, max_chars=8, overlap_chars=3)[0].char_end == 4


def test_a_chunk_never_ends_inside_a_word_when_a_space_is_available():
    document = make_document(" ".join(["alpha", "beta", "gamma", "delta"] * 6))

    chunks = chunk_document(document, max_chars=21, overlap_chars=0)

    for chunk in chunks[:-1]:
        assert chunk.text.endswith(" ")
    assert "".join(texts(chunks)) == document.content


# ------------------------------------------------------------
# hard splits
# ------------------------------------------------------------

def test_long_unbroken_string_is_hard_split_into_exact_pieces():
    document = make_document("x" * 25)

    chunks = chunk_document(document, max_chars=10, overlap_chars=0)

    assert texts(chunks) == ["x" * 10, "x" * 10, "x" * 5]
    assert spans(chunks) == [(0, 10), (10, 20), (20, 25)]


def test_hard_split_with_overlap_shares_exactly_the_overlap():
    document = make_document("x" * 25)

    chunks = chunk_document(document, max_chars=10, overlap_chars=3)

    assert spans(chunks) == [(0, 10), (7, 17), (14, 24), (21, 25)]
    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.char_end - later.char_start == 3


def test_distinct_characters_prove_the_hard_split_uses_the_right_slices():
    content = "".join(chr(ord("a") + i % 26) for i in range(60))  # abc...zabc...
    document = make_document(content)

    chunks = chunk_document(document, max_chars=13, overlap_chars=0)

    assert texts(chunks) == [content[i : i + 13] for i in range(0, 60, 13)]
    assert "".join(texts(chunks)) == content


# ------------------------------------------------------------
# size limit, offsets, forward progress, coverage
# ------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DOCUMENTS))
@pytest.mark.parametrize("max_chars,overlap", SETTINGS)
def test_every_chunk_is_bounded_exact_valid_and_the_loop_progresses(name, max_chars, overlap):
    document = make_document(DOCUMENTS[name])
    content = document.content

    chunks = chunk_document(document, max_chars=max_chars, overlap_chars=overlap)

    assert chunks
    for chunk in chunks:
        assert 0 < len(chunk.text) <= max_chars
        assert chunk.char_end - chunk.char_start == len(chunk.text)
        assert chunk.text == content[chunk.char_start:chunk.char_end]
        assert chunk.text.strip()
        assert document.contains_chunk(chunk)
    for earlier, later in zip(chunks, chunks[1:]):
        assert later.char_start > earlier.char_start  # forward progress
        assert later.char_end > earlier.char_end
        assert later.char_start >= earlier.char_end - overlap  # never more than the overlap
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(content)
    non_whitespace = {i for i, ch in enumerate(content) if not ch.isspace()}
    assert non_whitespace <= covered_indices(chunks)


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
@pytest.mark.parametrize("max_chars,overlap", [s for s in SETTINGS if s[0] >= 50])
def test_consecutive_chunks_share_exactly_the_overlap_and_nothing_more(
    name, max_chars, overlap
):
    document = make_document(DOCUMENTS[name])

    chunks = chunk_document(document, max_chars=max_chars, overlap_chars=overlap)

    for earlier, later in zip(chunks, chunks[1:]):
        assert earlier.char_end - later.char_start == overlap
        shared = document.content[later.char_start:earlier.char_end]
        assert earlier.text.endswith(shared) and later.text.startswith(shared)
    total = sum(len(chunk.text) for chunk in chunks)
    assert total == len(document.content) + overlap * (len(chunks) - 1)


@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_without_overlap_the_chunks_concatenate_to_the_content(name):
    document = make_document(DOCUMENTS[name])

    for max_chars in (7, 50, 333):
        chunks = chunk_document(document, max_chars=max_chars, overlap_chars=0)

        assert "".join(texts(chunks)) == document.content
        for earlier, later in zip(chunks, chunks[1:]):
            assert earlier.char_end == later.char_start


def test_overlap_one_less_than_max_still_advances_on_unbroken_text():
    document = make_document("x" * 20)

    chunks = chunk_document(document, max_chars=5, overlap_chars=4)

    assert [chunk.char_start for chunk in chunks] == list(range(16))
    assert spans(chunks)[-1] == (15, 20)
    assert all(len(chunk.text) <= 5 for chunk in chunks)


@pytest.mark.parametrize("max_chars", [2, 3, 7, 50])
@pytest.mark.parametrize("name", sorted(DOCUMENTS))
def test_maximum_overlap_never_loops_or_oversizes(name, max_chars):
    document = make_document(DOCUMENTS[name])

    chunks = chunk_document(document, max_chars=max_chars, overlap_chars=max_chars - 1)

    assert len(chunks) <= len(document.content)  # at least one character of progress each
    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(set(starts))
    assert all(len(chunk.text) <= max_chars for chunk in chunks)
    assert all(document.contains_chunk(chunk) for chunk in chunks)


def test_natural_boundary_too_close_to_the_start_cannot_stall_the_loop():
    # Single-letter words give a space boundary every two characters, which
    # is always within the overlap of the previous chunk.
    document = make_document(" ".join("abcdefghijklmnopqrstuvwxyz"))

    chunks = chunk_document(document, max_chars=6, overlap_chars=5)

    starts = [chunk.char_start for chunk in chunks]
    assert starts == sorted(set(starts))
    assert chunks[-1].char_end == len(document.content)
    assert all(len(chunk.text) <= 6 for chunk in chunks)


def test_natural_boundaries_are_used_with_overlap():
    document = make_document("aaaa bbbb cccc dddd eeee")

    chunks = chunk_document(document, max_chars=12, overlap_chars=4)

    assert spans(chunks) == [(0, 10), (6, 15), (11, 20), (16, 24)]
    assert texts(chunks) == ["aaaa bbbb ", "bbb cccc ", "ccc dddd ", "ddd eeee"]


# ------------------------------------------------------------
# empty, whitespace-only, short and exact-limit content
# ------------------------------------------------------------

@pytest.mark.parametrize("content", ["", " ", "   \n\t  \n", " " * 50])
def test_empty_and_whitespace_only_content_yield_no_chunks(content):
    # SourceDocument itself cannot hold such content, so build one
    # without validation to exercise the chunker's own behaviour.
    document = make_document("placeholder").model_copy(update={"content": content})

    assert chunk_document(document, max_chars=7, overlap_chars=2) == []
    assert chunk_document(document) == []


@pytest.mark.parametrize("content", ["", "   ", "\n\t\n"])
def test_the_schema_rejects_empty_documents_before_chunking_is_reachable(content):
    with pytest.raises(ValidationError, match="content must not be empty"):
        make_document(content)


def test_document_shorter_than_the_maximum_is_one_chunk():
    document = make_document("short text")

    chunks = chunk_document(document, max_chars=100, overlap_chars=10)

    assert len(chunks) == 1
    assert chunks[0].text == document.content
    assert spans(chunks) == [(0, 10)]
    assert chunks[0].chunk_id == "cwe:CWE-89#0000"


@pytest.mark.parametrize("overlap", [0, 3, 9])
def test_content_of_exactly_the_maximum_size_is_one_chunk_even_with_breaks(overlap):
    content = "ab\n\ncd ef\ngh"  # paragraph, line and space boundaries, 12 characters
    document = make_document(content)
    assert len(document.content) == 12

    chunks = chunk_document(document, max_chars=12, overlap_chars=overlap)

    assert texts(chunks) == [content]
    assert spans(chunks) == [(0, 12)]


@pytest.mark.parametrize("overlap,expected", [(0, [(0, 10), (10, 11)]), (4, [(0, 10), (6, 11)])])
def test_content_one_over_the_maximum_needs_a_second_chunk(overlap, expected):
    document = make_document("x" * 11)

    chunks = chunk_document(document, max_chars=10, overlap_chars=overlap)

    assert spans(chunks) == expected


def test_whitespace_only_windows_are_skipped_and_never_become_blank_chunks():
    content = "a" + " " * 30 + "b"
    document = make_document(content)
    assert document.content == content

    chunks = chunk_document(document, max_chars=10, overlap_chars=0)

    assert texts(chunks) == ["a" + " " * 9, " b"]
    assert spans(chunks) == [(0, 10), (30, 32)]
    assert all(chunk.text.strip() for chunk in chunks)
    assert {0, 31} <= covered_indices(chunks)  # both non-whitespace characters


def test_leading_indentation_of_the_content_is_preserved():
    document = make_document("    indented code()\nsecond")

    chunks = chunk_document(document, max_chars=100, overlap_chars=0)

    assert chunks[0].text == "    indented code()\nsecond"


# ------------------------------------------------------------
# identifiers, hashes and copied metadata
# ------------------------------------------------------------

def test_ordinals_are_sequential_and_ids_use_the_schema_helper():
    document = make_document(long_content())

    chunks = chunk_document(document, max_chars=100, overlap_chars=10)

    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunk_id for chunk in chunks] == [
        make_chunk_id(document.doc_id, i) for i in range(len(chunks))
    ]
    assert chunks[0].chunk_id == "cwe:CWE-89#0000"
    assert chunks[3].chunk_id == "cwe:CWE-89#0003"
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_hashes_are_independent_sha256_of_the_text():
    document = make_document(long_content())

    for chunk in chunk_document(document, max_chars=100, overlap_chars=10):
        assert chunk.chunk_sha256 == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()


def test_chunks_carry_the_parent_content_hash_and_document_metadata():
    document = make_document(long_content())

    chunks = chunk_document(document, max_chars=100, overlap_chars=10, chunker_version="v-test")

    assert len(chunks) > 1
    expected_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
    assert document.content_sha256 == expected_hash
    for chunk in chunks:
        assert chunk.doc_content_sha256 == expected_hash
        assert chunk.doc_id == "cwe:CWE-89"
        assert chunk.source == "cwe"
        assert chunk.source_id == "CWE-89"
        assert chunk.doc_type == "weakness"
        assert chunk.title == "SQL Injection"
        assert chunk.source_url == "https://cwe.mitre.org/data/definitions/89.html"
        assert chunk.chunker_version == "v-test"
        assert chunk.section is None


def test_missing_source_url_is_copied_as_none():
    document = make_document("some text", source_url=None)

    assert chunk_document(document)[0].source_url is None


def test_document_title_is_the_normalized_title():
    document = make_document("some text", title="  SQL \n Injection ")

    assert chunk_document(document)[0].title == "SQL Injection"


def test_chunks_of_changed_content_are_stale_for_the_new_document():
    old = make_document("alpha beta gamma delta")
    new = make_document("alpha beta gamma delta epsilon")

    old_chunks = chunk_document(old, max_chars=100, overlap_chars=0)
    new_chunks = chunk_document(new, max_chars=100, overlap_chars=0)

    assert new.content.startswith(old_chunks[0].text)  # same text, still a prefix
    assert not new.contains_chunk(old_chunks[0])  # but cut from the old content
    assert old_chunks[0].doc_content_sha256 != new_chunks[0].doc_content_sha256
    assert new.contains_chunk(new_chunks[0])


def test_chunker_version_can_be_changed_and_changes_nothing_else():
    document = make_document(long_content())

    first = chunk_document(document, max_chars=100, overlap_chars=0, chunker_version="v1")
    second = chunk_document(document, max_chars=100, overlap_chars=0, chunker_version="v2")

    assert spans(first) == spans(second)
    assert {c.chunker_version for c in first} == {"v1"}
    assert {c.chunker_version for c in second} == {"v2"}


# ------------------------------------------------------------
# invalid arguments
# ------------------------------------------------------------

@pytest.mark.parametrize("max_chars", [0, -5, True, 2.5, "10", None])
def test_invalid_max_chars_is_rejected(max_chars):
    document = make_document("some text")

    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(document, max_chars=max_chars, overlap_chars=0)


@pytest.mark.parametrize("overlap", [-1, True, 1.0, "2", None])
def test_invalid_overlap_type_or_sign_is_rejected(overlap):
    document = make_document("some text")

    with pytest.raises(ValueError, match="overlap_chars"):
        chunk_document(document, max_chars=10, overlap_chars=overlap)


@pytest.mark.parametrize("max_chars,overlap", [(10, 10), (10, 11), (10, 500), (1, 1)])
def test_overlap_equal_to_or_larger_than_max_is_rejected_not_adjusted(max_chars, overlap):
    document = make_document("some text")

    with pytest.raises(ValueError, match="smaller than max_chars"):
        chunk_document(document, max_chars=max_chars, overlap_chars=overlap)


def test_smallest_valid_settings_work():
    document = make_document("abc")

    chunks = chunk_document(document, max_chars=1, overlap_chars=0)

    assert texts(chunks) == ["a", "b", "c"]


@pytest.mark.parametrize("version", ["", "   ", None, 3])
def test_invalid_chunker_version_is_rejected(version):
    document = make_document("some text")

    with pytest.raises(ValueError, match="chunker_version"):
        chunk_document(document, chunker_version=version)


def test_invalid_settings_are_rejected_even_for_empty_content():
    empty = make_document("placeholder").model_copy(update={"content": ""})

    with pytest.raises(ValueError, match="max_chars"):
        chunk_document(empty, max_chars=0)


@pytest.mark.parametrize("not_a_document", [None, "text", {"content": "text"}])
def test_non_documents_are_rejected(not_a_document):
    with pytest.raises(TypeError, match="SourceDocument"):
        chunk_document(not_a_document)  # type: ignore[arg-type]


def test_settings_must_be_given_by_keyword():
    document = make_document("some text")

    with pytest.raises(TypeError):
        chunk_document(document, 10, 2)  # type: ignore[misc]


# ------------------------------------------------------------
# Unicode: offsets are code points, text is never altered
# ------------------------------------------------------------

def test_multibyte_characters_are_counted_as_code_points_not_bytes():
    emoji = "\U0001F600"  # 4 bytes in UTF-8, one code point
    document = make_document(emoji * 10)

    chunks = chunk_document(document, max_chars=4, overlap_chars=0)

    assert spans(chunks) == [(0, 4), (4, 8), (8, 10)]
    assert texts(chunks) == [emoji * 4, emoji * 4, emoji * 2]
    assert len(chunks[0].text.encode("utf-8")) == 16  # bytes would give a different cut
    assert all(document.contains_chunk(chunk) for chunk in chunks)


def test_cjk_text_splits_at_whitespace_with_code_point_offsets():
    words = "\u65e5\u672c\u8a9e \u30c6\u30b9\u30c8 \u65e5\u672c\u8a9e \u30c6\u30b9\u30c8"
    document = make_document(words)
    assert len(document.content) == 15

    chunks = chunk_document(document, max_chars=8, overlap_chars=0)

    assert spans(chunks) == [(0, 8), (8, 15)]
    assert texts(chunks) == [words[:8], words[8:]]
    assert len(chunks[0].text.encode("utf-8")) > len(chunks[0].text)
    assert all(document.contains_chunk(chunk) for chunk in chunks)


def test_hard_cut_does_not_separate_a_base_character_from_its_combining_mark():
    # "q" + U+0301 has no precomposed form, so NFC keeps both code points.
    content = "abcdq\u0301efgh"
    document = make_document(content)
    assert document.content == content and len(content) == 10

    chunks = chunk_document(document, max_chars=5, overlap_chars=0)

    # A raw cut at 5 would strand U+0301 at the start of the next chunk.
    assert texts(chunks) == ["abcd", "q\u0301efg", "h"]
    assert spans(chunks) == [(0, 4), (4, 9), (9, 10)]
    assert "".join(texts(chunks)) == content


def test_a_combining_mark_run_longer_than_the_window_is_cut_exactly_at_the_limit():
    marks = "\u0301" * 10
    document = make_document("q" + marks)

    chunks = chunk_document(document, max_chars=3, overlap_chars=0)

    assert spans(chunks) == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert "".join(texts(chunks)) == document.content
    assert all(len(chunk.text) <= 3 for chunk in chunks)
    assert all(document.contains_chunk(chunk) for chunk in chunks)


def test_text_inside_chunks_is_the_unaltered_normalized_content():
    # Input is NFD ("e" + U+0301); the schema normalizes it to NFC once and
    # the chunker must then reproduce that content byte for byte.
    document = make_document("cafe\u0301 au lait \u2013 caf\u00e9 \U0001F600 end")
    assert "\u00e9" in document.content and "e\u0301" not in document.content

    chunks = chunk_document(document, max_chars=9, overlap_chars=0)

    assert "".join(texts(chunks)) == document.content
    assert all(
        chunk.text == document.content[chunk.char_start:chunk.char_end] for chunk in chunks
    )


def test_unicode_tab_and_newline_content_is_preserved_exactly():
    document = make_document("col1\tcol2\n  \u2022 item one\n\t\u2022 item two")

    chunks = chunk_document(document, max_chars=12, overlap_chars=0)

    assert "".join(texts(chunks)) == document.content
    assert any("\t" in chunk.text for chunk in chunks)
