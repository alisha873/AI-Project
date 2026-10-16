import hashlib
import json

import pytest
from pydantic import ValidationError

from knowledge.schema import (
    Chunk,
    ExternalReference,
    ProvenanceRecord,
    SourceDocument,
    canonical_document_json,
    compute_document_sha256,
    content_sha256,
    make_chunk_id,
    make_doc_id,
    normalize_text,
    sha256_hex,
    validate_source_id,
)

GOOD_SHA = "a" * 64

CWE_89_TEXT = (
    "## Description\n"
    "The product constructs SQL commands using externally-influenced input.\n"
    "\n"
    "## Potential Mitigations\n"
    "Use parameterized queries.\n"
)


def make_provenance(**overrides) -> ProvenanceRecord:
    values = {
        "origin_kind": "bundled_fixture",
        "artifact_path": "tests/fixtures/knowledge/cwe_sample.jsonl",
        "artifact_sha256": GOOD_SHA,
        "adapter": "normalized_jsonl",
        "adapter_version": "0.1",
        "retrieved_at_utc": "2026-10-16T00:00:00Z",
        "ingested_at_utc": "2026-10-16T01:00:00Z",
    }
    values.update(overrides)
    return ProvenanceRecord(**values)


def make_document(**overrides) -> SourceDocument:
    values = {
        "source": "cwe",
        "source_id": "CWE-89",
        "source_version": "fixture-1",
        "doc_type": "weakness",
        "title": "SQL Injection",
        "content": CWE_89_TEXT,
        "source_url": "https://cwe.mitre.org/data/definitions/89.html",
        "provenance": make_provenance(),
    }
    values.update(overrides)
    return SourceDocument(**values)


# ------------------------------------------------------------
# normalize_text
# ------------------------------------------------------------

def test_normalization_is_independent_of_line_endings():
    unix = "line one\nline two\n\nline four"
    windows = unix.replace("\n", "\r\n")
    old_mac = unix.replace("\n", "\r")

    assert normalize_text(unix) == normalize_text(windows) == normalize_text(old_mac)
    assert "\r" not in normalize_text(windows)


def test_normalization_is_deterministic_across_repeated_calls():
    text = "Caf\u00e9\r\n  indented  \r\n\r\n\r\n\r\nend\x00"

    results = {normalize_text(text) for _ in range(5)}

    assert len(results) == 1


@pytest.mark.parametrize(
    "text",
    [
        "plain",
        "  padded  \n\n\n\nnext\t\n",
        "e\x00\u0301",  # control char between base letter and combining accent
        "\ufeffbom start",
        "a\u2028b\u2029c",
        "tab\tkept",
    ],
)
def test_normalization_is_idempotent(text):
    once = normalize_text(text)

    assert normalize_text(once) == once


def test_normalization_preserves_case_and_internal_spacing():
    text = "SELECT * FROM users WHERE id = '%s'  -- CWE-89"

    assert normalize_text(text) == text


def test_normalization_cleans_whitespace_bom_and_controls():
    text = "\ufeff  Title  \r\nbody   \x07text\r\n\r\n\r\n\r\nlast  \r\n"

    # Leading BOM dropped; first-line indentation kept; "Title  " keeps its
    # two trailing spaces (a Markdown hard break before another text line);
    # BEL deleted; blank-line run collapsed; last line's trailing spaces gone.
    assert normalize_text(text) == "  Title  \nbody   text\n\nlast"


@pytest.mark.parametrize(
    "separator",
    ["\x0b", "\x0c", "\x85", "\x1c", "\x1d", "\x1e", "\u2028", "\u2029"],
)
def test_line_break_controls_become_newlines_and_never_merge_words(separator):
    assert normalize_text(f"foo{separator}bar") == "foo\nbar"


def test_unit_separator_becomes_a_space():
    assert normalize_text("foo\x1fbar") == "foo bar"


def test_other_control_characters_are_deleted():
    # Intentionally lossy and documented: NUL, ESC and DEL are dropped.
    assert normalize_text("a\x00b\x1bc\x7fd") == "abcd"


def test_nel_and_form_feed_inside_a_document_do_not_change_word_count():
    text = "alpha\x0cbeta\x85gamma\x0bdelta"

    assert normalize_text(text).split() == ["alpha", "beta", "gamma", "delta"]


def test_first_line_indentation_is_preserved():
    assert normalize_text("    code()\n    more()\n") == "    code()\n    more()"
    assert normalize_text("\n\n  indented\n") == "  indented"


def test_inner_indentation_and_tabs_are_preserved():
    text = "list:\n  - item\n\t- tabbed\n      deep"

    assert normalize_text(text) == text


def test_markdown_hard_break_is_preserved_as_two_spaces():
    assert normalize_text("one  \ntwo") == "one  \ntwo"
    assert normalize_text("one      \ntwo") == "one  \ntwo"


def test_trailing_spaces_that_are_not_hard_breaks_are_removed():
    assert normalize_text("one \ntwo") == "one\ntwo"  # single space
    assert normalize_text("one\t \ntwo") == "one\ntwo"  # mixed whitespace
    assert normalize_text("one  \n\ntwo") == "one\n\ntwo"  # before a blank line
    assert normalize_text("one\ntwo  ") == "one\ntwo"  # last line of the text
    assert normalize_text("one\u00a0\ntwo") == "one\ntwo"  # NBSP


def test_blank_line_runs_collapse_to_one_blank_line_outside_code():
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"
    assert normalize_text("a\n\nb") == "a\n\nb"
    assert normalize_text("a\n \t \n\n b") == "a\n\n b"


def test_fenced_code_content_is_preserved_verbatim():
    text = "intro\n\n\n```sh\nA  \n\n\n\nB\t\n```  \n\n\n\nend"

    assert normalize_text(text) == "intro\n\n```sh\nA  \n\n\n\nB\t\n```\n\nend"


def test_tilde_fences_and_longer_closing_fences_are_supported():
    assert normalize_text("~~~\nx  \n\n\n~~~\n") == "~~~\nx  \n\n\n~~~"
    assert normalize_text("````\n```\ny  \n````") == "````\n```\ny  \n````"


@pytest.mark.parametrize("char", ["`", "~"])
def test_a_shorter_closing_marker_does_not_close_a_fence(char):
    opening, shorter = char * 4, char * 3
    # The 3-character line is code content (kept verbatim, trailing spaces
    # included), so the fence is still open for the blank-line run and the
    # trailing spaces after "B". Only the 4-character marker closes it.
    text = f"{opening}\nA\n{shorter}  \n\n\n\nB  \n{opening}  \n\n\n\nend"

    expected = f"{opening}\nA\n{shorter}  \n\n\n\nB  \n{opening}\n\nend"
    assert normalize_text(text) == expected
    assert normalize_text(expected) == expected


@pytest.mark.parametrize("char", ["`", "~"])
def test_an_equal_or_longer_closing_marker_closes_a_fence(char):
    for closing in (char * 4, char * 6):
        text = f"{char * 4}\nA  \n{closing}  \n\n\n\nB  \nC"
        # Closed at the marker: the blank-line run after it collapses and the
        # hard break after "B" is applied again.
        assert normalize_text(text) == f"{char * 4}\nA  \n{closing}\n\nB  \nC"


@pytest.mark.parametrize(
    "opening,other",
    [("```", "~~~"), ("~~~", "```"), ("````", "~~~~"), ("~~~~", "````")],
)
def test_a_closing_marker_of_the_other_character_does_not_close_a_fence(opening, other):
    # `other` is at least as long as `opening` but uses the other character,
    # so it is code content and the fence stays open until `opening` appears.
    text = f"{opening}\nA\n{other}  \n\n\n\nB  \n{opening}  \n\n\n\nend"

    expected = f"{opening}\nA\n{other}  \n\n\n\nB  \n{opening}\n\nend"
    assert normalize_text(text) == expected
    assert normalize_text(expected) == expected


def test_a_closing_marker_of_the_other_character_leaves_the_fence_unclosed():
    # Never closed: everything up to the end of the text stays verbatim
    # (apart from the final trailing whitespace of the whole text).
    text = "```\nA  \n~~~\n\n\n\nB  \n~~~~~~\n\n"

    assert normalize_text(text) == "```\nA  \n~~~\n\n\n\nB  \n~~~~~~"


# Fence detection is documented as simplified (see `normalize_text` and the
# comment above `_FENCE_OPEN_RE`): a marker may be indented by ANY number of
# spaces or tabs. This is NOT CommonMark, which treats 4+ spaces of indent as
# an indented code block rather than a fence. These tests pin the documented
# behaviour; they do not claim CommonMark compliance.
@pytest.mark.parametrize("indent", ["  ", "    ", "        ", "\t"])
def test_indented_fences_are_detected_and_their_indentation_is_preserved(indent):
    text = (
        f"intro\n\n\n{indent}```sh\n{indent}A  \n\n\n\n{indent}B\t\n"
        f"{indent}```  \n\n\n\nend"
    )

    expected = f"intro\n\n{indent}```sh\n{indent}A  \n\n\n\n{indent}B\t\n{indent}```\n\nend"
    assert normalize_text(text) == expected
    assert normalize_text(expected) == expected


@pytest.mark.parametrize(
    "open_indent,close_indent",
    [("    ", ""), ("", "    "), ("  ", "\t"), ("\t", "        ")],
)
def test_opening_and_closing_fence_indentation_do_not_have_to_match(
    open_indent, close_indent
):
    text = f"{open_indent}~~~\nA  \n\n\n\n{close_indent}~~~  \n\n\n\nend"

    expected = f"{open_indent}~~~\nA  \n\n\n\n{close_indent}~~~\n\nend"
    assert normalize_text(text) == expected


def test_an_indented_marker_with_other_text_before_it_is_not_a_fence():
    # Only whitespace may precede the marker.
    assert normalize_text("> ```\nA  \n\n\n\nB") == "> ```\nA\n\nB"
    assert normalize_text("- ```\nA  \n\n\n\nB") == "- ```\nA\n\nB"


def test_inline_triple_backticks_do_not_open_a_fence():
    assert normalize_text("```x```  \n\n\n\ny") == "```x```\n\ny"


def test_unclosed_fence_is_preserved_to_the_end_except_final_trailing_whitespace():
    assert normalize_text("```\na  \n\n\n\nb  \n\n") == "```\na  \n\n\n\nb"


def test_mid_text_bom_is_preserved_but_leading_bom_is_removed():
    assert normalize_text("a\ufeffb") == "a\ufeffb"
    assert normalize_text("\ufeffTitle") == "Title"
    assert normalize_text("\ufeff\ufeffTitle") == "Title"
    assert normalize_text("\n\ufeff\nbody") == "body"
    assert normalize_text("line one\n\ufeffline two") == "line one\n\ufeffline two"


def test_leading_bom_before_a_fence_does_not_change_fence_handling():
    assert normalize_text("\ufeff```\nx  \n```") == "```\nx  \n```"


def test_zero_width_and_bidi_format_characters_are_kept():
    text = "a\u200bb\u202eevil"

    assert normalize_text(text) == text


def test_nfc_unification_is_applied_and_documented_as_lossy():
    assert normalize_text("\u212b") == "\u00c5"  # ANGSTROM SIGN -> A with ring
    assert normalize_text("e\x00\u0301") == "\u00e9"  # control removed, then composed


def test_normalization_is_idempotent_on_pseudo_random_inputs():
    import random

    rng = random.Random(20261016)
    pieces = [
        "a", "b", " ", "  ", "\n", "\n\n", "\r", "\t", "\x00", "\x0b", "\x0c",
        "\x85", "\x1f", "\u00a0", "\u2028", "\ufeff", "\u0301", "e",
        "```", "~~~", "``", "`",
    ]

    for _ in range(3000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 16)))
        once = normalize_text(text)

        assert normalize_text(once) == once, repr(text)


def test_normalization_applies_unicode_nfc():
    decomposed = "cafe\u0301"
    composed = "caf\u00e9"

    assert normalize_text(decomposed) == composed


def test_normalization_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(b"bytes")  # type: ignore[arg-type]


# ------------------------------------------------------------
# hashing
# ------------------------------------------------------------

def test_sha256_hex_matches_known_vector():
    assert sha256_hex("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_content_sha256_is_deterministic_and_well_formed():
    first = content_sha256(CWE_89_TEXT)
    second = content_sha256(CWE_89_TEXT)

    assert first == second
    assert len(first) == 64
    assert first == first.lower()


def test_content_sha256_does_not_depend_on_line_endings():
    assert content_sha256(CWE_89_TEXT) == content_sha256(
        CWE_89_TEXT.replace("\n", "\r\n")
    )


def test_content_sha256_changes_when_content_changes():
    assert content_sha256("alpha") != content_sha256("beta")


def test_content_sha256_matches_independent_sha256_of_normalized_text():
    # "abc" is unchanged by normalization, so its content hash is the
    # published SHA-256 test vector.
    assert content_sha256("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert content_sha256("a\r\n\r\n\r\nb  \r\n") == hashlib.sha256(
        b"a\n\nb"
    ).hexdigest()


def test_content_sha256_is_the_same_for_nfc_and_nfd_input():
    assert content_sha256("cafe\u0301") == content_sha256("caf\u00e9")


def test_content_sha256_reflects_meaningful_whitespace_differences():
    assert content_sha256("    indented") != content_sha256("indented")
    assert content_sha256("one  \ntwo") != content_sha256("one\ntwo")


def test_lone_surrogates_are_rejected_by_documents_with_a_validation_error():
    with pytest.raises(ValidationError):
        make_document(content="broken \ud800 text")


# ------------------------------------------------------------
# identifiers
# ------------------------------------------------------------

def test_make_doc_id_is_deterministic_and_preserves_source_id():
    assert make_doc_id("cwe", "CWE-89") == "cwe:CWE-89"
    assert make_doc_id("cwe", "CWE-89") == make_doc_id("cwe", "CWE-89")
    assert make_doc_id("owasp", "A03:2021") == "owasp:A03:2021"


def test_source_id_is_never_rewritten():
    assert validate_source_id("cwe-89") == "cwe-89"
    assert make_doc_id("cwe", "cwe-89") != make_doc_id("cwe", "CWE-89")


def test_make_doc_id_rejects_unknown_source():
    with pytest.raises(ValueError):
        make_doc_id("wikipedia", "SQL_injection")


@pytest.mark.parametrize(
    "bad_id",
    ["", " CWE-89", "CWE-89 ", "CWE 89", "CWE-89\n", "CWE#89", "CWE\u200b-89", "X" * 201],
)
def test_invalid_source_ids_are_rejected_not_repaired(bad_id):
    with pytest.raises(ValueError):
        validate_source_id(bad_id)


def test_make_chunk_id_format():
    assert make_chunk_id("cwe:CWE-89", 0) == "cwe:CWE-89#0000"
    assert make_chunk_id("cwe:CWE-89", 12) == "cwe:CWE-89#0012"


@pytest.mark.parametrize("ordinal", [-1, True, 1.5, "1"])
def test_make_chunk_id_rejects_invalid_ordinals(ordinal):
    with pytest.raises(ValueError):
        make_chunk_id("cwe:CWE-89", ordinal)


# ------------------------------------------------------------
# ProvenanceRecord
# ------------------------------------------------------------

def test_provenance_accepts_valid_values():
    record = make_provenance()

    assert record.origin_kind == "bundled_fixture"
    assert record.artifact_sha256 == GOOD_SHA


@pytest.mark.parametrize(
    "overrides",
    [
        {"origin_kind": "scraped"},
        {"artifact_path": "/etc/passwd"},
        {"artifact_path": "C:/data/cwe.csv"},
        {"artifact_path": "..\\secret.csv"},
        {"artifact_path": "data/../../secret.csv"},
        {"artifact_path": "data\\cwe.csv"},
        {"artifact_path": ""},
        {"artifact_sha256": "A" * 64},
        {"artifact_sha256": "abc"},
        {"artifact_sha256": "a" * 64 + "\n"},
        {"adapter": ""},
        {"adapter_version": "  "},
        {"retrieved_at_utc": "2026-10-16"},
        {"retrieved_at_utc": "2026-10-16T00:00:00+05:30"},
        {"ingested_at_utc": "not a time"},
        {"ingested_at_utc": "2026-13-45T00:00:00Z"},
    ],
)
def test_provenance_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        make_provenance(**overrides)


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-1-5T01:02:03Z",  # unpadded month and day
        "2026-10-16T1:2:3Z",  # unpadded time fields
        "2026-10-6T01:02:03Z",  # unpadded day only
        "2026-10-16T01:02:03Z\n",  # trailing newline
        " 2026-10-16T01:02:03Z",  # leading space
        "2026-10-16t01:02:03Z",  # lowercase separator
        "2026-10-16T01:02:03z",  # lowercase zone
        "2026-10-16T01:02:03",  # no zone
        "2026-10-16T01:02:03.123Z",  # fractional seconds
        "\u0662026-10-16T01:02:03Z",  # non-ASCII digit
        "2026-02-30T01:02:03Z",  # well-shaped but not a real date
    ],
)
@pytest.mark.parametrize("field", ["retrieved_at_utc", "ingested_at_utc"])
def test_provenance_rejects_non_canonical_timestamps(field, timestamp):
    with pytest.raises(ValidationError, match="YYYY-MM-DDTHH:MM:SSZ"):
        make_provenance(**{field: timestamp})


def test_provenance_accepts_exact_zero_padded_timestamp():
    record = make_provenance(
        retrieved_at_utc="2026-01-05T01:02:03Z",
        ingested_at_utc="2026-12-31T23:59:59Z",
    )

    assert record.retrieved_at_utc == "2026-01-05T01:02:03Z"
    assert record.ingested_at_utc == "2026-12-31T23:59:59Z"


def test_provenance_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        make_provenance(extra_field="x")


# ------------------------------------------------------------
# ExternalReference
# ------------------------------------------------------------

def test_external_reference_resolves_target_doc_id():
    reference = ExternalReference(
        relation="related_weakness",
        target_source="cwe",
        target_source_id="CWE-89",
    )

    assert reference.target_doc_id == "cwe:CWE-89"


@pytest.mark.parametrize(
    "overrides",
    [
        {"relation": ""},
        {"relation": "  "},
        {"relation": " ChildOf"},
        {"relation": "ChildOf "},
        {"relation": "Child\nOf"},
        {"relation": "Child\u200bOf"},
        {"relation": "R" * 201},
        {"target_source": ""},
        {"target_source": "WASC "},
        {"target_source_id": ""},
        {"target_source_id": "CWE-89\n"},
        {"target_source": "cwe", "target_source_id": "CWE 89"},  # local: strict
        {"target_source": "owasp", "target_source_id": "A03#2021"},  # local: strict
        {"url": "ftp://example.org/ref"},
        {"url": "javascript:alert(1)"},
        {"url": "https://"},
        {"url": ""},
    ],
)
def test_external_reference_rejects_invalid_values(overrides):
    values = {
        "relation": "ChildOf",
        "target_source": "cwe",
        "target_source_id": "CWE-89",
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        ExternalReference(**values)


@pytest.mark.parametrize("label", ["ChildOf", "CanPrecede", "PeerOf", "CanAlsoBe", "Requires"])
def test_native_cwe_relationship_labels_are_preserved_exactly(label):
    reference = ExternalReference(
        relation=label, target_source="cwe", target_source_id="CWE-74"
    )

    assert reference.relation == label
    assert reference.target_doc_id == "cwe:CWE-74"


def test_snake_case_labels_are_also_accepted_unchanged():
    reference = ExternalReference(
        relation="related_weakness", target_source="cwe", target_source_id="CWE-89"
    )

    assert reference.relation == "related_weakness"


@pytest.mark.parametrize(
    "target_source,target_source_id",
    [
        ("WASC", "WASC-19"),
        ("OWASP Attacks", "SQL Injection"),  # spaces allowed for external labels
        ("CERT C", "STR31-C"),
        ("OWASP Top Ten 2021", "A03:2021"),
        ("ATTACK", "T1190"),  # source's own spelling; not a local doc id
        ("REF", "REF-62"),
    ],
)
def test_external_taxonomies_are_representable_but_not_resolved_to_local_docs(
    target_source, target_source_id
):
    reference = ExternalReference(
        relation="TaxonomyMapping",
        target_source=target_source,
        target_source_id=target_source_id,
    )

    assert reference.target_source == target_source
    assert reference.target_source_id == target_source_id
    assert reference.target_doc_id is None


def test_external_reference_accepts_optional_http_url_unchanged():
    url = "https://example.org/Papers/REF-62.html?Q=1"

    reference = ExternalReference(
        relation="Citation",
        target_source="REF",
        target_source_id="REF-62",
        url=url,
    )

    assert reference.url == url
    assert ExternalReference(
        relation="Citation", target_source="REF", target_source_id="REF-62"
    ).url is None


def test_external_reference_is_immutable_and_round_trips_through_json():
    reference = ExternalReference(
        relation="ChildOf",
        target_source="cwe",
        target_source_id="CWE-74",
        url="https://cwe.mitre.org/data/definitions/74.html",
    )

    with pytest.raises(ValidationError):
        reference.relation = "PeerOf"
    assert ExternalReference.model_validate_json(reference.model_dump_json()) == reference


# ------------------------------------------------------------
# SourceDocument
# ------------------------------------------------------------

def test_valid_document_derives_id_and_hash_and_preserves_source_id():
    document = make_document()

    assert document.schema_version == "1.0"
    assert document.doc_id == "cwe:CWE-89"
    assert document.source_id == "CWE-89"
    assert document.content_sha256 == content_sha256(CWE_89_TEXT)
    assert document.content_origin == "source_text"
    assert document.source_url == "https://cwe.mitre.org/data/definitions/89.html"


def test_document_content_is_normalized_on_construction():
    document = make_document(content="  Heading\r\n\r\n\r\n\r\nBody  \r\n")

    # first-line indentation is preserved; blank-line run collapsed;
    # final trailing spaces removed; CRLF converted
    assert document.content == "  Heading\n\nBody"
    assert document.content_sha256 == content_sha256("  Heading\n\nBody")


def test_windows_and_unix_line_endings_produce_identical_documents():
    unix_doc = make_document(content=CWE_89_TEXT)
    windows_doc = make_document(content=CWE_89_TEXT.replace("\n", "\r\n"))

    assert unix_doc == windows_doc
    assert unix_doc.content_sha256 == windows_doc.content_sha256


def test_document_accepts_matching_explicit_derived_fields():
    document = make_document(
        doc_id="cwe:CWE-89",
        content_sha256=content_sha256(CWE_89_TEXT),
    )

    assert document.doc_id == "cwe:CWE-89"


def test_document_rejects_mismatched_doc_id():
    with pytest.raises(ValidationError):
        make_document(doc_id="cwe:CWE-79")


def test_document_rejects_mismatched_content_hash():
    with pytest.raises(ValidationError):
        make_document(content_sha256="b" * 64)


def test_document_rejects_invalid_doc_type():
    with pytest.raises(ValidationError):
        make_document(doc_type="blog_post")


def test_document_rejects_invalid_source():
    with pytest.raises(ValidationError):
        make_document(source="wikipedia")


@pytest.mark.parametrize("content", ["", "   ", "\n\n\t\r\n", "\x00\x01\x02", "\ufeff"])
def test_document_rejects_empty_content(content):
    with pytest.raises(ValidationError):
        make_document(content=content)


@pytest.mark.parametrize("title", ["", "   ", "\n"])
def test_document_rejects_empty_title(title):
    with pytest.raises(ValidationError):
        make_document(title=title)


def test_document_title_whitespace_is_collapsed():
    assert make_document(title="  SQL \r\n Injection ").title == "SQL Injection"


def test_document_rejects_empty_source_version():
    with pytest.raises(ValidationError):
        make_document(source_version="  ")


def test_document_rejects_source_id_with_whitespace_instead_of_trimming():
    with pytest.raises(ValidationError):
        make_document(source_id=" CWE-89")


def test_document_rejects_missing_required_fields():
    with pytest.raises(ValidationError):
        SourceDocument(
            source="cwe",
            source_id="CWE-89",
            doc_type="weakness",
            title="SQL Injection",
            content=CWE_89_TEXT,
            provenance=make_provenance(),
        )  # source_version missing


def test_document_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        make_document(summary="LLM generated summary")


def test_document_cannot_claim_model_generated_origin():
    with pytest.raises(ValidationError):
        make_document(content_origin="model_generated")


@pytest.mark.parametrize(
    "url",
    [
        "https://cwe.mitre.org/data/definitions/89.html",
        "http://example.org/doc",
        "https://Example.ORG:8443/Path?Q=1#frag",
    ],
)
def test_document_accepts_http_and_https_urls_unchanged(url):
    assert make_document(source_url=url).source_url == url


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/file",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "example.org/no-scheme",
        "https://",
        "https:///path-only",
        "https://exa mple.org",
        "",
    ],
)
def test_document_rejects_invalid_urls(url):
    with pytest.raises(ValidationError):
        make_document(source_url=url)


def test_document_url_is_optional():
    assert make_document(source_url=None).source_url is None


@pytest.mark.parametrize("value", ["2026-10-16", "2026-10-16T08:30:00Z", "2026-10-16T08:30:00+00:00"])
def test_document_accepts_iso_source_modified(value):
    assert make_document(source_modified=value).source_modified == value


def test_document_rejects_invalid_source_modified():
    with pytest.raises(ValidationError):
        make_document(source_modified="last Tuesday")


def make_reference(
    relation="RelatedAttackPattern",
    target_source="capec",
    target_source_id="CAPEC-66",
    url=None,
) -> ExternalReference:
    return ExternalReference(
        relation=relation,
        target_source=target_source,
        target_source_id=target_source_id,
        url=url,
    )


def test_document_carries_source_declared_references():
    document = make_document(references=[make_reference()])

    assert document.references[0].target_doc_id == "capec:CAPEC-66"
    assert document.references[0].relation == "RelatedAttackPattern"


def test_document_references_default_to_an_empty_tuple():
    assert make_document().references == ()


def test_document_references_are_an_immutable_tuple_even_when_given_a_list():
    reference = make_reference()
    source_list = [reference]

    document = make_document(references=source_list)

    assert isinstance(document.references, tuple)
    source_list.append(make_reference(target_source_id="CAPEC-7"))
    assert document.references == (reference,)  # later list changes do not leak in


def test_document_references_cannot_be_mutated():
    document = make_document(
        references=[make_reference(), make_reference(target_source_id="CAPEC-7")]
    )

    with pytest.raises(AttributeError):
        document.references.append(make_reference())  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        document.references[0] = make_reference()  # type: ignore[index]
    with pytest.raises(ValidationError):
        document.references = ()
    with pytest.raises(ValidationError):
        document.references[0].relation = "Changed"


def test_document_is_immutable():
    document = make_document()

    with pytest.raises(ValidationError):
        document.title = "Changed"


def test_document_with_references_is_hashable_and_usable_as_a_dict_key():
    first = make_document(references=[make_reference()])
    second = make_document(references=[make_reference()])

    assert hash(first) == hash(second)
    assert {first: "kept"}[second] == "kept"


def test_document_json_round_trip_is_lossless():
    document = make_document(
        source_modified="2026-10-01",
        license="Fixture licence text",
        references=[
            make_reference(),
            make_reference(
                relation="Citation",
                target_source="REF",
                target_source_id="REF-62",
                url="https://example.org/ref-62",
            ),
        ],
    )

    payload = json.loads(document.model_dump_json())
    restored = SourceDocument.model_validate_json(document.model_dump_json())

    assert isinstance(payload["references"], list)  # tuples serialize as JSON arrays
    assert isinstance(restored.references, tuple)
    assert restored == document
    assert restored.document_sha256 == document.document_sha256


# ------------------------------------------------------------
# Document-level hash
# ------------------------------------------------------------

def test_document_sha256_is_derived_deterministic_and_well_formed():
    first = make_document()
    second = make_document()

    assert first.document_sha256 == second.document_sha256
    assert len(first.document_sha256) == 64
    assert first.document_sha256 != first.content_sha256


def test_content_sha256_is_still_the_hash_of_the_normalized_content_only():
    document = make_document()

    assert document.content_sha256 == hashlib.sha256(
        document.content.encode("utf-8")
    ).hexdigest()


def test_document_sha256_matches_the_documented_canonical_serialization():
    document = make_document(
        source_modified="2026-10-01",
        license="Fixture licence",
        references=[
            make_reference(relation="PeerOf", target_source="cwe", target_source_id="CWE-564"),
            make_reference(relation="ChildOf", target_source="cwe", target_source_id="CWE-74"),
        ],
    )
    expected_json = (
        '{"content_sha256":"%s",'
        '"doc_type":"weakness",'
        '"hash_version":1,'
        '"license":"Fixture licence",'
        '"references":['
        '{"relation":"ChildOf","target_source":"cwe","target_source_id":"CWE-74","url":null},'
        '{"relation":"PeerOf","target_source":"cwe","target_source_id":"CWE-564","url":null}],'
        '"schema_version":"1.0",'
        '"source":"cwe",'
        '"source_id":"CWE-89",'
        '"source_modified":"2026-10-01",'
        '"source_url":"https://cwe.mitre.org/data/definitions/89.html",'
        '"source_version":"fixture-1",'
        '"title":"SQL Injection"}'
    ) % document.content_sha256

    assert document.document_sha256 == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert canonical_document_json(
        schema_version=document.schema_version,
        source=document.source,
        source_id=document.source_id,
        source_version=document.source_version,
        doc_type=document.doc_type,
        title=document.title,
        content_sha256=document.content_sha256,
        source_url=document.source_url,
        license=document.license,
        source_modified=document.source_modified,
        references=document.references,
    ) == expected_json


def test_canonical_serialization_is_ascii_only():
    document = make_document(title="Injection \u2013 caf\u00e9", license="\u00a9 MITRE")
    canonical = canonical_document_json(
        schema_version=document.schema_version,
        source=document.source,
        source_id=document.source_id,
        source_version=document.source_version,
        doc_type=document.doc_type,
        title=document.title,
        content_sha256=document.content_sha256,
        source_url=document.source_url,
        license=document.license,
        source_modified=document.source_modified,
        references=document.references,
    )

    assert canonical.isascii()
    assert "\\u2013" in canonical


def test_reference_order_does_not_change_the_document_hash_but_is_preserved():
    first = make_reference(relation="ChildOf", target_source="cwe", target_source_id="CWE-74")
    second = make_reference(relation="PeerOf", target_source="cwe", target_source_id="CWE-564")
    third = make_reference(
        relation="Citation", target_source="REF", target_source_id="REF-1", url="https://example.org/a"
    )

    forward = make_document(references=[first, second, third])
    backward = make_document(references=[third, second, first])

    assert forward.document_sha256 == backward.document_sha256
    assert forward.references == (first, second, third)
    assert backward.references == (third, second, first)
    assert forward != backward  # the models themselves still record adapter order


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": "Structured Query Language Injection"},
        {"source_version": "fixture-2"},
        {"source_url": "https://cwe.mitre.org/data/definitions/89.html#x"},
        {"source_url": None},
        {"license": "Another licence"},
        {"source_modified": "2026-10-02"},
        {"doc_type": "guideline"},
        {"source_id": "CWE-90"},
        {"source": "capec", "source_id": "CAPEC-66"},
        {"content": CWE_89_TEXT + "\nExtra sentence."},
        {"references": [make_reference()]},
    ],
)
def test_metadata_or_content_change_changes_document_sha256(overrides):
    baseline = make_document(license="Baseline licence", source_modified="2026-10-01")
    values = {"license": "Baseline licence", "source_modified": "2026-10-01"}
    values.update(overrides)

    changed = make_document(**values)

    assert changed.document_sha256 != baseline.document_sha256


def _canonical(document: SourceDocument) -> str:
    return canonical_document_json(
        schema_version=document.schema_version,
        source=document.source,
        source_id=document.source_id,
        source_version=document.source_version,
        doc_type=document.doc_type,
        title=document.title,
        content_sha256=document.content_sha256,
        source_url=document.source_url,
        license=document.license,
        source_modified=document.source_modified,
        references=document.references,
    )


URL_A = "https://example.org/ref/a"
URL_B = "https://example.org/ref/b"


def test_references_that_differ_only_by_url_produce_different_document_hashes():
    # relation, target_source and target_source_id are identical in every case.
    no_url = make_document(references=[make_reference(url=None)])
    url_a = make_document(references=[make_reference(url=URL_A)])
    url_b = make_document(references=[make_reference(url=URL_B)])
    url_a_again = make_document(references=[make_reference(url=URL_A)])

    hashes = {no_url.document_sha256, url_a.document_sha256, url_b.document_sha256}
    assert len(hashes) == 3
    assert url_a.document_sha256 == url_a_again.document_sha256  # but deterministic


def test_the_url_is_part_of_each_canonical_reference_entry():
    document = make_document(references=[make_reference(url=URL_A)])

    assert (
        '"references":[{"relation":"RelatedAttackPattern","target_source":"capec",'
        '"target_source_id":"CAPEC-66","url":"%s"}]' % URL_A
    ) in _canonical(document)


@pytest.mark.parametrize(
    "urls",
    [
        (URL_A, URL_B),
        (None, URL_A),
        (None, URL_A, URL_B),
        (URL_B, None, URL_A),
    ],
)
def test_reference_url_is_the_tie_breaker_so_listing_order_does_not_change_the_hash(urls):
    references = [make_reference(url=url) for url in urls]
    forward = make_document(references=references)
    backward = make_document(references=list(reversed(references)))

    assert forward.document_sha256 == backward.document_sha256
    assert forward.references == tuple(references)  # adapter order is still kept
    # Canonical order: a missing URL sorts first, then URLs by code point.
    canonical_urls = [
        entry["url"] for entry in json.loads(_canonical(forward))["references"]
    ]
    assert canonical_urls == sorted(urls, key=lambda url: url or "")
    assert _canonical(forward) == _canonical(backward)


def test_document_hash_distinguishes_which_reference_carries_which_url():
    other_target = {"target_source_id": "CAPEC-7"}
    first = make_document(
        references=[make_reference(url=URL_A), make_reference(url=URL_B, **other_target)]
    )
    swapped = make_document(
        references=[make_reference(url=URL_B), make_reference(url=URL_A, **other_target)]
    )

    assert first.document_sha256 != swapped.document_sha256


def test_changing_a_reference_field_changes_document_sha256():
    base = make_document(references=[make_reference()])

    for changed_reference in (
        make_reference(relation="ChildOf"),
        make_reference(target_source="cwe", target_source_id="CWE-66"),
        make_reference(target_source_id="CAPEC-67"),
        make_reference(url="https://example.org/x"),
    ):
        assert make_document(references=[changed_reference]).document_sha256 != base.document_sha256


def test_metadata_change_leaves_content_sha256_unchanged_but_changes_document_sha256():
    base = make_document()
    retitled = make_document(title="A Different Title")

    assert retitled.content_sha256 == base.content_sha256
    assert retitled.document_sha256 != base.document_sha256


def test_provenance_changes_do_not_change_document_sha256():
    base = make_document()
    later = make_document(
        provenance=make_provenance(
            retrieved_at_utc="2027-01-01T00:00:00Z",
            ingested_at_utc="2027-01-02T03:04:05Z",
        )
    )
    other_artifact = make_document(
        provenance=make_provenance(artifact_sha256="b" * 64, adapter_version="9.9")
    )

    assert later.document_sha256 == base.document_sha256
    assert other_artifact.document_sha256 == base.document_sha256
    assert later != base  # the models differ (provenance), the document hash does not


def test_document_sha256_is_independent_of_line_endings_and_title_whitespace():
    unix = make_document(content=CWE_89_TEXT, title="SQL Injection")
    windows = make_document(
        content=CWE_89_TEXT.replace("\n", "\r\n"), title="  SQL \r\n Injection "
    )

    assert unix.document_sha256 == windows.document_sha256


def test_document_rejects_mismatched_or_malformed_document_sha256():
    with pytest.raises(ValidationError, match="document_sha256 does not match"):
        make_document(document_sha256="d" * 64)
    with pytest.raises(ValidationError, match="64 lowercase hexadecimal"):
        make_document(document_sha256="D" * 64)
    with pytest.raises(ValidationError, match="64 lowercase hexadecimal"):
        make_document(document_sha256="d" * 64 + "\n")


def test_document_accepts_matching_explicit_document_sha256():
    expected = make_document().document_sha256

    assert make_document(document_sha256=expected).document_sha256 == expected


def test_compute_document_sha256_matches_the_derived_field():
    document = make_document(references=[make_reference()])

    assert document.document_sha256 == compute_document_sha256(
        schema_version=document.schema_version,
        source=document.source,
        source_id=document.source_id,
        source_version=document.source_version,
        doc_type=document.doc_type,
        title=document.title,
        content_sha256=document.content_sha256,
        source_url=document.source_url,
        license=document.license,
        source_modified=document.source_modified,
        references=document.references,
    )


def test_document_without_derivable_hash_reports_the_real_field_error():
    with pytest.raises(ValidationError, match="doc_type"):
        make_document(doc_type="blog_post")


# ------------------------------------------------------------
# Chunk
# ------------------------------------------------------------

def make_chunk(document: SourceDocument, start: int, end: int, **overrides) -> Chunk:
    values = {
        "doc_id": document.doc_id,
        "ordinal": 0,
        "section": "Description",
        "text": document.content[start:end],
        "char_start": start,
        "char_end": end,
        "doc_content_sha256": document.content_sha256,
        "chunker_version": "test-0",
        "source": document.source,
        "source_id": document.source_id,
        "doc_type": document.doc_type,
        "title": document.title,
        "source_url": document.source_url,
    }
    values.update(overrides)
    return Chunk(**values)


def test_valid_chunk_derives_id_and_hash_and_copies_provenance():
    document = make_document()

    chunk = make_chunk(document, 0, 14, ordinal=3)

    assert chunk.chunk_id == "cwe:CWE-89#0003"
    assert chunk.chunk_sha256 == sha256_hex(chunk.text)
    assert chunk.source == "cwe"
    assert chunk.source_id == "CWE-89"
    assert chunk.doc_id == document.doc_id
    assert chunk.source_url == document.source_url
    assert chunk.doc_content_sha256 == document.content_sha256


def test_chunk_offsets_support_the_document_slice_invariant():
    document = make_document()
    start = document.content.index("## Potential Mitigations")
    end = len(document.content)

    chunk = make_chunk(document, start, end, ordinal=1, section="Potential Mitigations")

    assert document.content[chunk.char_start:chunk.char_end] == chunk.text
    assert document.contains_chunk(chunk)


def test_chunk_text_is_stored_exactly_without_stripping():
    document = make_document()
    start = document.content.index("\n\n")  # slice begins with newlines
    end = start + 12

    chunk = make_chunk(document, start, end)

    assert chunk.text == document.content[start:end]
    assert chunk.text.startswith("\n")
    assert document.contains_chunk(chunk)


def test_contains_chunk_detects_wrong_offsets():
    document = make_document()
    good = make_chunk(document, 0, 14)
    shifted = make_chunk(
        document,
        1,
        15,
        text=good.text,  # same text, wrong position
    )

    assert document.contains_chunk(good)
    assert not document.contains_chunk(shifted)


def test_contains_chunk_rejects_chunk_past_end_of_document():
    document = make_document()
    length = len(document.content)
    chunk = make_chunk(document, length - 3, length + 2, text="abcde")

    assert not document.contains_chunk(chunk)


def test_contains_chunk_accepts_a_chunk_covering_the_whole_document():
    document = make_document()

    assert document.contains_chunk(make_chunk(document, 0, len(document.content)))


def test_contains_chunk_rejects_chunk_cut_from_different_document_content():
    document = make_document()
    stale = make_chunk(document, 0, 14, doc_content_sha256="e" * 64)

    # text and offsets still match, but the chunk was cut from other content
    assert document.content[0:14] == stale.text
    assert not document.contains_chunk(stale)


def test_contains_chunk_detects_stale_chunk_after_content_update():
    old_document = make_document(content="Original text of the entry.")
    new_document = make_document(content="Original text of the entry. Plus an update.")
    chunk = make_chunk(old_document, 0, 13)  # "Original text"

    assert old_document.contains_chunk(chunk)
    assert new_document.content[0:13] == chunk.text  # same text, still a prefix
    assert not new_document.contains_chunk(chunk)  # but it records the old content hash


@pytest.mark.parametrize(
    "overrides",
    [
        {"doc_type": "guideline"},
        {"title": "A Different Title"},
        {"source_url": "https://example.org/other"},
        {"source_url": None},
    ],
)
def test_contains_chunk_rejects_mismatched_copied_metadata(overrides):
    document = make_document()
    chunk = make_chunk(document, 0, 14, **overrides)  # internally valid chunk

    assert not document.contains_chunk(chunk)


def test_contains_chunk_rejects_chunk_with_wrong_source_identity():
    document = make_document()
    other_source = make_document(
        source="capec", source_id="CAPEC-66", doc_type="attack_pattern", title="SQL Injection"
    )
    chunk = make_chunk(other_source, 0, 14)

    assert not document.contains_chunk(chunk)


def test_contains_chunk_rejects_invalid_offsets_on_unvalidated_chunks():
    document = make_document()
    good = make_chunk(document, 0, 14)
    fields = good.model_dump()

    negative_start = Chunk.model_construct(**{**fields, "char_start": -4, "char_end": 10})
    empty_span = Chunk.model_construct(**{**fields, "char_start": 5, "char_end": 5})
    past_end = Chunk.model_construct(
        **{**fields, "char_start": len(document.content) - 2, "char_end": len(document.content) + 12}
    )
    wrong_hash = Chunk.model_construct(**{**fields, "chunk_sha256": "f" * 64})

    assert document.contains_chunk(good)
    for broken in (negative_start, empty_span, past_end, wrong_hash):
        assert not document.contains_chunk(broken)


def test_contains_chunk_rejects_objects_that_are_not_chunks():
    document = make_document()

    assert not document.contains_chunk(None)  # type: ignore[arg-type]
    assert not document.contains_chunk(make_chunk(document, 0, 14).model_dump())  # type: ignore[arg-type]


def test_chunk_offsets_count_unicode_code_points_not_bytes():
    document = make_document(content="a\U0001F600b is fine")
    chunk = make_chunk(document, 1, 3)  # the emoji and "b"

    assert chunk.text == "\U0001F600b"
    assert document.contains_chunk(chunk)
    assert len(chunk.text.encode("utf-8")) != chunk.char_end - chunk.char_start


def test_contains_chunk_rejects_chunk_from_another_document():
    document = make_document()
    other = make_document(source_id="CWE-79", title="Cross-site Scripting")
    chunk = make_chunk(other, 0, 14)

    assert not document.contains_chunk(chunk)


def test_chunk_rejects_length_mismatch_between_offsets_and_text():
    document = make_document()

    with pytest.raises(ValidationError):
        make_chunk(document, 0, 20, text=document.content[0:14])


@pytest.mark.parametrize(
    "start,end",
    [(10, 10), (10, 5), (-1, 5)],
)
def test_chunk_rejects_invalid_offsets(start, end):
    document = make_document()

    with pytest.raises(ValidationError):
        make_chunk(document, start, end, text="abcde")


@pytest.mark.parametrize("text", ["", "   ", "\n\n", "line\rbreak", "bad\x00char"])
def test_chunk_rejects_blank_or_unnormalized_text(text):
    document = make_document()

    with pytest.raises(ValidationError):
        make_chunk(document, 0, max(len(text), 1), text=text)


def test_chunk_rejects_inconsistent_ids():
    document = make_document()

    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, doc_id="cwe:CWE-79")
    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, chunk_id="cwe:CWE-89#0099", ordinal=0)


def test_chunk_rejects_wrong_hash():
    document = make_document()

    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, chunk_sha256="c" * 64)


@pytest.mark.parametrize(
    "bad_hash",
    ["", "abc", "A" * 64, "a" * 63, "a" * 65, "g" * 64, "a" * 64 + "\n", " " + "a" * 64],
)
def test_chunk_rejects_malformed_doc_content_sha256(bad_hash):
    document = make_document()

    with pytest.raises(ValidationError, match="64 lowercase hexadecimal"):
        make_chunk(document, 0, 14, doc_content_sha256=bad_hash)


def test_chunk_requires_doc_content_sha256():
    document = make_document()
    values = make_chunk(document, 0, 14).model_dump()
    del values["doc_content_sha256"]

    with pytest.raises(ValidationError, match="doc_content_sha256"):
        Chunk(**values)


def test_chunk_accepts_well_formed_doc_content_sha256():
    document = make_document()

    chunk = make_chunk(document, 0, 14, doc_content_sha256="0123456789abcdef" * 4)

    assert chunk.doc_content_sha256 == "0123456789abcdef" * 4


def test_chunk_rejects_invalid_fields():
    document = make_document()

    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, ordinal=-1)
    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, doc_type="blog_post")
    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, source_url="ftp://example.org/x")
    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, chunker_version="")
    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, section="  ")
    with pytest.raises(ValidationError):
        make_chunk(document, 0, 14, unexpected="x")


def test_chunk_json_round_trip_is_lossless():
    document = make_document()
    chunk = make_chunk(document, 0, 14, ordinal=2)

    assert Chunk.model_validate_json(chunk.model_dump_json()) == chunk


def test_chunk_is_immutable():
    document = make_document()
    chunk = make_chunk(document, 0, 14)
    snapshot = chunk.model_dump()

    for field, value in [
        ("text", "tampered text"),
        ("char_start", 1),
        ("char_end", 99),
        ("chunk_sha256", "f" * 64),
        ("doc_content_sha256", "e" * 64),
        ("doc_id", "cwe:CWE-79"),
        ("source_url", None),
    ]:
        with pytest.raises(ValidationError):
            setattr(chunk, field, value)
    with pytest.raises(ValidationError):
        chunk.section = "Elsewhere"
    with pytest.raises(ValidationError):
        chunk.not_a_field = "x"  # type: ignore[attr-defined]

    assert chunk.model_dump() == snapshot  # nothing changed
    assert document.contains_chunk(chunk)


def test_chunk_is_hashable_and_equal_chunks_hash_equal():
    document = make_document()
    first = make_chunk(document, 0, 14)
    second = make_chunk(document, 0, 14)
    other = make_chunk(document, 0, 15)

    assert first == second and hash(first) == hash(second)
    assert {first: "kept"}[second] == "kept"
    assert other != first


# contains_chunk range checks. `Chunk.model_construct` skips validation, which
# is the only way to hand contains_chunk an out-of-range chunk. Each case
# below is built so that EVERY OTHER check still passes (identity, content
# hash, metadata, text == content[start:end], chunk_sha256), so only the
# range check under test can reject it. Text equality alone must not be
# enough: Python slicing clamps or wraps out-of-range indices.

def _unvalidated_chunk(document: SourceDocument, start: int, end: int, text: str) -> Chunk:
    fields = make_chunk(document, 0, 14).model_dump()
    fields.update(
        char_start=start,
        char_end=end,
        text=text,
        chunk_sha256=sha256_hex(text),
    )
    return Chunk.model_construct(**fields)


def test_contains_chunk_rejects_a_negative_start_even_when_the_text_matches():
    document = make_document()
    length = len(document.content)
    negative = _unvalidated_chunk(document, -5, length, document.content[-5:length])

    # The trap: negative indices wrap, so the slice and the text are equal.
    assert document.content[negative.char_start:negative.char_end] == negative.text
    assert negative.char_start < negative.char_end <= length
    assert not document.contains_chunk(negative)


def test_contains_chunk_rejects_an_end_past_the_document_even_when_the_text_matches():
    document = make_document()
    length = len(document.content)
    overlong = _unvalidated_chunk(document, length - 5, length + 7, document.content[-5:])

    # The trap: slicing clamps the end, so the slice and the text are equal.
    assert document.content[overlong.char_start:overlong.char_end] == overlong.text
    assert 0 <= overlong.char_start < overlong.char_end
    assert not document.contains_chunk(overlong)


@pytest.mark.parametrize("start,end", [(5, 5), (0, 0), (10, 5), (20, 3)])
def test_contains_chunk_rejects_an_empty_or_reversed_span_even_when_the_text_matches(
    start, end
):
    document = make_document()
    empty = _unvalidated_chunk(document, start, end, "")

    # The trap: an empty or reversed slice is "" and equals the empty text.
    assert document.content[empty.char_start:empty.char_end] == empty.text == ""
    assert 0 <= empty.char_start <= len(document.content)
    assert empty.char_end <= len(document.content)
    assert not document.contains_chunk(empty)


def test_contains_chunk_accepts_one_character_chunks_exactly_at_both_limits():
    document = make_document()
    length = len(document.content)

    # Guards against the range checks being made too strict (off by one).
    assert document.contains_chunk(_unvalidated_chunk(document, 0, 1, document.content[:1]))
    assert document.contains_chunk(
        _unvalidated_chunk(document, length - 1, length, document.content[-1:])
    )
