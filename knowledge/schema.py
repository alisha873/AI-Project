"""
Phase 10 — Security Knowledge Base: document and chunk schema.

This module defines the data contracts for *authoritative security
knowledge* (CWE, CAPEC, CVE, OWASP, Docker guidance, ...). It is
deliberately separate from incident evidence (`evidence/schema.py`):

- incident evidence  = what was observed in a specific incident
- security knowledge = reference material written by a trusted source
- model interpretation = anything an LLM produces; NEVER stored here

`SourceDocument.content` therefore holds source text only
(`content_origin` can only be "source_text").

This module contains pure data models and deterministic helpers. It does
no file, database, network, embedding or retrieval work.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"

SourceName = Literal["cve", "cwe", "capec", "owasp", "docker_docs", "attack"]
DocType = Literal[
    "vulnerability",
    "weakness",
    "attack_pattern",
    "guideline",
    "technique",
]
OriginKind = Literal["local_file", "http_download", "bundled_fixture"]

_VALID_SOURCES: frozenset[str] = frozenset(
    ["cve", "cwe", "capec", "owasp", "docker_docs", "attack"]
)

MAX_SOURCE_ID_LENGTH = 200
MAX_LABEL_LENGTH = 200

# Bumped if the canonical document-hash serialization ever changes.
DOCUMENT_HASH_VERSION = 1

# All patterns are used with `fullmatch` (never `match`), because `$` would
# also accept a trailing newline. Digits are spelled [0-9] because `\d`
# matches non-ASCII digits in Python.
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_UTC_TIMESTAMP_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_UTC_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_CHUNK_ID_RE = re.compile(r"^(?P<doc_id>.+)#(?P<ordinal>[0-9]{4,})$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

# Controls that act as line breaks. Deleting them would silently glue the
# neighbouring words together, so they are converted to "\n" instead.
_LINE_BREAK_CONTROLS = ("\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")

# Markdown code fences (simplified: the marker may be indented by any
# amount of spaces/tabs, so more content is preserved rather than less).
_FENCE_OPEN_RE = re.compile(r"[ \t]*(?P<marker>`{3,}|~{3,})(?P<rest>.*)")
_FENCE_CLOSE_RE = re.compile(r"[ \t]*(?P<marker>`{3,}|~{3,})[ \t]*")


# ============================================================
# DETERMINISTIC HELPERS
# ============================================================

def _is_disallowed_control(char: str) -> bool:
    """Control characters other than newline and tab."""
    return unicodedata.category(char) == "Cc" and char not in ("\n", "\t")


def normalize_text(text: str) -> str:
    """
    Return a canonical form of `text` (deterministic and idempotent).

    Transformations, in order:

    1. Line breaks. CRLF, CR, VT (\\x0b), FF (\\x0c), FS/GS/RS
       (\\x1c-\\x1e), NEL (\\x85), U+2028 and U+2029 all become "\\n", so
       Windows and Unix inputs normalize identically and no words are
       glued together. US (\\x1f) becomes a space.
    2. Every other control character except "\\n" and "\\t" is DELETED
       (NUL, ESC, DEL, other C0/C1 controls).
    3. Unicode NFC (canonically equivalent code points are unified).
    4. Leading blank lines and a leading byte-order mark (U+FEFF) are
       removed. A U+FEFF anywhere else is a real (invisible) character of
       the source and is kept, like zero-width spaces and bidi controls.
    5. Outside fenced code blocks (``` or ~~~):
       - trailing whitespace is removed from each line, EXCEPT that a
         Markdown hard break (two or more trailing spaces before another
         non-blank line) is kept as exactly two spaces;
       - runs of blank lines collapse to a single blank line.
    6. Trailing blank lines and trailing whitespace of the text are
       removed.

    Preserved: letter case, punctuation, tabs, internal spacing, the
    indentation of every line (including the first), and everything inside
    fenced code blocks (trailing spaces and blank-line runs included).
    An unclosed fence extends to the end of the text.

    INTENTIONALLY LOSSY (the source text is changed, never reported):
    - deletion of control characters other than the line-break controls
      above (a raw ESC or NUL byte in a source is dropped);
    - NFC unification, e.g. U+212B ANGSTROM SIGN becomes U+00C5;
    - U+FEFF at the very start is dropped;
    - trailing tabs/NBSP/other whitespace and single trailing spaces are
      dropped outside code fences, and three or more trailing spaces are
      reduced to two;
    - blank-line runs collapse outside code fences (including inside
      Markdown indented code blocks, which are not detected);
    - characters inside fenced blocks are still subject to steps 1-3 and
      to the final trailing-whitespace removal at the end of the text;
    - fence detection is simplified (not full CommonMark).

    The result depends on the Unicode database of the running Python
    (NFC), which only matters for newly assigned characters.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text).__name__}")

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for control in _LINE_BREAK_CONTROLS:
        text = text.replace(control, "\n")
    text = text.replace("\x1f", " ")
    text = "".join(ch for ch in text if not _is_disallowed_control(ch))
    text = unicodedata.normalize("NFC", text)

    lines = _drop_leading_blank_lines(text.split("\n"))
    output = _normalize_lines(lines)

    while output and not output[-1].strip():
        output.pop()
    if output:
        output[-1] = output[-1].rstrip()
    return "\n".join(output)


def _drop_leading_blank_lines(lines: list[str]) -> list[str]:
    """Drop leading blank lines and a leading BOM; keep first-line indentation."""
    lines = list(lines)
    while lines:
        head = lines[0].lstrip("\ufeff")
        if not head.strip():
            lines.pop(0)
        else:
            lines[0] = head
            break
    return lines


def _normalize_lines(lines: list[str]) -> list[str]:
    """Whitespace handling for step 5 of `normalize_text`."""
    output: list[str] = []
    fence_char: str | None = None
    fence_length = 0

    for index, line in enumerate(lines):
        if fence_char is not None:
            closing = _FENCE_CLOSE_RE.fullmatch(line)
            if (
                closing
                and closing.group("marker")[0] == fence_char
                and len(closing.group("marker")) >= fence_length
            ):
                fence_char = None
                output.append(line.rstrip())
            else:
                output.append(line)  # code content: verbatim
            continue

        opening = _FENCE_OPEN_RE.fullmatch(line)
        if opening:
            marker = opening.group("marker")
            # Like CommonMark, a backtick fence's info string has no backtick
            # (otherwise it is inline code such as ```x```).
            if marker[0] != "`" or "`" not in opening.group("rest"):
                fence_char, fence_length = marker[0], len(marker)
                output.append(line.rstrip())
                continue

        body = line.rstrip()
        if not body:
            if output and not output[-1]:
                continue  # collapse blank-line run
            output.append("")
            continue

        trailing = line[len(body):]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if len(trailing) >= 2 and trailing == " " * len(trailing) and next_line.strip():
            body += "  "  # Markdown hard break
        output.append(body)

    return output


def sha256_hex(text: str) -> str:
    """SHA-256 of `text` encoded as UTF-8, with no normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_sha256(text: str) -> str:
    """
    SHA-256 of the *normalized* text.

    Because normalization is applied first, the hash does not depend on
    the platform's line-ending convention.
    """
    return sha256_hex(normalize_text(text))


def validate_source_id(source_id: str) -> str:
    """
    Check an original source identifier and return it UNCHANGED.

    Identifiers such as "CWE-89" or "A03:2021" are never rewritten
    (no trimming, no case change). Identifiers that cannot be used safely
    inside `doc_id` / `chunk_id` are rejected instead of silently altered.
    """
    if not isinstance(source_id, str):
        raise ValueError("source_id must be a string")
    if not source_id:
        raise ValueError("source_id must not be empty")
    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"source_id exceeds {MAX_SOURCE_ID_LENGTH} characters")
    for char in source_id:
        if char.isspace() or unicodedata.category(char) in ("Cc", "Cf"):
            raise ValueError(
                "source_id must not contain whitespace or control characters"
            )
    if "#" in source_id:
        raise ValueError("source_id must not contain '#' (reserved for chunk IDs)")
    return source_id


def make_doc_id(source: str, source_id: str) -> str:
    """
    Deterministic document ID: "<source>:<source_id>", e.g. "cwe:CWE-89".

    Derived only from the source's own identifier, so re-ingesting the
    same record always yields the same ID.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"unknown source {source!r}; expected one of {sorted(_VALID_SOURCES)}"
        )
    return f"{source}:{validate_source_id(source_id)}"


def make_chunk_id(doc_id: str, ordinal: int) -> str:
    """Deterministic chunk ID: "<doc_id>#<ordinal, zero-padded to 4>"."""
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")
    return f"{doc_id}#{ordinal:04d}"


# ============================================================
# FIELD VALIDATION HELPERS
# ============================================================

def _require_text(value: str, field_name: str) -> str:
    """Non-empty text; surrounding whitespace is not allowed to hide emptiness."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


def _validate_http_url(value: str) -> str:
    """Accept only http/https URLs with a host. The URL is returned unchanged."""
    if any(char.isspace() for char in value):
        raise ValueError("URL must not contain whitespace")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("URL must include a host")
    return value


def _validate_utc_timestamp(value: str, field_name: str) -> str:
    """
    Require exactly 'YYYY-MM-DDTHH:MM:SSZ' with zero-padded fields (the
    format used by existing analysis outputs), and a real calendar time.

    The shape is checked first: `strptime` alone accepts unpadded fields
    such as '2026-1-5T1:2:3Z'.
    """
    message = f"{field_name} must be UTC in the form YYYY-MM-DDTHH:MM:SSZ (zero-padded)"
    if not _UTC_TIMESTAMP_RE.fullmatch(value):
        raise ValueError(message)
    try:
        datetime.strptime(value, _UTC_TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ValueError(message) from exc
    return value


def _validate_sha256(value: str, field_name: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be 64 lowercase hexadecimal characters")
    return value


def _validate_label(value: str, field_name: str) -> str:
    """
    A source-provided label (relationship name, taxonomy name, external
    identifier). Returned UNCHANGED: internal spaces are allowed (e.g.
    'OWASP Attacks'), but surrounding whitespace, control or invisible
    format characters, and absurd lengths are rejected instead of repaired.
    """
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{field_name} must not have leading or trailing whitespace")
    if len(value) > MAX_LABEL_LENGTH:
        raise ValueError(f"{field_name} exceeds {MAX_LABEL_LENGTH} characters")
    if any(unicodedata.category(char) in ("Cc", "Cf") for char in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _normalize_title(value: str) -> str:
    """Titles are single-line: all whitespace runs collapse to one space."""
    return " ".join(normalize_text(value).split())


def _validate_relative_path(value: str) -> str:
    """Relative, forward-slash path with no parent-directory segments."""
    _require_text(value, "artifact_path")
    if "\\" in value:
        raise ValueError("artifact_path must use forward slashes")
    if value.startswith("/") or _WINDOWS_DRIVE_RE.match(value):
        raise ValueError("artifact_path must be relative")
    if ".." in value.split("/"):
        raise ValueError("artifact_path must not contain '..'")
    return value


# ============================================================
# PROVENANCE AND REFERENCES
# ============================================================

class ProvenanceRecord(BaseModel):
    """
    How AutoShield obtained and processed a document.

    This records OUR handling of the raw artifact (file hash, adapter,
    timestamps). It does not claim anything about the content's truth.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    origin_kind: OriginKind
    artifact_path: str
    artifact_sha256: str
    adapter: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    retrieved_at_utc: str
    ingested_at_utc: str

    @field_validator("artifact_path")
    @classmethod
    def _check_artifact_path(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("artifact_sha256")
    @classmethod
    def _check_artifact_sha256(cls, value: str) -> str:
        return _validate_sha256(value, "artifact_sha256")

    @field_validator("adapter", "adapter_version")
    @classmethod
    def _check_nonblank(cls, value: str) -> str:
        return _require_text(value, "value")

    @field_validator("retrieved_at_utc")
    @classmethod
    def _check_retrieved(cls, value: str) -> str:
        return _validate_utc_timestamp(value, "retrieved_at_utc")

    @field_validator("ingested_at_utc")
    @classmethod
    def _check_ingested(cls, value: str) -> str:
        return _validate_utc_timestamp(value, "ingested_at_utc")


class ExternalReference(BaseModel):
    """
    A relationship or reference that the SOURCE ITSELF declares (e.g. a
    CWE entry's "ChildOf" link, or a CAPEC mapping to a WASC or ATT&CK
    entry). It is a source fact, not an AutoShield inference.

    All labels are stored exactly as the source wrote them:
    - `relation`: the source's own label, e.g. "ChildOf", "CanPrecede".
    - `target_source`: the target's taxonomy or source name. Lowercase
      names from `SourceName` (e.g. "cwe") denote documents AutoShield can
      hold; any other value (e.g. "WASC", "CERT C", "REF") is an external
      taxonomy that is recorded but not resolvable to a local document.
    - `target_source_id`: the target's identifier in that taxonomy. It is
      validated strictly (like `source_id`) only for local sources.
    - `url`: optional http/https link for the reference.

    The target need not exist in our corpus.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation: str
    target_source: str
    target_source_id: str
    url: str | None = None

    @field_validator("relation")
    @classmethod
    def _check_relation(cls, value: str) -> str:
        return _validate_label(value, "relation")

    @field_validator("target_source")
    @classmethod
    def _check_target_source(cls, value: str) -> str:
        return _validate_label(value, "target_source")

    @field_validator("target_source_id")
    @classmethod
    def _check_target_source_id(cls, value: str) -> str:
        return _validate_label(value, "target_source_id")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        return None if value is None else _validate_http_url(value)

    @model_validator(mode="after")
    def _check_local_target(self) -> "ExternalReference":
        if self.target_source in _VALID_SOURCES:
            validate_source_id(self.target_source_id)
        return self

    @property
    def target_doc_id(self) -> str | None:
        """The local `doc_id` this reference points to, or None for external taxonomies."""
        if self.target_source in _VALID_SOURCES:
            return make_doc_id(self.target_source, self.target_source_id)
        return None


# ============================================================
# DOCUMENT-LEVEL HASH
# ============================================================

def canonical_document_json(
    *,
    schema_version: str,
    source: str,
    source_id: str,
    source_version: str,
    doc_type: str,
    title: str,
    content_sha256: str,
    source_url: str | None,
    license: str | None,
    source_modified: str | None,
    references: "Iterable[ExternalReference]",
) -> str:
    """
    The exact text that `document_sha256` hashes.

    Canonical serialization:
    - a JSON object with keys sorted alphabetically, no whitespace
      (separators "," and ":"), pure-ASCII output (`ensure_ascii=True`),
      absent optional values as `null`;
    - it contains `hash_version`, `schema_version`, `source`, `source_id`,
      `source_version`, `doc_type`, `title` (already normalized),
      `content_sha256` (which stands for the full normalized content),
      `source_url`, `license`, `source_modified` and `references`;
    - `references` is a list of objects with keys `relation`,
      `target_source`, `target_source_id` and `url`, SORTED by
      (relation, target_source, target_source_id, url or "") using
      Python string (code point) order, so the hash does not depend on the
      order in which an adapter listed them. Duplicates are not removed.

    Deliberately excluded: `doc_id` and `content_origin` (derived /
    constant) and ALL of `provenance`. Provenance describes how we
    obtained the record (timestamps, artifact hash, adapter), not what the
    source said, so re-ingesting an unchanged record leaves the hash
    unchanged.
    """
    entries = sorted(
        (
            {
                "relation": ref.relation,
                "target_source": ref.target_source,
                "target_source_id": ref.target_source_id,
                "url": ref.url,
            }
            for ref in references
        ),
        key=lambda e: (e["relation"], e["target_source"], e["target_source_id"], e["url"] or ""),
    )
    payload = {
        "hash_version": DOCUMENT_HASH_VERSION,
        "schema_version": schema_version,
        "source": source,
        "source_id": source_id,
        "source_version": source_version,
        "doc_type": doc_type,
        "title": title,
        "content_sha256": content_sha256,
        "source_url": source_url,
        "license": license,
        "source_modified": source_modified,
        "references": entries,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_document_sha256(**fields) -> str:
    """SHA-256 of `canonical_document_json(**fields)` (see that function)."""
    return sha256_hex(canonical_document_json(**fields))


# ============================================================
# SOURCE DOCUMENT
# ============================================================

class SourceDocument(BaseModel):
    """
    One normalized security-knowledge document (e.g. a single CWE entry).

    - `source_id` is the source's own identifier, preserved exactly.
    - `doc_id`, `content_sha256` and `document_sha256` are derived. They
      may be omitted when constructing (they are then computed); if
      supplied they must match.
    - `content` is normalized with `normalize_text` on construction (see
      that function for the intentionally lossy steps); `title` has its
      whitespace collapsed to single spaces.
    - `content_sha256` hashes the normalized content only.
      `document_sha256` hashes the canonical source-derived fields
      (identifiers, version, type, title, content hash, URL, license,
      modification date, references) and excludes `provenance`; see
      `canonical_document_json`. Use it to detect that a record changed.
    - `source_url` is kept exactly as given (only validated), so it is
      never silently rewritten by URL canonicalization.
    - `references` is an immutable tuple that keeps the adapter's order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = SCHEMA_VERSION

    doc_id: str
    source: SourceName
    source_id: str
    source_version: str
    doc_type: DocType

    title: str
    content: str
    content_sha256: str
    document_sha256: str
    # Source text only. Model-generated text is not representable here.
    content_origin: Literal["source_text"] = "source_text"

    source_url: str | None = None
    license: str | None = None
    source_modified: str | None = None

    references: tuple[ExternalReference, ...] = ()
    provenance: ProvenanceRecord

    @model_validator(mode="before")
    @classmethod
    def _fill_derived_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        source = data.get("source")
        source_id = data.get("source_id")
        if "doc_id" not in data and isinstance(source, str) and isinstance(source_id, str):
            data["doc_id"] = make_doc_id(source, source_id)
        content = data.get("content")
        if "content_sha256" not in data and isinstance(content, str):
            data["content_sha256"] = content_sha256(content)
        if "document_sha256" not in data:
            derived = cls._derive_document_sha256(data)
            if derived is not None:
                data["document_sha256"] = derived
        return data

    @staticmethod
    def _derive_document_sha256(data: dict) -> str | None:
        """
        Compute the document hash from raw constructor input, applying the
        same normalization the field validators apply. Returns None when the
        input is too broken to hash; the field validators then report the
        real problem.
        """
        try:
            references = tuple(
                ExternalReference.model_validate(item)
                for item in (data.get("references") or ())
            )
            return compute_document_sha256(
                schema_version=data.get("schema_version", SCHEMA_VERSION),
                source=data["source"],
                source_id=data["source_id"],
                source_version=data["source_version"],
                doc_type=data["doc_type"],
                title=_normalize_title(data["title"]),
                content_sha256=content_sha256(data["content"]),
                source_url=data.get("source_url"),
                license=data.get("license"),
                source_modified=data.get("source_modified"),
                references=references,
            )
        except (KeyError, TypeError, ValueError):
            return None

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, value: str) -> str:
        return validate_source_id(value)

    @field_validator("source_version")
    @classmethod
    def _check_source_version(cls, value: str) -> str:
        return _require_text(value, "source_version")

    @field_validator("title")
    @classmethod
    def _check_title(cls, value: str) -> str:
        return _require_text(_normalize_title(value), "title")

    @field_validator("content")
    @classmethod
    def _normalize_content(cls, value: str) -> str:
        return _require_text(normalize_text(value), "content")

    @field_validator("content_sha256")
    @classmethod
    def _check_content_sha256_format(cls, value: str) -> str:
        return _validate_sha256(value, "content_sha256")

    @field_validator("document_sha256")
    @classmethod
    def _check_document_sha256_format(cls, value: str) -> str:
        return _validate_sha256(value, "document_sha256")

    @field_validator("source_url")
    @classmethod
    def _check_source_url(cls, value: str | None) -> str | None:
        return None if value is None else _validate_http_url(value)

    @field_validator("license")
    @classmethod
    def _check_license(cls, value: str | None) -> str | None:
        return None if value is None else _require_text(value, "license")

    @field_validator("source_modified")
    @classmethod
    def _check_source_modified(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("source_modified must be an ISO 8601 date or datetime") from exc
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "SourceDocument":
        expected_id = make_doc_id(self.source, self.source_id)
        if self.doc_id != expected_id:
            raise ValueError(
                f"doc_id {self.doc_id!r} does not match source/source_id ({expected_id!r})"
            )
        if self.content_sha256 != content_sha256(self.content):
            raise ValueError("content_sha256 does not match the normalized content")
        expected_document_hash = compute_document_sha256(
            schema_version=self.schema_version,
            source=self.source,
            source_id=self.source_id,
            source_version=self.source_version,
            doc_type=self.doc_type,
            title=self.title,
            content_sha256=self.content_sha256,
            source_url=self.source_url,
            license=self.license,
            source_modified=self.source_modified,
            references=self.references,
        )
        if self.document_sha256 != expected_document_hash:
            raise ValueError("document_sha256 does not match the document fields")
        return self

    def contains_chunk(self, chunk: "Chunk") -> bool:
        """
        True only if `chunk` is faithful to this document:

        - same document identity (`doc_id`, `source`, `source_id`);
        - cut from this exact content (`doc_content_sha256` equals this
          document's `content_sha256`), so stale chunks are detected;
        - copied metadata still matches (`doc_type`, `title`, `source_url`);
        - offsets are valid and
          `content[char_start:char_end] == chunk.text` exactly;
        - `chunk_sha256` matches `text` (guards chunks built without
          validation).

        Offsets are Python string indices (Unicode code points).
        """
        if not isinstance(chunk, Chunk):
            return False
        return (
            chunk.doc_id == self.doc_id
            and chunk.source == self.source
            and chunk.source_id == self.source_id
            and chunk.doc_content_sha256 == self.content_sha256
            and chunk.doc_type == self.doc_type
            and chunk.title == self.title
            and chunk.source_url == self.source_url
            and 0 <= chunk.char_start < chunk.char_end <= len(self.content)
            and self.content[chunk.char_start:chunk.char_end] == chunk.text
            and chunk.chunk_sha256 == sha256_hex(chunk.text)
        )


# ============================================================
# CHUNK
# ============================================================

class Chunk(BaseModel):
    """
    A contiguous slice of one `SourceDocument.content`.

    `text` is stored EXACTLY as sliced (never stripped or re-normalized)
    so that a later chunker can guarantee

        document.content[char_start:char_end] == chunk.text

    (see `SourceDocument.contains_chunk`). The schema itself enforces what
    can be checked without the document: `char_end - char_start == len(text)`,
    text is not blank, and text contains no carriage returns or control
    characters (which normalized content never contains).

    The source fields are copied from the document so a retrieved chunk
    carries its provenance without a database join. `doc_content_sha256`
    is the `content_sha256` of the document content the offsets refer to:
    if the document's content changes, chunks carrying the old value are
    stale (the offsets and text no longer describe the current document).
    Offsets are Python string indices (Unicode code points), not bytes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    doc_id: str
    ordinal: int = Field(ge=0)
    section: str | None = None

    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    chunk_sha256: str
    doc_content_sha256: str
    chunker_version: str

    source: SourceName
    source_id: str
    doc_type: DocType
    title: str
    source_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _fill_derived_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        doc_id = data.get("doc_id")
        ordinal = data.get("ordinal")
        if (
            "chunk_id" not in data
            and isinstance(doc_id, str)
            and isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
        ):
            data["chunk_id"] = make_chunk_id(doc_id, ordinal)
        text = data.get("text")
        if "chunk_sha256" not in data and isinstance(text, str):
            data["chunk_sha256"] = sha256_hex(text)
        return data

    @field_validator("text")
    @classmethod
    def _check_text(cls, value: str) -> str:
        _require_text(value, "text")
        if "\r" in value or any(_is_disallowed_control(ch) for ch in value):
            raise ValueError(
                "text must be normalized (no carriage returns or control characters)"
            )
        return value

    @field_validator("section")
    @classmethod
    def _check_section(cls, value: str | None) -> str | None:
        return None if value is None else _require_text(value, "section")

    @field_validator("chunker_version", "title")
    @classmethod
    def _check_nonblank(cls, value: str) -> str:
        return _require_text(value, "value")

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, value: str) -> str:
        return validate_source_id(value)

    @field_validator("source_url")
    @classmethod
    def _check_source_url(cls, value: str | None) -> str | None:
        return None if value is None else _validate_http_url(value)

    @field_validator("chunk_sha256")
    @classmethod
    def _check_chunk_sha256_format(cls, value: str) -> str:
        return _validate_sha256(value, "chunk_sha256")

    @field_validator("doc_content_sha256")
    @classmethod
    def _check_doc_content_sha256_format(cls, value: str) -> str:
        return _validate_sha256(value, "doc_content_sha256")

    @model_validator(mode="after")
    def _check_consistency(self) -> "Chunk":
        if self.doc_id != make_doc_id(self.source, self.source_id):
            raise ValueError("doc_id does not match source/source_id")
        if self.chunk_id != make_chunk_id(self.doc_id, self.ordinal):
            raise ValueError("chunk_id does not match doc_id/ordinal")
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        if self.char_end - self.char_start != len(self.text):
            raise ValueError("char_end - char_start must equal len(text)")
        if self.chunk_sha256 != sha256_hex(self.text):
            raise ValueError("chunk_sha256 does not match text")
        return self