"""
Phase 10 — Milestone 4B: shared helpers for the native MITRE XML adapters.

`mitre_cwe.py` and `mitre_capec.py` read two DIFFERENT official XML formats
(different namespaces and element vocabularies). What they share is not the
format but the surrounding discipline, which lives here:

- reading a pinned ZIP archive safely (checksum first, exactly one XML
  member, nothing is ever extracted to disk);
- verifying the catalog identity, version and date declared INSIDE the XML;
- locating the curated records without trusting element positions;
- flattening the sources' rich text (XHTML fragments) into readable text;
- building validated `SourceDocument` objects through the existing schema.

Nothing here touches SQLite, chunking, hashing algorithms or the network.
Hashes and normalization come from `knowledge.schema`.

Failure policy: every problem raises a subclass of `MitreSourceError`. Nothing
is skipped, repaired or defaulted, so a curated corpus can never silently
become a partial one.
"""

from __future__ import annotations

import hashlib
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from knowledge.schema import ExternalReference, ProvenanceRecord, SourceDocument

XHTML_NS = "http://www.w3.org/1999/xhtml"

# Official per-record page patterns. Checked to answer HTTP 200 on 2026-10-16
# (see data/knowledge_sources/NOTICE.md). They point at the CURRENT site, which
# may be a newer release than the pinned archive.
OFFICIAL_RECORD_URLS = {
    "cwe": "https://cwe.mitre.org/data/definitions/{id}.html",
    "capec": "https://capec.mitre.org/data/definitions/{id}.html",
}

# The official archives are 18 MB (CWE) and 4 MB (CAPEC) uncompressed. A
# member far beyond that is not the file we pinned; refuse it (zip-bomb guard).
MAX_XML_BYTES = 64 * 1024 * 1024

_ISO_DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_DIGITS_RE = re.compile(r"[0-9]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


# ============================================================
# ERRORS
# ============================================================

class MitreSourceError(Exception):
    """Base class for every error raised while reading MITRE source data."""


class ArchiveError(MitreSourceError):
    """The ZIP archive is missing, unreadable, ambiguous or not the pinned one."""


class ChecksumMismatchError(ArchiveError):
    """The archive's SHA-256 differs from the pinned value."""


class SourceFormatError(MitreSourceError):
    """The XML is malformed, has the wrong structure, or lacks required content."""


class SourceVersionError(MitreSourceError):
    """The version or date declared inside the XML is not the pinned one."""


class SelectionError(MitreSourceError):
    """Curated record IDs are missing, duplicated or do not match the pinned names."""


# ============================================================
# DATA TYPES
# ============================================================

@dataclass(frozen=True)
class SelectedRecord:
    """One curated record: numeric ID as in the XML ("89"), verified name, optional status."""

    record_id: str
    name: str
    status: str | None = None

    def __post_init__(self) -> None:
        if not _DIGITS_RE.fullmatch(self.record_id):
            raise SelectionError(f"record_id must be digits only, got {self.record_id!r}")
        if not self.name.strip():
            raise SelectionError(f"record {self.record_id}: name must not be empty")


@dataclass(frozen=True)
class ArtifactSpec:
    """
    What a pinned archive must be, plus the metadata stamped on its documents.
    Normally built from the manifest by `mitre_corpus.load_manifest`.

    - `archive_sha256`: expected SHA-256 of the archive file bytes.
    - `xml_member`: the one XML file expected inside the archive.
    - `catalog_date`: the `Date` attribute expected on the XML root.
    - `artifact_filename`: recorded as `provenance.artifact_path` (a pure
      identifier; the checksum, not the location, proves identity).
    - `retrieved_at_utc` / `ingested_at_utc`: provenance timestamps, supplied
      by the caller (never read from the clock) so repeated builds are
      byte-identical and the store sees no change.
    """

    source_version: str
    catalog_date: str
    archive_sha256: str
    xml_member: str
    artifact_filename: str
    license: str
    retrieved_at_utc: str
    ingested_at_utc: str

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.archive_sha256):
            raise ArchiveError("archive_sha256 must be 64 lowercase hexadecimal characters")


# ============================================================
# ARCHIVE READING
# ============================================================

def _plain_member_name(name: str) -> bool:
    """A bare file name: no directories, no parent segments, no drive letters."""
    return bool(name) and not any(ch in name for ch in ("/", "\\", ":", "\x00")) and name not in (".", "..")


def read_xml_member(
    archive_path: str | Path, *, expected_sha256: str, expected_member: str
) -> tuple[bytes, str]:
    """
    Return `(xml_bytes, archive_sha256)` for the single XML member of a pinned archive.

    Order of checks: file exists -> SHA-256 of the archive bytes equals the pin
    -> it is a ZIP -> it holds exactly ONE `.xml` member (other members, such
    as the `.xsd` shipped with CAPEC, are ignored) -> that member has a plain
    file name equal to `expected_member` -> its size is sane. The bytes that
    were hashed are the bytes that are parsed (the file is read once), and
    nothing is extracted to disk.
    """
    path = Path(archive_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ArchiveError(f"archive not found: {path}") from exc
    except OSError as exc:
        raise ArchiveError(f"cannot read archive {path}: {exc}") from exc

    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"checksum mismatch for {path.name}: expected {expected_sha256}, got {actual}. "
            "The file is not the pinned release. Re-download it from the official URL in the "
            "manifest, or, if you intend to move to another release, update the manifest "
            "deliberately and re-verify the selection."
        )

    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"{path.name} is not a valid ZIP archive: {exc}") from exc

    with archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        xml_members = [info for info in members if info.filename.lower().endswith(".xml")]
        if len(xml_members) != 1:
            raise ArchiveError(
                f"{path.name} must contain exactly one .xml member, found "
                f"{[info.filename for info in xml_members]} (all members: "
                f"{[info.filename for info in members]})"
            )
        info = xml_members[0]
        if not _plain_member_name(info.filename):
            raise ArchiveError(f"{path.name}: unsafe member name {info.filename!r}")
        if info.filename != expected_member:
            raise ArchiveError(
                f"{path.name}: expected XML member {expected_member!r}, found {info.filename!r}"
            )
        if info.file_size > MAX_XML_BYTES:
            raise ArchiveError(
                f"{path.name}: member {info.filename!r} declares {info.file_size} bytes "
                f"(limit {MAX_XML_BYTES})"
            )
        try:
            with archive.open(info) as handle:
                data = handle.read(MAX_XML_BYTES + 1)
        except (zipfile.BadZipFile, OSError, RuntimeError, EOFError) as exc:
            raise ArchiveError(f"{path.name}: cannot read member {info.filename!r}: {exc}") from exc
    if len(data) > MAX_XML_BYTES:
        raise ArchiveError(f"{path.name}: member {info.filename!r} exceeds {MAX_XML_BYTES} bytes")
    return data, actual


def parse_catalog_root(
    xml_bytes: bytes,
    *,
    namespace: str,
    root_tag: str,
    catalog_name: str,
    expected_version: str,
    expected_date: str,
) -> ET.Element:
    """
    Parse the XML and verify the identity the FILE declares about itself.

    The root element must be `{namespace}root_tag` with `Name == catalog_name`,
    `Version == expected_version` and `Date == expected_date`. A filename is
    never trusted for any of these.
    """
    # The official files declare no DTD. Entity declarations are how XML
    # "billion laughs" expansion works, so refuse them instead of parsing.
    if b"<!DOCTYPE" in xml_bytes or b"<!ENTITY" in xml_bytes:
        raise SourceFormatError("the XML contains a DOCTYPE or ENTITY declaration; refusing to parse")
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise SourceFormatError(f"malformed XML: {exc}") from exc

    expected_tag = f"{{{namespace}}}{root_tag}"
    if root.tag != expected_tag:
        raise SourceFormatError(f"unexpected root element {root.tag!r}; expected {expected_tag!r}")
    if root.get("Name") != catalog_name:
        raise SourceFormatError(f"root Name is {root.get('Name')!r}; expected {catalog_name!r}")
    version = root.get("Version")
    if version is None:
        raise SourceVersionError("the XML root declares no Version attribute")
    if version != expected_version:
        raise SourceVersionError(
            f"the XML declares {catalog_name} version {version!r} but {expected_version!r} is pinned; "
            "a version change needs an explicit manifest update and fresh verification"
        )
    if root.get("Date") != expected_date:
        raise SourceVersionError(
            f"the XML declares date {root.get('Date')!r} but {expected_date!r} is pinned"
        )
    return root


# ============================================================
# ELEMENT HELPERS (namespace-aware; never positional)
# ============================================================

def qname(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def children(element: ET.Element | None, namespace: str, name: str) -> list[ET.Element]:
    """Direct children called `name` in `namespace` (an absent parent gives [])."""
    if element is None:
        return []
    wanted = qname(namespace, name)
    return [child for child in element if child.tag == wanted]


def optional_child(
    element: ET.Element | None, namespace: str, name: str, *, label: str
) -> ET.Element | None:
    """The single direct child `name`, None if absent; more than one is an error."""
    found = children(element, namespace, name)
    if len(found) > 1:
        raise SourceFormatError(f"{label}: element <{name}> occurs {len(found)} times; expected at most one")
    return found[0] if found else None


def collapse(text: str | None) -> str:
    """Whitespace runs (including XML pretty-printing newlines) become single spaces."""
    return " ".join((text or "").split())


def child_text(
    element: ET.Element | None, namespace: str, name: str, *, label: str
) -> str | None:
    """Flattened text of the single child `name`, or None when it is absent or blank."""
    child = optional_child(element, namespace, name, label=label)
    if child is None:
        return None
    return rich_text(child) or None


_BLOCK_TAGS = frozenset({"p", "div", "ul", "ol", "table", "pre"})


def rich_text(element: ET.Element) -> str:
    """
    Flatten an element that may hold XHTML (`p`, `ul`/`li`, `br`, `div`, `b`, ...)
    into readable plain text.

    Only elements in the XHTML namespace are treated as markup; text of every
    element is kept. Paragraph-like blocks are separated by a blank line,
    list items become "- " lines, `br` starts a new line, and all other
    whitespace collapses to single spaces. No wording is changed.
    """
    tokens: list[tuple[str, str]] = []

    def walk(node: ET.Element) -> None:
        name = local_name(node) if node.tag.startswith(f"{{{XHTML_NS}}}") else ""
        block = name in _BLOCK_TAGS
        if block:
            tokens.append(("para", ""))
        if name == "li":
            tokens.append(("item", ""))
        if name == "br":
            tokens.append(("line", ""))
        if node.text:
            tokens.append(("text", node.text))
        for child in node:
            walk(child)
            if child.tail:
                tokens.append(("text", child.tail))
        if block:
            tokens.append(("para", ""))

    walk(element)

    lines: list[str] = []
    current: list[str] = []
    bullet = False

    def end_line() -> None:
        nonlocal bullet
        text = collapse("".join(current))
        current.clear()
        if text:
            lines.append(f"- {text}" if bullet else text)
            bullet = False

    for kind, value in tokens:
        if kind == "text":
            current.append(value)
        elif kind == "line":
            end_line()
        elif kind == "item":
            end_line()
            bullet = True
        else:  # "para"
            end_line()
            if lines and lines[-1] != "":
                lines.append("")
    end_line()
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def one_line(element: ET.Element) -> str:
    """`rich_text` collapsed onto one line, for list entries."""
    return collapse(rich_text(element))


def latest_modification_date(
    record: ET.Element, namespace: str, *, label: str
) -> str | None:
    """
    The most recent `Modification_Date` in the record's `Content_History`
    (an ISO `YYYY-MM-DD` date), or None if the source lists no modification.
    The submission date is deliberately not used as a modification date.
    """
    history = optional_child(record, namespace, "Content_History", label=label)
    dates: list[str] = []
    for modification in children(history, namespace, "Modification"):
        raw = child_text(modification, namespace, "Modification_Date", label=label)
        if raw is None:
            raise SourceFormatError(f"{label}: a <Modification> has no Modification_Date")
        if not _ISO_DATE_RE.fullmatch(raw):
            raise SourceFormatError(f"{label}: Modification_Date {raw!r} is not YYYY-MM-DD")
        try:
            date.fromisoformat(raw)
        except ValueError as exc:
            raise SourceFormatError(f"{label}: Modification_Date {raw!r} is not a real date") from exc
        dates.append(raw)
    return max(dates) if dates else None


def require_digits(value: str | None, *, what: str, label: str) -> str:
    if value is None or not _DIGITS_RE.fullmatch(value.strip()):
        raise SourceFormatError(f"{label}: {what} must be a numeric ID, got {value!r}")
    return value.strip()


# ============================================================
# SELECTION OF THE CURATED RECORDS
# ============================================================

def select_records(
    container: ET.Element | None,
    namespace: str,
    record_tag: str,
    selection: Sequence[SelectedRecord],
    *,
    catalog: str,
) -> list[tuple[SelectedRecord, ET.Element]]:
    """
    Find every curated record among the direct children of `container`.

    Raises `SelectionError` (listing ALL problems at once) if the selection
    repeats an ID, a selected ID is absent, a selected ID occurs more than
    once in the source, or the record's Name/Status differs from the pinned
    value. Returns `(selected, element)` pairs in selection order.
    """
    if container is None:
        raise SourceFormatError(f"{catalog}: the XML has no container element for <{record_tag}> records")
    problems: list[str] = []
    ids = [item.record_id for item in selection]
    repeated = sorted({i for i in ids if ids.count(i) > 1}, key=int)
    if repeated:
        problems.append(f"selection repeats ID(s) {repeated}")

    wanted = set(ids)
    found: dict[str, list[ET.Element]] = {}
    tag = qname(namespace, record_tag)
    for element in container:
        if element.tag == tag and element.get("ID") in wanted:
            found.setdefault(element.get("ID"), []).append(element)

    result: list[tuple[SelectedRecord, ET.Element]] = []
    for item in selection:
        elements = found.get(item.record_id, [])
        if not elements:
            problems.append(f"{catalog} ID {item.record_id} is missing from the source")
            continue
        if len(elements) > 1:
            problems.append(f"{catalog} ID {item.record_id} occurs {len(elements)} times in the source")
            continue
        element = elements[0]
        actual_name = collapse(element.get("Name"))
        if actual_name != item.name:
            problems.append(
                f"{catalog} ID {item.record_id}: name is {actual_name!r}, pinned name is {item.name!r}"
            )
        if item.status is not None and element.get("Status") != item.status:
            problems.append(
                f"{catalog} ID {item.record_id}: status is {element.get('Status')!r}, "
                f"pinned status is {item.status!r}"
            )
        result.append((item, element))
    if problems:
        raise SelectionError("; ".join(problems))
    return result


# ============================================================
# EXTERNAL REFERENCES AND SHARED SECTION FORMATTING
# ============================================================

@dataclass(frozen=True)
class SourceCitation:
    """One entry of the catalog's `External_References` table (a book, paper, web page ...)."""

    ref_id: str
    title: str | None
    authors: tuple[str, ...]
    url: str | None


def citation_table(root: ET.Element, namespace: str, *, catalog: str) -> dict[str, SourceCitation]:
    """All `External_Reference` entries, keyed by `Reference_ID`; a duplicate ID is an error."""
    label = f"{catalog} External_References"
    table: dict[str, SourceCitation] = {}
    container = optional_child(root, namespace, "External_References", label=label)
    for element in children(container, namespace, "External_Reference"):
        ref_id = element.get("Reference_ID")
        if not ref_id or not ref_id.strip():
            raise SourceFormatError(f"{label}: an External_Reference has no Reference_ID")
        if ref_id in table:
            raise SourceFormatError(f"{label}: Reference_ID {ref_id!r} occurs more than once")
        title = child_text(element, namespace, "Title", label=f"{label} {ref_id}")
        url = child_text(element, namespace, "URL", label=f"{label} {ref_id}")
        authors = tuple(
            collapse(rich_text(author)) for author in children(element, namespace, "Author")
        )
        table[ref_id] = SourceCitation(
            ref_id=ref_id,
            title=collapse(title) if title else None,
            authors=tuple(a for a in authors if a),
            url=url,
        )
    return table


def reference_dict(
    relation: str, target_source: str, target_source_id: str, url: str | None = None
) -> dict[str, str | None]:
    """An `ExternalReference` as plain data (validated later by the schema)."""
    return {
        "relation": relation,
        "target_source": target_source,
        "target_source_id": target_source_id,
        "url": url,
    }


def local_target_reference(
    relation: str, target_source: str, record_id: str
) -> dict[str, str | None]:
    """Reference to a CWE/CAPEC record the SOURCE named (it may be outside the curated subset)."""
    return reference_dict(
        relation,
        target_source,
        f"{target_source.upper()}-{record_id}",
        OFFICIAL_RECORD_URLS[target_source].format(id=record_id),
    )


def citation_entries(
    record: ET.Element,
    namespace: str,
    table: dict[str, SourceCitation],
    *,
    ref_source: str,
    label: str,
) -> tuple[list[str], list[dict[str, str | None]]]:
    """
    The record's `References`: (content lines, ExternalReference dicts).

    A reference to an ID missing from the catalog's table is an error. Several
    cites of one ID with different `Section`s give several content lines but
    one ExternalReference.
    """
    lines: list[str] = []
    refs: list[dict[str, str | None]] = []
    container = optional_child(record, namespace, "References", label=label)
    for element in children(container, namespace, "Reference"):
        ref_id = element.get("External_Reference_ID")
        if ref_id is None or ref_id not in table:
            raise SourceFormatError(
                f"{label}: reference to {ref_id!r}, which is not in the catalog's External_References"
            )
        citation = table[ref_id]
        section = collapse(element.get("Section"))
        line = f"- {ref_id}: {citation.title or '(no title)'}"
        if citation.authors:
            line += f" ({'; '.join(citation.authors)})"
        if section:
            line += f", {section}"
        lines.append(line)
        refs.append(reference_dict("Reference", ref_source, ref_id, citation.url))
    return lines, refs


def taxonomy_entries(
    record: ET.Element, namespace: str, *, label: str
) -> tuple[list[str], list[dict[str, str | None]]]:
    """
    The record's `Taxonomy_Mappings` (e.g. WASC, OWASP, CERT): content lines and
    ExternalReference dicts. The target ID is the source's `Entry_ID`, or its
    `Entry_Name` when the mapping gives no ID.
    """
    lines: list[str] = []
    refs: list[dict[str, str | None]] = []
    container = optional_child(record, namespace, "Taxonomy_Mappings", label=label)
    for element in children(container, namespace, "Taxonomy_Mapping"):
        taxonomy = collapse(element.get("Taxonomy_Name"))
        if not taxonomy:
            raise SourceFormatError(f"{label}: a Taxonomy_Mapping has no Taxonomy_Name")
        entry_id = collapse(child_text(element, namespace, "Entry_ID", label=label))
        entry_name = collapse(child_text(element, namespace, "Entry_Name", label=label))
        fit = collapse(child_text(element, namespace, "Mapping_Fit", label=label))
        target = entry_id or entry_name
        if not target:
            raise SourceFormatError(f"{label}: Taxonomy_Mapping {taxonomy!r} has neither Entry_ID nor Entry_Name")
        line = f"- {taxonomy}"
        if entry_id:
            line += f" {entry_id}"
        if entry_name:
            line += f": {entry_name}"
        if fit:
            line += f" (fit: {fit})"
        lines.append(line)
        refs.append(reference_dict("TaxonomyMapping", taxonomy, target))
    return lines, refs


def consequence_lines(container: ET.Element | None, namespace: str, *, label: str) -> list[str]:
    """`Consequence` entries (shared by CWE `Common_Consequences` and CAPEC `Consequences`)."""
    lines: list[str] = []
    for element in children(container, namespace, "Consequence"):
        scopes = [one_line(x) for x in children(element, namespace, "Scope")]
        impacts = [one_line(x) for x in children(element, namespace, "Impact")]
        note = child_text(element, namespace, "Note", label=label)
        parts: list[str] = []
        if scopes:
            parts.append("Scope: " + ", ".join(scopes))
        if impacts:
            parts.append("Impact: " + "; ".join(impacts))
        if note:
            parts.append("Note: " + collapse(note))
        if parts:
            lines.append("- " + " | ".join(parts))
    return lines


def alternate_term_lines(record: ET.Element, namespace: str, *, label: str) -> list[str]:
    container = optional_child(record, namespace, "Alternate_Terms", label=label)
    lines: list[str] = []
    for element in children(container, namespace, "Alternate_Term"):
        term = collapse(child_text(element, namespace, "Term", label=label))
        description = collapse(child_text(element, namespace, "Description", label=label))
        if term:
            lines.append(f"- {term}: {description}" if description else f"- {term}")
    return lines


def note_lines(record: ET.Element, namespace: str, *, label: str) -> list[str]:
    container = optional_child(record, namespace, "Notes", label=label)
    lines: list[str] = []
    for element in children(container, namespace, "Note"):
        text = one_line(element)
        kind = collapse(element.get("Type"))
        if text:
            lines.append(f"- {kind}: {text}" if kind else f"- {text}")
    return lines


def assemble_content(
    heading: str,
    header_lines: Sequence[str],
    sections: Sequence[tuple[str, str | Sequence[str]]],
) -> str:
    """
    Readable, sectioned text: a `# heading`, simple `Key: value` header lines,
    then one `## Section` per non-empty section (a string is a paragraph block,
    a list of lines is a bullet list). Empty sections are omitted, never invented.
    """
    blocks = [f"# {heading}"]
    header = [line for line in header_lines if line]
    if header:
        blocks.append("\n".join(header))
    for title, body in sections:
        text = body if isinstance(body, str) else "\n".join(body)
        if text.strip():
            blocks.append(f"## {title}\n{text}")
    return "\n\n".join(blocks)


# ============================================================
# DOCUMENT CONSTRUCTION
# ============================================================

def make_document(
    *,
    source: str,
    record_id: str,
    doc_type: str,
    title: str,
    content: str,
    source_modified: str | None,
    references: Sequence[dict[str, str | None]],
    spec: ArtifactSpec,
    adapter: str,
    adapter_version: str,
    archive_sha256: str,
) -> SourceDocument:
    """
    Build one `SourceDocument` through the existing schema (nothing is
    re-implemented: IDs, hashes and normalization are the schema's).

    Identical references are listed once, in first-seen order. Any schema
    violation is reported as a `SourceFormatError` naming the record.
    """
    source_id = f"{source.upper()}-{record_id}"
    seen: set[tuple[str | None, ...]] = set()
    unique: list[dict[str, str | None]] = []
    for ref in references:
        key = (ref["relation"], ref["target_source"], ref["target_source_id"], ref["url"])
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    try:
        return SourceDocument(
            source=source,
            source_id=source_id,
            source_version=spec.source_version,
            doc_type=doc_type,
            title=title,
            content=content,
            source_url=OFFICIAL_RECORD_URLS[source].format(id=record_id),
            license=spec.license,
            source_modified=source_modified,
            references=tuple(ExternalReference.model_validate(ref) for ref in unique),
            provenance=ProvenanceRecord(
                origin_kind="local_file",
                artifact_path=spec.artifact_filename,
                artifact_sha256=archive_sha256,
                adapter=adapter,
                adapter_version=adapter_version,
                retrieved_at_utc=spec.retrieved_at_utc,
                ingested_at_utc=spec.ingested_at_utc,
            ),
        )
    except (ValidationError, ValueError) as exc:
        raise SourceFormatError(
            f"{source_id}: the record does not satisfy the SourceDocument schema: {exc}"
        ) from exc
