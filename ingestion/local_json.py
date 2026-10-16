"""
Phase 10 — Knowledge ingestion: minimal local JSON document loader.

`load_documents(path)` reads ONE local JSON file and returns validated
`knowledge.schema.SourceDocument` objects. It exists to establish and test
the adapter contract that later source-specific adapters (CWE, CAPEC, ...)
will follow: *read an artifact, return validated documents*. It does no
database work, no chunking and no network access, and it does not know the
store or the pipeline.

File format (UTF-8, with or without a byte-order mark)
-------------------------------------------------------
::

    {
      "format": "autoshield.local_documents.v1",
      "description": "optional free text",
      "documents": [
        {
          "source": "cwe",
          "source_id": "EXAMPLE-1",
          "source_version": "v1",
          "doc_type": "weakness",
          "title": "...",
          "content": "...",
          "source_url": null,
          "license": "...",
          "source_modified": null,
          "references": [
            {"relation": "ChildOf", "target_source": "cwe", "target_source_id": "EXAMPLE-2"}
          ],
          "provenance": {
            "origin_kind": "bundled_fixture",
            "artifact_path": "relative/path.json",
            "artifact_sha256": "<64 hex digits>",
            "adapter": "local_json",
            "adapter_version": "1",
            "retrieved_at_utc": "2026-10-16T00:00:00Z",
            "ingested_at_utc": "2026-10-16T00:00:00Z"
          }
        }
      ]
    }

Each record has exactly the shape of a serialized `SourceDocument`
(`SourceDocument.model_dump(mode="json")`). `doc_id`, `content_sha256` and
`document_sha256` may be omitted (the schema computes them); if present
they must be correct. Unknown keys are errors (the schema forbids them).

The loader never invents metadata: provenance, URLs, licences and
timestamps come from the record exactly as written, and a missing
required field is an error, not a default. Nothing checks that
`artifact_sha256` matches any real file; that is the job of an adapter
that actually downloads or reads a source artifact.

Strictness (nothing is skipped or repaired silently)
----------------------------------------------------
- unreadable file, bad UTF-8, malformed JSON: `LocalJsonError`;
- duplicate keys inside any JSON object: error (plain `json` would keep
  only the last one);
- top level must be an object with `format` equal to `FORMAT_ID`,
  a `documents` list and optionally a string `description`; any other key
  is an error;
- every record is validated by `SourceDocument`; the first invalid record
  raises and NOTHING is returned;
- duplicate `doc_id` values within one file are an error (even if the two
  records are identical), reported with both record indices;
- an empty `documents` list is valid and returns `[]`.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from knowledge.schema import SourceDocument

FORMAT_ID = "autoshield.local_documents.v1"

_ALLOWED_TOP_LEVEL_KEYS = frozenset({"format", "description", "documents"})


class LocalJsonError(ValueError):
    """The file could not be read, parsed or validated. The message names the path."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: set[str] = set()
    for key, _value in pairs:
        if key in seen:
            raise ValueError(f"duplicate JSON key {key!r}")
        seen.add(key)
    return dict(pairs)


def _describe_validation_error(exc: ValidationError) -> str:
    """Short 'field: message' list; never echoes (possibly huge) input values."""
    parts = []
    for error in exc.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in error["loc"]) or "(document)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)


def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise LocalJsonError(f"{path}: cannot read file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise LocalJsonError(f"{path}: file is not valid UTF-8: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise LocalJsonError(f"{path}: malformed JSON: {exc}") from exc
    except ValueError as exc:  # raised by _reject_duplicate_keys
        raise LocalJsonError(f"{path}: invalid JSON: {exc}") from exc


def _check_envelope(path: Path, data: object) -> list[object]:
    """Validate the top-level structure and return the raw record list."""
    if not isinstance(data, dict):
        raise LocalJsonError(
            f"{path}: unsupported top-level JSON ({type(data).__name__}); "
            f"expected an object with 'format' and 'documents'"
        )
    unknown = sorted(set(data) - _ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise LocalJsonError(f"{path}: unknown top-level keys {unknown}")
    if data.get("format") != FORMAT_ID:
        raise LocalJsonError(
            f"{path}: 'format' must be {FORMAT_ID!r}, got {data.get('format')!r}"
        )
    if "description" in data and not isinstance(data["description"], str):
        raise LocalJsonError(f"{path}: 'description' must be a string")
    records = data.get("documents")
    if not isinstance(records, list):
        raise LocalJsonError(f"{path}: 'documents' must be a list")
    return records


def load_documents(path: str | Path) -> list[SourceDocument]:
    """
    Load and validate every document in a local JSON file (format above).

    Args:
        path: explicit filesystem path of the JSON file.

    Returns:
        Validated `SourceDocument` objects in file order (possibly empty).

    Raises:
        TypeError: `path` is not a str or Path.
        LocalJsonError: the file is unreadable, malformed or unsupported, a
            record is invalid (message gives `documents[<index>]`; the
            underlying `ValidationError` is the `__cause__`), or two
            records share a `doc_id`.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}")
    path = Path(path)

    records = _check_envelope(path, _read_json(path))

    documents: list[SourceDocument] = []
    first_index: dict[str, int] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise LocalJsonError(
                f"{path}: documents[{index}] must be a JSON object, got {type(record).__name__}"
            )
        try:
            document = SourceDocument.model_validate(record)
        except ValidationError as exc:
            raise LocalJsonError(
                f"{path}: documents[{index}] is not a valid SourceDocument: "
                f"{_describe_validation_error(exc)}"
            ) from exc
        if document.doc_id in first_index:
            raise LocalJsonError(
                f"{path}: documents[{index}] repeats doc_id {document.doc_id!r} "
                f"(first seen at documents[{first_index[document.doc_id]}])"
            )
        first_index[document.doc_id] = index
        documents.append(document)
    return documents
