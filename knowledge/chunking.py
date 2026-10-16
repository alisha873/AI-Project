"""
Phase 10 — Security Knowledge Base: deterministic chunking.

`chunk_document` cuts one validated `SourceDocument` into validated
`Chunk` objects. It is pure and dependency-free: no tokenization, no
embeddings, no files, no database, no network.

Units
-----
Every size and offset is counted in Python string indices, i.e. Unicode
code points, exactly like `Chunk.char_start` / `Chunk.char_end`. They are
NOT UTF-8 bytes and NOT grapheme clusters.

Fidelity
--------
Chunk text is always the exact slice `document.content[start:end]`. The
chunker never strips, normalizes or rewrites text (the document content was
already normalized by the schema), so `SourceDocument.contains_chunk` holds
for every chunk. IDs, hashes and copied metadata come from the schema's own
`Chunk` validators and helpers; nothing is re-implemented here.

Algorithm (all steps deterministic)
-----------------------------------
Walk the content from `pos` (initially 0). For each chunk:

1. The window is `content[pos : pos + max_chars]`.
2. If the window reaches the end of the content, the chunk is the rest.
3. Otherwise the chunk end is the LAST boundary inside the window, trying
   in this order (the first kind that has a usable boundary wins):
   a. paragraph: after the last "\\n\\n" in the window;
   b. line: after the last "\\n" in the window;
   c. whitespace: after the last whitespace character in the window;
   d. hard cut at the window end. A hard cut is moved back so it does not
      separate a base character from its following combining marks, unless
      that would leave no room for progress.
4. The next chunk starts `overlap_chars` before the previous chunk's end.

Invariants (they make the loop terminate and every chunk valid)
---------------------------------------------------------------
- `0 <= overlap_chars < max_chars` is required; other values are rejected
  rather than adjusted.
- A boundary is usable only if the chunk would end MORE than
  `overlap_chars` after its start, so the next start is strictly greater
  than the current start (forward progress by at least one character).
  The hard cut always satisfies this because `overlap_chars < max_chars`.
- A boundary is usable only if the chunk contains at least one
  non-whitespace character, because `Chunk` rejects blank text. A window
  that is entirely whitespace is skipped (see "Limitations").
- Hence every chunk is non-empty, non-blank and at most `max_chars` long.
- Consecutive chunks share exactly `overlap_chars` characters and nothing
  else (the one exception: no overlap is added across a skipped
  whitespace-only window).

Limitations
-----------
- Overlap is an exact character count, so a chunk may start in the middle
  of a word. Boundaries are chosen for the chunk END only.
- Natural boundaries are preferred strictly in the order above; the
  chunker does not try to balance chunk sizes, so a chunk can be much
  shorter than `max_chars`.
- Whitespace-only windows (only possible for whitespace runs of at least
  `max_chars` characters, e.g. inside code fences) are not emitted. Such
  characters are not covered by any chunk; non-whitespace is always covered.
- Combining marks are kept with their base character only for hard cuts.
  Other grapheme clusters (emoji ZWJ sequences, Hangul jamo, regional
  indicator pairs) can still be split by a hard cut.
- Markdown structure is only used through blank lines and newlines. Fences
  and headings are not interpreted, and `Chunk.section` is left as None.
"""

from __future__ import annotations

import unicodedata

from knowledge.schema import Chunk, SourceDocument

CHUNKER_VERSION = "char-structure-v1"
DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 120


def _validate_settings(max_chars: int, overlap_chars: int, chunker_version: str) -> None:
    """Reject invalid settings; never adjust them."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise ValueError("max_chars must be an integer")
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    if isinstance(overlap_chars, bool) or not isinstance(overlap_chars, int):
        raise ValueError("overlap_chars must be an integer")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must not be negative")
    if overlap_chars >= max_chars:
        raise ValueError(
            f"overlap_chars ({overlap_chars}) must be smaller than max_chars ({max_chars})"
        )
    if not isinstance(chunker_version, str) or not chunker_version.strip():
        raise ValueError("chunker_version must be a non-blank string")


def _is_combining(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def _choose_end(content: str, pos: int, limit: int, lo: int) -> int:
    """
    Pick the chunk end in `(lo, limit]` for a window that stops before the
    end of the content. `lo` is the smallest end that is NOT allowed (it
    encodes both forward progress and a non-blank chunk), so the result is
    always greater than `lo`. `limit` must be greater than `lo`.
    """
    # a. paragraph boundary
    index = content.rfind("\n\n", pos, limit)
    if index != -1 and index + 2 > lo:
        return index + 2

    # b. line boundary
    index = content.rfind("\n", pos, limit)
    if index != -1 and index + 1 > lo:
        return index + 1

    # c. whitespace boundary
    for index in range(limit - 1, lo - 1, -1):
        if content[index].isspace():
            return index + 1

    # d. hard cut, kept clear of a base character's combining marks
    end = limit
    while end > lo + 1 and _is_combining(content[end]):
        end -= 1
    if _is_combining(content[end]):
        return limit  # a mark run longer than the window: cut exactly at the limit
    return end


def chunk_document(
    document: SourceDocument,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    chunker_version: str = CHUNKER_VERSION,
) -> list[Chunk]:
    """
    Split `document.content` into validated chunks (see the module docstring
    for the exact algorithm and invariants).

    Args:
        document: a validated `SourceDocument`.
        max_chars: maximum chunk length in code points (default 1200, >= 1).
        overlap_chars: code points shared by consecutive chunks (default 120,
            `0 <= overlap_chars < max_chars`).
        chunker_version: recorded on every chunk; change it whenever the
            chunking behaviour changes (default `"char-structure-v1"`).

    Returns:
        Chunks in document order with ordinals 0, 1, 2, ... Each one
        satisfies `document.contains_chunk(chunk)`. Content that is empty or
        only whitespace (not constructible through the schema) yields `[]`.

    Raises:
        TypeError: `document` is not a `SourceDocument`.
        ValueError: invalid settings.
        RuntimeError: an internal consistency check failed (a bug, or a
            `SourceDocument` built without validation).
    """
    if not isinstance(document, SourceDocument):
        raise TypeError(
            f"chunk_document expects a SourceDocument, got {type(document).__name__}"
        )
    _validate_settings(max_chars, overlap_chars, chunker_version)

    content = document.content
    length = len(content)
    chunks: list[Chunk] = []
    pos = 0

    while pos < length:
        limit = min(pos + max_chars, length)

        first_text = pos
        while first_text < limit and content[first_text].isspace():
            first_text += 1
        if first_text == limit:
            pos = limit  # whitespace-only window: nothing a Chunk could hold
            continue

        if limit == length:
            end = length
        else:
            end = _choose_end(content, pos, limit, max(pos + overlap_chars, first_text))

        chunk = Chunk(
            doc_id=document.doc_id,
            ordinal=len(chunks),
            text=content[pos:end],
            char_start=pos,
            char_end=end,
            doc_content_sha256=document.content_sha256,
            chunker_version=chunker_version,
            source=document.source,
            source_id=document.source_id,
            doc_type=document.doc_type,
            title=document.title,
            source_url=document.source_url,
        )
        if not document.contains_chunk(chunk):
            raise RuntimeError(f"internal error: chunk {chunk.chunk_id} is not in its document")
        chunks.append(chunk)

        if end == length:
            break
        next_pos = end - overlap_chars
        if next_pos <= pos:  # unreachable by construction; never loop forever
            raise RuntimeError("internal error: chunking made no forward progress")
        pos = next_pos

    return chunks
