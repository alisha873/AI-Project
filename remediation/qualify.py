"""
Shorten a compromise sentence the model already denied, then let grounding validate it again.

`remediation.grounding` is not modified. Its negation window is short on purpose, so a real denial
can still be reported: "do not confirm that an attack succeeded or that the target system was
compromised", or "No ... logs were available to confirm successful exploitation".

This module does not turn an affirmative claim into a denial. It only replaces a sentence that
already denies the flagged phrase with one canonical unconfirmed statement. The unchanged
`validate_plan` then decides whether that statement is acceptable. If a flagged phrase is not a
clear denial, nothing is rewritten.
"""

from __future__ import annotations

import re

from remediation.contract import PlanDraft
from remediation.grounding import PlanValidation

CANONICAL_DENIAL = (
    "The finding status is hypothesis_not_confirmed, and the supplied material "
    "does not establish that outcome."
)

_SENTENCE_BREAK = re.compile(r"[.!?;:](?=\s|$)|\n")
_ASSERTED = re.compile(r"asserts '([^']+)'")
_PART = re.compile(r"^([A-Za-z_]+)(?:\[(\d+)\])?$")

# A lead that denies the words after it. "but"/"however" between the lead and the phrase ends the denial.
_DENIAL_LEAD = re.compile(
    r"(?:(?:does|do|did)\s+not|cannot|can't|can\s+not)\s+"
    r"(?:(?:currently|yet|directly)\s+)?"
    r"(?:establish|prove|show|confirm|demonstrate)\b"
    r"|no\s+(?:evidence|indication|sign|proof)s?\s+(?:of|that)\b"
    r"|\bno\b(?:\s+\w+){0,16}\s+(?:were|was|are|is)\s+available\s+to\s+confirm\b"
    r"|(?:were|was|are|is)\s+not\s+available\s+to\s+confirm\b",
    re.IGNORECASE,
)
_REASSERT = re.compile(r"\b(?:but|however|although|though)\b", re.IGNORECASE)


def qualify_denied_compromise(draft: PlanDraft, validation: PlanValidation) -> PlanDraft | None:
    """Return a draft with denied compromise sentences shortened, or None if any phrase is affirmative.

    None means the caller's text must not be rewritten. The caller still rejects that plan.
    """
    if not validation.issues:
        return None
    by_location: dict[str, list[str]] = {}
    for issue in validation.issues:
        if issue.code != "unsupported_compromise_claim":
            return None
        matched = _ASSERTED.search(issue.message)
        if matched is None:
            return None
        by_location.setdefault(issue.location, []).append(matched.group(1))

    data = draft.model_dump()
    for location, phrases in by_location.items():
        try:
            text = _read(data, location)
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(text, str):
            return None
        rewritten = _rewrite_denied_phrases(text, phrases)
        if rewritten is None:
            return None
        _write(data, location, rewritten)
    try:
        return PlanDraft.model_validate(data)
    except ValueError:
        return None


def _rewrite_denied_phrases(text: str, phrases: list[str]) -> str | None:
    spans: list[tuple[int, int]] = []
    for phrase in phrases:
        found = False
        start = 0
        needle = phrase.casefold()
        folded = text.casefold()
        while True:
            at = folded.find(needle, start)
            if at < 0:
                break
            found = True
            end = at + len(phrase)
            begin, stop = _sentence_bounds(text, at, end)
            if not _governed_by_denial(text[begin:stop], at - begin):
                return None
            spans.append((begin, stop))
            start = end
        if not found:
            return None
    if not spans:
        return None
    return _replace_spans(text, spans)


def _governed_by_denial(sentence: str, phrase_at: int) -> bool:
    leads = list(_DENIAL_LEAD.finditer(sentence, 0, phrase_at))
    if not leads:
        return False
    between = sentence[leads[-1].end():phrase_at]
    return _REASSERT.search(between) is None


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    begin = 0
    for match in _SENTENCE_BREAK.finditer(text, 0, start):
        begin = match.end()
    following = _SENTENCE_BREAK.search(text, end)
    stop = following.end() if following else len(text)
    return begin, stop


def _replace_spans(text: str, spans: list[tuple[int, int]]) -> str:
    merged: list[tuple[int, int]] = []
    for begin, end in sorted(spans):
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    pieces: list[str] = []
    cursor = 0
    for begin, end in merged:
        prefix = text[cursor:begin]
        pieces.append(prefix)
        if prefix and not prefix[-1].isspace():
            pieces.append(" ")
        pieces.append(CANONICAL_DENIAL + " ")
        cursor = end
    rest = text[cursor:].lstrip()
    if pieces and rest[:1].islower():
        rest = rest[0].upper() + rest[1:]
    pieces.append(rest)
    rewritten = re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()
    return rewritten or CANONICAL_DENIAL


def _read(data: object, location: str) -> object:
    cursor = data
    for part in location.split("."):
        cursor = _step(cursor, part)
    return cursor


def _write(data: dict, location: str, text: str) -> None:
    parts = location.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = _step(cursor, part)
    last = parts[-1]
    matched = _PART.fullmatch(last)
    if matched is None:
        raise KeyError(location)
    name, index = matched.group(1), matched.group(2)
    if index is None:
        cursor[name] = text
    else:
        cursor[name][int(index)] = text


def _step(cursor: object, part: str) -> object:
    matched = _PART.fullmatch(part)
    if matched is None or not isinstance(cursor, dict):
        raise KeyError(part)
    name, index = matched.group(1), matched.group(2)
    value = cursor[name]
    if index is None:
        return value
    if not isinstance(value, list):
        raise TypeError(part)
    return value[int(index)]
