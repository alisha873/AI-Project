"""
Phase 11 (Milestone 11D) - deterministic citation validation of a grounded answer.

`validate_answer(answer, context)` checks a parsed answer against the
`GroundedContext` it was generated from and returns EVERY problem it finds
(`CitationValidation.issues`), not just the first.

What is checked
---------------
Structure     exact top-level keys, exact item keys, string/array types, allowed
              enum values, a bounded statement length, the right incident id.
Citations     every `evidence_ids` entry is a citable observed event of THIS
              context; every `reference_ids` entry is a retrieved chunk of THIS
              context; every `hypothesis_id` is a supplied hypothesis, addressed
              exactly once. Evidence IDs Phase 9 cited but Phase 8 did not
              describe (`unresolved_evidence_ids`) are not citable. A statement
              that has no citation of the required kind is rejected.
Categories    observed_evidence cites only events and mentions no knowledge-base
              identifier; reference_guidance cites only passages and never an
              incident event; a hypothesis may cite both. A hypothesis whose
              status is "hypothesis_only" needs at least one evidence ID.
              Identifiers or IDs written INSIDE statement text must also be known
              and must be listed in that item's own citations.
Relevance     every retrieved passage is assessed exactly once; a passage assessed
              "not_relevant" may not be cited. With no retrieved passages the
              guidance and relevance arrays must be empty (no invented support).
Identifiers   a CWE/CAPEC/CVE identifier in a statement must come from the
              hypothesis text, a cited passage's own source id, or an identifier
              that appears in a cited passage's text. Anything else is rejected.
Facts         HTTP methods (upper case), status codes and URL paths written in an
              observed or hypothesis statement must appear in the cited events'
              request context. A positive claim of successful exploitation, data
              access, exfiltration, privilege escalation or compromise is rejected
              (this evidence has no event-level impact marker); negated wording
              ("no evidence of ...", "not confirmed") is allowed. A statement that
              presents the root cause or vulnerability as confirmed is rejected
              unless the supplied finding is confirmed.

What this does NOT prove
------------------------
Passing validation shows that the citations exist, belong to this context and
sit in the right category, and that a few high-risk patterns are absent. It does
NOT prove that a statement is entailed by the cited text, that a passage is
actually relevant to the incident (the relevance label is the model's own
judgement), or that the answer is complete. The fact checks are regular
expressions and can both miss paraphrases and flag unusual wording.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rag.context import GroundedContext
from rag.prompt import ANSWER_KEYS, ANSWER_SCHEMA_VERSION, HYPOTHESIS_STATUSES, RELEVANCE_ASSESSMENTS

MAX_STATEMENT_CHARS = 2000

VALIDATION_NOTE = (
    "Validation checks that citations exist, belong to this context and sit in the right category, "
    "and screens a few high-risk statement patterns. It does not prove that a statement is entailed "
    "by its citations or that a passage is relevant."
)

_OBSERVED_KEYS = ("statement", "evidence_ids")
_HYPOTHESIS_KEYS = ("hypothesis_id", "statement", "status", "evidence_ids", "reference_ids")
_GUIDANCE_KEYS = ("statement", "reference_ids")
_RELEVANCE_KEYS = ("reference_id", "assessment")

_IDENTIFIER_RE = re.compile(r"\b(?:CWE|CAPEC|CVE)-[0-9]+(?:-[0-9]+)?\b", re.IGNORECASE)
_EVIDENCE_TOKEN_RE = re.compile(
    r"\b(?:application|nginx|api|postgres|docker|packet|auth)(?:-[A-Za-z]+)*-[0-9]+(?:-[A-Za-z0-9_]+)*\b"
)
_REFERENCE_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_]*:[A-Za-z][A-Za-z0-9_]*-[0-9]+(?:-[0-9]+)?#[0-9]{4}\b")
_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b")
_STATUS_RE = re.compile(r"\b(?:HTTP\s+|status(?:\s+code)?\s*[:=]?\s*)([1-5][0-9]{2})\b", re.IGNORECASE)
_PATH_RE = re.compile(r"(?<![A-Za-z0-9/])(/[A-Za-z0-9._~!$&'()*+,;=:@%/-]*)")

_HIGH_IMPACT = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bsuccessful(?:ly)?\s+exploit(?:ed|ation)\b",
        r"\b(?:was|were|been|is|are)\s+(?:successfully\s+)?exploited\b",
        r"\b(?:database|db)\s+(?:was\s+|were\s+)?accessed\b",
        r"\bdata\s+(?:was\s+|were\s+)?exfiltrated\b",
        r"\bprivilege\s+escalation\s+(?:occurred|succeeded|was\s+successful)\b",
        r"\b(?:system|application|server|host|database)\s+(?:was\s+|were\s+|has\s+been\s+)?compromised\b",
        r"\b(?:attacker|intruder)\s+(?:obtained|gained)\s+(?:admin|administrator|root)\b",
    )
)
_CONFIRMATION = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:root cause|vulnerability|sql injection|injection|exploitation|attack)\b[^.;\n]{0,40}\b(?:is|was|has been|have been|were|are)\s+confirmed\b",
        r"\bconfirmed\s+(?:root cause|vulnerability|sql injection|exploitation|attack|compromise)\b",
        r"\bconfirms\s+(?:that\s+)?(?:the\s+)?(?:root cause|vulnerability|attack|exploitation|sql injection)\b",
        r"\bhas\s+been\s+(?:proven|confirmed)\b",
    )
)
# Matches the END of the text just before a pattern hit when that text is an explicit hedge
# ("no evidence of <hit>", "does not establish <hit>", "not <hit>"). Deliberately narrow: a bare
# "no"/"not" only counts when at most one word separates it from the hit.
_NEGATION_BEFORE = re.compile(
    r"(?:"
    r"no\s+(?:evidence|indication|sign|proof)s?\s+(?:of|that)\s+(?:\w+\s+){0,2}"
    r"|(?:does|do|did)\s+not\s+(?:establish|prove|show|confirm|demonstrate)\s+(?:that\s+)?(?:\w+\s+){0,2}"
    r"|(?:cannot|can't|can\s+not)\s+(?:confirm|establish|prove)\s+(?:that\s+)?(?:\w+\s+){0,2}"
    r"|not\s+(?:evidence\s+of|shown|established|demonstrated|observed|proven|confirmed)\s+(?:\w+\s+){0,2}"
    r"|without\s+evidence\s+of\s+(?:\w+\s+){0,2}"
    r"|(?:has|have|had|was|were|is|are)\s+not\s+(?:been\s+)?(?:\w+\s+)?"
    r"|never\s+(?:\w+\s+)?"
    r"|(?:no|not|without)\s+(?:\w+\s+)?"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class CitationValidation:
    passed: bool
    issues: tuple[ValidationIssue, ...]
    note: str = VALIDATION_NOTE

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


class InvalidAnswerError(ValueError):
    """The model response is not a JSON object."""


def parse_answer_json(response: str) -> dict[str, Any]:
    """Strict JSON parse of a model response (surrounding whitespace only; no code fences)."""
    if not isinstance(response, str) or not response.strip():
        raise InvalidAnswerError("the model returned an empty response")
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise InvalidAnswerError(f"the model response is not valid JSON ({exc.msg} at position {exc.pos})") from exc
    if not isinstance(value, dict):
        raise InvalidAnswerError("the model response must be a JSON object")
    return value


# Hedge AFTER a phrase: "successful exploitation is not confirmed / cannot be established".
_NEGATION_AFTER = re.compile(
    r"^\s*(?:(?:is|are|was|were|has\s+been|have\s+been)\s+(?:not|never)(?:\s+been)?"
    r"|cannot\s+be|can't\s+be|can\s+not\s+be)\s+"
    r"(?:confirmed|established|shown|demonstrated|proven|observed|evidenced|supported)\b",
    re.IGNORECASE,
)


def _is_negated(text: str, start: int, end: int | None = None) -> bool:
    if _NEGATION_BEFORE.search(text[max(0, start - 70):start]) is not None:
        return True
    return end is not None and _NEGATION_AFTER.match(text[end:end + 60]) is not None


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ValidationIssue] = []

    def add(self, code: str, location: str, message: str) -> None:
        self.issues.append(ValidationIssue(code, location, message))


def _check_object(value: Any, keys: tuple[str, ...], where: str, out: _Collector) -> bool:
    if not isinstance(value, dict):
        out.add("invalid_structure", where, f"must be an object, got {type(value).__name__}")
        return False
    missing = [key for key in keys if key not in value]
    extra = sorted(str(key) for key in value if key not in keys)
    for key in missing:
        out.add("missing_key", where, f"missing required key {key!r}")
    for key in extra:
        out.add("unknown_key", where, f"unexpected key {key!r}")
    return not missing


def _string_list(value: Any, where: str, out: _Collector) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        out.add("invalid_structure", where, "must be an array of non-empty strings")
        return []
    return list(value)


def _statement(value: Any, where: str, out: _Collector) -> str:
    if not isinstance(value, str) or not value.strip():
        out.add("invalid_structure", where, "statement must be a non-empty string")
        return ""
    if len(value) > MAX_STATEMENT_CHARS:
        out.add("statement_too_long", where, f"statement exceeds {MAX_STATEMENT_CHARS} characters")
        return ""
    return value


def _identifiers(text: str) -> set[str]:
    return {match.group(0).upper() for match in _IDENTIFIER_RE.finditer(text)}


# ============================================================
# CONTEXT FACTS USED BY THE CHECKS
# ============================================================

class _Facts:
    """What a context makes citable, precomputed once."""

    def __init__(self, context: GroundedContext) -> None:
        self.context = context
        self.evidence = {item.evidence_id: item for item in context.observed_evidence}
        self.unresolved = set(context.unresolved_evidence_ids)
        self.references = {ref.reference_id: ref for ref in context.references}
        self.hypotheses = {item.hypothesis_id: item for item in context.hypotheses}
        self.any_confirmed = any(item.is_confirmed for item in context.hypotheses)

        self.hypothesis_identifiers: set[str] = set()
        for item in context.hypotheses:
            for text in (item.vulnerability_type, item.root_cause, item.affected_component, item.rationale,
                         *item.evidence_needed_to_confirm):
                if text:
                    self.hypothesis_identifiers |= _identifiers(text)
        self.reference_identifiers = {
            ref_id: {ref.source_id.upper()} | _identifiers(ref.text) | _identifiers(ref.title)
            for ref_id, ref in self.references.items()
        }
        self.global_identifiers = set(self.hypothesis_identifiers)
        for identifiers in self.reference_identifiers.values():
            self.global_identifiers |= identifiers

    def cited_identifiers(self, reference_ids: set[str]) -> set[str]:
        found: set[str] = set()
        for ref_id in reference_ids:
            found |= self.reference_identifiers.get(ref_id, set())
        return found

    def requests(self, evidence_ids: set[str]) -> list[dict[str, str | None]]:
        return [self.evidence[e].request for e in sorted(evidence_ids) if e in self.evidence]


# ============================================================
# STATEMENT TEXT CHECKS
# ============================================================

def _check_http_facts(text: str, where: str, requests: list[dict[str, str | None]], out: _Collector) -> None:
    methods = {str(r.get("method")).upper() for r in requests if r.get("method")}
    statuses = {str(r.get(k)) for r in requests for k in ("status", "upstream_status") if r.get(k)}
    paths = {str(r.get("path")) for r in requests if r.get("path")}
    for method in sorted(set(_METHOD_RE.findall(text))):
        if method not in methods:
            out.add("http_fact_mismatch", where, f"mentions HTTP method {method} not found in the cited request evidence")
    for status in sorted(set(_STATUS_RE.findall(text))):
        if status not in statuses:
            out.add("http_fact_mismatch", where, f"mentions HTTP status {status} not found in the cited request evidence")
    for raw in sorted(set(_PATH_RE.findall(text))):
        path = raw.rstrip(".,;:)")
        if path not in {"", "/", "//"} and path not in paths:
            out.add("http_fact_mismatch", where, f"mentions path {path!r} not found in the cited request evidence")


def _check_prose(
    text: str,
    where: str,
    out: _Collector,
    facts: _Facts,
    *,
    own_evidence: set[str],
    own_references: set[str],
    allowed_identifiers: set[str],
    requests: list[dict[str, str | None]] | None,
) -> None:
    for token in sorted(set(_EVIDENCE_TOKEN_RE.findall(text))):
        if token not in facts.evidence:
            out.add("unknown_evidence_id", where, f"the text mentions {token!r}, which is not a citable event of this context")
        elif token not in own_evidence:
            out.add("prose_citation_mismatch", where, f"the text mentions {token!r} but the item does not cite it")
    for token in sorted(set(_REFERENCE_TOKEN_RE.findall(text))):
        if token not in facts.references:
            out.add("unknown_reference_id", where, f"the text mentions {token!r}, which was not retrieved for this context")
        elif token not in own_references:
            out.add("prose_citation_mismatch", where, f"the text mentions {token!r} but the item does not cite it")

    for identifier in sorted(_identifiers(text)):
        if identifier not in allowed_identifiers:
            out.add("unsupported_identifier", where,
                    f"{identifier} is not stated by the hypothesis or by a passage this item cites")

    for pattern in _HIGH_IMPACT:
        for match in pattern.finditer(text):
            if not _is_negated(text, match.start(), match.end()):
                out.add("high_impact_claim", where,
                        f"asserts {match.group(0)!r}; no event in this context states that outcome")
    if not facts.any_confirmed:
        for pattern in _CONFIRMATION:
            for match in pattern.finditer(text):
                if not _is_negated(text, match.start()):
                    out.add("confirmation_claim", where,
                            f"presents {match.group(0)!r} as confirmed, but the finding is only a hypothesis")
    if requests is not None:
        _check_http_facts(text, where, requests, out)


def _check_evidence_ids(ids: list[str], where: str, out: _Collector, facts: _Facts) -> set[str]:
    valid: set[str] = set()
    for evidence_id in ids:
        if evidence_id in facts.evidence:
            valid.add(evidence_id)
        elif evidence_id in facts.references:
            out.add("category_mixing", where, f"{evidence_id!r} is a reference passage, not incident evidence")
        elif evidence_id in facts.unresolved:
            out.add("unknown_evidence_id", where, f"{evidence_id!r} has no description in the reconstruction and is not citable")
        else:
            out.add("unknown_evidence_id", where, f"{evidence_id!r} is not an event of this context")
    return valid


def _check_reference_ids(
    ids: list[str], where: str, out: _Collector, facts: _Facts, not_relevant: set[str]
) -> set[str]:
    valid: set[str] = set()
    for reference_id in ids:
        if reference_id in facts.references:
            if reference_id in not_relevant:
                out.add("not_relevant_reference_cited", where,
                        f"{reference_id!r} was assessed as not relevant and may not be cited")
            else:
                valid.add(reference_id)
        elif reference_id in facts.evidence:
            out.add("category_mixing", where, f"{reference_id!r} is incident evidence, not a reference passage")
        else:
            out.add("unknown_reference_id", where, f"{reference_id!r} was not retrieved for this context")
    return valid


# ============================================================
# SECTIONS
# ============================================================

def _relevance_assessments(answer: dict, out: _Collector, facts: _Facts) -> dict[str, str]:
    """Validate `reference_relevance`; return {reference_id: assessment} for the valid entries."""
    entries = answer.get("reference_relevance")
    if not isinstance(entries, list):
        out.add("invalid_structure", "reference_relevance", "must be an array")
        return {}
    assessed: dict[str, str] = {}
    for index, entry in enumerate(entries):
        where = f"reference_relevance[{index}]"
        if not _check_object(entry, _RELEVANCE_KEYS, where, out):
            continue
        reference_id, assessment = entry.get("reference_id"), entry.get("assessment")
        if assessment not in RELEVANCE_ASSESSMENTS:
            out.add("invalid_value", where, f"assessment must be one of {list(RELEVANCE_ASSESSMENTS)}")
            continue
        if not isinstance(reference_id, str) or reference_id not in facts.references:
            out.add("unknown_reference_id", where, f"{reference_id!r} was not retrieved for this context")
            continue
        if reference_id in assessed:
            out.add("duplicate_assessment", where, f"{reference_id!r} is assessed more than once")
            continue
        assessed[reference_id] = assessment
    for reference_id in facts.references:
        if reference_id not in assessed:
            out.add("missing_assessment", "reference_relevance", f"retrieved passage {reference_id!r} is not assessed")
    return assessed


def _observed_section(answer: dict, out: _Collector, facts: _Facts) -> None:
    items = answer.get("observed_evidence")
    if not isinstance(items, list):
        out.add("invalid_structure", "observed_evidence", "must be an array")
        return
    for index, item in enumerate(items):
        where = f"observed_evidence[{index}]"
        if not _check_object(item, _OBSERVED_KEYS, where, out):
            continue
        text = _statement(item.get("statement"), where, out)
        ids = _string_list(item.get("evidence_ids"), f"{where}.evidence_ids", out)
        if not ids:
            out.add("missing_citation", where, "an observed statement must cite at least one evidence_id")
        valid = _check_evidence_ids(ids, f"{where}.evidence_ids", out, facts)
        if text:
            _check_prose(text, where, out, facts, own_evidence=valid, own_references=set(),
                         allowed_identifiers=set(), requests=facts.requests(valid))


def _hypothesis_section(answer: dict, out: _Collector, facts: _Facts, not_relevant: set[str]) -> None:
    items = answer.get("hypotheses")
    if not isinstance(items, list):
        out.add("invalid_structure", "hypotheses", "must be an array")
        return
    seen: set[str] = set()
    for index, item in enumerate(items):
        where = f"hypotheses[{index}]"
        if not _check_object(item, _HYPOTHESIS_KEYS, where, out):
            continue
        hypothesis_id = item.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or hypothesis_id not in facts.hypotheses:
            out.add("unknown_hypothesis_id", where, f"{hypothesis_id!r} is not a hypothesis of this context")
        elif hypothesis_id in seen:
            out.add("duplicate_hypothesis", where, f"{hypothesis_id!r} is addressed more than once")
        else:
            seen.add(hypothesis_id)
        status = item.get("status")
        if status not in HYPOTHESIS_STATUSES:
            out.add("invalid_value", where, f"status must be one of {list(HYPOTHESIS_STATUSES)}")
        text = _statement(item.get("statement"), where, out)
        evidence_ids = _string_list(item.get("evidence_ids"), f"{where}.evidence_ids", out)
        reference_ids = _string_list(item.get("reference_ids"), f"{where}.reference_ids", out)
        valid_evidence = _check_evidence_ids(evidence_ids, f"{where}.evidence_ids", out, facts)
        valid_references = _check_reference_ids(reference_ids, f"{where}.reference_ids", out, facts, not_relevant)
        if status == "hypothesis_only" and not valid_evidence:
            out.add("missing_citation", where, "a hypothesis_only statement must cite at least one evidence_id")
        if text:
            _check_prose(text, where, out, facts, own_evidence=valid_evidence, own_references=valid_references,
                         allowed_identifiers=facts.hypothesis_identifiers | facts.cited_identifiers(valid_references),
                         requests=facts.requests(valid_evidence))
    for hypothesis_id in facts.hypotheses:
        if hypothesis_id not in seen:
            out.add("missing_hypothesis", "hypotheses", f"hypothesis {hypothesis_id!r} is not addressed")


def _guidance_section(answer: dict, out: _Collector, facts: _Facts, not_relevant: set[str]) -> None:
    items = answer.get("reference_guidance")
    if not isinstance(items, list):
        out.add("invalid_structure", "reference_guidance", "must be an array")
        return
    if items and not facts.references:
        out.add("no_reference_material", "reference_guidance",
                "no reference passage was retrieved, so no reference guidance may be given")
    for index, item in enumerate(items):
        where = f"reference_guidance[{index}]"
        if not _check_object(item, _GUIDANCE_KEYS, where, out):
            continue
        text = _statement(item.get("statement"), where, out)
        ids = _string_list(item.get("reference_ids"), f"{where}.reference_ids", out)
        if not ids:
            out.add("missing_citation", where, "a guidance statement must cite at least one reference_id")
        valid = _check_reference_ids(ids, f"{where}.reference_ids", out, facts, not_relevant)
        if text:
            _check_prose(text, where, out, facts, own_evidence=set(), own_references=valid,
                         allowed_identifiers=facts.cited_identifiers(valid), requests=None)


def _free_text_section(answer: dict, key: str, out: _Collector, facts: _Facts) -> None:
    value = answer.get(key)
    if not isinstance(value, list):
        out.add("invalid_structure", key, "must be an array of strings")
        return
    for index, item in enumerate(value):
        where = f"{key}[{index}]"
        text = _statement(item, where, out)
        if text:
            _check_prose(text, where, out, facts, own_evidence=set(facts.evidence), own_references=set(facts.references),
                         allowed_identifiers=facts.global_identifiers, requests=None)


# ============================================================
# PUBLIC API
# ============================================================

def validate_answer(answer: object, context: GroundedContext) -> CitationValidation:
    """Validate a parsed answer against `context`; returns every issue found (see the module docstring)."""
    out = _Collector()
    if not isinstance(answer, dict):
        out.add("invalid_structure", "answer", f"must be an object, got {type(answer).__name__}")
        return CitationValidation(False, tuple(out.issues))
    facts = _Facts(context)

    _check_object(answer, ANSWER_KEYS, "answer", out)
    if answer.get("schema_version") != ANSWER_SCHEMA_VERSION:
        out.add("invalid_value", "schema_version", f"must be {ANSWER_SCHEMA_VERSION!r}")
    if answer.get("incident_id") != context.incident_id:
        out.add("wrong_incident", "incident_id", f"must be {context.incident_id!r}")

    assessed = _relevance_assessments(answer, out, facts)
    not_relevant = {ref_id for ref_id, value in assessed.items() if value == "not_relevant"}
    _observed_section(answer, out, facts)
    _hypothesis_section(answer, out, facts, not_relevant)
    _guidance_section(answer, out, facts, not_relevant)
    for key in ("evidence_gaps", "limitations"):
        _free_text_section(answer, key, out, facts)
    return CitationValidation(not out.issues, tuple(out.issues))
