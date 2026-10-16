"""
Phase 11 (Milestone 11A) - deterministic retrieval queries from Phase 8/9 output.

Input: the plain JSON dictionaries the Phase 8 and Phase 9 scripts write
(`analysis/root_cause_analysis.json`, `analysis/incident_reconstruction.json`;
see `scripts/run_incident_analysis.py`). Neither has a formal schema yet, so
this module reads only the fields listed in the `*_FIELDS_READ` constants,
validates their types, and treats missing, empty or null optional fields as
absent: it never invents content.

One query is built per root-cause finding, from these fields, in this order
(each becomes a `QueryComponent` that records where it came from and what kind
of statement it is):

| order | field                                         | basis                           |
|-------|-----------------------------------------------|---------------------------------|
| 1     | finding.vulnerability_type                    | inferred_hypothesis / confirmed |
| 2     | finding.root_cause                            | inferred_hypothesis / confirmed |
| 3     | finding.affected_component                    | inferred_hypothesis / confirmed |
| 4     | reconstruction claims citing the finding's    | observed_evidence               |
|       | evidence IDs (`statement`); else the timeline |                                 |
|       | `action` of each supporting event             |                                 |
| 5     | finding.rationale                             | inferred_hypothesis / confirmed |
| 6     | finding.evidence_needed_to_confirm (each)     | evidence_gap                    |

Rules that keep the query honest
--------------------------------
- Hypothesis vs finding. A finding whose `status` is not exactly "confirmed" is
  a HYPOTHESIS; its components carry basis "inferred_hypothesis" and the
  `IncidentQuery` says `is_confirmed == False`. Retrieval results are reference
  knowledge; they never confirm the hypothesis (`RESULT_MEANING`).
- Placeholders are not content. Phase 9 stores literal strings such as
  "Not confirmed from the available evidence." (root_cause) and
  "Undetermined from request evidence alone." (affected_component), or
  "unknown" when it has nothing. They are detected by `is_placeholder` and
  dropped. A finding with no informative subject (no usable vulnerability type,
  root cause, component or observed evidence) produces NO query and is listed
  in `QueryBuildResult.skipped`; rationale and evidence gaps alone are not a
  subject, so the `insufficient_evidence` boilerplate never becomes a query.
- Raw attack payloads are not copied into queries. Claim statements describe
  the observation in words ("... contains SQL-injection-like syntax in
  /search"); the percent-encoded query string is not natural-language
  knowledge-base vocabulary and would only add noise. (A design choice, not a
  claim that payload text is never useful.)
- Evidence IDs (for example "nginx-19") are removed from claim text; they are
  not knowledge-base vocabulary. They are kept in `IncidentQuery.evidence_ids`.
- Explicit identifiers the ANALYSIS itself states (for example "CWE-89" inside
  Phase 9's `vulnerability_type`) stay in the text; the keyword backend turns
  them into phrase terms. They are the analysis's hypothesis, not an answer key.
- The LLM narrative (`llm_incident_narrative.json`) is model-written text and is
  NOT a query source.

Separation from evaluation
--------------------------
This module imports nothing from `retrieval.evaluation` and never reads a
benchmark, a relevance label or any `ground_truth` data. `load_analysis_artifacts`
opens exactly two named files under `<incident>/analysis/`. As defence in
depth, input dictionaries that contain a key from `FORBIDDEN_INPUT_KEYS` at
their top level (or at the top level of a finding) are refused, not ignored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from retrieval.contract import (
    DEFAULT_LIMIT,
    MAX_QUERY_TEXT_CHARS,
    RESULT_MEANING,
    RetrievalError,
    RetrievalQuery,
)

ROOT_CAUSE_FILENAME = "root_cause_analysis.json"
RECONSTRUCTION_FILENAME = "incident_reconstruction.json"

# The ONLY keys this module reads (a test checks that nothing else is accessed).
ROOT_CAUSE_FIELDS_READ = frozenset({"incident_id", "findings"})
FINDING_FIELDS_READ = frozenset(
    {
        "status",
        "root_cause",
        "affected_component",
        "vulnerability_type",
        "supporting_evidence_ids",
        "rationale",
        "evidence_needed_to_confirm",
    }
)
RECONSTRUCTION_FIELDS_READ = frozenset({"incident_id", "claims", "timeline"})
CLAIM_FIELDS_READ = frozenset({"claim_id", "statement", "evidence_ids"})
TIMELINE_FIELDS_READ = frozenset({"event_id", "action"})

# Keys that would indicate an evaluation answer key; input containing one is refused.
FORBIDDEN_INPUT_KEYS = frozenset(
    {
        "ground_truth",
        "expected_answer",
        "expected_answers",
        "answer_key",
        "relevance_labels",
        "relevant_doc_ids",
        "relevant_source_ids",
    }
)

# Words Phase 9 uses to hedge a type name; removed from the type PHRASE only.
_HEDGE_WORDS = frozenset({"potential", "possible", "suspected", "likely", "probable", "hypothesis"})
# Leading words of the placeholder strings Phase 9 emits today. Whole-word match, so a real
# name such as "Unknownhost proxy" is NOT a placeholder; "Unknown; no evidence" is.
_PLACEHOLDER_RE = re.compile(
    r"(?:unknown|undetermined|not confirmed|not determined|not available|n/a|none)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

Basis = Literal["inferred_hypothesis", "confirmed_finding", "observed_evidence", "evidence_gap"]


# ============================================================
# ERRORS AND VALUE TYPES
# ============================================================

class QueryBuildError(RetrievalError, ValueError):
    """The analysis input is malformed, inconsistent or unreadable."""


class ForbiddenQueryInputError(QueryBuildError):
    """The input contains an evaluation-answer key; runtime queries must never use one."""


@dataclass(frozen=True)
class QueryComponent:
    """One piece of query text, with its origin and the kind of statement it is."""

    field_path: str
    text: str
    basis: Basis


@dataclass(frozen=True)
class IncidentQuery:
    """
    A retrieval query plus everything needed to explain it.

    `finding_status` is copied from Phase 9 unchanged; `is_confirmed` is True
    only when it is exactly "confirmed". `omitted_fields` lists components that
    did not fit within the query-text limit.
    """

    query: RetrievalQuery
    incident_id: str
    finding_index: int
    finding_status: str
    evidence_ids: tuple[str, ...]
    components: tuple[QueryComponent, ...]
    omitted_fields: tuple[str, ...] = ()

    @property
    def is_confirmed(self) -> bool:
        return self.finding_status == "confirmed"

    @property
    def meaning(self) -> str:
        return RESULT_MEANING


@dataclass(frozen=True)
class SkippedFinding:
    finding_index: int
    reason: str


@dataclass(frozen=True)
class QueryBuildResult:
    incident_id: str
    queries: tuple[IncidentQuery, ...]
    skipped: tuple[SkippedFinding, ...]


# ============================================================
# INPUT READING (type checks; absent optional fields are not an error)
# ============================================================

def is_placeholder(text: str | None) -> bool:
    """True for empty text and for the literal 'nothing known' strings Phase 9 emits."""
    if text is None or not text.strip():
        return True
    return _PLACEHOLDER_RE.match(text.strip()) is not None


def _mapping(value: object, where: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise QueryBuildError(f"{where} must be an object, got {type(value).__name__}")
    return value


def _refuse_forbidden_keys(mapping: Mapping, where: str) -> None:
    found = sorted(FORBIDDEN_INPUT_KEYS.intersection(str(key) for key in mapping))
    if found:
        raise ForbiddenQueryInputError(
            f"{where} contains evaluation-answer key(s) {found}; runtime retrieval queries "
            "must never be derived from ground truth or relevance labels"
        )


def _optional_text(mapping: Mapping, key: str, where: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise QueryBuildError(f"{where}.{key} must be a string or null, got {type(value).__name__}")
    return value if value.strip() else None


def _optional_text_list(mapping: Mapping, key: str, where: str) -> list[str]:
    value = mapping.get(key)
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise QueryBuildError(f"{where}.{key} must be a list of strings or null")
    if any(not isinstance(item, str) for item in value):
        raise QueryBuildError(f"{where}.{key} must contain only strings")
    return [item for item in value if item.strip()]


def _required_incident_id(mapping: Mapping, where: str) -> str:
    value = mapping.get("incident_id")
    if not isinstance(value, str) or not value.strip():
        raise QueryBuildError(f"{where}.incident_id must be a non-empty string")
    return value


# ============================================================
# TEXT HELPERS
# ============================================================

def _squash(text: str) -> str:
    return " ".join(text.split())


def _type_phrase(vulnerability_type: str) -> str | None:
    """'potential_sql_injection (CWE-89 hypothesis)' -> 'sql injection' (None if nothing is left)."""
    without_parentheses = re.sub(r"\([^)]*\)", " ", vulnerability_type)
    words = re.findall(r"[^\W_]+", without_parentheses.lower())
    words = [w for w in words if w not in _HEDGE_WORDS]
    return " ".join(words) if words else None


def _strip_ids(text: str, ids: set[str]) -> str:
    for evidence_id in sorted(ids, key=lambda value: (-len(value), value)):
        text = text.replace(evidence_id, " ")
    return _squash(text)


# ============================================================
# BUILDING
# ============================================================

def _observed_components(
    reconstruction: Mapping | None, supporting: list[str]
) -> list[QueryComponent]:
    """Claims that cite a supporting event; for uncovered events, the timeline action."""
    if reconstruction is None or not supporting:
        return []
    wanted = set(supporting)
    components: list[QueryComponent] = []
    covered: set[str] = set()

    claims = reconstruction.get("claims")
    if claims is not None and not isinstance(claims, (list, tuple)):
        raise QueryBuildError("reconstruction.claims must be a list or null")
    ordered_claims = []
    for index, raw in enumerate(claims or ()):
        claim = _mapping(raw, f"reconstruction.claims[{index}]")
        evidence_ids = _optional_text_list(claim, "evidence_ids", f"reconstruction.claims[{index}]")
        if wanted.intersection(evidence_ids):
            statement = _optional_text(claim, "statement", f"reconstruction.claims[{index}]")
            claim_id = _optional_text(claim, "claim_id", f"reconstruction.claims[{index}]") or f"#{index}"
            if statement:
                ordered_claims.append((claim_id, statement, evidence_ids))
    for claim_id, statement, evidence_ids in sorted(ordered_claims, key=lambda row: row[0]):
        text = _strip_ids(statement, set(evidence_ids) | wanted)
        if text:
            components.append(QueryComponent(f"claims[{claim_id}].statement", text, "observed_evidence"))
            covered.update(wanted.intersection(evidence_ids))

    uncovered = wanted - covered
    if uncovered:
        timeline = reconstruction.get("timeline")
        if timeline is not None and not isinstance(timeline, (list, tuple)):
            raise QueryBuildError("reconstruction.timeline must be a list or null")
        by_id: dict[str, str] = {}
        for index, raw in enumerate(timeline or ()):
            entry = _mapping(raw, f"reconstruction.timeline[{index}]")
            event_id = _optional_text(entry, "event_id", f"reconstruction.timeline[{index}]")
            action = _optional_text(entry, "action", f"reconstruction.timeline[{index}]")
            if event_id in uncovered and action and event_id not in by_id:
                by_id[event_id] = action
        for event_id in sorted(by_id):
            components.append(
                QueryComponent(f"timeline[{event_id}].action", _squash(by_id[event_id]), "observed_evidence")
            )
    return components


def _build_one(
    index: int, finding: Mapping, incident_id: str, reconstruction: Mapping | None, limit: int
) -> IncidentQuery | SkippedFinding:
    where = f"root_cause.findings[{index}]"
    _refuse_forbidden_keys(finding, where)

    status = _optional_text(finding, "status", where) or "unspecified"
    asserted: Basis = "confirmed_finding" if status == "confirmed" else "inferred_hypothesis"
    path = f"findings[{index}]"

    vulnerability_type = _optional_text(finding, "vulnerability_type", where)
    root_cause = _optional_text(finding, "root_cause", where)
    component = _optional_text(finding, "affected_component", where)
    rationale = _optional_text(finding, "rationale", where)
    gaps = _optional_text_list(finding, "evidence_needed_to_confirm", where)
    raw_supporting = _optional_text_list(finding, "supporting_evidence_ids", where)
    supporting = list(dict.fromkeys(raw_supporting))

    components: list[QueryComponent] = []
    phrases: list[str] = []
    if not is_placeholder(vulnerability_type):
        components.append(
            QueryComponent(f"{path}.vulnerability_type", _squash(vulnerability_type.replace("_", " ")), asserted)
        )
        phrase = _type_phrase(vulnerability_type)
        if phrase:
            phrases.append(phrase)
    if not is_placeholder(root_cause):
        components.append(QueryComponent(f"{path}.root_cause", _squash(root_cause), asserted))
    if not is_placeholder(component):
        components.append(QueryComponent(f"{path}.affected_component", _squash(component), asserted))
    components.extend(_observed_components(reconstruction, supporting))

    if not components:
        return SkippedFinding(index, "no_informative_subject")

    if rationale:
        components.append(QueryComponent(f"{path}.rationale", _squash(_strip_ids(rationale, set(supporting))), asserted))
    for gap_index, gap in enumerate(gaps):
        components.append(
            QueryComponent(f"{path}.evidence_needed_to_confirm[{gap_index}]", _squash(gap), "evidence_gap")
        )

    kept, omitted, used = [], [], 0
    overflowed = False
    for item in components:
        cost = len(item.text) + (1 if kept else 0)
        if not overflowed and used + cost <= MAX_QUERY_TEXT_CHARS:
            kept.append(item)
            used += cost
        elif not kept:  # even the first component is too long: truncate it
            kept.append(QueryComponent(item.field_path, item.text[:MAX_QUERY_TEXT_CHARS].rstrip(), item.basis))
            omitted.append(f"{item.field_path} (truncated)")
            used = len(kept[0].text)
            overflowed = True
        else:
            overflowed = True
            omitted.append(item.field_path)

    query = RetrievalQuery(
        text="\n".join(item.text for item in kept),
        phrases=tuple(dict.fromkeys(phrases)),
        limit=limit,
    )
    return IncidentQuery(
        query=query,
        incident_id=incident_id,
        finding_index=index,
        finding_status=status,
        evidence_ids=tuple(supporting),
        components=tuple(kept),
        omitted_fields=tuple(omitted),
    )


def build_retrieval_queries(
    root_cause: Mapping,
    reconstruction: Mapping | None = None,
    *,
    limit: int = DEFAULT_LIMIT,
) -> QueryBuildResult:
    """
    Build one `IncidentQuery` per usable root-cause finding.

    Args:
        root_cause: the Phase 9 dictionary (`root_cause_analysis.json`).
        reconstruction: optional Phase 8 dictionary (`incident_reconstruction.json`),
            used for observed-evidence descriptions. Its `incident_id` must match.
        limit: maximum documents per query (see `RetrievalQuery.limit`).

    Findings without an informative subject are listed in `skipped`. Raises
    `QueryBuildError` for malformed input and `ForbiddenQueryInputError` if an
    answer-key field is present. The result depends only on the input.
    """
    root_cause = _mapping(root_cause, "root_cause")
    _refuse_forbidden_keys(root_cause, "root_cause")
    incident_id = _required_incident_id(root_cause, "root_cause")

    if reconstruction is not None:
        reconstruction = _mapping(reconstruction, "reconstruction")
        _refuse_forbidden_keys(reconstruction, "reconstruction")
        other = _required_incident_id(reconstruction, "reconstruction")
        if other != incident_id:
            raise QueryBuildError(
                f"reconstruction is for {other!r} but the root-cause analysis is for {incident_id!r}"
            )

    findings = root_cause.get("findings")
    if not isinstance(findings, (list, tuple)):
        raise QueryBuildError("root_cause.findings must be a list")

    queries: list[IncidentQuery] = []
    skipped: list[SkippedFinding] = []
    for index, raw in enumerate(findings):
        finding = _mapping(raw, f"root_cause.findings[{index}]")
        outcome = _build_one(index, finding, incident_id, reconstruction, limit)
        (skipped if isinstance(outcome, SkippedFinding) else queries).append(outcome)
    return QueryBuildResult(incident_id, tuple(queries), tuple(skipped))


# ============================================================
# LOADING REAL ARTIFACTS
# ============================================================

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QueryBuildError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QueryBuildError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise QueryBuildError(f"{path} must contain a JSON object")
    return data


def load_analysis_artifacts(incident_dir: str | Path) -> tuple[dict, dict | None]:
    """
    Read `<incident_dir>/analysis/root_cause_analysis.json` (required) and
    `.../incident_reconstruction.json` (optional). These are the only two files
    this module ever opens; it does not list or read other directories (in
    particular not `ground_truth/`).
    """
    analysis = Path(incident_dir) / "analysis"
    root_cause = _read_json(analysis / ROOT_CAUSE_FILENAME)
    reconstruction_path = analysis / RECONSTRUCTION_FILENAME
    reconstruction = _read_json(reconstruction_path) if reconstruction_path.is_file() else None
    return root_cause, reconstruction


def build_queries_for_incident(incident_dir: str | Path, *, limit: int = DEFAULT_LIMIT) -> QueryBuildResult:
    """`load_analysis_artifacts` followed by `build_retrieval_queries`."""
    root_cause, reconstruction = load_analysis_artifacts(incident_dir)
    return build_retrieval_queries(root_cause, reconstruction, limit=limit)
