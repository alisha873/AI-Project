"""
Phase 11 (Milestone 11B) - the versioned retrieval benchmark, format `v2`.

`retrieval.evaluation` still loads and evaluates format v1 unchanged
(`retrieval_benchmark_v1.json` stays on disk and reproducible). This module
loads `retrieval_benchmark_v2.json`, which adds, per case:

- `split`        "development" (may be used to tune a retriever) or "heldout"
                 (run once, after the implementation is frozen). Cases carried
                 over from v1 are always development, because their results
                 had already been seen.
- `category`     what kind of need the question expresses (see CATEGORIES).
- `expectation`  "relevant_documents" (scored with Recall@k / MRR),
                 "low_information" or "no_relevant_documents" (behaviour cases:
                 never scored for recall, only checked for whether the retriever
                 refused, found nothing or flagged the query).
- `label_provenance`  who labelled it and what review happened: the author's
                 labels, any reviews, an adjudication, and a `review_status`.

Validation policy (strict; any breach raises `BenchmarkError` naming the case)
-----------------------------------------------------------------------------
- unknown, missing or duplicate JSON keys; wrong types;
- a malformed or duplicate `case_id`;
- `split`, `category`, `query_style`, `expectation`, `review_status` outside
  the documented sets; a carried-from-v1 case outside the development split;
- a scored case without labels, a behaviour case WITH labels, duplicate labels,
  or a label that is not a well-formed `<source>:<source_id>` document ID;
- a question containing a CWE-/CAPEC-/CVE- identifier, EXCEPT in category
  `identifier_lookup`, where every identifier must name a labelled record
  (identifiers are compared after Unicode normalization, so a lookalike hyphen
  cannot hide one);
- label provenance that contradicts the final labels (for example status
  `ai_review_agreed` while a review differs from the author, or an
  adjudication whose final labels differ from `relevant_doc_ids`);
- a `human_*` review status without a human reviewer (none exists yet);
- a `review_summary` whose counts do not match the cases.
`evaluate_v2` additionally refuses a label that is not in the evaluated corpus
(`UnknownLabelError`) and a corpus whose fingerprint differs from the one the
benchmark records.

Limits: a small, manually curated benchmark. It shows how a keyword baseline
behaves on these cases, not general retrieval quality, and none of its labels
has been reviewed by a human.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge.schema import make_doc_id
from retrieval.evaluation import BenchmarkError

BENCHMARK_FORMAT_V2 = "autoshield.retrieval_benchmark.v2"
SPLITS = ("development", "heldout")
CATEGORIES = (
    "forensic_observation",
    "terminology",
    "weakness_consequence",
    "attack_technique",
    "mitigation_detection",
    "disambiguation",
    "identifier_lookup",
    "low_information",
    "unanswerable",
)
QUERY_STYLES_V2 = ("descriptive", "terminology", "short", "identifier")
EXPECTATIONS = ("relevant_documents", "low_information", "no_relevant_documents")
REVIEW_STATUSES = (
    "author_only_awaiting_review",
    "ai_review_agreed",
    "ai_review_adjudicated",
    "human_review_agreed",
    "human_review_adjudicated",
    "not_applicable_no_labels",
)
ORIGINS = ("v1", "v2")

_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_IDENTIFIER_RE = re.compile(r"\b(CWE|CAPEC|CVE)[-_ ]([0-9]+)", re.IGNORECASE)
_DASH_RE = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ReviewRecord:
    reviewer_id: str
    reviewer_type: str
    method: str
    labels: tuple[str, ...]
    note: str

    @property
    def is_human(self) -> bool:
        return self.reviewer_type.startswith("human")


@dataclass(frozen=True)
class Adjudication:
    decision: str
    reason: str
    rationale: str
    final_labels: tuple[str, ...]
    contested_doc_ids: tuple[str, ...]
    adjudicated_by: str
    independent: bool
    awaiting_human_adjudication: bool


@dataclass(frozen=True)
class LabelProvenance:
    authored_by: str
    author_labels: tuple[str, ...]
    reviews: tuple[ReviewRecord, ...]
    adjudication: Adjudication | None
    review_status: str


@dataclass(frozen=True)
class BenchmarkCaseV2:
    case_id: str
    split: str
    origin: str
    category: str
    query_style: str
    question: str
    task: str
    expectation: str
    relevant_doc_ids: tuple[str, ...]
    rationale: str
    provenance: LabelProvenance

    @property
    def scored(self) -> bool:
        return self.expectation == "relevant_documents"

    @property
    def agreed_doc_ids(self) -> tuple[str, ...]:
        """Labels given by the author AND every reviewer (the strict set used for sensitivity checks)."""
        agreed = set(self.provenance.author_labels)
        for review in self.provenance.reviews:
            agreed &= set(review.labels)
        return tuple(label for label in self.provenance.author_labels if label in agreed)

    @property
    def contested_doc_ids(self) -> tuple[str, ...]:
        """Documents on which the author and some reviewer disagreed (empty if none)."""
        if self.provenance.adjudication is not None:
            return self.provenance.adjudication.contested_doc_ids
        return ()


@dataclass(frozen=True)
class BenchmarkV2:
    benchmark_id: str
    supersedes: str
    description: str
    protocol: str
    corpus: Mapping[str, Any]
    cases: tuple[BenchmarkCaseV2, ...]

    @property
    def corpus_fingerprint(self) -> str:
        return self.corpus["corpus_fingerprint"]

    def select(self, *, split: str | None = None, scored: bool | None = None) -> tuple[BenchmarkCaseV2, ...]:
        return tuple(
            case for case in self.cases
            if (split is None or case.split == split) and (scored is None or case.scored == scored)
        )


# ============================================================
# LOADER: parsing helpers
# ============================================================

def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError(f"duplicate key {key!r} in the benchmark file")
        result[key] = value
    return result


def _exact_keys(value: object, where: str, keys: set[str]) -> dict:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{where} must be an object")
    missing, unknown = sorted(keys - value.keys()), sorted(value.keys() - keys)
    if missing:
        raise BenchmarkError(f"{where} is missing key(s) {missing}")
    if unknown:
        raise BenchmarkError(f"{where} has unknown key(s) {unknown}")
    return value


def _text(value: object, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise BenchmarkError(f"{where} must be a non-empty string")
    return value


def _label_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise BenchmarkError(f"{where} must be a list")
    labels = tuple(value)
    for label in labels:
        if not isinstance(label, str) or not label:
            raise BenchmarkError(f"{where} must contain non-empty strings")
        source, _, source_id = label.partition(":")
        try:
            make_doc_id(source, source_id)
        except ValueError as exc:
            raise BenchmarkError(f"{where}: invalid document ID {label!r}: {exc}") from exc
    if len(set(labels)) != len(labels):
        raise BenchmarkError(f"{where} contains duplicate labels")
    return labels


def _question_identifiers(question: str) -> set[str]:
    normalized = _DASH_RE.sub("-", unicodedata.normalize("NFKC", question))
    return {f"{m.group(1).upper()}-{m.group(2)}" for m in _IDENTIFIER_RE.finditer(normalized)}


def _parse_review(raw: object, where: str) -> ReviewRecord:
    data = _exact_keys(raw, where, {"reviewer_id", "reviewer_type", "method", "labels", "note"})
    return ReviewRecord(
        reviewer_id=_text(data["reviewer_id"], f"{where}.reviewer_id"),
        reviewer_type=_text(data["reviewer_type"], f"{where}.reviewer_type"),
        method=_text(data["method"], f"{where}.method"),
        labels=_label_list(data["labels"], f"{where}.labels"),
        note=_text(data["note"], f"{where}.note", allow_empty=True),
    )


def _parse_adjudication(raw: object, where: str) -> Adjudication:
    data = _exact_keys(
        raw, where,
        {"decision", "reason", "rationale", "final_labels", "contested_doc_ids", "adjudicated_by",
         "independent", "awaiting_human_adjudication"},
    )
    if not isinstance(data["independent"], bool) or not isinstance(data["awaiting_human_adjudication"], bool):
        raise BenchmarkError(f"{where}: independent and awaiting_human_adjudication must be booleans")
    return Adjudication(
        decision=_text(data["decision"], f"{where}.decision"),
        reason=_text(data["reason"], f"{where}.reason"),
        rationale=_text(data["rationale"], f"{where}.rationale"),
        final_labels=_label_list(data["final_labels"], f"{where}.final_labels"),
        contested_doc_ids=_label_list(data["contested_doc_ids"], f"{where}.contested_doc_ids"),
        adjudicated_by=_text(data["adjudicated_by"], f"{where}.adjudicated_by"),
        independent=data["independent"],
        awaiting_human_adjudication=data["awaiting_human_adjudication"],
    )


# ============================================================
# LOADER: label provenance, cases, file
# ============================================================

def _parse_provenance(raw: object, where: str, final: tuple[str, ...], scored: bool) -> LabelProvenance:
    data = _exact_keys(raw, where, {"authored_by", "author_labels", "reviews", "adjudication", "review_status"})
    status = data["review_status"]
    if status not in REVIEW_STATUSES:
        raise BenchmarkError(f"{where}.review_status must be one of {REVIEW_STATUSES}, got {status!r}")
    if not isinstance(data["reviews"], list):
        raise BenchmarkError(f"{where}.reviews must be a list")
    author = _label_list(data["author_labels"], f"{where}.author_labels")
    reviews = tuple(_parse_review(r, f"{where}.reviews[{i}]") for i, r in enumerate(data["reviews"]))
    adjudication = (
        None if data["adjudication"] is None
        else _parse_adjudication(data["adjudication"], f"{where}.adjudication")
    )

    if scored and not author:
        raise BenchmarkError(f"{where}: a scored case needs author labels")
    if not scored and author:
        raise BenchmarkError(f"{where}: a behaviour case must not have author labels")
    if status.startswith("human") and not any(r.is_human for r in reviews):
        raise BenchmarkError(f"{where}: status {status!r} requires a human reviewer, and none is recorded")

    differs = [r for r in reviews if set(r.labels) != set(author)]
    if status == "author_only_awaiting_review":
        if reviews or adjudication is not None:
            raise BenchmarkError(f"{where}: an author-only case cannot have reviews or an adjudication")
        expected = author
    elif status == "not_applicable_no_labels":
        if scored or reviews or adjudication is not None:
            raise BenchmarkError(f"{where}: status {status!r} is only for unscored cases without reviews")
        expected = ()
    elif status.endswith("_agreed"):
        if not reviews or differs or adjudication is not None:
            raise BenchmarkError(
                f"{where}: status {status!r} needs reviews that all equal the author labels and no adjudication"
            )
        expected = author
    else:  # *_adjudicated
        if not reviews or not differs or adjudication is None:
            raise BenchmarkError(f"{where}: status {status!r} needs a differing review and an adjudication")
        contested: set[str] = set()
        for review in reviews:
            contested |= set(author) ^ set(review.labels)
        if set(adjudication.contested_doc_ids) != contested:
            raise BenchmarkError(f"{where}: contested_doc_ids do not match the author/review differences")
        expected = adjudication.final_labels
    if set(final) != set(expected):
        raise BenchmarkError(
            f"{where}: relevant_doc_ids {list(final)} contradict the recorded provenance {list(expected)}"
        )
    return LabelProvenance(_text(data["authored_by"], f"{where}.authored_by"), author, reviews, adjudication, status)


def _build_case(raw: object, index: int) -> BenchmarkCaseV2:
    keys = {"case_id", "split", "origin", "category", "query_style", "question", "task", "expectation",
            "relevant_doc_ids", "rationale", "label_provenance"}
    data = _exact_keys(raw, f"cases[{index}]", keys)
    case_id = data["case_id"]
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise BenchmarkError(f"cases[{index}]: invalid case_id {case_id!r}")
    where = f"case {case_id!r}"
    for key, allowed in (("split", SPLITS), ("origin", ORIGINS), ("category", CATEGORIES),
                         ("query_style", QUERY_STYLES_V2), ("expectation", EXPECTATIONS)):
        if data[key] not in allowed:
            raise BenchmarkError(f"{where}: {key} must be one of {allowed}, got {data[key]!r}")
    if data["origin"] == "v1" and data["split"] != "development":
        raise BenchmarkError(f"{where}: a case carried over from v1 must be in the development split")
    question = _text(data["question"], f"{where}: question")
    task = _text(data["task"], f"{where}: task")
    rationale = _text(data["rationale"], f"{where}: rationale")
    labels = _label_list(data["relevant_doc_ids"], f"{where}: relevant_doc_ids")
    scored = data["expectation"] == "relevant_documents"
    if scored and not labels:
        raise BenchmarkError(f"{where}: a scored case needs at least one relevant document")
    if not scored and labels:
        raise BenchmarkError(f"{where}: a behaviour case ({data['expectation']}) must have no relevant documents")
    if (data["category"] in ("low_information", "unanswerable")) == scored:
        raise BenchmarkError(f"{where}: category {data['category']!r} does not fit expectation {data['expectation']!r}")

    named = _question_identifiers(question)
    if named and data["category"] != "identifier_lookup":
        raise BenchmarkError(
            f"{where}: the question contains identifier(s) {sorted(named)}; only category "
            "identifier_lookup may name identifiers"
        )
    if data["category"] == "identifier_lookup":
        labelled = {label.partition(":")[2].upper() for label in labels}
        if not named or not named <= labelled:
            raise BenchmarkError(f"{where}: an identifier_lookup question must name only labelled records")
    provenance = _parse_provenance(data["label_provenance"], f"{where}: label_provenance", labels, scored)
    return BenchmarkCaseV2(
        case_id, data["split"], data["origin"], data["category"], data["query_style"], question, task,
        data["expectation"], labels, rationale, provenance,
    )


def load_benchmark_v2(path: str | Path) -> BenchmarkV2:
    """Read and strictly validate a format-v2 benchmark file (see the module docstring)."""
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except OSError as exc:
        raise BenchmarkError(f"cannot read benchmark {file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"benchmark {file} is not valid JSON: {exc}") from exc

    top = _exact_keys(raw, "benchmark", {
        "format", "benchmark_id", "supersedes", "description", "protocol", "corpus", "splits",
        "annotation_guide", "review_summary", "cases"})
    if top["format"] != BENCHMARK_FORMAT_V2:
        raise BenchmarkError(f"format must be {BENCHMARK_FORMAT_V2!r}, got {top['format']!r}")
    for key in ("benchmark_id", "supersedes", "description", "protocol", "annotation_guide"):
        _text(top[key], key)
    corpus = _exact_keys(top["corpus"], "corpus", {
        "cwe_version", "capec_version", "is_complete_catalogue", "document_count", "manifest",
        "corpus_fingerprint"})
    fingerprint = corpus["corpus_fingerprint"]
    if not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
        raise BenchmarkError("corpus.corpus_fingerprint must be a 64-character lower-case hex SHA-256")
    splits = _exact_keys(top["splits"], "splits", set(SPLITS))
    for name in SPLITS:
        _text(splits[name], f"splits.{name}")
    if not isinstance(top["cases"], list) or not top["cases"]:
        raise BenchmarkError("cases must be a non-empty list")

    cases = tuple(_build_case(entry, index) for index, entry in enumerate(top["cases"]))
    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise BenchmarkError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)

    summary = _exact_keys(top["review_summary"], "review_summary", {
        "human_review", "ai_review_passes", "note", "items_reviewed", "exact_set_agreement",
        "adjudicated", "adjudicated_case_ids"})
    reviewed = sum(1 for c in cases if c.provenance.reviews)
    agreed = sum(1 for c in cases if c.provenance.review_status.endswith("_agreed"))
    adjudicated = [c.case_id for c in cases if c.provenance.review_status.endswith("_adjudicated")]
    if (summary["items_reviewed"], summary["exact_set_agreement"], summary["adjudicated"],
            summary["adjudicated_case_ids"]) != (reviewed, agreed, len(adjudicated), adjudicated):
        raise BenchmarkError("review_summary counts do not match the cases")
    any_human = any(r.is_human for c in cases for r in c.provenance.reviews)
    if (summary["human_review"] == "not_performed") == any_human:
        raise BenchmarkError("review_summary.human_review contradicts the recorded reviewers")
    return BenchmarkV2(top["benchmark_id"], top["supersedes"], top["description"], top["protocol"], corpus, cases)
