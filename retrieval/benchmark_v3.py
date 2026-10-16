"""
Phase 11 (Milestone 11C) - strict loader for the frozen final held-out benchmark, format `v3`.

`data/retrieval_benchmark/preflight_11c/benchmark_11c.json` was written and frozen before any 11C code existed.
Its SHA-256 is pinned in `retrieval.freeze.FROZEN_V3_SHA256`. This module is the ONLY retrieval module that
reads the v3 cases, and only the final-evaluation script imports it (a test enforces that the development
script and every other retrieval module do not).

Loading refuses a file whose SHA-256 differs from the pinned value (`require_frozen_hash`, on by default), so
an edited, regenerated or re-serialised copy cannot be mistaken for the frozen set. The loader never rewrites
the file.

The v3 file is converted to the existing v2 data classes (`BenchmarkV2`) so the already-tested evaluation code
(`retrieval.evaluation_v2.evaluate_v2`: source-level Recall@k, MRR, behaviour cases, paired comparison) is reused
unchanged:

- strict labels (`relevant_doc_ids`) are the PRIMARY label set;
- "strict plus lenient" (`+ lenient_extra_doc_ids`, other-lane records the sources themselves declare related)
  is a SENSITIVITY set only.

The v3 labels are single-pass labels by the AI assistant, grounded in verbatim source quotes. They are not blind,
not independently annotated and not human-reviewed. The adapter records this in `authored_by`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge.schema import make_doc_id
from retrieval.benchmark_v2 import (
    CATEGORIES,
    EXPECTATIONS,
    QUERY_STYLES_V2,
    BenchmarkCaseV2,
    BenchmarkV2,
    LabelProvenance,
    _question_identifiers,
)
from retrieval.evaluation import BenchmarkError
from retrieval.freeze import FROZEN_V3_SHA256, FrozenBenchmarkError

BENCHMARK_FORMAT_V3 = "autoshield.retrieval_benchmark.v3"
V3_SPLIT = "final_heldout"
LABEL_SETS = ("strict", "strict_plus_lenient")
AUTHOR = "ai_assistant_single_pass_not_independently_reviewed"

_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = {
    "format", "benchmark_id", "supersedes", "relation_to_previous", "description", "protocol", "corpus",
    "annotation_guide", "label_method", "label_audit", "review_summary", "label_provenance", "cases",
    "behaviour_cases",
}
_SCORED_KEYS = {
    "case_id", "split", "origin", "category", "query_style", "question", "task", "expectation",
    "relevant_doc_ids", "lenient_extra_doc_ids", "rationale", "evidence", "considered_not_relevant",
    "exclusivity_checks", "close_call", "review_status",
}
_BEHAVIOUR_KEYS = {
    "case_id", "split", "origin", "category", "query_style", "question", "task", "expectation",
    "relevant_doc_ids", "absence_checks", "rationale", "review_status",
}


@dataclass(frozen=True)
class CaseV3:
    case_id: str
    category: str
    query_style: str
    question: str
    task: str
    expectation: str
    relevant_doc_ids: tuple[str, ...]
    lenient_extra_doc_ids: tuple[str, ...]
    rationale: str
    close_call: bool

    @property
    def scored(self) -> bool:
        return self.expectation == "relevant_documents"


@dataclass(frozen=True)
class BenchmarkV3:
    benchmark_id: str
    description: str
    protocol: str
    corpus: dict[str, Any]
    cases: tuple[CaseV3, ...]
    sha256: str

    @property
    def corpus_fingerprint(self) -> str:
        return self.corpus["corpus_fingerprint"]

    def as_benchmark_v2(self, label_set: str = "strict") -> BenchmarkV2:
        """The cases as a `BenchmarkV2` for the shared evaluator, using the chosen label set."""
        if label_set not in LABEL_SETS:
            raise ValueError(f"label_set must be one of {LABEL_SETS}")
        cases = []
        for case in self.cases:
            labels = case.relevant_doc_ids
            if label_set == "strict_plus_lenient":
                labels = labels + tuple(x for x in case.lenient_extra_doc_ids if x not in labels)
            provenance = LabelProvenance(AUTHOR, labels, (), None, "author_only_awaiting_review" if labels
                                         else "not_applicable_no_labels")
            cases.append(BenchmarkCaseV2(
                case.case_id, V3_SPLIT, "v2", case.category, case.query_style, case.question, case.task,
                case.expectation, labels, case.rationale, provenance,
            ))
        return BenchmarkV2(self.benchmark_id, "none", self.description, self.protocol, self.corpus, tuple(cases))


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError(f"duplicate key {key!r} in the benchmark file")
        result[key] = value
    return result


def _exact(value: object, where: str, keys: set[str]) -> dict:
    if not isinstance(value, dict):
        raise BenchmarkError(f"{where} must be an object")
    missing, unknown = sorted(keys - value.keys()), sorted(value.keys() - keys)
    if missing:
        raise BenchmarkError(f"{where} is missing key(s) {missing}")
    if unknown:
        raise BenchmarkError(f"{where} has unknown key(s) {unknown}")
    return value


def _labels(value: object, where: str) -> tuple[str, ...]:
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


def _case(raw: object, index: int, behaviour: bool) -> CaseV3:
    data = _exact(raw, f"{'behaviour_cases' if behaviour else 'cases'}[{index}]",
                  _BEHAVIOUR_KEYS if behaviour else _SCORED_KEYS)
    case_id = data["case_id"]
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise BenchmarkError(f"invalid case_id {case_id!r}")
    where = f"case {case_id!r}"
    if data["split"] != V3_SPLIT:
        raise BenchmarkError(f"{where}: split must be {V3_SPLIT!r}")
    for key, allowed in (("category", CATEGORIES), ("query_style", QUERY_STYLES_V2), ("expectation", EXPECTATIONS)):
        if data[key] not in allowed:
            raise BenchmarkError(f"{where}: {key} must be one of {allowed}, got {data[key]!r}")
    for key in ("question", "task", "rationale"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise BenchmarkError(f"{where}: {key} must be a non-empty string")
    labels = _labels(data["relevant_doc_ids"], f"{where}: relevant_doc_ids")
    lenient = () if behaviour else _labels(data["lenient_extra_doc_ids"], f"{where}: lenient_extra_doc_ids")
    scored = data["expectation"] == "relevant_documents"
    if behaviour == scored:
        raise BenchmarkError(f"{where}: expectation {data['expectation']!r} does not fit its list")
    if scored and not labels:
        raise BenchmarkError(f"{where}: a scored case needs at least one relevant document")
    if not scored and labels:
        raise BenchmarkError(f"{where}: a behaviour case must have no relevant documents")
    if set(labels) & set(lenient):
        raise BenchmarkError(f"{where}: strict and lenient labels overlap")
    named = _question_identifiers(data["question"])
    if named and data["category"] != "identifier_lookup":
        raise BenchmarkError(f"{where}: only identifier_lookup questions may name identifiers")
    close = False if behaviour else data["close_call"] is not None
    return CaseV3(case_id, data["category"], data["query_style"], data["question"], data["task"],
                  data["expectation"], labels, lenient, data["rationale"], close)


def load_benchmark_v3(
    path: str | Path,
    *,
    require_frozen_hash: bool = True,
    expected_sha256: str | None = None,
) -> BenchmarkV3:
    """
    Read and strictly validate the v3 file. Raises `FrozenBenchmarkError` if its SHA-256 differs from the
    pinned value (unless `require_frozen_hash=False`, which only tests use) and `BenchmarkError` for any
    structural problem.
    """
    file = Path(path)
    try:
        raw_bytes = file.read_bytes()
    except OSError as exc:
        raise BenchmarkError(f"cannot read benchmark {file}: {exc}") from exc
    digest = hashlib.sha256(raw_bytes).hexdigest()
    pinned = expected_sha256 if expected_sha256 is not None else FROZEN_V3_SHA256
    if require_frozen_hash and digest != pinned:
        raise FrozenBenchmarkError(
            f"benchmark file SHA-256 is {digest}, expected the frozen value {pinned}; "
            "the file was edited or is not the frozen v3 benchmark"
        )
    try:
        raw = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"benchmark {file} is not valid JSON: {exc}") from exc
    top = _exact(raw, "benchmark", _TOP_KEYS)
    if top["format"] != BENCHMARK_FORMAT_V3:
        raise BenchmarkError(f"format must be {BENCHMARK_FORMAT_V3!r}, got {top['format']!r}")
    for key in ("benchmark_id", "description", "protocol"):
        if not isinstance(top[key], str) or not top[key].strip():
            raise BenchmarkError(f"{key} must be a non-empty string")
    corpus = top["corpus"]
    if not isinstance(corpus, dict) or not isinstance(corpus.get("corpus_fingerprint"), str) \
            or not _FINGERPRINT_RE.fullmatch(corpus["corpus_fingerprint"]):
        raise BenchmarkError("corpus.corpus_fingerprint must be a 64-character lower-case hex SHA-256")
    for key in ("cases", "behaviour_cases"):
        if not isinstance(top[key], list):
            raise BenchmarkError(f"{key} must be a list")
    if not top["cases"]:
        raise BenchmarkError("cases must be a non-empty list")
    cases = tuple(_case(entry, i, False) for i, entry in enumerate(top["cases"])) + tuple(
        _case(entry, i, True) for i, entry in enumerate(top["behaviour_cases"])
    )
    ids = [case.case_id for case in cases]
    if len(set(ids)) != len(ids):
        raise BenchmarkError("duplicate case_id in the benchmark")
    return BenchmarkV3(top["benchmark_id"], top["description"], top["protocol"], corpus, cases, digest)
