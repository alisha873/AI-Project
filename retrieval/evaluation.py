"""
Phase 11 (Milestone 11A) - retrieval evaluation: metrics, benchmark loader, harness.

This module measures how well a `Retriever` finds labelled source documents.
It is a SMALL, MANUALLY CURATED benchmark harness, not evidence of general
retrieval quality (see the limits at the end).

Source-level metrics
--------------------
Retrievers return one item per source document (`RetrievalResult`), and the
metric functions defensively remove repeated IDs from a ranking (first
occurrence wins), so a source is never counted twice because several of its
chunks matched. A "rank" is the position in that list (1-based); the harness
never re-sorts what a retriever returned, so tie-breaking is the retriever's
(deterministic) responsibility.

- Recall@k = |relevant IDs found in the top k| / |relevant IDs|.
- Reciprocal rank = 1 / rank of the FIRST relevant ID in the returned list
  (0 if none was returned); MRR is its mean over cases. Because the harness
  retrieves at most `retrieve_limit` documents, this is "MRR over the top
  `retrieve_limit`".
- Empty relevant set: the pure functions return None ("undefined"), never 0.
  `BenchmarkCase` refuses empty label sets, so reports never contain this.
- Empty result: Recall@k = 0 and reciprocal rank = 0 for that case, and it is
  counted in `num_no_results`.
- A case whose query cannot be run (a `RetrievalError`) scores 0 for every
  metric, is flagged `status="error"`, and is counted in `num_errors`; failures
  are never silently excluded from the means.

Benchmark file (`autoshield.retrieval_benchmark.v1`) validation
---------------------------------------------------------------
The loader is strict and raises `BenchmarkError` naming the case for: unknown or
duplicate keys, a malformed or duplicate `case_id`, an empty question, task or
rationale, an invalid `query_style`, an empty label list, duplicate labels within
a case, a label that is not a well-formed `<source>:<source_id>` document ID,
and a question that contains any CWE-/CAPEC-/CVE- identifier (so the answer
cannot be written into the query). `evaluate_benchmark` additionally raises
`UnknownLabelError` before running anything if a label names a document that is
not in the corpus being evaluated, because that would silently lower recall.

Separation: this module never builds queries from analysis output, and
`retrieval.query_builder` never imports this module or reads the benchmark.

Limits of what the numbers mean
-------------------------------
Few cases, labels written by one author after seeing the corpus titles and
declared relationships (so not blind), binary relevance, no judgements for
unlabelled-but-relevant documents (parents/siblings are deliberately left
unlabelled), no negative cases, and a corpus of only 23 documents. The numbers
describe this baseline on this benchmark, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from knowledge.schema import make_doc_id
from retrieval.contract import RetrievalError, RetrievalQuery, Retriever

BENCHMARK_FORMAT = "autoshield.retrieval_benchmark.v1"
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_RETRIEVE_LIMIT = 10
QUERY_STYLES = ("terminology", "descriptive")

_CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_IDENTIFIER_RE = re.compile(r"\b(?:CWE|CAPEC|CVE)-[0-9]+", re.IGNORECASE)
_ROUND = 6


# ============================================================
# ERRORS
# ============================================================

class BenchmarkError(RetrievalError, ValueError):
    """A benchmark file, case or label is invalid."""


class UnknownLabelError(BenchmarkError):
    """A relevance label names a document that is not in the evaluated corpus."""


# ============================================================
# METRICS (pure functions)
# ============================================================

def dedupe_ranking(ranked_ids: Iterable[str]) -> list[str]:
    """Remove repeated IDs, keeping first occurrences, so each source counts once."""
    return list(dict.fromkeys(ranked_ids))


def _checked_relevant(relevant: Iterable[str]) -> list[str]:
    labels = list(relevant)
    if any(not isinstance(label, str) or not label for label in labels):
        raise BenchmarkError("relevance labels must be non-empty strings")
    if len(set(labels)) != len(labels):
        raise BenchmarkError(f"duplicate relevance labels: {sorted({x for x in labels if labels.count(x) > 1})}")
    return labels


def recall_at_k(ranked_ids: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Fraction of the relevant IDs present in the top `k` distinct IDs; None if there are none."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    labels = _checked_relevant(relevant)
    if not labels:
        return None
    top = set(dedupe_ranking(ranked_ids)[:k])
    return sum(1 for label in labels if label in top) / len(labels)


def first_relevant_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> int | None:
    """1-based rank of the first relevant ID in the de-duplicated ranking, or None."""
    wanted = set(_checked_relevant(relevant))
    for position, doc_id in enumerate(dedupe_ranking(ranked_ids), start=1):
        if doc_id in wanted:
            return position
    return None


def reciprocal_rank(ranked_ids: Sequence[str], relevant: Iterable[str]) -> float | None:
    """1/rank of the first relevant ID; 0.0 if none was returned; None if there are no labels."""
    labels = _checked_relevant(relevant)
    if not labels:
        return None
    rank = first_relevant_rank(ranked_ids, labels)
    return 0.0 if rank is None else 1.0 / rank


def _mean(values: Iterable[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return None if not present else sum(present) / len(present)


# ============================================================
# BENCHMARK
# ============================================================

@dataclass(frozen=True)
class BenchmarkCase:
    """One labelled retrieval question (validated on construction)."""

    case_id: str
    question: str
    task: str
    relevant_doc_ids: tuple[str, ...]
    rationale: str
    query_style: str

    def __post_init__(self) -> None:
        where = f"case {self.case_id!r}"
        if not isinstance(self.case_id, str) or not _CASE_ID_RE.fullmatch(self.case_id):
            raise BenchmarkError(f"invalid case_id {self.case_id!r}")
        for name in ("question", "task", "rationale"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise BenchmarkError(f"{where}: {name} must be a non-empty string")
        if self.query_style not in QUERY_STYLES:
            raise BenchmarkError(f"{where}: query_style must be one of {QUERY_STYLES}")
        labels = list(self.relevant_doc_ids)
        if not labels:
            raise BenchmarkError(f"{where}: at least one relevant document ID is required")
        try:
            _checked_relevant(labels)
        except BenchmarkError as exc:
            raise BenchmarkError(f"{where}: {exc}") from exc
        for label in labels:
            source, _, source_id = label.partition(":")
            try:
                make_doc_id(source, source_id)  # validates the source name and the identifier
            except ValueError as exc:
                raise BenchmarkError(f"{where}: invalid document ID {label!r}: {exc}") from exc
        leaked = _IDENTIFIER_RE.findall(self.question)
        if leaked:
            raise BenchmarkError(
                f"{where}: the question contains identifier(s) {sorted(set(leaked))}; questions must "
                "describe observations and must not contain CWE/CAPEC/CVE identifiers"
            )
        object.__setattr__(self, "relevant_doc_ids", tuple(labels))


@dataclass(frozen=True)
class Benchmark:
    benchmark_id: str
    description: str
    protocol: str
    corpus: Mapping[str, Any]
    cases: tuple[BenchmarkCase, ...]


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


def load_benchmark(path: str | Path) -> Benchmark:
    """Read and strictly validate a benchmark file (see the module docstring)."""
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except OSError as exc:
        raise BenchmarkError(f"cannot read benchmark {file}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"benchmark {file} is not valid JSON: {exc}") from exc

    top = _exact_keys(raw, "benchmark", {"format", "benchmark_id", "description", "protocol", "corpus", "cases"})
    if top["format"] != BENCHMARK_FORMAT:
        raise BenchmarkError(f"format must be {BENCHMARK_FORMAT!r}, got {top['format']!r}")
    for key in ("benchmark_id", "description", "protocol"):
        if not isinstance(top[key], str) or not top[key].strip():
            raise BenchmarkError(f"{key} must be a non-empty string")
    if not isinstance(top["corpus"], dict):
        raise BenchmarkError("corpus must be an object")
    if not isinstance(top["cases"], list) or not top["cases"]:
        raise BenchmarkError("cases must be a non-empty list")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, entry in enumerate(top["cases"]):
        data = _exact_keys(
            entry, f"cases[{index}]",
            {"case_id", "question", "task", "relevant_doc_ids", "rationale", "query_style"},
        )
        if not isinstance(data["relevant_doc_ids"], list):
            raise BenchmarkError(f"cases[{index}].relevant_doc_ids must be a list")
        case = BenchmarkCase(
            case_id=data["case_id"],
            question=data["question"],
            task=data["task"],
            relevant_doc_ids=tuple(data["relevant_doc_ids"]),
            rationale=data["rationale"],
            query_style=data["query_style"],
        )
        if case.case_id in seen:
            raise BenchmarkError(f"duplicate case_id {case.case_id!r}")
        seen.add(case.case_id)
        cases.append(case)
    return Benchmark(top["benchmark_id"], top["description"], top["protocol"], top["corpus"], tuple(cases))


# ============================================================
# REPORT
# ============================================================

@dataclass(frozen=True)
class CaseResult:
    case_id: str
    query_style: str
    status: str  # "ok", "no_results" or "error"
    relevant_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    first_relevant_rank: int | None
    recall_at_k: tuple[tuple[int, float], ...]
    reciprocal_rank: float
    possibly_incomplete: bool = False
    error: str | None = None


@dataclass(frozen=True)
class EvaluationReport:
    benchmark_id: str
    retriever: str
    ks: tuple[int, ...]
    retrieve_limit: int
    corpus_doc_count: int
    corpus_fingerprint: str | None
    case_results: tuple[CaseResult, ...]

    @property
    def num_cases(self) -> int:
        return len(self.case_results)

    @property
    def num_no_results(self) -> int:
        return sum(1 for r in self.case_results if r.status == "no_results")

    @property
    def num_errors(self) -> int:
        return sum(1 for r in self.case_results if r.status == "error")

    @property
    def num_possibly_incomplete(self) -> int:
        return sum(1 for r in self.case_results if r.possibly_incomplete)

    def mean_recall_at_k(self, style: str | None = None) -> dict[int, float | None]:
        rows = [r for r in self.case_results if style is None or r.query_style == style]
        return {k: _mean(dict(r.recall_at_k)[k] for r in rows) for k in self.ks}

    def mrr(self, style: str | None = None) -> float | None:
        return _mean(r.reciprocal_rank for r in self.case_results if style is None or r.query_style == style)

    def to_dict(self) -> dict[str, Any]:
        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, _ROUND)

        def summary(style: str | None) -> dict[str, Any]:
            count = sum(1 for r in self.case_results if style is None or r.query_style == style)
            return {
                "cases": count,
                "mean_recall_at_k": {str(k): rounded(v) for k, v in self.mean_recall_at_k(style).items()},
                "mrr": rounded(self.mrr(style)),
            }

        return {
            "benchmark_id": self.benchmark_id,
            "retriever": self.retriever,
            "ks": list(self.ks),
            "retrieve_limit": self.retrieve_limit,
            "corpus_doc_count": self.corpus_doc_count,
            "corpus_fingerprint": self.corpus_fingerprint,
            "num_cases": self.num_cases,
            "num_no_results": self.num_no_results,
            "num_errors": self.num_errors,
            "num_possibly_incomplete": self.num_possibly_incomplete,
            "overall": summary(None),
            "by_query_style": {style: summary(style) for style in QUERY_STYLES},
            "cases": [
                {
                    "case_id": r.case_id,
                    "query_style": r.query_style,
                    "status": r.status,
                    "relevant_doc_ids": list(r.relevant_doc_ids),
                    "retrieved_doc_ids": list(r.retrieved_doc_ids),
                    "first_relevant_rank": r.first_relevant_rank,
                    "recall_at_k": {str(k): rounded(v) for k, v in r.recall_at_k},
                    "reciprocal_rank": rounded(r.reciprocal_rank),
                    "possibly_incomplete": r.possibly_incomplete,
                    "error": r.error,
                }
                for r in self.case_results
            ],
        }

    def to_json(self) -> str:
        """Deterministic JSON (sorted keys, no timestamps)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def render_text(self) -> str:
        """A compact, deterministic text summary."""
        data = self.to_dict()

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        lines = [
            f"Benchmark {self.benchmark_id} | retriever {self.retriever} | "
            f"{self.num_cases} cases | corpus {self.corpus_doc_count} documents",
            f"  no-result cases: {self.num_no_results}, errors: {self.num_errors}, "
            f"possibly incomplete: {self.num_possibly_incomplete}",
            f"  retrieve_limit {self.retrieve_limit}; MRR is over the top {self.retrieve_limit} documents",
        ]
        for label, block in [("overall", data["overall"])] + [
            (style, data["by_query_style"][style]) for style in QUERY_STYLES
        ]:
            recalls = "  ".join(f"R@{k}={fmt(block['mean_recall_at_k'][str(k)])}" for k in self.ks)
            lines.append(f"  {label:12s} ({block['cases']:2d} cases)  {recalls}  MRR={fmt(block['mrr'])}")
        lines.append("  per case (first relevant rank):")
        for r in self.case_results:
            rank = "-" if r.first_relevant_rank is None else str(r.first_relevant_rank)
            lines.append(f"    {r.case_id:7s} {r.query_style:11s} {r.status:10s} rank={rank:>2s}  {','.join(r.relevant_doc_ids)}")
        return "\n".join(lines) + "\n"


# ============================================================
# HARNESS
# ============================================================

def corpus_fingerprint(store: Any) -> str:
    """SHA-256 over every stored (doc_id, document_sha256) pair, so a report names its corpus."""
    lines = [f"{doc_id}\t{store.get_document(doc_id).document_sha256}" for doc_id in sorted(store.list_doc_ids())]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def evaluate_benchmark(
    benchmark: Benchmark,
    retriever: Retriever,
    *,
    corpus_doc_ids: Iterable[str],
    ks: Sequence[int] = DEFAULT_KS,
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT,
    fingerprint: str | None = None,
) -> EvaluationReport:
    """
    Run every case through `retriever` and compute source-level metrics from
    what it actually returned. Nothing is hard-coded or cached.

    Raises `UnknownLabelError` (before any retrieval) if a label is not in
    `corpus_doc_ids`, and `ValueError` for invalid `ks` / `retrieve_limit`.
    """
    corpus = set(corpus_doc_ids)
    ks_checked = tuple(sorted(set(ks)))
    if not ks_checked or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in ks_checked):
        raise ValueError("ks must be a non-empty collection of positive integers")
    if isinstance(retrieve_limit, bool) or not isinstance(retrieve_limit, int) or retrieve_limit < 1:
        raise ValueError("retrieve_limit must be a positive integer")
    if ks_checked[-1] > retrieve_limit:
        raise ValueError(f"the largest k ({ks_checked[-1]}) exceeds retrieve_limit ({retrieve_limit})")

    for case in benchmark.cases:
        missing = [label for label in case.relevant_doc_ids if label not in corpus]
        if missing:
            raise UnknownLabelError(f"case {case.case_id!r} has label(s) {missing} that are not in the corpus")

    results: list[CaseResult] = []
    for case in benchmark.cases:
        labels = case.relevant_doc_ids
        error: str | None = None
        ranking: list[str] = []
        status = "ok"
        incomplete = False
        try:
            result = retriever.retrieve(RetrievalQuery(text=case.question, limit=retrieve_limit))
            ranking = dedupe_ranking(result.doc_ids)[:retrieve_limit]
            status = result.status
            incomplete = result.possibly_incomplete
        except RetrievalError as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        results.append(
            CaseResult(
                case_id=case.case_id,
                query_style=case.query_style,
                status=status,
                relevant_doc_ids=labels,
                retrieved_doc_ids=tuple(ranking),
                first_relevant_rank=first_relevant_rank(ranking, labels),
                recall_at_k=tuple((k, recall_at_k(ranking, labels, k) or 0.0) for k in ks_checked),
                reciprocal_rank=reciprocal_rank(ranking, labels) or 0.0,
                possibly_incomplete=incomplete,
                error=error,
            )
        )
    return EvaluationReport(
        benchmark_id=benchmark.benchmark_id,
        retriever=retriever.name,
        ks=ks_checked,
        retrieve_limit=retrieve_limit,
        corpus_doc_count=len(corpus),
        corpus_fingerprint=fingerprint,
        case_results=tuple(results),
    )
