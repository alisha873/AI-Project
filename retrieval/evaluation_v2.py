"""
Phase 11 (Milestone 11B) - grouped evaluation over benchmark v2.

Reuses the pure source-level metrics of `retrieval.evaluation` (Recall@k,
reciprocal rank, de-duplication by document ID). `retrieval.evaluation` and
`retrieve_limit`-style v1 reports are unchanged; this module adds splits,
categories, behaviour cases, a random-ranking comparator and a paired
comparison between two runs.

Metrics (computed only from what the retriever actually returned)
-----------------------------------------------------------------
- Primary cutoffs k = 1, 3, 5 (`DEFAULT_KS_V2`) and MRR. On a 23-document
  corpus these are the informative ones.
- Recall@10 is a DIAGNOSTIC only (`diagnostic_ks`). A random ranking of 23
  documents already puts about 43% of any relevant document in its top 10, so
  a high Recall@10 says little; it is reported under its own key.
- Recall@k = |relevant documents in the top k distinct documents| / |relevant|.
  With several relevant documents the denominator is all of them.
- Reciprocal rank = 1 / rank of the first relevant document in the returned,
  de-duplicated list; 0 if none was returned. MRR is the mean over cases and is
  "over the top `retrieve_limit`".
- Rankings are de-duplicated by `doc_id` (first occurrence wins) before any
  rank or recall is computed, so several chunks of one source count once.
- The harness never re-sorts: tie-breaking belongs to the retriever.
- Empty result: recall 0 and reciprocal rank 0, counted in `num_no_results`.
  A `RetrievalError` for a case: all zeros, `status="error"`, counted in
  `num_errors`, never silently excluded. An empty relevant set is impossible
  for a scored case (the loader refuses it).
- Missing expected documents are listed per case (`missing_doc_ids`: relevant
  documents absent from the returned list).

Random comparator: for a corpus of N documents a uniformly random ranking has
expected Recall@k = min(k, N) / N and an expected reciprocal rank that depends
on N and the number of relevant documents R (`random_expected_reciprocal_rank`).
Both are reported next to the measured values so the reader can see how much
better than chance a result is. It is an analytic expectation, not a run.

Label modes: "final" (the adjudicated `relevant_doc_ids`, the primary mode) and
"agreed" (only documents given by the author AND every reviewer, a strict
sensitivity check). In "agreed" mode a case whose agreed set is empty is
excluded and listed in `excluded_case_ids`.

Behaviour cases are never scored for recall. A case is `handled` when the
retriever raised a typed error, returned no documents, or flagged the query
with a `LOW_INFORMATION_PREFIX` note. A keyword retriever has no relevance
threshold, so it can only abstain when nothing matches or the query is
obviously thin; the report shows how many out-of-corpus questions were NOT
handled.

Statistical honesty: group sizes are printed and small groups (n < 10) are
marked. With a few dozen cases, differences between retrievers are descriptive;
no significance test is made and none should be claimed.

The harness sends ONLY `case.question` to the retriever, never labels, task,
rationale or provenance. Query construction from incident output lives in
`retrieval.query_builder`, which imports nothing from this module.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from retrieval.benchmark_v2 import SPLITS, BenchmarkCaseV2, BenchmarkV2
from retrieval.contract import LOW_INFORMATION_PREFIX, RetrievalError, RetrievalQuery, Retriever
from retrieval.evaluation import (
    BenchmarkError,
    UnknownLabelError,
    dedupe_ranking,
    first_relevant_rank,
    recall_at_k,
    reciprocal_rank,
)

DEFAULT_KS_V2: tuple[int, ...] = (1, 3, 5)
DEFAULT_DIAGNOSTIC_KS: tuple[int, ...] = (10,)
DEFAULT_RETRIEVE_LIMIT_V2 = 10
LABEL_MODES = ("final", "agreed")
SMALL_GROUP = 10
_ROUND = 6

CAVEATS = (
    "Small, manually curated benchmark over a 23-document corpus; not evidence of general retrieval quality.",
    "No human reviewer has labelled any case; the second review was an AI pass and adjudication was by the author.",
    "Recall@10 is a diagnostic only: a random ranking of 23 documents already reaches about 0.43.",
    "Groups with fewer than 10 cases are indicative only; no significance test is made.",
)


class CorpusMismatchError(BenchmarkError):
    """The evaluated corpus is not the corpus the benchmark was written for."""


# ============================================================
# PURE FUNCTIONS
# ============================================================

def random_expected_recall_at_k(n_docs: int, k: int) -> float:
    """Expected Recall@k of a uniformly random ranking of `n_docs` documents."""
    if n_docs < 1 or k < 1:
        raise ValueError("n_docs and k must be positive")
    return min(k, n_docs) / n_docs


def random_expected_reciprocal_rank(n_docs: int, n_relevant: int) -> float:
    """
    Expected reciprocal rank of the first relevant document in a uniformly random
    ranking: sum over r of (1/r) * C(n - r, R - 1) / C(n, R).
    """
    if n_docs < 1 or not 1 <= n_relevant <= n_docs:
        raise ValueError("need 1 <= n_relevant <= n_docs")
    total = math.comb(n_docs, n_relevant)
    return sum(
        math.comb(n_docs - r, n_relevant - 1) / (r * total) for r in range(1, n_docs - n_relevant + 2)
    )


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return None if not items else sum(items) / len(items)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, _ROUND)


# ============================================================
# RESULT TYPES
# ============================================================

@dataclass(frozen=True)
class ScoredCaseResult:
    case_id: str
    split: str
    category: str
    query_style: str
    status: str  # "ok", "no_results" or "error"
    relevant_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    first_relevant_rank: int | None
    missing_doc_ids: tuple[str, ...]
    recall_at_k: tuple[tuple[int, float], ...]
    reciprocal_rank: float
    possibly_incomplete: bool
    contested: bool
    query_notes: tuple[str, ...]
    error: str | None

    def recall(self, k: int) -> float:
        return dict(self.recall_at_k)[k]


@dataclass(frozen=True)
class BehaviourCaseResult:
    case_id: str
    split: str
    category: str
    expectation: str
    status: str
    handled: bool
    reason: str
    num_returned: int
    top_doc_id: str | None
    top_score: float | None
    query_notes: tuple[str, ...]
    error: str | None


# ============================================================
# REPORT
# ============================================================

@dataclass(frozen=True)
class EvaluationReportV2:
    benchmark_id: str
    benchmark_sha256: str | None
    retriever: str
    label_mode: str
    split: str | None
    ks: tuple[int, ...]
    diagnostic_ks: tuple[int, ...]
    retrieve_limit: int
    corpus_doc_count: int
    corpus_fingerprint: str | None
    scored: tuple[ScoredCaseResult, ...]
    behaviour: tuple[BehaviourCaseResult, ...]
    excluded_case_ids: tuple[str, ...] = ()

    @property
    def all_ks(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.ks) | set(self.diagnostic_ks)))

    def _group(self, rows: Sequence[ScoredCaseResult]) -> dict[str, Any]:
        n_docs = self.corpus_doc_count
        primary = {str(k): _rounded(_mean(r.recall(k) for r in rows)) for k in self.ks}
        diagnostic = {str(k): _rounded(_mean(r.recall(k) for r in rows)) for k in self.diagnostic_ks}
        random_recall = {
            str(k): _rounded(_mean(random_expected_recall_at_k(n_docs, k) for _ in rows)) for k in self.ks
        }
        random_mrr = _mean(random_expected_reciprocal_rank(n_docs, len(r.relevant_doc_ids)) for r in rows)
        return {
            "cases": len(rows),
            "small_group": len(rows) < SMALL_GROUP,
            "mean_recall_at_k": primary,
            "diagnostic_mean_recall_at_k": diagnostic,
            "mrr": _rounded(_mean(r.reciprocal_rank for r in rows)),
            "random_expected_recall_at_k": random_recall,
            "random_expected_mrr": _rounded(random_mrr),
            "num_no_results": sum(1 for r in rows if r.status == "no_results"),
            "num_errors": sum(1 for r in rows if r.status == "error"),
            "num_with_missing_relevant": sum(1 for r in rows if r.missing_doc_ids),
        }

    def _grouped(self, key: str) -> dict[str, Any]:
        names = sorted({getattr(r, key) for r in self.scored})
        return {name: self._group([r for r in self.scored if getattr(r, key) == name]) for name in names}

    def to_dict(self) -> dict[str, Any]:
        behaviour_groups: dict[str, Any] = {}
        for expectation in sorted({b.expectation for b in self.behaviour}):
            rows = [b for b in self.behaviour if b.expectation == expectation]
            behaviour_groups[expectation] = {
                "cases": len(rows),
                "handled": sum(1 for b in rows if b.handled),
                "not_handled": sum(1 for b in rows if not b.handled),
            }
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_sha256": self.benchmark_sha256,
            "retriever": self.retriever,
            "label_mode": self.label_mode,
            "split": self.split,
            "ks": list(self.ks),
            "diagnostic_ks": list(self.diagnostic_ks),
            "retrieve_limit": self.retrieve_limit,
            "corpus_doc_count": self.corpus_doc_count,
            "corpus_fingerprint": self.corpus_fingerprint,
            "excluded_case_ids": list(self.excluded_case_ids),
            "overall": self._group(self.scored),
            "by_split": self._grouped("split"),
            "by_category": self._grouped("category"),
            "by_query_style": self._grouped("query_style"),
            "by_label_multiplicity": {
                "single_relevant": self._group([r for r in self.scored if len(r.relevant_doc_ids) == 1]),
                "multiple_relevant": self._group([r for r in self.scored if len(r.relevant_doc_ids) > 1]),
            },
            "behaviour_summary": behaviour_groups,
            "scored_cases": [
                {
                    "case_id": r.case_id, "split": r.split, "category": r.category, "query_style": r.query_style,
                    "status": r.status, "relevant_doc_ids": list(r.relevant_doc_ids),
                    "retrieved_doc_ids": list(r.retrieved_doc_ids),
                    "first_relevant_rank": r.first_relevant_rank,
                    "missing_doc_ids": list(r.missing_doc_ids),
                    "recall_at_k": {str(k): _rounded(v) for k, v in r.recall_at_k},
                    "reciprocal_rank": _rounded(r.reciprocal_rank),
                    "possibly_incomplete": r.possibly_incomplete, "contested": r.contested,
                    "query_notes": list(r.query_notes), "error": r.error,
                }
                for r in self.scored
            ],
            "behaviour_cases": [
                {
                    "case_id": b.case_id, "split": b.split, "category": b.category,
                    "expectation": b.expectation, "status": b.status, "handled": b.handled,
                    "reason": b.reason, "num_returned": b.num_returned, "top_doc_id": b.top_doc_id,
                    "top_score": _rounded(b.top_score), "query_notes": list(b.query_notes), "error": b.error,
                }
                for b in self.behaviour
            ],
            "caveats": list(CAVEATS),
        }

    def to_json(self) -> str:
        """Deterministic JSON: sorted keys, no timestamps."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    def render_text(self) -> str:
        data = self.to_dict()

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        def line(label: str, block: dict[str, Any]) -> str:
            recalls = "  ".join(f"R@{k}={fmt(block['mean_recall_at_k'][str(k)])}" for k in self.ks)
            diag = "  ".join(f"[diag R@{k}={fmt(block['diagnostic_mean_recall_at_k'][str(k)])}]" for k in self.diagnostic_ks)
            mark = "*" if block["small_group"] else " "
            return f"  {label:22s}{mark}n={block['cases']:2d}  {recalls}  MRR={fmt(block['mrr'])}  {diag}"

        scope = self.split or "all splits"
        lines = [
            f"Benchmark {self.benchmark_id} | retriever {self.retriever} | labels: {self.label_mode} | {scope}",
            f"  corpus {self.corpus_doc_count} documents, fingerprint {self.corpus_fingerprint}",
            f"  {len(self.scored)} scored case(s), {len(self.behaviour)} behaviour case(s); retrieve_limit {self.retrieve_limit}",
            "  (* = fewer than 10 cases: indicative only)",
            line("OVERALL", data["overall"]),
            "  random-ranking expectation (analytic): "
            + "  ".join(f"R@{k}={fmt(data['overall']['random_expected_recall_at_k'][str(k)])}" for k in self.ks)
            + f"  MRR={fmt(data['overall']['random_expected_mrr'])}",
        ]
        for title, key in (("split", "by_split"), ("category", "by_category"), ("query style", "by_query_style")):
            lines.append(f"  by {title}:")
            lines.extend(line(name, block) for name, block in data[key].items())
        lines.append("  by label multiplicity:")
        lines.extend(line(name, block) for name, block in data["by_label_multiplicity"].items())
        lines.append("  behaviour cases (not scored for recall):")
        for expectation, block in data["behaviour_summary"].items():
            lines.append(f"    {expectation:22s} handled {block['handled']}/{block['cases']}")
        for b in self.behaviour:
            lines.append(f"      {b.case_id:9s} {b.expectation:22s} handled={b.handled!s:5s} {b.reason}")
        lines.append("  scored cases (first relevant rank; missing = relevant documents not returned):")
        for r in self.scored:
            rank = "-" if r.first_relevant_rank is None else str(r.first_relevant_rank)
            missing = f" missing={','.join(r.missing_doc_ids)}" if r.missing_doc_ids else ""
            lines.append(f"    {r.case_id:9s} {r.split:11s} {r.category:20s} {r.status:10s} rank={rank:>2s}{missing}")
        lines.extend(f"  caveat: {c}" for c in CAVEATS)
        return "\n".join(lines) + "\n"


# ============================================================
# HARNESS
# ============================================================

def _checked_ks(ks: Sequence[int], name: str) -> tuple[int, ...]:
    checked = tuple(sorted(set(ks)))
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in checked):
        raise ValueError(f"{name} must contain only positive integers")
    return checked


def _retrieve(retriever: Retriever, case: BenchmarkCaseV2, limit: int):
    """Run one case. Only the question text reaches the retriever. Returns (result, error)."""
    try:
        return retriever.retrieve(RetrievalQuery(text=case.question, limit=limit)), None
    except RetrievalError as exc:
        return None, f"{type(exc).__name__}: {exc}"


def evaluate_v2(
    benchmark: BenchmarkV2,
    retriever: Retriever,
    *,
    corpus_doc_ids: Iterable[str],
    fingerprint: str | None = None,
    split: str | None = None,
    label_mode: str = "final",
    ks: Sequence[int] = DEFAULT_KS_V2,
    diagnostic_ks: Sequence[int] = DEFAULT_DIAGNOSTIC_KS,
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT_V2,
    benchmark_sha256: str | None = None,
    allow_corpus_mismatch: bool = False,
) -> EvaluationReportV2:
    """
    Run the selected cases through `retriever` and compute metrics from what it returned.

    Raises `UnknownLabelError` (before any retrieval) for a label outside `corpus_doc_ids`,
    `CorpusMismatchError` if `fingerprint` differs from the benchmark's recorded corpus
    fingerprint (unless `allow_corpus_mismatch`), and `ValueError` for invalid options.
    """
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be None or one of {SPLITS}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")
    primary, diagnostic = _checked_ks(ks, "ks"), _checked_ks(diagnostic_ks, "diagnostic_ks")
    if not primary:
        raise ValueError("ks must not be empty")
    if isinstance(retrieve_limit, bool) or not isinstance(retrieve_limit, int) or retrieve_limit < 1:
        raise ValueError("retrieve_limit must be a positive integer")
    if max(primary + diagnostic) > retrieve_limit:
        raise ValueError(f"the largest k ({max(primary + diagnostic)}) exceeds retrieve_limit ({retrieve_limit})")
    if fingerprint is not None and fingerprint != benchmark.corpus_fingerprint and not allow_corpus_mismatch:
        raise CorpusMismatchError(
            f"corpus fingerprint {fingerprint} differs from the benchmark's recorded "
            f"{benchmark.corpus_fingerprint}; results would not be comparable"
        )

    corpus = set(corpus_doc_ids)
    selected = benchmark.select(split=split)
    for case in selected:
        used = case.relevant_doc_ids if label_mode == "final" else case.provenance.author_labels
        missing = [label for label in used if label not in corpus]
        if missing:
            raise UnknownLabelError(f"case {case.case_id!r} has label(s) {missing} that are not in the corpus")

    all_ks = tuple(sorted(set(primary) | set(diagnostic)))
    scored: list[ScoredCaseResult] = []
    behaviour: list[BehaviourCaseResult] = []
    excluded: list[str] = []
    for case in selected:
        result, error = _retrieve(retriever, case, retrieve_limit)
        ranking = [] if result is None else dedupe_ranking(result.doc_ids)[:retrieve_limit]
        status = "error" if result is None else result.status
        notes = () if result is None else tuple(result.query_notes)
        if case.scored:
            labels = case.relevant_doc_ids if label_mode == "final" else case.agreed_doc_ids
            if not labels:
                excluded.append(case.case_id)
                continue
            scored.append(
                ScoredCaseResult(
                    case_id=case.case_id, split=case.split, category=case.category,
                    query_style=case.query_style, status=status, relevant_doc_ids=tuple(labels),
                    retrieved_doc_ids=tuple(ranking),
                    first_relevant_rank=first_relevant_rank(ranking, labels),
                    missing_doc_ids=tuple(label for label in labels if label not in ranking),
                    recall_at_k=tuple((k, recall_at_k(ranking, labels, k) or 0.0) for k in all_ks),
                    reciprocal_rank=reciprocal_rank(ranking, labels) or 0.0,
                    possibly_incomplete=False if result is None else result.possibly_incomplete,
                    contested=bool(set(case.contested_doc_ids) & set(labels)),
                    query_notes=notes, error=error,
                )
            )
            continue
        if result is None:
            handled, reason = True, "refused with a typed error"
        elif result.status == "no_results":
            handled, reason = True, "returned no documents"
        elif any(note.startswith(LOW_INFORMATION_PREFIX) for note in notes):
            handled, reason = True, "flagged the query as low-information"
        else:
            handled, reason = False, f"returned {len(result.items)} document(s) without a low-information flag"
        top = None if not ranking else result.items[0]
        behaviour.append(
            BehaviourCaseResult(
                case_id=case.case_id, split=case.split, category=case.category,
                expectation=case.expectation, status=status, handled=handled, reason=reason,
                num_returned=len(ranking), top_doc_id=None if top is None else top.doc_id,
                top_score=None if top is None else top.score, query_notes=notes, error=error,
            )
        )
    return EvaluationReportV2(
        benchmark_id=benchmark.benchmark_id, benchmark_sha256=benchmark_sha256, retriever=retriever.name,
        label_mode=label_mode, split=split, ks=primary, diagnostic_ks=diagnostic,
        retrieve_limit=retrieve_limit, corpus_doc_count=len(corpus), corpus_fingerprint=fingerprint,
        scored=tuple(scored), behaviour=tuple(behaviour), excluded_case_ids=tuple(excluded),
    )


# ============================================================
# PAIRED COMPARISON OF TWO RUNS
# ============================================================

@dataclass(frozen=True)
class CaseChange:
    case_id: str
    split: str
    category: str
    old_rank: int | None
    new_rank: int | None
    old_recall: float
    new_recall: float
    outcome: str  # "improved", "worsened" or "unchanged"


def _rank_key(rank: int | None) -> float:
    return math.inf if rank is None else float(rank)


def compare_reports(old: EvaluationReportV2, new: EvaluationReportV2) -> dict[str, Any]:
    """
    Compare two runs case by case on the same benchmark, labels, split and corpus.

    A case IMPROVED if the first relevant document moved to a better rank (found
    where it was missing counts as better), WORSENED if it moved to a worse rank,
    and otherwise improved/worsened if Recall@(largest primary k) rose/fell;
    anything else is UNCHANGED. The counts are descriptive; no significance is
    claimed. Raises `BenchmarkError` if the runs are not comparable.
    """
    same = (old.benchmark_id, old.benchmark_sha256, old.label_mode, old.split, old.ks, old.retrieve_limit,
            old.corpus_fingerprint)
    other = (new.benchmark_id, new.benchmark_sha256, new.label_mode, new.split, new.ks, new.retrieve_limit,
             new.corpus_fingerprint)
    if same != other:
        raise BenchmarkError("the two reports use different benchmark, labels, split, cutoffs or corpus")
    old_by = {r.case_id: r for r in old.scored}
    new_by = {r.case_id: r for r in new.scored}
    if old_by.keys() != new_by.keys():
        raise BenchmarkError("the two reports cover different cases")
    k_ref = max(old.ks)
    changes: list[CaseChange] = []
    for case_id in sorted(old_by):
        a, b = old_by[case_id], new_by[case_id]
        if _rank_key(b.first_relevant_rank) < _rank_key(a.first_relevant_rank):
            outcome = "improved"
        elif _rank_key(b.first_relevant_rank) > _rank_key(a.first_relevant_rank):
            outcome = "worsened"
        elif b.recall(k_ref) > a.recall(k_ref):
            outcome = "improved"
        elif b.recall(k_ref) < a.recall(k_ref):
            outcome = "worsened"
        else:
            outcome = "unchanged"
        changes.append(CaseChange(case_id, a.split, a.category, a.first_relevant_rank, b.first_relevant_rank,
                                  a.recall(k_ref), b.recall(k_ref), outcome))
    old_data, new_data = old.to_dict()["overall"], new.to_dict()["overall"]

    def delta(key: str) -> dict[str, float | None]:
        return {
            str(k): _rounded(new_data[key][str(k)] - old_data[key][str(k)])
            for k in (old.ks if key == "mean_recall_at_k" else old.diagnostic_ks)
        }

    return {
        "old_retriever": old.retriever,
        "new_retriever": new.retriever,
        "benchmark_id": old.benchmark_id,
        "split": old.split,
        "label_mode": old.label_mode,
        "corpus_fingerprint": old.corpus_fingerprint,
        "reference_k": k_ref,
        "counts": {name: sum(1 for c in changes if c.outcome == name) for name in ("improved", "worsened", "unchanged")},
        "delta_mean_recall_at_k": delta("mean_recall_at_k"),
        "delta_diagnostic_mean_recall_at_k": delta("diagnostic_mean_recall_at_k"),
        "delta_mrr": _rounded(new_data["mrr"] - old_data["mrr"]),
        "changes": [
            {
                "case_id": c.case_id, "split": c.split, "category": c.category, "old_first_relevant_rank": c.old_rank,
                "new_first_relevant_rank": c.new_rank, "old_recall_at_ref_k": _rounded(c.old_recall),
                "new_recall_at_ref_k": _rounded(c.new_recall), "outcome": c.outcome,
            }
            for c in changes
        ],
        "caveat": "Descriptive counts on a small benchmark; no significance is claimed.",
    }
