"""
Phase 11 (Milestone 11C) - run several retrievers on one benchmark and assemble a deterministic comparison.

This module contains no model code, no benchmark file and no v3-specific logic: it takes a `BenchmarkV2`
(the shared case type; the v3 loader converts to it) and a dictionary of `Retriever`s, runs the existing
`retrieval.evaluation_v2.evaluate_v2` on each, and adds what that harness does not report:

- `by_lane_group`: results for the cases whose relevant documents are only CWE, only CAPEC, or in both lanes;
- COUNTS beside every percentage: `hit_at_k` (questions with at least one relevant document in the top k) and
  `full_recall_at_k` (questions whose every relevant document is in the top k). With a few dozen questions a
  change of one question is within noise, so counts are reported first;
- the lane-specific system (S4): the weakness-lane retriever on the CWE-only cases, the attack-pattern-lane
  retriever on the CAPEC-only cases (mixed cases need both lanes and are listed as not applicable);
- paired, case-by-case comparisons between chosen pairs of systems (`compare_reports`: improved / worsened /
  unchanged; descriptive, no significance claim);
- a question-by-question table of the first relevant rank of every system.

Every number is computed from what the retrievers actually returned; nothing is hard-coded. The payload has no
timestamps, so two runs on the same inputs give identical JSON.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from retrieval.benchmark_v2 import BenchmarkV2
from retrieval.contract import Retriever
from retrieval.evaluation_v2 import (
    DEFAULT_DIAGNOSTIC_KS,
    DEFAULT_KS_V2,
    DEFAULT_RETRIEVE_LIMIT_V2,
    EvaluationReportV2,
    ScoredCaseResult,
    compare_reports,
    evaluate_v2,
)
from retrieval.lanes import ATTACK_PATTERN_LANE, WEAKNESS_LANE
from retrieval.rrf import LaneRetriever

LANE_GROUPS = ("cwe_only", "capec_only", "both")
_ROUND = 6
MANDATORY_CAVEAT = (
    "A 23-document, hand-curated corpus and AI-labelled questions; results do not show generalisation, and the "
    "labels are not independently reviewed."
)


def lane_group_of(relevant_doc_ids: Iterable[str]) -> str:
    """"cwe_only", "capec_only" or "both", from the document-id prefixes of a case's relevant documents."""
    sources = {doc_id.partition(":")[0] for doc_id in relevant_doc_ids}
    if sources == {"cwe"}:
        return "cwe_only"
    if sources == {"capec"}:
        return "capec_only"
    return "both"


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else round(sum(values) / len(values), _ROUND)


def summarize_rows(rows: Sequence[ScoredCaseResult], ks: Sequence[int]) -> dict[str, Any]:
    """Counts first, then means, for a set of scored case results."""
    return {
        "cases": len(rows),
        "hit_at_k": {str(k): sum(1 for r in rows if r.first_relevant_rank is not None and r.first_relevant_rank <= k)
                     for k in ks},
        "full_recall_at_k": {str(k): sum(1 for r in rows if r.recall(k) == 1.0) for k in ks},
        "mean_recall_at_k": {str(k): _mean([r.recall(k) for r in rows]) for k in ks},
        "mrr": _mean([r.reciprocal_rank for r in rows]),
        "num_no_results": sum(1 for r in rows if r.status == "no_results"),
        "num_errors": sum(1 for r in rows if r.status == "error"),
    }


def summarize_report(report: EvaluationReportV2) -> dict[str, Any]:
    """Overall and per-lane-group summaries (with counts) for one system's report."""
    ks = report.ks
    rows = report.scored
    return {
        "overall": summarize_rows(rows, ks),
        "by_lane_group": {
            group: summarize_rows([r for r in rows if lane_group_of(r.relevant_doc_ids) == group], ks)
            for group in LANE_GROUPS
        },
    }


def _subset(benchmark: BenchmarkV2, group: str) -> BenchmarkV2:
    cases = tuple(c for c in benchmark.cases if c.scored and lane_group_of(c.relevant_doc_ids) == group)
    return replace(benchmark, cases=cases)


def compare_systems(
    benchmark: BenchmarkV2,
    *,
    systems: Mapping[str, Retriever],
    lane_retrievers: Mapping[str, Retriever] | None,
    corpus_doc_ids: Sequence[str],
    fingerprint: str,
    benchmark_sha256: str | None,
    split: str | None = None,
    label_mode: str = "final",
    ks: Sequence[int] = DEFAULT_KS_V2,
    diagnostic_ks: Sequence[int] = DEFAULT_DIAGNOSTIC_KS,
    retrieve_limit: int = DEFAULT_RETRIEVE_LIMIT_V2,
    pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """
    Evaluate every system on `benchmark` and return a deterministic payload (see the module docstring).

    `systems` maps a label to a retriever that is run on EVERY case. `lane_retrievers` (optional, keys
    "weakness" and "attack_pattern", each a retriever; it is wrapped with a lane filter here) adds the
    lane-specific system. `pairs` lists (old, new) system labels for paired comparisons.
    """
    common = dict(
        corpus_doc_ids=list(corpus_doc_ids), fingerprint=fingerprint, split=split, label_mode=label_mode,
        ks=ks, diagnostic_ks=diagnostic_ks, retrieve_limit=retrieve_limit, benchmark_sha256=benchmark_sha256,
    )
    reports: dict[str, EvaluationReportV2] = {}
    for label, retriever in systems.items():
        reports[label] = evaluate_v2(benchmark, retriever, **common)

    payload: dict[str, Any] = {
        "benchmark_id": benchmark.benchmark_id,
        "benchmark_sha256": benchmark_sha256,
        "corpus_fingerprint": fingerprint,
        "label_mode": label_mode,
        "ks": list(ks),
        "retrieve_limit": retrieve_limit,
        "scored_cases": sum(1 for c in benchmark.cases if c.scored),
        "behaviour_cases": sum(1 for c in benchmark.cases if not c.scored),
        "lane_groups": {
            group: sum(1 for c in benchmark.cases if c.scored and lane_group_of(c.relevant_doc_ids) == group)
            for group in LANE_GROUPS
        },
        "caveat": MANDATORY_CAVEAT,
        "systems": {},
    }
    for label, report in reports.items():
        payload["systems"][label] = {
            "retriever": report.retriever,
            "summary": summarize_report(report),
            "report": report.to_dict(),
        }

    if lane_retrievers is not None:
        lane_map = {"cwe_only": (WEAKNESS_LANE, lane_retrievers["weakness"]),
                    "capec_only": (ATTACK_PATTERN_LANE, lane_retrievers["attack_pattern"])}
        lane_payload: dict[str, Any] = {
            "definition": "weakness-lane retriever on CWE-only cases; attack-pattern-lane retriever on CAPEC-only "
                          "cases; cases whose relevant documents span both lanes are not applicable here (see RRF)",
            "not_applicable_cases": [c.case_id for c in benchmark.cases
                                     if c.scored and lane_group_of(c.relevant_doc_ids) == "both"],
            "lanes": {},
        }
        for group, (lane, retriever) in lane_map.items():
            sub = _subset(benchmark, group)
            if not sub.cases:
                lane_payload["lanes"][group] = {"lane": lane.name, "cases": 0}
                continue
            report = evaluate_v2(sub, LaneRetriever(retriever, lane), **common)
            lane_payload["lanes"][group] = {
                "lane": lane.name, "retriever": report.retriever,
                "summary": summarize_rows(report.scored, report.ks), "report": report.to_dict(),
            }
        payload["lane_specific"] = lane_payload

    payload["paired_comparisons"] = [
        compare_reports(reports[old], reports[new]) for old, new in pairs if old in reports and new in reports
    ]
    first_rank = {label: {r.case_id: r.first_relevant_rank for r in report.scored} for label, report in reports.items()}
    case_ids = [c.case_id for c in benchmark.cases if c.scored]
    payload["first_relevant_rank_by_question"] = {
        case_id: {label: first_rank[label].get(case_id) for label in reports} for case_id in case_ids
    }
    return payload
