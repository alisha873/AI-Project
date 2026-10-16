"""
Compare the frozen 11A keyword baseline with the 11B hardened retriever on
benchmark v2, using one corpus build and one corpus fingerprint.

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.run_retrieval_comparison

By default only the DEVELOPMENT split is evaluated, because that is the only
split a retriever may be tuned on. `--split heldout` or `--split all` is refused
unless `--allow-heldout` is passed: run it only after the retrieval code is
frozen. Every run prints a fingerprint of the retrieval source files and the
SHA-256 of the benchmark file, so "the implementation was frozen" and "the
labels were not edited" can be checked afterwards.

Retrievers compared (same store, same questions, same labels):
  v1            `fts5-bm25-v1`, the Milestone 11A baseline (unchanged code)
  v2-bestchunk  `fts5-bm25-v2-bestchunk`, v2 query handling with v1's document ranking
  v2            `fts5-bm25-v2`, v2 query handling plus top-2-chunk document scoring

The corpus is built in memory from the two verified archives (exactly like
`scripts.run_retrieval_baseline`); no database file is created or modified. With
`--output-dir` the deterministic JSON reports (no timestamps) are written too.
`--incident-dir` additionally prints the ranking for the query built from a real
incident's Phase 8/9 analysis files (a diagnostic only: no labels are involved).

Local files only; no network, no model, no embeddings.
Exit status: 0 on success; 1 for any failure; 2 for command-line misuse.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.mitre_corpus import (  # noqa: E402
    build_corpus_documents,
    ingest_corpus_documents,
    load_manifest,
)
from ingestion.mitre_xml import MitreSourceError  # noqa: E402
from ingestion.pipeline import IngestionError  # noqa: E402
from knowledge.store import KnowledgeStore, KnowledgeStoreError  # noqa: E402
from retrieval.benchmark_v2 import SPLITS, load_benchmark_v2  # noqa: E402
from retrieval.contract import RetrievalError, RetrievalQuery  # noqa: E402
from retrieval.evaluation import corpus_fingerprint  # noqa: E402
from retrieval.evaluation_v2 import LABEL_MODES, compare_reports, evaluate_v2  # noqa: E402
from retrieval.fts5 import Fts5Retriever  # noqa: E402
from retrieval.fts5_v2 import Fts5RetrieverV2  # noqa: E402
from retrieval.query_builder import build_queries_for_incident  # noqa: E402

DEFAULT_MANIFEST = _REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"
DEFAULT_BENCHMARK = _REPO_ROOT / "data" / "retrieval_benchmark" / "retrieval_benchmark_v2.json"
DEFAULT_CWE = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "cwec_v4.20.xml.zip"
DEFAULT_CAPEC = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "capec_v3.9.zip"
RETRIEVER_KEYS = ("v1", "v2-bestchunk", "v2")


def code_fingerprint() -> str:
    """SHA-256 over the retrieval package's Python sources (sorted by name), to prove a freeze."""
    digest = hashlib.sha256()
    for path in sorted((_REPO_ROOT / "retrieval").glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _make_retriever(key: str, store: KnowledgeStore):
    if key == "v1":
        return Fts5Retriever(store)
    if key == "v2-bestchunk":
        return Fts5RetrieverV2(store, aggregation="best_chunk")
    return Fts5RetrieverV2(store)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_retrieval_comparison",
        description="Compare the 11A FTS5 baseline with the 11B hardened retriever on benchmark v2.",
    )
    parser.add_argument("--cwe-archive", type=Path, default=DEFAULT_CWE)
    parser.add_argument("--capec-archive", type=Path, default=DEFAULT_CAPEC)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="development")
    parser.add_argument("--allow-heldout", action="store_true",
                        help="required for --split heldout or all; use only after the code is frozen")
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="final")
    parser.add_argument("--retrievers", nargs="+", choices=RETRIEVER_KEYS, default=list(RETRIEVER_KEYS))
    parser.add_argument("--output-dir", type=Path, help="also write the deterministic JSON reports here")
    parser.add_argument("--incident-dir", type=Path, help="diagnostic: rank the query built from this incident")
    parser.add_argument("--show-changes", action="store_true", help="list every case whose outcome changed")
    return parser


def _comparison_text(comparison: dict, show_changes: bool) -> str:
    counts = comparison["counts"]
    lines = [
        f"Comparison {comparison['old_retriever']} -> {comparison['new_retriever']} "
        f"({comparison['split'] or 'all splits'}, labels {comparison['label_mode']}):",
        f"  cases improved {counts['improved']}, worsened {counts['worsened']}, unchanged {counts['unchanged']} "
        f"(reference k={comparison['reference_k']})",
        "  delta mean recall: "
        + "  ".join(f"R@{k}={v:+.3f}" for k, v in comparison["delta_mean_recall_at_k"].items())
        + f"  MRR={comparison['delta_mrr']:+.3f}"
        + "  [diag "
        + "  ".join(f"R@{k}={v:+.3f}" for k, v in comparison["delta_diagnostic_mean_recall_at_k"].items())
        + "]",
        f"  {comparison['caveat']}",
    ]
    for change in comparison["changes"]:
        if show_changes and change["outcome"] != "unchanged":
            lines.append(
                f"    {change['case_id']:9s} {change['outcome']:9s} first relevant rank "
                f"{change['old_first_relevant_rank']} -> {change['new_first_relevant_rank']}"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.split in ("heldout", "all") and not args.allow_heldout:
        print("Refused: the held-out split may only be run once, after the retrieval code is frozen. "
              "Pass --allow-heldout to confirm.", file=sys.stderr)
        return 2
    split = None if args.split == "all" else args.split
    try:
        manifest = load_manifest(args.manifest)
        benchmark = load_benchmark_v2(args.benchmark)
        benchmark_hash = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
        documents = build_corpus_documents(
            manifest, cwe_archive=args.cwe_archive, capec_archive=args.capec_archive
        )
        with KnowledgeStore(":memory:") as store:
            ingest_corpus_documents(manifest, documents, store=store)
            fingerprint = corpus_fingerprint(store)
            print(f"retrieval code fingerprint: {code_fingerprint()}")
            print(f"benchmark file sha256:      {benchmark_hash}")
            print(f"corpus fingerprint:         {fingerprint}\n")
            reports = {}
            for key in args.retrievers:
                report = evaluate_v2(
                    benchmark, _make_retriever(key, store), corpus_doc_ids=store.list_doc_ids(),
                    fingerprint=fingerprint, split=split, label_mode=args.label_mode,
                    benchmark_sha256=benchmark_hash,
                )
                reports[key] = report
                print(report.render_text())
                if args.output_dir:
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    name = f"{key}_{args.split}_{args.label_mode}.json"
                    (args.output_dir / name).write_text(report.to_json(), encoding="utf-8")
            for old, new in (("v1", "v2"), ("v1", "v2-bestchunk"), ("v2-bestchunk", "v2")):
                if old in reports and new in reports:
                    comparison = compare_reports(reports[old], reports[new])
                    print(_comparison_text(comparison, args.show_changes))
            if args.incident_dir:
                built = build_queries_for_incident(args.incident_dir)
                for incident_query in built.queries:
                    print(f"incident diagnostic ({built.incident_id}, finding {incident_query.finding_index}, "
                          f"{incident_query.finding_status}):")
                    for key in args.retrievers:
                        result = _make_retriever(key, store).retrieve(
                            RetrievalQuery(text=incident_query.query.text, phrases=incident_query.query.phrases, limit=10)
                        )
                        print(f"  {key:13s} " + ", ".join(result.doc_ids))
    except (MitreSourceError, IngestionError, KnowledgeStoreError, RetrievalError, OSError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
