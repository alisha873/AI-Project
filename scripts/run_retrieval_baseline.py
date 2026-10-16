"""
Run the FTS5 retrieval baseline against the pinned MITRE corpus.

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.run_retrieval_baseline

What it does: verifies and parses the two pinned archives (exactly like
`scripts.ingest_mitre_corpus`), ingests the 23 curated records into a PRIVATE
IN-MEMORY knowledge store (no database file is created or modified), loads the
fixed benchmark `data/retrieval_benchmark/retrieval_benchmark_v1.json`, runs
every question through `Fts5Retriever`, and prints Recall@k and MRR computed
from the documents actually returned. `--output` also writes the deterministic
JSON report (no timestamps) so two runs on the same inputs are byte-identical.

Local files only; no network, no model, no embeddings.

Exit status: 0 on success; 1 for any failure (manifest, archive, benchmark,
store or evaluation); 2 for command-line misuse.
"""

from __future__ import annotations

import argparse
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
from retrieval.contract import RetrievalError  # noqa: E402
from retrieval.evaluation import (  # noqa: E402
    DEFAULT_KS,
    DEFAULT_RETRIEVE_LIMIT,
    corpus_fingerprint,
    evaluate_benchmark,
    load_benchmark,
)
from retrieval.fts5 import Fts5Retriever  # noqa: E402

DEFAULT_MANIFEST = _REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"
DEFAULT_BENCHMARK = _REPO_ROOT / "data" / "retrieval_benchmark" / "retrieval_benchmark_v1.json"
DEFAULT_CWE = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "cwec_v4.20.xml.zip"
DEFAULT_CAPEC = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "capec_v3.9.zip"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_retrieval_baseline",
        description="Measure the FTS5 keyword baseline (Recall@k, MRR) on the pinned MITRE corpus.",
    )
    parser.add_argument("--cwe-archive", type=Path, default=DEFAULT_CWE)
    parser.add_argument("--capec-archive", type=Path, default=DEFAULT_CAPEC)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, help="also write the deterministic JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        benchmark = load_benchmark(args.benchmark)
        documents = build_corpus_documents(
            manifest, cwe_archive=args.cwe_archive, capec_archive=args.capec_archive
        )
        with KnowledgeStore(":memory:") as store:
            ingest_corpus_documents(manifest, documents, store=store)
            report = evaluate_benchmark(
                benchmark,
                Fts5Retriever(store),
                corpus_doc_ids=store.list_doc_ids(),
                ks=DEFAULT_KS,
                retrieve_limit=DEFAULT_RETRIEVE_LIMIT,
                fingerprint=corpus_fingerprint(store),
            )
        print(report.render_text())
        print(f"corpus fingerprint: {report.corpus_fingerprint}")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report.to_json(), encoding="utf-8")
            print(f"report written to {args.output}")
    except (MitreSourceError, IngestionError, KnowledgeStoreError, RetrievalError, OSError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
