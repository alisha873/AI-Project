"""
Phase 11 (Milestone 11C), Phases B and C - preregister, then run the frozen final benchmark (v3) exactly once.

    python -m scripts.run_semantic_final freeze     # Phase B: write the preregistration record (no v3 cases are read)
    python -m scripts.run_semantic_final run        # Phase C: verify everything, then run all systems once

`freeze` hashes the benchmark file (bytes only), records the live state (corpus, model, dependencies, cache, retrieval
settings, metrics, and a SHA-256 of every implementation file) and refuses to overwrite an existing record.

`run` refuses to start unless ALL of these hold (each is enforced in code, not only documented):
  1. the preregistration exists, is unedited, and records the pinned v3 hash;
  2. the benchmark file's SHA-256 equals the pinned value (the loader refuses anything else);
  3. the live state (implementation file hashes, model, dependency versions, corpus fingerprint, cache format,
     retrieval settings, metrics) equals the recorded state, field by field;
  4. no final report exists yet (the report is created exclusively; it is never overwritten).
It then runs FTS5 v1, FTS5 v2, semantic (all documents), semantic lane-specific and RRF on the same questions and
labels, for the strict label set (primary) and the strict-plus-lenient set (sensitivity), and writes ONE
deterministic report. Nothing is tuned: there is no option that changes a retrieval setting.

This is a process guard, not a cryptographic one: someone determined can delete the record and the report and start
over, and the labels themselves are single-pass AI labels that were not independently reviewed.

Exit status: 0 success; 1 a run or environment failure; 2 refused by a guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.mitre_xml import MitreSourceError  # noqa: E402
from ingestion.pipeline import IngestionError  # noqa: E402
from knowledge.chunking import CHUNKER_VERSION  # noqa: E402
from knowledge.store import KnowledgeStore, KnowledgeStoreError  # noqa: E402
from retrieval import freeze  # noqa: E402
from retrieval.benchmark_v3 import LABEL_SETS, load_benchmark_v3  # noqa: E402
from retrieval.comparison import MANDATORY_CAVEAT, compare_systems  # noqa: E402
from retrieval.contract import RetrievalError  # noqa: E402
from retrieval.embeddings import Embedder  # noqa: E402
from retrieval.evaluation import corpus_fingerprint  # noqa: E402
from retrieval.freeze import (  # noqa: E402
    FinalBenchmarkProtectedError,
    FrozenBenchmarkError,
    PreregistrationError,
    build_frozen_state,
    build_preregistration,
    check_pinned_requirements,
    git_head,
    implementation_hashes,
    require_valid_preregistration,
)
from retrieval.fts5 import Fts5Retriever  # noqa: E402
from retrieval.fts5_v2 import Fts5RetrieverV2  # noqa: E402
from retrieval.lanes import RRF_DEFAULT_K  # noqa: E402
from retrieval.vector_cache import CACHE_FORMAT, DEFAULT_CACHE_DIR  # noqa: E402
from scripts.run_semantic_comparison import (  # noqa: E402
    DEFAULT_CAPEC,
    DEFAULT_CWE,
    DEFAULT_MANIFEST,
    build_semantic,
    open_corpus,
    real_embedder,
    render_summary,
    retrieval_settings,
)

REPORT_FORMAT = "autoshield.semantic_comparison.final.v1"
DEFAULT_BENCHMARK = _REPO_ROOT / "data" / "retrieval_benchmark" / "preflight_11c" / freeze.FROZEN_V3_FILENAME
DEFAULT_RESULTS_DIR = _REPO_ROOT / "data" / "retrieval_benchmark" / "results_11c" / "final"
DEFAULT_PREREGISTRATION = DEFAULT_RESULTS_DIR / "preregistration_11c.json"
DEFAULT_REPORT = DEFAULT_RESULTS_DIR / "final_report_v3.json"
REQUIREMENTS_FILE = "requirements-embeddings.txt"

EmbedderFactory = Callable[[Path | None, bool], Embedder]
CorpusOpener = Callable[[argparse.Namespace], KnowledgeStore]


class RefusedError(RetrievalError):
    """A guard stopped the run (exit status 2)."""


def _default_corpus_opener(args: argparse.Namespace) -> KnowledgeStore:
    return open_corpus(args.manifest, args.cwe_archive, args.capec_archive)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_state(
    args: argparse.Namespace,
    *,
    store: KnowledgeStore,
    embedder: Embedder,
    repo_root: Path,
    benchmark_sha256: str,
) -> tuple[dict[str, Any], Any, Any]:
    """
    The live state that the preregistration freezes (see `retrieval.freeze.build_frozen_state`), plus the semantic
    and RRF retrievers that were built to obtain it (so `run` reuses exactly those).
    """
    installed, problems = check_pinned_requirements(repo_root / REQUIREMENTS_FILE)
    if problems:
        raise RefusedError("the installed packages do not match requirements-embeddings.txt: " + "; ".join(problems))
    semantic, rrf = build_semantic(store, embedder, args.cache_dir, rebuild_cache=False, rrf_k=RRF_DEFAULT_K)
    cache_report = semantic.cache_report
    frozen = build_frozen_state(
        benchmark_sha256=benchmark_sha256,
        corpus_fingerprint=corpus_fingerprint(store),
        document_count=len(store.list_doc_ids()),
        chunk_count=semantic.chunk_count,
        chunker={"version": CHUNKER_VERSION},
        embedding_identity=embedder.config.identity(),
        embedding_fingerprint=embedder.config.fingerprint(),
        runtime_versions=dict(embedder.runtime_versions),
        installed_pins=installed,
        cache_format=CACHE_FORMAT,
        cache_index_fingerprint=cache_report.index_fingerprint if cache_report else "",
        retrieval_settings=retrieval_settings(RRF_DEFAULT_K),
        implementation=implementation_hashes(repo_root),
    )
    return frozen, semantic, rrf


def _benchmark_hash_or_refuse(path: Path) -> str:
    """Hash the file's bytes (no case is parsed) and refuse anything that is not the pinned v3 file."""
    if not path.is_file():
        raise RefusedError(f"benchmark file {path} does not exist")
    digest = _file_sha256(path)
    if digest != freeze.FROZEN_V3_SHA256:
        raise RefusedError(
            f"benchmark file SHA-256 is {digest}, expected the frozen value {freeze.FROZEN_V3_SHA256}"
        )
    return digest


def _write_exclusive(path: Path, text: str) -> None:
    """Create `path` (never overwrite) and write `text`; raises `FileExistsError` if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


# ============================================================
# PHASE B: FREEZE
# ============================================================

def do_freeze(args: argparse.Namespace, embedder_factory: EmbedderFactory, corpus_opener: CorpusOpener,
              repo_root: Path) -> int:
    if args.preregistration.exists():
        raise RefusedError(
            f"{args.preregistration} already exists; a preregistration is never overwritten. "
            "Starting over requires a new benchmark version."
        )
    benchmark_sha256 = _benchmark_hash_or_refuse(args.benchmark)
    store = corpus_opener(args)
    try:
        embedder = embedder_factory(args.model_dir, True)
        frozen, _, _ = collect_state(args, store=store, embedder=embedder, repo_root=repo_root,
                                     benchmark_sha256=benchmark_sha256)
    finally:
        store.close()
    record = build_preregistration(
        frozen, frozen_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), head=git_head(repo_root)
    )
    _write_exclusive(args.preregistration, _json_text(record))
    print(f"preregistration written: {args.preregistration}")
    print(f"  benchmark sha256        {frozen['benchmark']['sha256']}")
    print(f"  corpus fingerprint      {frozen['corpus']['fingerprint']}")
    print(f"  embedding fingerprint   {frozen['embedding']['fingerprint']}")
    print(f"  cache index fingerprint {frozen['cache']['index_fingerprint']}")
    print(f"  implementation          {frozen['implementation']['fingerprint']} "
          f"({len(frozen['implementation']['files'])} files)")
    print(f"  frozen block sha256     {record['frozen_sha256']}")
    print(f"  record sha256           {record['record_sha256']}")
    print(f"  git HEAD (informational) {record['git_head_informational']}")
    return 0


# ============================================================
# PHASE C: RUN ONCE
# ============================================================

def do_run(args: argparse.Namespace, embedder_factory: EmbedderFactory, corpus_opener: CorpusOpener,
           repo_root: Path) -> int:
    if args.report.exists():
        raise RefusedError(f"{args.report} already exists: the final evaluation is run once and never repeated")
    benchmark_sha256 = _benchmark_hash_or_refuse(args.benchmark)
    store = corpus_opener(args)
    try:
        embedder = embedder_factory(args.model_dir, True)
        frozen, semantic, rrf = collect_state(args, store=store, embedder=embedder, repo_root=repo_root,
                                              benchmark_sha256=benchmark_sha256)
        record = require_valid_preregistration(args.preregistration, frozen)  # raises with every difference

        benchmark = load_benchmark_v3(args.benchmark)  # verifies the pinned hash again before parsing
        if benchmark.corpus_fingerprint != frozen["corpus"]["fingerprint"]:
            raise RefusedError("the benchmark was written for a different corpus than the one loaded")

        systems = {
            "S1_fts5_v1": Fts5Retriever(store),
            "S2_fts5_v2": Fts5RetrieverV2(store),
            "S3_semantic": semantic,
            "S5_rrf": rrf,
        }
        pairs = [("S1_fts5_v1", "S3_semantic"), ("S2_fts5_v2", "S3_semantic"), ("S3_semantic", "S5_rrf"),
                 ("S2_fts5_v2", "S5_rrf"), ("S1_fts5_v1", "S5_rrf")]
        results: dict[str, Any] = {}
        for label_set in LABEL_SETS:
            results[label_set] = compare_systems(
                benchmark.as_benchmark_v2(label_set), systems=systems,
                lane_retrievers={"weakness": semantic, "attack_pattern": semantic},
                corpus_doc_ids=store.list_doc_ids(), fingerprint=frozen["corpus"]["fingerprint"],
                benchmark_sha256=benchmark_sha256, split=None, label_mode="final", pairs=pairs,
            )
        errors = sum(
            system["summary"]["overall"]["num_errors"]
            for payload in results.values() for system in payload["systems"].values()
        )
        cache_report = semantic.cache_report.to_dict() if semantic.cache_report else None
    finally:
        store.close()

    report = {
        "format": REPORT_FORMAT,
        "purpose": "the one-time final evaluation on the frozen v3 benchmark, run after preregistration",
        "preregistration": {
            "record_sha256": record["record_sha256"], "frozen_sha256": record["frozen_sha256"],
            "frozen_at_utc": record["frozen_at_utc"],
        },
        "benchmark": {"id": benchmark.benchmark_id, "sha256": benchmark_sha256,
                      "scored_cases": sum(c.scored for c in benchmark.cases),
                      "behaviour_cases": sum(not c.scored for c in benchmark.cases)},
        "primary_label_set": "strict",
        "label_sets": results,
        "cache": cache_report,
        "limitations": [
            MANDATORY_CAVEAT,
            "The v3 labels are single-pass AI labels, not blind and not independently reviewed.",
            "No statistical significance is claimed; read the counts, not only the percentages.",
            "Behaviour cases are reported separately and no abstention rule was chosen, so no abstention claim is made.",
            "A process guard, not a cryptographic one, kept the configuration frozen.",
        ],
    }
    _write_exclusive(args.report, _json_text(report))  # exclusive: a second run can never overwrite this
    print(f"final report written: {args.report}")
    print(f"preregistration record sha256 {record['record_sha256']}")
    for label_set, payload in results.items():
        print(f"\n===== label set: {label_set}{' (primary)' if label_set == 'strict' else ' (sensitivity)'} =====")
        print(render_summary(payload))
    if errors:
        print(f"Result: FAILED - {errors} question(s) ended in a retrieval error", file=sys.stderr)
        return 1
    return 0


# ============================================================
# COMMAND LINE
# ============================================================

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_semantic_final",
        description="Preregister (freeze) and run the frozen final benchmark once.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("freeze", "write the preregistration record (reads no benchmark cases)"),
                       ("run", "verify the preregistration, then run the final evaluation once")):
        command = sub.add_parser(name, help=text, description=text)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--cwe-archive", type=Path, default=DEFAULT_CWE)
        command.add_argument("--capec-archive", type=Path, default=DEFAULT_CAPEC)
        command.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
        command.add_argument("--model-dir", type=Path)
        command.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
        command.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
        if name == "run":
            command.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    embedder_factory: EmbedderFactory = real_embedder,
    corpus_opener: CorpusOpener = _default_corpus_opener,
    repo_root: Path = _REPO_ROOT,
) -> int:
    args = _parser().parse_args(argv)
    handler = do_freeze if args.command == "freeze" else do_run
    try:
        return handler(args, embedder_factory, corpus_opener, repo_root)
    except (RefusedError, PreregistrationError, FrozenBenchmarkError, FinalBenchmarkProtectedError, FileExistsError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except (MitreSourceError, IngestionError, KnowledgeStoreError, RetrievalError, OSError, ValueError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
