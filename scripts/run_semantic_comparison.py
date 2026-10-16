"""
Phase 11 (Milestone 11C), Phase A - DEVELOPMENT comparison of FTS5 v1, FTS5 v2, semantic retrieval and RRF.

    python -m scripts.run_semantic_comparison --split development --output-dir <dir>

This script is for development data only. It REFUSES the frozen final benchmark (v3) by file name, by SHA-256
and by format (`retrieval.freeze.refuse_final_benchmark`), and it does not import the v3 loader. The final
evaluation is a different script (`scripts/run_semantic_final.py`) that first checks a preregistration.

Systems compared on the same questions and labels:
    S1  fts5-bm25-v1                 the 11A baseline
    S2  fts5-bm25-v2 (top2_sum)      the 11B hardened baseline
    S3  semantic, all documents      BGE embeddings, best chunk per document
    S4  semantic, lane-specific      CWE-only questions on the weakness lane, CAPEC-only on the attack-pattern lane
    S5  RRF (k=60) of the two lane rankings of the same semantic retriever

`--instruction both` additionally runs the semantic systems WITHOUT the BGE query instruction. That ablation is
informational: the default (instruction on) was fixed before any 11C result, and nothing here changes it.

The v2 held-out split was already observed during Milestone 11B and is spent. It can be run only with
`--include-spent-heldout`, and the report labels it "observed, spent": it is never independent final evidence.

Outputs
  * a deterministic JSON report (no timestamps, no absolute paths): identical inputs give identical bytes;
  * optionally (`--timing-output`) a separate, NON-deterministic file with wall-clock seconds and peak memory.

Exit status: 0 success; 1 a run, model, cache or retrieval failure (or any question that errored); 2 refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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
from knowledge.chunking import CHUNKER_VERSION  # noqa: E402
from knowledge.store import KnowledgeStore, KnowledgeStoreError  # noqa: E402
from retrieval.benchmark_v2 import SPLITS, load_benchmark_v2  # noqa: E402
from retrieval.comparison import compare_systems, summarize_rows  # noqa: E402,F401
from retrieval.contract import RetrievalError  # noqa: E402
from retrieval.embeddings import Embedder  # noqa: E402
from retrieval.evaluation import corpus_fingerprint  # noqa: E402
from retrieval.evaluation_v2 import DEFAULT_DIAGNOSTIC_KS, DEFAULT_KS_V2, DEFAULT_RETRIEVE_LIMIT_V2, LABEL_MODES  # noqa: E402
from retrieval.freeze import FinalBenchmarkProtectedError, implementation_hashes, refuse_final_benchmark  # noqa: E402
from retrieval.fts5 import Fts5Retriever  # noqa: E402
from retrieval.fts5_v2 import Fts5RetrieverV2  # noqa: E402
from retrieval.lanes import LANES, RRF_DEFAULT_K  # noqa: E402
from retrieval.rrf import RrfRetriever  # noqa: E402
from retrieval.semantic import AGGREGATION, DEFAULT_MAX_CHUNKS_PER_ITEM, SemanticRetriever  # noqa: E402
from retrieval.vector_cache import CACHE_FORMAT, DEFAULT_CACHE_DIR, VectorCache  # noqa: E402

REPORT_FORMAT = "autoshield.semantic_comparison.dev.v1"
DEFAULT_MANIFEST = _REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"
DEFAULT_BENCHMARK = _REPO_ROOT / "data" / "retrieval_benchmark" / "retrieval_benchmark_v2.json"
DEFAULT_CWE = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "cwec_v4.20.xml.zip"
DEFAULT_CAPEC = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "capec_v3.9.zip"
LANE_DEPTH = 100

EmbedderFactory = Callable[[Path | None, bool], Embedder]


# ============================================================
# SHARED HELPERS (also used by scripts/run_semantic_final.py)
# ============================================================

def open_corpus(manifest_path: Path, cwe_archive: Path, capec_archive: Path) -> KnowledgeStore:
    """Build the pinned corpus in an in-memory store (the caller closes it)."""
    manifest = load_manifest(manifest_path)
    documents = build_corpus_documents(manifest, cwe_archive=cwe_archive, capec_archive=capec_archive)
    store = KnowledgeStore(":memory:")
    try:
        ingest_corpus_documents(manifest, documents, store=store)
    except BaseException:
        store.close()
        raise
    return store


def real_embedder(model_dir: Path | None, query_instruction: bool) -> Embedder:
    """The approved BGE model (imported lazily so the script's refusals and --help never load it)."""
    from retrieval.embeddings_st import BGE_QUERY_INSTRUCTION, SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(model_dir, query_instruction=BGE_QUERY_INSTRUCTION if query_instruction else "")


def cache_for(base_dir: Path, embedder: Embedder) -> VectorCache:
    """One cache directory per embedding fingerprint, so ablations never invalidate each other."""
    return VectorCache(base_dir / embedder.config.fingerprint()[:16])


def build_semantic(
    store: KnowledgeStore, embedder: Embedder, cache_base: Path, *, rebuild_cache: bool, rrf_k: int = RRF_DEFAULT_K
) -> tuple[SemanticRetriever, RrfRetriever]:
    semantic = SemanticRetriever(
        store, embedder, cache=cache_for(cache_base, embedder), rebuild_cache=rebuild_cache,
        max_chunks_per_item=DEFAULT_MAX_CHUNKS_PER_ITEM,
    )
    return semantic, RrfRetriever(semantic, k=rrf_k, lane_depth=LANE_DEPTH)


def retrieval_settings(rrf_k: int) -> dict[str, Any]:
    return {
        "chunk_to_document_rule": AGGREGATION,
        "max_chunks_per_item": DEFAULT_MAX_CHUNKS_PER_ITEM,
        "similarity": "cosine (dot product of unit vectors, float64)",
        "chunker_version": CHUNKER_VERSION,
        "fts5_v1": "fts5-bm25-v1",
        "fts5_v2": "fts5-bm25-v2, default aggregation (top2_sum)",
        "lanes": {lane.name: list(lane.doc_types) for lane in LANES},
        "lane_depth": LANE_DEPTH,
        "rrf": {"k": rrf_k, "fusion_level": "source documents, ranks only",
                "tie_break": "higher rrf; smaller best lane rank; lane order; doc_id"},
        "retrieve_limit": DEFAULT_RETRIEVE_LIMIT_V2,
        "ks": list(DEFAULT_KS_V2),
        "diagnostic_ks": list(DEFAULT_DIAGNOSTIC_KS),
        "cache_format": CACHE_FORMAT,
    }


def peak_working_set_bytes() -> int | None:
    """Peak resident memory of this process in bytes (Windows PeakWorkingSetSize; POSIX ru_maxrss), else None."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit handle must not be truncated to int
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
            return None
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak * (1 if sys.platform == "darwin" else 1024))
    except Exception:  # measurement is optional; never let it break a run
        return None


def render_summary(payload: dict[str, Any]) -> str:
    """Counts first, then means, per system, overall and by lane group (plain text for the console)."""
    ks = payload["ks"]
    lines = []
    for label, system in payload["systems"].items():
        overall = system["summary"]["overall"]
        lines.append(f"{label:24s} ({system['retriever']})")
        lines.append(
            "  overall n=%d  hits %s  full-recall %s  mean recall %s  MRR %s  no_results %d errors %d" % (
                overall["cases"],
                " ".join(f"@{k}={overall['hit_at_k'][str(k)]}" for k in ks),
                " ".join(f"@{k}={overall['full_recall_at_k'][str(k)]}" for k in ks),
                " ".join(f"@{k}={overall['mean_recall_at_k'][str(k)]}" for k in ks),
                overall["mrr"], overall["num_no_results"], overall["num_errors"],
            )
        )
        for group, summary in system["summary"]["by_lane_group"].items():
            if summary["cases"]:
                lines.append(
                    f"  {group:10s} n={summary['cases']:<3d} hits "
                    + " ".join(f"@{k}={summary['hit_at_k'][str(k)]}" for k in ks)
                    + f"  MRR {summary['mrr']}"
                )
    lane = payload.get("lane_specific")
    if lane:
        lines.append("S4 lane-specific (CWE-only -> weakness lane; CAPEC-only -> attack-pattern lane)")
        for group, entry in lane["lanes"].items():
            if entry.get("cases", 1) == 0:
                lines.append(f"  {group:10s} no cases")
                continue
            summary = entry["summary"]
            lines.append(
                f"  {group:10s} n={summary['cases']:<3d} hits "
                + " ".join(f"@{k}={summary['hit_at_k'][str(k)]}" for k in ks) + f"  MRR {summary['mrr']}"
            )
        lines.append(f"  not applicable (relevant documents in both lanes): {len(lane['not_applicable_cases'])} cases")
    for comparison in payload["paired_comparisons"]:
        counts = comparison["counts"]
        lines.append(
            f"paired {comparison['old_retriever']} -> {comparison['new_retriever']}: "
            f"improved {counts['improved']}, worsened {counts['worsened']}, unchanged {counts['unchanged']} "
            f"(MRR {comparison['delta_mrr']:+.3f})"
        )
    lines.append(payload["caveat"])
    return "\n".join(lines) + "\n"


# ============================================================
# COMMAND LINE
# ============================================================

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_semantic_comparison",
        description="Development comparison of FTS5 v1/v2, semantic retrieval and RRF (refuses the frozen v3 benchmark).",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cwe-archive", type=Path, default=DEFAULT_CWE)
    parser.add_argument("--capec-archive", type=Path, default=DEFAULT_CAPEC)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--split", choices=SPLITS, default="development")
    parser.add_argument("--include-spent-heldout", action="store_true",
                        help="required for --split heldout: that split was observed in 11B and is spent")
    parser.add_argument("--label-mode", choices=LABEL_MODES, default="final")
    parser.add_argument("--instruction", choices=("on", "off", "both"), default="on",
                        help="BGE query instruction; 'off' and 'both' are informational ablations")
    parser.add_argument("--model-dir", type=Path, help="local model directory (default: data/models/... or $AUTOSHIELD_MODEL_DIR)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="base directory for embedding caches (one subdirectory per embedding fingerprint)")
    parser.add_argument("--rebuild-cache", action="store_true", help="ignore existing cached vectors and recompute")
    parser.add_argument("--rrf-k", type=int, default=RRF_DEFAULT_K, help="RRF constant (development ablation only)")
    parser.add_argument("--output-dir", type=Path, help="write the deterministic JSON report here")
    parser.add_argument("--timing-output", type=Path, help="write a separate NON-deterministic timing/memory file here")
    return parser


def run(args: argparse.Namespace, embedder_factory: EmbedderFactory) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Run the comparison; returns (deterministic report, timing info, number of errored questions)."""
    timing: dict[str, Any] = {}
    started = time.perf_counter()
    benchmark = load_benchmark_v2(args.benchmark)
    benchmark_sha = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    store = open_corpus(args.manifest, args.cwe_archive, args.capec_archive)
    try:
        fingerprint = corpus_fingerprint(store)
        doc_ids = store.list_doc_ids()
        timing["corpus_seconds"] = round(time.perf_counter() - started, 3)

        variants = {"on": [True], "off": [False], "both": [True, False]}[args.instruction]
        embedders: dict[bool, Embedder] = {}
        semantic: dict[bool, Any] = {}
        rrf: dict[bool, Any] = {}
        cache_reports: dict[str, Any] = {}
        for with_instruction in variants:
            phase = time.perf_counter()
            embedder = embedder_factory(args.model_dir, with_instruction)
            embedders[with_instruction] = embedder
            semantic[with_instruction], rrf[with_instruction] = build_semantic(
                store, embedder, args.cache_dir, rebuild_cache=args.rebuild_cache, rrf_k=args.rrf_k
            )
            key = "instruction_on" if with_instruction else "instruction_off"
            report = semantic[with_instruction].cache_report
            cache_reports[key] = None if report is None else report.to_dict()
            timing[f"embedding_setup_seconds_{key}"] = round(time.perf_counter() - phase, 3)

        primary = variants[0]
        systems: dict[str, Any] = {
            "S1_fts5_v1": Fts5Retriever(store),
            "S2_fts5_v2": Fts5RetrieverV2(store),
            "S3_semantic": semantic[primary],
            "S5_rrf": rrf[primary],
        }
        ablation_labels: list[str] = []
        if args.instruction == "both":
            systems["S3x_semantic_no_instruction"] = semantic[False]
            systems["S5x_rrf_no_instruction"] = rrf[False]
            ablation_labels = ["S3x_semantic_no_instruction", "S5x_rrf_no_instruction"]
        pairs = [("S1_fts5_v1", "S3_semantic"), ("S2_fts5_v2", "S3_semantic"), ("S3_semantic", "S5_rrf"),
                 ("S2_fts5_v2", "S5_rrf"), ("S1_fts5_v1", "S5_rrf")]
        if ablation_labels:
            pairs.append(("S3x_semantic_no_instruction", "S3_semantic"))

        phase = time.perf_counter()
        comparison = compare_systems(
            benchmark, systems=systems,
            lane_retrievers={"weakness": semantic[primary], "attack_pattern": semantic[primary]},
            corpus_doc_ids=doc_ids, fingerprint=fingerprint, benchmark_sha256=benchmark_sha,
            split=args.split, label_mode=args.label_mode, pairs=pairs,
        )
        timing["evaluation_seconds"] = round(time.perf_counter() - phase, 3)
        chunk_count = semantic[primary].chunk_count
        config = embedders[primary].config
        report = {
            "format": REPORT_FORMAT,
            "purpose": "development comparison; not final evidence and not independent of the code's development",
            "benchmark": {
                "id": benchmark.benchmark_id, "sha256": benchmark_sha, "split": args.split,
                "split_status": "observed_spent_heldout" if args.split == "heldout" else "development",
                "label_mode": args.label_mode,
            },
            "corpus": {"fingerprint": fingerprint, "documents": len(doc_ids), "chunks": chunk_count},
            "embedding": {
                "identity": config.identity(), "fingerprint": config.fingerprint(),
                "runtime_versions": dict(embedders[primary].runtime_versions),
                "query_instruction": "on" if primary else "off",
            },
            "cache": cache_reports,
            "retrieval_settings": retrieval_settings(args.rrf_k),
            "ablation_systems": ablation_labels,
            "ablation_note": "informational only; the default (query instruction on) was fixed before any 11C result",
            "implementation_files": implementation_hashes(_REPO_ROOT),
            "comparison": comparison,
        }
        errors = sum(
            system["summary"]["overall"]["num_errors"] for system in comparison["systems"].values()
        )
        return report, timing, errors
    finally:
        store.close()


def main(argv: Sequence[str] | None = None, *, embedder_factory: EmbedderFactory = real_embedder) -> int:
    args = _parser().parse_args(argv)
    try:
        refuse_final_benchmark(args.benchmark)
    except FinalBenchmarkProtectedError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    if args.split == "heldout" and not args.include_spent_heldout:
        print("Refused: the v2 held-out split was observed in Milestone 11B and is spent. Pass "
              "--include-spent-heldout to run it as supplementary, non-independent information.", file=sys.stderr)
        return 2
    try:
        report, timing, errors = run(args, embedder_factory)
    except (MitreSourceError, IngestionError, KnowledgeStoreError, RetrievalError, OSError, ValueError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    comparison = report["comparison"]
    print(f"benchmark {report['benchmark']['id']} sha256 {report['benchmark']['sha256']}")
    print(f"split {args.split} ({report['benchmark']['split_status']}), labels {args.label_mode}")
    print(f"corpus fingerprint {report['corpus']['fingerprint']} "
          f"({report['corpus']['documents']} documents, {report['corpus']['chunks']} chunks)")
    print(f"embedding {report['embedding']['identity']['model_id']}@{report['embedding']['identity']['model_revision']} "
          f"fingerprint {report['embedding']['fingerprint'][:16]} instruction {report['embedding']['query_instruction']}")
    for key, cache in report["cache"].items():
        if cache:
            print(f"cache[{key}]: reused {cache['reused']} created {cache['created']} "
                  f"invalidated {cache['invalidated']} rejected {cache['rejected']} (before: {cache['status_before']})")
    print()
    print(render_summary(comparison))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = f"dev_{args.split}_{args.label_mode}_instruction-{args.instruction}.json"
        (args.output_dir / name).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_dir / name}")
    peak = peak_working_set_bytes()
    timing["peak_working_set_bytes"] = peak
    print(f"timing (non-deterministic): {json.dumps(timing, sort_keys=True)}")
    if args.timing_output:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        args.timing_output.write_text(json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        print(f"Result: FAILED - {errors} question(s) ended in a retrieval error", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
