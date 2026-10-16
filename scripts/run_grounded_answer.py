"""
Phase 11D: retrieve CWE/CAPEC reference knowledge for an analysed incident and, optionally, generate a
citation-validated grounded answer.

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.run_grounded_answer incident_001

By default this builds the context ONLY (no language model is called, no key is needed):

    analysis/root_cause_analysis.json + analysis/incident_reconstruction.json
        -> deterministic queries -> FTS5 v1 keyword retrieval (the production baseline)
        -> a structured context with document ids, chunk ids, provenance and retrieved text.

Add `--provider groq` or `--provider ollama` to also generate an answer through the existing LLM
clients (`llm/groq.py` needs a key in `.env`; `llm/ollama.py` needs a local Ollama server). The answer is
accepted only if citation validation passes; otherwise it is reported as rejected. Neither provider
was called while this script was written; only mocked responses were tested.

Knowledge base: by default the 23 curated pinned MITRE records are built into a private IN-MEMORY
store from the archives in data/knowledge_sources/raw (like scripts.run_retrieval_baseline); pass
`--database` to use a store built earlier by scripts.ingest_mitre_corpus. Nothing is written unless
`--output` is given. `ground_truth/` is never read.

Retrieval method: always FTS5 v1 (`fts5-bm25-v1`). There is deliberately no option to select semantic
retrieval or RRF.

Exit status: 0 success; 1 failure; 2 command-line misuse (argparse); 3 generation was requested and at
least one answer was rejected or the model call failed.
"""

from __future__ import annotations

import argparse
import json
import re
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
from llm.base import LLMClient  # noqa: E402
from rag.context import ContextBuildError, ContextLimits, RagError  # noqa: E402
from rag.report import IncidentRagReport, run_incident_rag  # noqa: E402
from retrieval.contract import RetrievalError  # noqa: E402

DEFAULT_MANIFEST = _REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"
DEFAULT_CWE = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "cwec_v4.20.xml.zip"
DEFAULT_CAPEC = _REPO_ROOT / "data" / "knowledge_sources" / "raw" / "capec_v3.9.zip"
_INCIDENT_RE = re.compile(r"incident_[0-9]{3}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_grounded_answer",
        description="Retrieve CWE/CAPEC knowledge (FTS5 v1) for an incident and optionally generate a grounded answer.",
    )
    parser.add_argument("incident_id", help="for example incident_001")
    parser.add_argument("--incident-dir", type=Path, help="explicit incident directory (its name must equal incident_id)")
    parser.add_argument("--database", type=Path, help="an existing knowledge store (SQLite) instead of the in-memory build")
    parser.add_argument("--cwe-archive", type=Path, default=DEFAULT_CWE)
    parser.add_argument("--capec-archive", type=Path, default=DEFAULT_CAPEC)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--provider", choices=("none", "groq", "ollama"), default="none",
                        help="language-model provider; 'none' (default) builds the context only")
    parser.add_argument("--model", help="model name override for the provider")
    parser.add_argument("--max-documents", type=int, default=ContextLimits().max_documents)
    parser.add_argument("--max-chunks-per-document", type=int, default=ContextLimits().max_chunks_per_document)
    parser.add_argument("--max-passage-chars", type=int, default=None,
                        help="shorten each reference passage's text to at most this many characters in the "
                             "model prompt (ids, titles and provenance are kept; off by default)")
    parser.add_argument("--output", type=Path, help="write the full JSON report here")
    return parser


def make_llm_client(provider: str, model: str | None) -> LLMClient | None:
    """None for 'none'. The provider modules are imported lazily so the default path needs neither."""
    if provider == "none":
        return None
    if provider == "groq":
        from llm.groq import GroqLLMClient

        return GroqLLMClient(model=model)
    if provider == "ollama":
        from llm.ollama import OllamaLLMClient

        return OllamaLLMClient(model=model) if model else OllamaLLMClient()
    raise ValueError(f"unknown provider {provider!r}")


def _resolve_incident_dir(incident_id: str, explicit: Path | None) -> Path:
    if not _INCIDENT_RE.fullmatch(incident_id):
        raise ValueError("incident_id must look like incident_001")
    path = (explicit or _REPO_ROOT / incident_id).resolve()
    if path.name != incident_id:
        raise ValueError(f"the incident directory name must be {incident_id!r}")
    if not path.is_dir():
        raise ValueError(f"incident directory not found: {path}")
    return path


def _open_store(args: argparse.Namespace) -> KnowledgeStore:
    if args.database:
        if not args.database.is_file():
            raise ValueError(f"knowledge store not found: {args.database} (build it with scripts.ingest_mitre_corpus)")
        return KnowledgeStore(args.database)
    manifest = load_manifest(args.manifest)
    documents = build_corpus_documents(manifest, cwe_archive=args.cwe_archive, capec_archive=args.capec_archive)
    store = KnowledgeStore(":memory:")
    try:
        ingest_corpus_documents(manifest, documents, store=store)
    except Exception:
        store.close()
        raise
    return store


def render_summary(report: IncidentRagReport) -> str:
    lines = [
        f"Incident: {report.incident_id}",
        f"Retrieval method used: {', '.join(report.retrievers) or '(no query was built)'}"
        f"{'  [production baseline]' if report.uses_only_production_retriever else '  [NOT the production baseline]'}",
        f"Findings skipped (no informative subject): {len(report.skipped_findings)}",
    ]
    for context in report.contexts:
        lines.append(f"\nFinding {context.finding_index} - status {context.hypotheses[0].status} "
                     f"(confirmed: {context.hypotheses[0].is_confirmed})")
        lines.append(f"  observed evidence: {', '.join(context.evidence_ids) or 'none'}"
                     f"; unresolved ids: {', '.join(context.unresolved_evidence_ids) or 'none'}")
        lines.append(f"  retrieval: {context.retrieval_status}, {len(context.references)} passage(s) shown, "
                     f"{context.omitted_documents} document(s) and {context.omitted_chunks} chunk(s) left out")
        for ref in context.references:
            lines.append(f"    #{ref.document_rank} {ref.reference_id}  {ref.title}")
    if report.generation_requested:
        lines.append("")
        for result in report.results:
            lines.append(f"Answer for finding {result.context.finding_index}: {result.status} - {result.message}")
            if result.validation and not result.validation.passed:
                for issue in result.validation.issues[:10]:
                    lines.append(f"    [{issue.code}] {issue.location}: {issue.message}")
    else:
        lines.append("\nNo language model was called (--provider none).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        limits = ContextLimits(args.max_documents, args.max_chunks_per_document)
        incident_dir = _resolve_incident_dir(args.incident_id, args.incident_dir)
        try:
            llm_client = make_llm_client(args.provider, args.model)
        except ImportError as exc:
            raise ValueError(f"provider {args.provider!r} is not available: {exc}") from exc
        except Exception as exc:  # e.g. a missing API key; the provider's own message, no secrets
            raise ValueError(f"cannot create the {args.provider!r} client: {type(exc).__name__}: {exc}") from exc
        with _open_store(args) as store:
            report = run_incident_rag(incident_dir, store, llm_client, limits=limits,
                                      max_characters_per_passage=args.max_passage_chars)
        print(render_summary(report))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"report written to {args.output}")
    except (MitreSourceError, IngestionError, KnowledgeStoreError, RetrievalError, RagError,
            ContextBuildError, ValueError, OSError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if report.generation_requested and any(not r.accepted and r.status in {
            "llm_error", "rejected_invalid_response", "rejected_citations"} for r in report.results):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
