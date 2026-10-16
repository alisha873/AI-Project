"""
Phase 12: propose evidence-grounded remediations for an analysed incident (ADVISORY ONLY).

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.run_remediation_plan incident_001 --provider groq --output evaluation/results/incident_001_remediation_plan_live.json

What it does, per usable finding:

    analysis/*.json -> FTS5 v1 retrieval -> grounded context -> validated Phase 11D answer
        -> planner input (relevance filter, configuration facts, sufficiency gate)
        -> remediation plan (one model call) -> grounding validation

With `--provider none` (the default) NO language model is called and nothing is planned: the command only
confirms that the analysis files can be read and the retrieval context built. With `--provider groq` or
`--provider ollama` the existing LLM clients are used (Groq needs a key in the git-ignored `.env`); each
finding costs two calls (grounded answer, then plan). A plan is accepted only if grounding validation passes.

Nothing is executed: no file or container is changed and no test is launched. The knowledge base is built
in memory from the pinned archives (like scripts.run_grounded_answer) unless `--database` is given. Nothing
is written unless `--output` is given. `ground_truth/` is never read. Retrieval is always FTS5 v1.

Exit status: 0 success (an explicit insufficient-evidence plan is a success); 1 failure; 2 command-line
misuse (argparse); 3 planning was requested and a model call failed or an answer or plan was rejected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.mitre_xml import MitreSourceError  # noqa: E402
from ingestion.pipeline import IngestionError  # noqa: E402
from knowledge.store import KnowledgeStoreError  # noqa: E402
from rag.context import ContextBuildError, ContextLimits, RagError  # noqa: E402
from remediation.configuration import ConfigurationError  # noqa: E402
from remediation.report import IncidentRemediationReport, run_incident_remediation  # noqa: E402
from retrieval.contract import RetrievalError  # noqa: E402
from scripts.run_grounded_answer import (  # noqa: E402
    DEFAULT_CAPEC,
    DEFAULT_CWE,
    DEFAULT_MANIFEST,
    _open_store,
    _resolve_incident_dir,
    make_llm_client,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_remediation_plan",
        description="Propose evidence-grounded remediations for an incident (advisory only; nothing is executed).",
    )
    parser.add_argument("incident_id", help="for example incident_001")
    parser.add_argument("--incident-dir", type=Path, help="explicit incident directory (its name must equal incident_id)")
    parser.add_argument("--database", type=Path, help="an existing knowledge store (SQLite) instead of the in-memory build")
    parser.add_argument("--cwe-archive", type=Path, default=DEFAULT_CWE)
    parser.add_argument("--capec-archive", type=Path, default=DEFAULT_CAPEC)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--provider", choices=("none", "groq", "ollama"), default="none",
                        help="language-model provider; 'none' (default) calls no model and plans nothing")
    parser.add_argument("--model", help="model name override for the provider")
    parser.add_argument("--max-documents", type=int, default=ContextLimits().max_documents)
    parser.add_argument("--max-chunks-per-document", type=int, default=ContextLimits().max_chunks_per_document)
    parser.add_argument("--output", type=Path, help="write the full JSON report here")
    return parser


def render_summary(report: IncidentRemediationReport) -> str:
    lines = [
        f"Incident: {report.incident_id}",
        f"Retrieval method used: {', '.join(report.retrievers) or '(no query was built)'}"
        f"{'  [production baseline]' if report.uses_only_production_retriever else '  [NOT the production baseline]'}",
        f"Findings skipped (no informative subject): {len(report.skipped_findings)}",
        "ADVISORY ONLY: nothing was executed.",
    ]
    if not report.planning_requested:
        lines.append("\nNo language model was called (--provider none); no remediation was planned.")
        return "\n".join(lines)
    answers = dict(report.grounded_answer_statuses)
    upstream = {item.finding_index: item for item in report.grounded_answer_diagnostics}
    if report.nothing_planned_reason:
        lines.append(f"\nNOTHING WAS PLANNED: {report.nothing_planned_reason}.")
        for index, reason in report.skipped_findings:
            lines.append(f"  finding {index} skipped: {reason}")
    for result in report.results:
        lines.append(f"\nFinding {result.finding_index}: Phase 11D answer {answers.get(result.finding_index, 'n/a')}; "
                     f"plan {result.status} - {result.message}")
        failure = upstream.get(result.finding_index)
        if failure is not None:
            lines.append(f"  UPSTREAM (Phase 11D) answer not accepted [{failure.status}]"
                         f"{f' from {failure.llm}' if failure.llm else ''}: {failure.message}")
            if failure.issue_codes:
                lines.append(f"    validation issue codes: {', '.join(failure.issue_codes)}")
            lines.append("    The planner's model was "
                         f"{'called' if failure.planner_model_called else 'NOT called'} for this finding.")
        if result.validation and not result.validation.passed:
            for issue in result.validation.issues[:10]:
                lines.append(f"    [{issue.code}] {issue.location}: {issue.message}")
        plan = result.plan
        if plan is None:
            continue
        for reason in plan.insufficient_reasons:
            lines.append(f"  insufficient evidence [{reason.code}]: {reason.message}")
        summary = plan.input_summary
        lines.append(f"  configuration: {summary.configuration_status}; usable passages: "
                     f"{', '.join(summary.usable_reference_ids) or 'none'}; excluded: "
                     f"{', '.join(f'{e.reference_id} ({e.label})' for e in summary.excluded_references) or 'none'}")
        for proposal in plan.proposals:
            basis = proposal.priority_basis
            lines.append(f"  #{proposal.priority_rank} {proposal.proposal_id}  {proposal.title}")
            lines.append(f"      {basis.evidence_linkage}; severity {basis.severity}; disruption {basis.disruption}; "
                         f"reversibility {basis.reversibility}; confidence {proposal.confidence.level} (model's own judgement)")
            lines.append(f"      evidence: {', '.join(proposal.observed_evidence_ids) or 'none'}; "
                         f"references: {', '.join(proposal.reference_ids)}; status: {proposal.status}, human review required")
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
            report = run_incident_remediation(incident_dir, store, llm_client, limits=limits)
        print(render_summary(report))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"report written to {args.output}")
    except (MitreSourceError, IngestionError, KnowledgeStoreError, RetrievalError, RagError,
            ContextBuildError, ConfigurationError, ValueError, OSError) as exc:
        print(f"Result: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 3 if report.any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
