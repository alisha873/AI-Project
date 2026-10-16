"""
Phase 14: judge a saved Phase 13 validation report. Offline only.

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.run_judge --report incident_001\\analysis\\validation_report.json --output incident_001\\analysis\\judge_decision.json

This command reads a JSON file and writes a judge assessment. It does not send a request, start a container,
call a model, or add --live. A dry-run or blocked report stays a dry-run judgment.

Exit status: 0 an assessment was written (including not_evaluated, blocked and inconclusive); 2 command-line misuse.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from judge.agent import JudgeAgent  # noqa: E402
from judge.contract import serialize  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_judge",
        description="Judge a saved Phase 13 validation report. No network and no model.",
    )
    parser.add_argument("--report", type=Path, required=True, help="path to a validation report JSON file")
    parser.add_argument("--output", type=Path, help="write the judge assessment JSON here")
    parser.add_argument(
        "--claim",
        help="optional claim id (objective_observed_under_tested_conditions, "
        "objective_not_observed_under_tested_conditions, remediation_applied, application_is_secure)",
    )
    return parser


def _emit(assessment, output: Path | None) -> None:
    text = serialize(assessment)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(
        f"judge {assessment.decision} provenance={assessment.evidence_provenance} "
        f"eligible={assessment.eligible_for_downstream} live_evidence={assessment.live_evidence}",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    agent = JudgeAgent()
    if not args.report.is_file():
        assessment = agent.evaluate_payload(None, claim_id=args.claim)
        _emit(assessment, args.output)
        return 0
    try:
        payload = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        assessment = agent.evaluate_payload([], claim_id=args.claim)
        _emit(assessment, args.output)
        return 0
    assessment = agent.evaluate_payload(payload, claim_id=args.claim)
    _emit(assessment, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
