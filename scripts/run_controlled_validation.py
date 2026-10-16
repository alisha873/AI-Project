"""
Phase 13: run the authorized validation workflow for one incident/scenario.

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.run_controlled_validation incident_001

The default is a DRY RUN. It uses the offline orchestrator, sends zero network requests, starts no
container, and writes a `not_validated` or `blocked` report. `--live` is not a substitute for reviewer
approval: live execution also requires `--i-accept-authorized-live-validation` AND
`AUTOSHIELD_LIVE_VALIDATION=1`, plus a genuine approved scenario, approved lab, live-suitable identity
and operator authorization. Those are not supplied by this script.

Exit status: 0 a report was produced (including blocked / not_validated); 1 unexpected failure;
2 command-line misuse; 3 live was requested but the opt-in or safety gates refused it before any send.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.inputs import RemediationApplicationAttestation  # noqa: E402
from validation.orchestrator import ValidationRequest, build_validation_report  # noqa: E402
from validation.runner import run_controlled_validation  # noqa: E402
from validation.scenario import SQLI_SCENARIO_ID, default_registry  # noqa: E402
from validation.scope import LabIdentityEvidence, OperatorAuthorization, ScopePolicy  # noqa: E402
from validation_live.opt_in import live_opt_in_granted  # noqa: E402

_INCIDENT_RE = re.compile(r"incident_[0-9]{3}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_controlled_validation",
        description="Produce a Phase 13 validation report. Default: dry run, zero network requests.",
    )
    parser.add_argument("incident_id", help="for example incident_001")
    parser.add_argument("--scenario-id", default=SQLI_SCENARIO_ID)
    parser.add_argument("--target", help="approved lab origin, for example http://127.0.0.1:8080")
    parser.add_argument("--plan-json", type=Path, help="optional Phase 12 result or plan JSON")
    parser.add_argument("--attestation-json", type=Path, help="optional remediation-application attestation")
    parser.add_argument("--identity-json", type=Path, help="optional lab-identity evidence JSON")
    parser.add_argument("--authorization-json", type=Path, help="optional operator authorization JSON")
    parser.add_argument("--output", type=Path, help="write the report JSON here")
    parser.add_argument(
        "--live", action="store_true",
        help="request live execution (still requires the acknowledgement flag, the env opt-in, and every gate)",
    )
    parser.add_argument(
        "--i-accept-authorized-live-validation", action="store_true",
        help="acknowledgement half of the live opt-in; not a substitute for reviewer approval",
    )
    return parser


def _load_json(path: Path | None) -> object | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_request(args: argparse.Namespace) -> ValidationRequest:
    plan = _load_json(args.plan_json)
    application = None
    if args.attestation_json is not None:
        application = RemediationApplicationAttestation.model_validate(_load_json(args.attestation_json))
    identity = None
    if args.identity_json is not None:
        identity = LabIdentityEvidence.model_validate(_load_json(args.identity_json))
    authorization = None
    if args.authorization_json is not None:
        authorization = OperatorAuthorization.model_validate(_load_json(args.authorization_json))
    return ValidationRequest(
        incident_id=args.incident_id,
        scenario_id=args.scenario_id,
        plan_result=plan,
        application=application,
        target_url=args.target,
        lab_identity=identity,
        authorization=authorization,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _INCIDENT_RE.fullmatch(args.incident_id):
        print("incident_id must look like incident_001", file=sys.stderr)
        return 2

    request = _build_request(args)
    registry = default_registry()
    policy = ScopePolicy()  # deny by default; an operator-configured policy is a future, explicit step

    if not args.live:
        report = build_validation_report(request, registry, policy)
        mode = "dry-run"
    elif not live_opt_in_granted(acknowledgement=args.i_accept_authorized_live_validation):
        report = build_validation_report(request, registry, policy)
        mode = "live-refused-opt-in"
    else:
        # Opt-in is complete. The live transport is still not constructed unless a target origin is present;
        # missing target / approvals / identity keep the runner blocked and send nothing.
        from validation.scope import parse_target_origin
        from validation_live.http_transport import LiveHttpTransport, LiveTransportConfigError

        origin = parse_target_origin(args.target) if args.target else None
        if origin is None:
            report = build_validation_report(request, registry, policy)
            mode = "live-refused-target"
        else:
            try:
                from validation.scope import normalized_origin
                transport = LiveHttpTransport(normalized_origin(*origin))
            except LiveTransportConfigError:
                report = build_validation_report(request, registry, policy)
                mode = "live-refused-origin"
            else:
                report = run_controlled_validation(
                    request, registry, policy, transport, live_opt_in=True,
                )
                mode = "live"

    payload = report.model_dump(mode="json")
    payload["execution_mode"] = mode
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    print(
        f"validation {report.status}/{report.reason_code} mode={mode} "
        f"attempted={report.execution.attempted} requests_sent={report.execution.requests_sent}",
        file=sys.stderr,
    )
    if mode != "dry-run" and not report.execution.attempted:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
