"""
Offline tests for the Phase 13 validation CLI and runner live opt-in.

No real network call is made. A dry run must produce a report and send nothing. `--live` without the
acknowledgement flag and AUTOSHIELD_LIVE_VALIDATION=1 must not construct or call a live transport.
"""

from __future__ import annotations

import json

import pytest

from tests.validation_helpers import approved_registry, fixed_clock, full_policy, full_request


class _LooksLive:
    kind = "live"
    transport_id = "pretend-live"

    def __init__(self) -> None:
        self.requests: list = []

    def send(self, request):
        self.requests.append(request)
        raise AssertionError("a live transport must not be called when gates refuse the run")
from validation.orchestrator import build_validation_report
from validation.runner import LIVE_TRANSPORT_ENABLED, run_controlled_validation
from validation.scenario import default_registry
from validation.scope import ScopePolicy
from validation_live.opt_in import live_opt_in_granted
from scripts import run_controlled_validation as cli


def test_opt_in_requires_both_the_flag_and_the_env_value():
    assert live_opt_in_granted(acknowledgement=True, environ={"AUTOSHIELD_LIVE_VALIDATION": "1"}) is True
    assert live_opt_in_granted(acknowledgement=True, environ={"AUTOSHIELD_LIVE_VALIDATION": "yes"}) is False
    assert live_opt_in_granted(acknowledgement=False, environ={"AUTOSHIELD_LIVE_VALIDATION": "1"}) is False
    assert live_opt_in_granted(acknowledgement=True, environ={}) is False


def test_the_build_lock_stays_closed():
    assert LIVE_TRANSPORT_ENABLED is False


def test_a_dry_run_sends_nothing_and_does_not_claim_live_evidence():
    request = full_request()
    report = build_validation_report(request, default_registry(), ScopePolicy())
    assert report.execution.attempted is False and report.execution.requests_sent == 0
    assert report.execution.transport_kind in ("none", "fake") or report.status in ("not_validated", "blocked")
    from validation.handoff import build_phase14_handoff
    assert build_phase14_handoff(report).usable_as_post_remediation_evidence is False


def test_live_opt_in_without_an_approved_builtin_scenario_sends_nothing():
    transport = _LooksLive()
    report = run_controlled_validation(
        full_request(), default_registry(), full_policy(), transport,
        clock=fixed_clock(), live_opt_in=True,
    )
    assert report.execution.attempted is False
    assert transport.requests == []
    assert report.status in ("not_validated", "blocked")


def test_a_live_transport_without_opt_in_is_still_blocked():
    transport = _LooksLive()
    report = run_controlled_validation(
        full_request(), approved_registry(), full_policy(), transport, clock=fixed_clock(),
    )
    assert (report.status, report.reason_code) == ("blocked", "live_transport_not_enabled")
    assert transport.requests == []


def test_cli_default_is_a_dry_run_with_zero_network(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("network or process access attempted")

    monkeypatch.setattr("socket.getaddrinfo", boom)
    monkeypatch.setattr("requests.Session.get", boom)
    output = tmp_path / "report.json"
    code = cli.main(["incident_001", "--output", str(output)])
    assert code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "dry-run"
    assert payload["execution"]["attempted"] is False
    assert payload["execution"]["requests_sent"] == 0
    assert payload["status"] in ("not_validated", "blocked")
    assert "GROQ_API_KEY" not in json.dumps(payload)


def test_cli_live_without_opt_in_does_not_enable_transport(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOSHIELD_LIVE_VALIDATION", raising=False)
    created = []
    monkeypatch.setattr(
        "validation_live.http_transport.LiveHttpTransport",
        lambda *a, **k: created.append(True) or (_ for _ in ()).throw(AssertionError("must not construct")),
    )
    output = tmp_path / "report.json"
    code = cli.main(["incident_001", "--live", "--target", "http://127.0.0.1:8080", "--output", str(output)])
    assert code == 3
    assert created == []
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "live-refused-opt-in"
    assert payload["execution"]["attempted"] is False


def test_cli_live_with_opt_in_but_no_approvals_still_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOSHIELD_LIVE_VALIDATION", "1")
    sent = []

    class Recording:
        kind = "live"
        transport_id = "test-live"

        def send(self, request):
            sent.append(request)
            raise AssertionError("must not send")

    monkeypatch.setattr(
        "validation_live.http_transport.LiveHttpTransport",
        lambda *a, **k: Recording(),
    )
    output = tmp_path / "report.json"
    code = cli.main([
        "incident_001", "--live", "--i-accept-authorized-live-validation",
        "--target", "http://127.0.0.1:8080", "--output", str(output),
    ])
    assert code == 3
    assert sent == []
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["execution"]["attempted"] is False
    assert payload["scenario"]["eligible_for_execution"] is False
