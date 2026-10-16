"""
Offline tests for the live HTTP transport. No real socket, DNS or HTTP call is made.

The transport is constructed with injected resolver / HTTP functions so a test cannot be mistaken
for evidence from a real request. `kind` is always `live`; the fake `ScriptedTransport` stays `fake`.
"""

from __future__ import annotations

import json

import pytest
import requests

from tests.validation_helpers import approved_scenario
from validation.probe_requests import build_probe_requests
from validation.transport import ScriptedTransport
from validation_live.http_transport import (
    LiveHttpTransport,
    LiveTransportConfigError,
    destination_forbidden,
)

ORIGIN = "http://127.0.0.1:8080"


def _probe():
    return build_probe_requests(approved_scenario(), ORIGIN).probe


class _Response:
    def __init__(self, status_code=200, text='{"data":[]}', headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


def _transport(*, resolver=None, http_get=None, origin=ORIGIN):
    return LiveHttpTransport(
        origin,
        resolver=resolver or (lambda host: ["127.0.0.1"]),
        http_get=http_get or (lambda url, headers, timeout: _Response()),
    )


def test_live_and_fake_kinds_cannot_be_confused():
    assert LiveHttpTransport(ORIGIN, resolver=lambda h: ["127.0.0.1"],
                             http_get=lambda *a: _Response()).kind == "live"
    assert ScriptedTransport().kind == "fake"


def test_a_public_or_metadata_origin_cannot_be_constructed():
    with pytest.raises(LiveTransportConfigError):
        LiveHttpTransport("http://8.8.8.8:8080")
    with pytest.raises(LiveTransportConfigError):
        LiveHttpTransport("http://169.254.169.254:80")
    with pytest.raises(LiveTransportConfigError):
        LiveHttpTransport("http://evil.example:8080")


@pytest.mark.parametrize("address,host,reason", [
    ("169.254.169.254", "localhost", "destination_not_a_lab_address"),
    ("8.8.8.8", "localhost", "destination_is_public"),
    ("10.0.0.9", "localhost", "localhost_resolved_off_loopback"),
    ("10.0.0.2", "10.0.0.1", "resolved_address_mismatch"),
])
def test_forbidden_or_rebound_destinations_are_named(address, host, reason):
    assert destination_forbidden(address, origin_host=host) == reason


def test_localhost_resolving_to_metadata_never_sends():
    called = []
    transport = _transport(
        resolver=lambda host: ["169.254.169.254"],
        http_get=lambda url, headers, timeout: called.append(url) or _Response(),
        origin="http://localhost:8080",
    )
    request = build_probe_requests(approved_scenario(), "http://localhost:8080").control
    assert transport.send(request).outcome == "unreachable"
    assert called == []


def test_a_redirect_off_origin_is_not_followed():
    calls = []

    def http_get(url, headers, timeout):
        calls.append(url)
        return _Response(302, "", {"Location": "http://evil.example/steal"})

    transport = _transport(http_get=http_get)
    assert transport.send(_probe()).outcome == "error"
    assert len(calls) == 1
    assert "evil.example" not in calls[0]


def test_an_on_origin_redirect_is_returned_and_not_followed():
    calls = []

    def http_get(url, headers, timeout):
        calls.append(url)
        return _Response(302, "", {"Location": "/other"})

    transport = _transport(http_get=http_get)
    response = transport.send(_probe())
    assert response.outcome == "response" and response.status_code == 302
    assert len(calls) == 1


def test_timeout_and_connection_failure_are_not_success():
    transport = _transport(http_get=lambda *a: (_ for _ in ()).throw(requests.Timeout()))
    assert transport.send(_probe()).outcome == "timeout"
    transport = _transport(http_get=lambda *a: (_ for _ in ()).throw(requests.ConnectionError()))
    assert transport.send(_probe()).outcome == "unreachable"


def test_a_mismatched_origin_or_forbidden_path_is_refused_without_sending():
    called = []
    transport = _transport(http_get=lambda url, headers, timeout: called.append(url) or _Response())
    other = build_probe_requests(approved_scenario(), ORIGIN).probe
    other_wrong_origin = other.model_copy(update={"origin": "http://10.0.0.1:8080"})
    assert transport.send(other_wrong_origin).outcome == "error"
    assert called == []


def test_no_authorization_cookie_or_secret_is_attached():
    seen = {}

    def http_get(url, headers, timeout):
        seen["headers"] = headers
        return _Response()

    transport = _transport(http_get=http_get)
    assert transport.send(_probe()).outcome == "response"
    assert set(seen["headers"]) <= {"Host", "Accept"}
    dumped = json.dumps(seen)
    assert "Authorization" not in dumped and "Cookie" not in dumped and "gsk_" not in dumped
