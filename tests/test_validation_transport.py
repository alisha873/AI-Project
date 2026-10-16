"""Phase 13B.1: the typed transport interface and the deterministic fake transport."""

from __future__ import annotations

import pytest

from tests.validation_helpers import approved_scenario, lab_body, simulated_lab
from validation.probe_requests import build_probe_requests
from validation.transport import (
    MAX_BODY_CHARS,
    ScriptedTransport,
    ScriptExhausted,
    TransportResponse,
    ValidationTransport,
)

ORIGIN = "http://127.0.0.1:8080"


def pair():
    return build_probe_requests(approved_scenario(), ORIGIN)


def test_a_response_needs_a_status_and_a_body():
    assert TransportResponse.ok("{}").status_code == 200
    for bad in ({"outcome": "response"}, {"outcome": "response", "status_code": 200},
                {"outcome": "response", "body": "x"}, {"outcome": "response", "status_code": 99, "body": ""},
                {"outcome": "response", "status_code": 600, "body": ""}):
        with pytest.raises(ValueError):
            TransportResponse(**bad)


@pytest.mark.parametrize("make", [TransportResponse.timeout, TransportResponse.unreachable, TransportResponse.error])
def test_a_failed_exchange_carries_neither_status_nor_body(make):
    assert make().status_code is None and make().body is None
    with pytest.raises(ValueError):
        TransportResponse(outcome=make().outcome, status_code=200, body="x")


def test_the_body_is_capped_and_the_model_is_strict():
    TransportResponse.ok("x" * MAX_BODY_CHARS)
    with pytest.raises(ValueError):
        TransportResponse.ok("x" * (MAX_BODY_CHARS + 1))
    with pytest.raises(ValueError):
        TransportResponse(outcome="response", status_code="200", body="x")  # no coercion
    with pytest.raises(ValueError):
        TransportResponse(outcome="weird")


def test_the_fake_transport_declares_itself_fake_and_matches_the_protocol():
    fake = ScriptedTransport()
    assert fake.kind == "fake" and fake.transport_id == "fake-scripted-transport"
    assert isinstance(fake, ValidationTransport)


def test_scripted_answers_are_returned_in_order_and_requests_are_recorded():
    requests = pair()
    fake = ScriptedTransport(TransportResponse.ok(lab_body(["a"])), TransportResponse.timeout())
    assert fake.send(requests.control).body == lab_body(["a"])
    assert fake.send(requests.probe).outcome == "timeout"
    assert fake.requests == [requests.control, requests.probe]


def test_a_callable_item_can_simulate_a_lab_deterministically():
    requests = pair()
    lab = simulated_lab(vulnerable=True)
    assert lab(requests.control) == lab(requests.control)
    assert lab(requests.control).body == lab_body([])
    assert lab(requests.probe).body == lab_body(["first note", "second note", "hello world"])
    assert simulated_lab(vulnerable=False)(requests.probe).body == lab_body([])


def test_an_exception_item_is_raised():
    fake = ScriptedTransport(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        fake.send(pair().control)


def test_running_out_of_script_is_a_loud_test_error_not_an_ordinary_exception():
    fake = ScriptedTransport()
    with pytest.raises(ScriptExhausted):
        fake.send(pair().control)
    assert not issubclass(ScriptExhausted, Exception), "the runner must not be able to swallow it"


def test_an_object_without_a_kind_or_send_is_not_a_transport():
    class NoKind:
        transport_id = "x"

        def send(self, request):
            return None

    assert not isinstance(NoKind(), ValidationTransport)
    assert not isinstance(object(), ValidationTransport)
