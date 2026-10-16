"""
Phase 13B.1 - the typed transport interface and a deterministic FAKE transport.

A transport is the ONLY thing that may ever carry a `ProbeRequest` to a target. In Phase 13B.1 no real transport
exists: this module defines the interface (`ValidationTransport`), the typed answer (`TransportResponse`) and
`ScriptedTransport`, an in-memory transport that returns scripted answers and records the requests it was given.
It opens no socket, resolves no name, starts no process and reads no file.

Every transport declares what it is:
  kind == "fake"   an in-memory stand-in. Results obtained through it are SIMULATIONS.
  kind == "live"   a real transport (does not exist yet; the runner refuses it while `LIVE_TRANSPORT_ENABLED`
                   is False).

A transport answers with a `TransportResponse`; it signals a timeout, an unreachable lab or another failure as an
`outcome`, never as success. It never receives a URL typed by a person: only the origin the scope policy
approved reaches it, inside the `ProbeRequest`.
"""

from __future__ import annotations

from typing import Callable, Literal, Protocol, runtime_checkable

from pydantic import model_validator

from validation.contract import _Strict
from validation.probe_requests import ProbeRequest

MAX_BODY_CHARS = 100_000

TransportOutcome = Literal["response", "timeout", "unreachable", "error"]


class TransportResponse(_Strict):
    """What a transport reports. `response` carries a status and a body; the other outcomes carry neither."""

    outcome: TransportOutcome
    status_code: int | None = None
    body: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "TransportResponse":
        if self.outcome == "response":
            if self.status_code is None or not 100 <= self.status_code <= 599 or self.body is None:
                raise ValueError("a response needs an HTTP status (100-599) and a body")
            if len(self.body) > MAX_BODY_CHARS:
                raise ValueError(f"a response body is limited to {MAX_BODY_CHARS} characters")
        elif self.status_code is not None or self.body is not None:
            raise ValueError("a failed exchange carries neither a status nor a body")
        return self

    @classmethod
    def ok(cls, body: str, status_code: int = 200) -> "TransportResponse":
        return cls(outcome="response", status_code=status_code, body=body)

    @classmethod
    def timeout(cls) -> "TransportResponse":
        return cls(outcome="timeout")

    @classmethod
    def unreachable(cls) -> "TransportResponse":
        return cls(outcome="unreachable")

    @classmethod
    def error(cls) -> "TransportResponse":
        return cls(outcome="error")


@runtime_checkable
class ValidationTransport(Protocol):
    transport_id: str
    kind: Literal["fake", "live"]

    def send(self, request: ProbeRequest) -> TransportResponse: ...


class ScriptExhausted(BaseException):
    """A fake transport was asked for more answers than its script holds. It derives from BaseException on
    purpose: the runner converts ordinary transport exceptions into an inconclusive result, and this is a test
    bug that must never be swallowed that way."""


ScriptItem = TransportResponse | BaseException | Callable[[ProbeRequest], TransportResponse]


class ScriptedTransport:
    """A deterministic fake. Items are consumed in order; a callable may inspect the request (for example to
    simulate a lab); an exception item is raised. Every request is recorded in `requests`."""

    kind: Literal["fake", "live"] = "fake"

    def __init__(self, *script: ScriptItem, transport_id: str = "fake-scripted-transport") -> None:
        self.transport_id = transport_id
        self._script = list(script)
        self.requests: list[ProbeRequest] = []

    def send(self, request: ProbeRequest) -> TransportResponse:
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._script):
            raise ScriptExhausted(f"no scripted answer for request {index + 1}")
        item = self._script[index]
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(request)
        return item
