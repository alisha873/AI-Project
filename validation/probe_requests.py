"""
Phase 13B.1 - scenario-specific request CONSTRUCTION (pure data; nothing is sent).

`build_probe_requests(scenario, origin)` turns an approved scenario and a scope-approved origin into the two
`ProbeRequest` values a runner would hand to a transport: a benign CONTROL request and the original PROBE. It
builds data only. It opens no connection, resolves no name and never touches a file.

It FAILS CLOSED, with a stable code, instead of guessing:

  scenario_not_eligible        the scenario is not approved or still has an open blocking ambiguity
  unsupported_method           only GET is known
  endpoint_malformed           the scenario path is not a plain absolute path
  wire_encoding_unspecified    the reviewer has not fixed how the value is put on the wire
  control_value_unspecified    the reviewer has not supplied a concrete control value
  probe_value_unusable         the original probe value is empty, over-long or contains control characters
  origin_not_approved_form     the origin is not a plain `http://host:port` origin

Only the encoding named by the scenario is implemented. A request carries NO host of its own beyond the
origin the scope policy approved, and no headers. This module does not use the third-party `requests` library.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import quote

from validation.contract import Identifier, _Strict
from validation.scenario import ScenarioDefinition
from validation.scope import normalized_origin, parse_target_origin

REQUEST_TIMEOUT_SECONDS = 5.0
MAX_VALUE_CHARS = 200

# Endpoints that must never be built for the search scenario. `/api/data` returns the whole table unconditionally;
# it is not the search endpoint under test and must not be selected here even if a definition names it.
FORBIDDEN_PATHS = ("/api/data",)

RequestRole = Literal["control", "probe"]
CONSTRUCTION_CODES = (
    "scenario_not_eligible",
    "unsupported_method",
    "endpoint_malformed",
    "endpoint_not_allowed",
    "wire_encoding_unspecified",
    "control_value_unspecified",
    "probe_value_unusable",
    "origin_not_approved_form",
)


class RequestConstructionError(ValueError):
    """A request could not be built safely. `code` is one of `CONSTRUCTION_CODES`."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class ProbeRequest(_Strict):
    """One request as DATA. `target` is the full origin-relative request line, nothing else."""

    request_id: Identifier
    role: RequestRole
    method: Literal["GET"]
    origin: Identifier
    path: Identifier
    query: str
    timeout_seconds: float
    attack_shaped: bool

    @property
    def target(self) -> str:
        return f"{self.path}?{self.query}"


class RequestPair(_Strict):
    control: ProbeRequest
    probe: ProbeRequest


def _encode(parameter: str, value: str) -> str:
    # RFC 3986 unreserved characters stay as they are; everything else (including space) becomes %XX of its
    # UTF-8 bytes. `quote(..., safe="")` is a pure string function.
    return f"{quote(parameter, safe='')}={quote(value, safe='')}"


def _usable_value(value: object) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= MAX_VALUE_CHARS
            and not any(ord(ch) < 32 or ord(ch) == 127 for ch in value))


def check_constructible(scenario: ScenarioDefinition) -> str | None:
    """The first construction problem code for `scenario`, or None. Needs no origin."""
    if not scenario.eligible_for_execution:
        return "scenario_not_eligible"
    endpoint, probe = scenario.endpoint, scenario.probe
    if endpoint.method != "GET":
        return "unsupported_method"
    if not endpoint.path.startswith("/") or any(ch in endpoint.path for ch in "?#\\ \t\r\n"):
        return "endpoint_malformed"
    if endpoint.path in FORBIDDEN_PATHS:
        return "endpoint_not_allowed"
    if probe.wire_encoding != "percent_encoded_utf8_query":
        return "wire_encoding_unspecified"
    if not _usable_value(probe.control_value):
        return "control_value_unspecified"
    if not _usable_value(probe.parameter_value_candidate):
        return "probe_value_unusable"
    return None


def build_probe_requests(scenario: object, origin: object) -> RequestPair:
    if not isinstance(scenario, ScenarioDefinition):
        raise RequestConstructionError("scenario_not_eligible", "a ScenarioDefinition is required")
    problem = check_constructible(scenario)
    if problem is not None:
        raise RequestConstructionError(problem, f"scenario {scenario.scenario_id!r} can not be built safely")
    parsed = parse_target_origin(origin) if isinstance(origin, str) else None
    if parsed is None or normalized_origin(*parsed) != origin:
        raise RequestConstructionError("origin_not_approved_form", "the origin must be a normalized http origin")
    endpoint, probe = scenario.endpoint, scenario.probe
    assert probe.control_value is not None  # check_constructible guarantees it

    def make(role: RequestRole, value: str) -> ProbeRequest:
        return ProbeRequest(
            request_id=f"{scenario.scenario_id}:{role}", role=role, method="GET", origin=origin,
            path=endpoint.path, query=_encode(endpoint.parameter, value),
            timeout_seconds=REQUEST_TIMEOUT_SECONDS, attack_shaped=role == "probe",
        )

    return RequestPair(control=make("control", probe.control_value),
                       probe=make("probe", probe.parameter_value_candidate))


def build_baseline_requests(scenario: object, origin: object) -> tuple[ProbeRequest, ...]:
    """The reviewed legitimate baseline searches as INERT control requests (one per `baseline_searches` term).

    Fails closed exactly like `build_probe_requests` (same codes). Returns an empty tuple when the scenario
    declares no baseline searches. Nothing here is sent; these are controlled comparison inputs as data."""
    if not isinstance(scenario, ScenarioDefinition):
        raise RequestConstructionError("scenario_not_eligible", "a ScenarioDefinition is required")
    problem = check_constructible(scenario)
    if problem is not None:
        raise RequestConstructionError(problem, f"scenario {scenario.scenario_id!r} can not be built safely")
    parsed = parse_target_origin(origin) if isinstance(origin, str) else None
    if parsed is None or normalized_origin(*parsed) != origin:
        raise RequestConstructionError("origin_not_approved_form", "the origin must be a normalized http origin")
    endpoint = scenario.endpoint
    requests: list[ProbeRequest] = []
    for baseline in scenario.probe.baseline_searches:
        if not _usable_value(baseline.term):
            raise RequestConstructionError("probe_value_unusable", f"baseline term {baseline.term!r} is unusable")
        requests.append(ProbeRequest(
            request_id=f"{scenario.scenario_id}:baseline:{baseline.term}", role="control", method="GET",
            origin=origin, path=endpoint.path, query=_encode(endpoint.parameter, baseline.term),
            timeout_seconds=REQUEST_TIMEOUT_SECONDS, attack_shaped=False,
        ))
    return tuple(requests)
