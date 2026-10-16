"""
Phase 13 - the fail-closed scope policy (pure decision logic; no I/O).

The policy answers one question: "would a future live check against THIS target, for THIS scenario, be
inside an approved, isolated lab?" It never connects, resolves a name, probes a port or runs a command; it
only compares data it is given. Phase 13A has no component that could act on a "yes".

Principles
----------
- DENY BY DEFAULT. A policy with no approved lab denies everything; `execution_enabled` defaults to False.
- A URL or host name grants nothing. A target is only ever compared with an `ApprovedLab` that an operator
  configured, and the comparison is exact (canonical host text and an explicit port). `localhost` or
  `127.0.0.1` does not prove a lab, and a private address is not treated as safe. Approved lab hosts may not be
  globally routable addresses at all.
- Identity is separate from address. A future runner must supply `LabIdentityEvidence` that was established
  INDEPENDENTLY of the target string (for example container names, network name and a compose-file hash read
  from the Docker engine by a separate component) and it must match the approved lab.
- A person must authorize it: an `OperatorAuthorization` for this lab, scenario and incident that explicitly
  permits live execution.
- All checks run and every failure is reported, so a denial is auditable. The decision has two parts:
  `scope_conditions_met` (everything about the target, scenario, identity and authorization is in order) and
  `execution_permitted` (that AND the policy's own `execution_enabled` switch). Phase 13A's orchestrator
  additionally refuses to execute whatever the policy says, because no runner exists.

What a live runner (Phase 13B) will need to supply is exactly the `ScopeRequest`: the target, the scenario,
the incident id, the independently established identity and the operator's authorization.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlsplit

from pydantic import model_validator

from validation.contract import Identifier, Sha256, _Strict
from validation.scenario import ScenarioDefinition
from validation.seals import IssuedIdentity, identity_key, lookup_identities

POLICY_VERSION = "1.0"

SCOPE_REASON_CODES = (
    "execution_disabled",
    "no_approved_lab",
    "target_missing",
    "target_malformed",
    "target_not_approved",
    "scenario_missing",
    "scenario_not_eligible",
    "scenario_not_approved_for_lab",
    "lab_identity_missing",
    "lab_identity_not_independent",
    "lab_identity_mismatch",
    "lab_identity_not_assessed",
    "lab_identity_stale",
    "lab_identity_from_future",
    "lab_identity_freshness_unchecked",
    "lab_identity_verifier_conflict",
    "lab_identity_not_live_suitable",
    "lab_state_unverified",
    "lab_state_mismatch",
    "operator_authorization_missing",
    "operator_authorization_not_live",
    "operator_authorization_mismatch",
)
_MAX_TARGET_CHARS = 200


# ============================================================
# CONFIGURATION AND EVIDENCE MODELS
# ============================================================

class ApprovedLab(_Strict):
    """An operator-approved isolated lab. Constructing one is the act of approving it; nothing is probed."""

    approval_id: Identifier
    lab_id: Identifier
    approved_by: Identifier
    host: Identifier
    port: int
    expected_network: Identifier
    expected_containers: list[Identifier]
    compose_sha256: Sha256 | None = None
    approved_scenario_ids: list[Identifier]

    @model_validator(mode="after")
    def _valid(self) -> "ApprovedLab":
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be 1-65535")
        if not self.expected_containers or len(set(self.expected_containers)) != len(self.expected_containers):
            raise ValueError("an approved lab lists its expected containers once each")
        if not self.approved_scenario_ids:
            raise ValueError("an approved lab names the scenarios it approves")
        if self.host != "localhost":
            try:
                address = ipaddress.ip_address(self.host)
            except ValueError as exc:
                raise ValueError("an approved lab host must be 'localhost' or a literal IP address") from exc
            if str(address) != self.host:
                raise ValueError("an approved lab IP address must be written in canonical form")
            if address.is_global or address.is_unspecified or address.is_multicast:
                raise ValueError("an approved lab host can not be a public, unspecified or multicast address")
        return self


class LabIdentityEvidence(_Strict):
    """What an independent component observed about the lab. Phase 13A never produces this."""

    lab_id: Identifier
    observed_network: Identifier
    observed_containers: list[Identifier]
    compose_sha256: Sha256 | None = None
    # The lab-state identifier the independent component observed (for example a snapshot id). Optional; a
    # post-remediation check compares it with the state the operator attested the remediation produced.
    lab_state_id: Identifier | None = None
    established_by: Identifier
    established_independently: bool


class OperatorAuthorization(_Strict):
    authorization_id: Identifier
    operator_id: Identifier
    lab_id: Identifier
    scenario_id: Identifier
    incident_id: Identifier
    live_execution_authorized: bool


@dataclass(frozen=True)
class ScopeRequest:
    target_url: object = None
    scenario: ScenarioDefinition | None = None
    incident_id: str | None = None
    identity: LabIdentityEvidence | None = None
    authorization: OperatorAuthorization | None = None
    # When set (a post-remediation check), the independently observed lab state must equal this attested state.
    expected_lab_state_id: str | None = None
    # Phase 13B.3: use-time checks of the identity evidence. `now` is the caller's clock (the policy never reads one);
    # `forbidden_verifier_ids` are ids (the runner, the operator, the attester) that may not be the verifier;
    # `require_live_suitable_identity` is set when the transport is live.
    now: datetime | None = None
    forbidden_verifier_ids: tuple[str, ...] = ()
    require_live_suitable_identity: bool = False


@dataclass(frozen=True)
class ScopeDecision:
    policy_version: str
    scope_conditions_met: bool
    execution_permitted: bool
    reason_codes: tuple[str, ...]
    origin: str | None = None  # normalized scheme://host:port, only when the target parsed
    lab_id: str | None = None  # only when the target matched an approved lab


# ============================================================
# TARGET PARSING (string handling only)
# ============================================================

def parse_target_origin(target: object) -> tuple[str, int] | None:
    """Return (host, port) for a plain `http://host:port[/]` origin, or None if it is anything else.

    Pure string parsing: no DNS, no socket. Credentials, query, fragment, a path, a missing port, backslashes,
    whitespace and control characters are all refused.
    """
    if not isinstance(target, str) or not target or len(target) > _MAX_TARGET_CHARS:
        return None
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 or ch in "\\%" for ch in target):
        return None
    if not target.startswith("http://"):
        return None
    try:
        parts = urlsplit(target)
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if not host or port is None or parts.username is not None or parts.password is not None:
        return None
    if parts.query or parts.fragment or parts.path not in ("", "/") or "?" in target or "#" in target:
        return None
    return host, port


def normalized_origin(host: str, port: int) -> str:
    return f"http://{host}:{port}"


# ============================================================
# THE POLICY
# ============================================================

def _issued_records(identity: LabIdentityEvidence) -> tuple[IssuedIdentity, ...]:
    return lookup_identities(identity_key(
        lab_id=identity.lab_id, observed_network=identity.observed_network,
        observed_containers=identity.observed_containers, compose_sha256=identity.compose_sha256,
        lab_state_id=identity.lab_state_id, established_by=identity.established_by,
        established_independently=identity.established_independently,
    ))


def issued_identity_for(identity: LabIdentityEvidence, now: datetime | None = None) -> IssuedIdentity | None:
    """The record `assess_lab_identity` created for exactly this evidence, or None (hand-built / altered / evicted).
    With `now`, the best record whose validity window covers it (live-suitable first); without it, the newest."""
    records = _issued_records(identity)
    if not records:
        return None
    if now is not None and now.tzinfo is not None:
        moment = now.astimezone(timezone.utc)
        covering = [r for r in records if r.valid_from <= moment <= r.valid_until]
        if covering:
            return max(covering, key=lambda r: (r.authoritative, r.live_suitable, r.valid_until))
    return max(records, key=lambda r: r.valid_until)


class ScopePolicy:
    def __init__(self, approved_labs: tuple[ApprovedLab, ...] | list[ApprovedLab] = (),
                 execution_enabled: bool = False) -> None:
        if not all(isinstance(lab, ApprovedLab) for lab in approved_labs):
            raise TypeError("approved_labs must contain ApprovedLab objects")
        self._labs = tuple(approved_labs)
        self._execution_enabled = execution_enabled is True

    def evaluate(self, request: ScopeRequest) -> ScopeDecision:
        failures: list[str] = []
        lab: ApprovedLab | None = None
        origin: str | None = None

        if not self._labs:
            failures.append("no_approved_lab")
        if request.target_url is None:
            failures.append("target_missing")
        else:
            parsed = parse_target_origin(request.target_url)
            if parsed is None:
                failures.append("target_malformed")
            else:
                origin = normalized_origin(*parsed)
                lab = next((item for item in self._labs if (item.host, item.port) == parsed), None)
                if lab is None and self._labs:
                    failures.append("target_not_approved")

        scenario = request.scenario
        if scenario is None:
            failures.append("scenario_missing")
        else:
            if not scenario.eligible_for_execution:
                failures.append("scenario_not_eligible")
            if lab is not None and scenario.scenario_id not in lab.approved_scenario_ids:
                failures.append("scenario_not_approved_for_lab")

        failures.extend(self._identity_failures(request, lab))
        failures.extend(self._authorization_failures(request, lab, scenario))

        if not self._execution_enabled:
            failures.append("execution_disabled")

        conditions_met = all(code == "execution_disabled" for code in failures)
        return ScopeDecision(
            policy_version=POLICY_VERSION,
            scope_conditions_met=conditions_met,
            execution_permitted=not failures,
            reason_codes=tuple(failures),
            origin=origin,
            lab_id=lab.lab_id if lab is not None else None,
        )

    @staticmethod
    def _identity_failures(request: ScopeRequest, lab: ApprovedLab | None) -> list[str]:
        identity = request.identity
        if identity is None:
            return ["lab_identity_missing"]
        failures: list[str] = []
        if not identity.established_independently:
            failures.append("lab_identity_not_independent")
        else:
            failures.extend(ScopePolicy._issued_identity_failures(request, identity))
        if identity.established_by in request.forbidden_verifier_ids:
            failures.append("lab_identity_verifier_conflict")
        if lab is None:
            # nothing to compare with: fail closed rather than guess
            failures.append("lab_identity_mismatch")
        else:
            matches = (
                identity.lab_id == lab.lab_id
                and identity.observed_network == lab.expected_network
                and sorted(identity.observed_containers) == sorted(lab.expected_containers)
                and (lab.compose_sha256 is None or identity.compose_sha256 == lab.compose_sha256)
            )
            if not matches:
                failures.append("lab_identity_mismatch")
        expected = request.expected_lab_state_id
        if expected is not None:
            if identity.lab_state_id is None:
                failures.append("lab_state_unverified")
            elif identity.lab_state_id != expected:
                failures.append("lab_state_mismatch")
        return failures

    @staticmethod
    def _issued_identity_failures(request: ScopeRequest, identity: LabIdentityEvidence) -> list[str]:
        """`established_independently` is a claim anyone can write. It only counts if `assess_lab_identity` issued
        this exact evidence (see `validation.seals`); the checks that follow use the issued record, not the claim."""
        records = _issued_records(identity)
        if not records:
            return ["lab_identity_not_assessed"]
        failures: list[str] = []
        now = request.now
        issued = issued_identity_for(identity, now)
        assert issued is not None
        if now is None or now.tzinfo is None:
            if request.require_live_suitable_identity:
                failures.append("lab_identity_freshness_unchecked")
        else:
            moment = now.astimezone(timezone.utc)
            if not any(r.valid_from <= moment <= r.valid_until for r in records):
                # no observation of this exact evidence covers `now`: all are older (stale) or all are newer (future)
                failures.append("lab_identity_from_future" if moment < min(r.valid_from for r in records)
                                else "lab_identity_stale")
        if request.require_live_suitable_identity and not issued.live_suitable:
            failures.append("lab_identity_not_live_suitable")
        return failures

    @staticmethod
    def _authorization_failures(
        request: ScopeRequest, lab: ApprovedLab | None, scenario: ScenarioDefinition | None
    ) -> list[str]:
        auth = request.authorization
        if auth is None:
            return ["operator_authorization_missing"]
        failures: list[str] = []
        if not auth.live_execution_authorized:
            failures.append("operator_authorization_not_live")
        if (
            lab is None
            or scenario is None
            or request.incident_id is None
            or (auth.lab_id, auth.scenario_id, auth.incident_id)
            != (lab.lab_id, scenario.scenario_id, request.incident_id)
        ):
            failures.append("operator_authorization_mismatch")
        return failures
