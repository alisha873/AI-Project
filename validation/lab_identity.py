"""
Phase 13B.2 preparation - the versioned lab-identity contract (pure; no I/O).

Question this module answers: "what was observed about the lab, by whom, how recently, and may Phase 13 call that
independent verification?"

Honest starting point. **No independent identity verifier exists in this build.** Nothing here inspects Docker, a
network or a file. A `LabIdentityObservation` is therefore only a *record that somebody supplied*; whether it counts
as independently verified is decided by `assess_lab_identity`, and only a verifier that is REGISTERED in a
`VerifierRegistry` (and is not the runner or the operator) can make it count. `default_verifier_registry()` is empty
and `IDENTITY_VERIFIER_AVAILABLE` is False, so with the default configuration the level can never reach
`verifier_registered` and `independently_verified` is always False. That "unavailable" state is reported explicitly
(`verifier_unavailable`), never papered over.

Levels (`IdentityAssessment.level`), weakest first:

  none_supplied            nothing (or something malformed) was supplied.
  operator_asserted        a person or tool typed the observation; nothing checked it.
  verifier_unavailable     the observation claims a verifier, but the registry is empty (no verifier exists).
  verifier_unregistered    it claims a verifier the registry does not know.
  verifier_not_independent the named verifier is the runner or the operator (`distinct_from`).
  verifier_registered      a registered verifier, distinct from the runner and operator, made the observation.

Only `verifier_registered` AND a fresh observation gives `independently_verified`. `live_suitable` additionally needs
the registry entry to say it is suitable for live use; a `test_double` entry never is, so a test can exercise the
logic without ever being mistaken for live evidence.

Freshness: an observation older than `max_age_seconds`, or dated in the future beyond the allowed clock skew, is not
fresh and is not passed on to the scope policy (which then sees no identity and denies).

The bridge to Phase 13A: a fresh observation becomes a `LabIdentityEvidence` (the scope policy's input) whose
`established_independently` is exactly `independently_verified`. The scope policy and its fail-closed behaviour are
unchanged.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Annotated, Iterable, Literal

from pydantic import AfterValidator, StringConstraints, model_validator

from validation import seals
from validation.contract import Identifier, Sha256, _Strict
from validation.scope import LabIdentityEvidence

IDENTITY_CONTRACT_VERSION = "1.0"
IDENTITY_MAX_AGE_SECONDS = 900
CLOCK_SKEW_SECONDS = 60

# The repository's own list of identity verifiers. It is EMPTY: no independent lab-identity verifier is implemented or
# reviewed in this build. Adding an entry is a code change to be reviewed; a caller can not do it by building a
# `VerifierRegistry` of its own (such a registry is not `authoritative`, so it can never yield live-suitable evidence).
_BUILTIN_VERIFIERS: tuple["VerifierEntry", ...] = ()

_TIMESTAMP = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


def parse_utc(value: object) -> datetime:
    """Parse a canonical UTC timestamp (`2026-10-16T12:00:00Z`). Raises ValueError for anything else."""
    if not isinstance(value, str) or not re.match(_TIMESTAMP, value):
        raise ValueError("a timestamp must look like 2026-10-16T12:00:00Z")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("a timestamp must be a real calendar date and time") from exc


def _checked_timestamp(value: str) -> str:
    parse_utc(value)
    return value


UtcTimestamp = Annotated[str, AfterValidator(_checked_timestamp)]
LabStateId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")]

VerificationClaim = Literal["operator_asserted", "verifier_attested"]
VerificationLevel = Literal[
    "none_supplied", "operator_asserted", "verifier_unavailable", "verifier_unregistered",
    "verifier_not_independent", "verifier_registered",
]
IDENTITY_REASON_CODES = (
    "identity_missing",
    "identity_malformed",
    "identity_operator_asserted_only",
    "verifier_unavailable",
    "verifier_not_registered",
    "verifier_not_independent",
    "identity_stale",
    "identity_from_future",
)


def require_aware(now: object) -> datetime:
    """`now` is always injected by the caller; these modules never read a clock."""
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise TypeError("now must be a timezone-aware datetime supplied by the caller")
    return now.astimezone(timezone.utc)


# ------------------------------------------------------------
# the observation (versioned data a verifier or an operator would supply)
# ------------------------------------------------------------

class VerificationRecord(_Strict):
    """Who made the observation and what they claim about it. A claim, not a proof."""

    claim: VerificationClaim
    source_id: Identifier  # the operator (operator_asserted) or the verifier (verifier_attested)


class LabIdentityObservation(_Strict):
    contract_version: Literal["1.0"] = IDENTITY_CONTRACT_VERSION
    observation_id: Identifier
    lab_id: Identifier
    observed_network: Identifier
    observed_containers: list[Identifier]
    compose_sha256: Sha256 | None = None
    lab_state_id: LabStateId
    observed_at_utc: UtcTimestamp
    verification: VerificationRecord

    @model_validator(mode="after")
    def _containers(self) -> "LabIdentityObservation":
        if not self.observed_containers or len(set(self.observed_containers)) != len(self.observed_containers):
            raise ValueError("an observation lists its containers once each and at least one")
        return self


# ------------------------------------------------------------
# the registry of verifiers that actually exist
# ------------------------------------------------------------

class VerifierEntry(_Strict):
    verifier_id: Identifier
    kind: Literal["read_only_environment_inspector", "test_double"]
    suitable_for_live: bool = False

    @model_validator(mode="after")
    def _doubles_are_never_live(self) -> "VerifierEntry":
        if self.kind == "test_double" and self.suitable_for_live:
            raise ValueError("a test double can never be suitable for live use")
        return self


class VerifierRegistry:
    """A set of verifiers. A registry built by a caller is NOT authoritative: it lets logic be exercised (and its
    results are simulation-grade) but its verifiers can never be live-suitable. Only `default_verifier_registry()`,
    built from the repository's `_BUILTIN_VERIFIERS`, is authoritative."""

    def __init__(self, *entries: VerifierEntry) -> None:
        self._authoritative = False
        self._entries: dict[str, VerifierEntry] = {}
        for entry in entries:
            if not isinstance(entry, VerifierEntry):
                raise TypeError("a registry holds VerifierEntry objects")
            if entry.verifier_id in self._entries:
                raise ValueError(f"verifier {entry.verifier_id!r} is registered twice")
            self._entries[entry.verifier_id] = entry

    @property
    def is_empty(self) -> bool:
        return not self._entries

    @property
    def authoritative(self) -> bool:
        return self._authoritative

    def get(self, verifier_id: str) -> VerifierEntry | None:
        return self._entries.get(verifier_id)


def default_verifier_registry() -> VerifierRegistry:
    """The repository's registry, built from `_BUILTIN_VERIFIERS` (empty: no independent verifier exists)."""
    registry = VerifierRegistry(*_BUILTIN_VERIFIERS)
    registry._authoritative = True
    return registry


IDENTITY_VERIFIER_AVAILABLE = bool(_BUILTIN_VERIFIERS)


# ------------------------------------------------------------
# the assessment
# ------------------------------------------------------------

class IdentityAssessment(_Strict):
    contract_version: Literal["1.0"] = IDENTITY_CONTRACT_VERSION
    level: VerificationLevel
    fresh: bool
    independently_verified: bool
    live_suitable: bool
    authoritative_registry: bool = False
    reason_codes: list[Identifier]
    observation_id: Identifier | None = None
    lab_id: Identifier | None = None
    lab_state_id: LabStateId | None = None
    observed_at_utc: UtcTimestamp | None = None
    # The Phase 13A scope-policy input; present only for a fresh observation. Its `established_independently`
    # equals `independently_verified`, so an unverified observation is still denied by the scope policy.
    scope_identity: LabIdentityEvidence | None = None


def _none(code: str) -> IdentityAssessment:
    return IdentityAssessment(level="none_supplied", fresh=False, independently_verified=False,
                              live_suitable=False, reason_codes=[code])


def assess_lab_identity(
    observation: object,
    registry: VerifierRegistry | None = None,
    *,
    now: datetime,
    distinct_from: Iterable[str] = (),
    max_age_seconds: int = IDENTITY_MAX_AGE_SECONDS,
) -> IdentityAssessment:
    """Judge one observation. Never raises for bad DATA; raises TypeError only for a naive `now`."""
    moment = require_aware(now)
    registry = registry if registry is not None else default_verifier_registry()
    if observation is None:
        return _none("identity_missing")
    if not isinstance(observation, LabIdentityObservation):
        return _none("identity_malformed")

    reasons: list[str] = []
    max_age = min(max_age_seconds, IDENTITY_MAX_AGE_SECONDS)  # a caller can only tighten the limit, never extend it
    age = (moment - parse_utc(observation.observed_at_utc)).total_seconds()
    fresh = True
    if age < -CLOCK_SKEW_SECONDS:
        fresh = False
        reasons.append("identity_from_future")
    elif age > max_age:
        fresh = False
        reasons.append("identity_stale")

    claim, source = observation.verification.claim, observation.verification.source_id
    entry = None
    if claim == "operator_asserted":
        level: VerificationLevel = "operator_asserted"
        reasons.append("identity_operator_asserted_only")
    elif registry.is_empty:
        level = "verifier_unavailable"
        reasons.append("verifier_unavailable")
    else:
        entry = registry.get(source)
        if entry is None:
            level = "verifier_unregistered"
            reasons.append("verifier_not_registered")
        elif source in set(distinct_from):
            level = "verifier_not_independent"
            reasons.append("verifier_not_independent")
        else:
            level = "verifier_registered"

    independent = level == "verifier_registered" and fresh
    # live suitability needs the repository's own registry AND an entry that declares itself live-suitable
    live_suitable = independent and registry.authoritative and entry is not None and entry.suitable_for_live
    evidence = LabIdentityEvidence(
        lab_id=observation.lab_id, observed_network=observation.observed_network,
        observed_containers=list(observation.observed_containers), compose_sha256=observation.compose_sha256,
        lab_state_id=observation.lab_state_id, established_by=source, established_independently=independent,
    ) if fresh else None
    if evidence is not None and independent:
        # The ONLY place an independently-established identity is issued (see validation.seals).
        observed = parse_utc(observation.observed_at_utc)
        seals.issue_identity(
            seals.identity_key(
                lab_id=evidence.lab_id, observed_network=evidence.observed_network,
                observed_containers=evidence.observed_containers, compose_sha256=evidence.compose_sha256,
                lab_state_id=evidence.lab_state_id, established_by=evidence.established_by,
                established_independently=True),
            seals.IssuedIdentity(
                source_id=source, observed_at_utc=observation.observed_at_utc,
                valid_from=observed - timedelta(seconds=CLOCK_SKEW_SECONDS),
                valid_until=observed + timedelta(seconds=max_age),
                live_suitable=live_suitable, authoritative=registry.authoritative))
    return IdentityAssessment(
        level=level, fresh=fresh, independently_verified=independent, live_suitable=live_suitable,
        authoritative_registry=registry.authoritative, reason_codes=reasons,
        observation_id=observation.observation_id, lab_id=observation.lab_id,
        lab_state_id=observation.lab_state_id, observed_at_utc=observation.observed_at_utc,
        scope_identity=evidence,
    )
