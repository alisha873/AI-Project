"""
Phase 13B.3 - in-process issuance records that close the "hand-built trust" paths (pure; no I/O).

Problem. Data in this package is plain, strict, serializable models. A caller can construct any of them by hand, or
load them from JSON: `LabIdentityEvidence(established_independently=True)`, a `ValidationReport` that says it came
from a live transport, a "verified" flag in a test fixture. A boolean written by the caller can not be trusted.

Mechanism. Trust is not carried by a field a caller can set; it is carried by a record that exists only because a
specific function ran in this process:

  identity issuance   `issue_identity(...)` is called ONLY by `validation.lab_identity.assess_lab_identity`, and only
                      for a fresh observation that a registered, distinct verifier made. The record stores whether the
                      verifier is live-suitable and until when the observation counts. `ScopePolicy` trusts
                      `established_independently` only when `lookup_identities(...)` finds such a record.
  report issuance     `issue_report(...)` is called ONLY by `validation.runner`, for a report it just built.
                      `validation.evidence_gate` treats a report as usable evidence only if `is_report_issued(...)`.

What this does and does not give.
  * A caller-supplied boolean, a hand-written fixture, a deserialized JSON object, an edited copy (`model_copy`,
    changed fields) or a report built by hand is NOT trusted: its content digest was never issued.
  * It does not stop a malicious caller who imports this module and calls the issue functions, or who patches module
    constants, inside the same Python process. That is outside what in-process code can prevent; an AST test only
    pins that the issue functions are called from exactly one module each.
  * Issuance is per process and is never persisted, so a stored or reloaded report is never "usable" until a signed,
    persistent design replaces this one (recorded as an open limitation).

Records are bounded; the oldest are evicted, which only ever makes evidence LESS trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

_MAX_RECORDS = 10_000
_MAX_PER_KEY = 16


@dataclass(frozen=True)
class IssuedIdentity:
    source_id: str
    observed_at_utc: str
    valid_from: datetime   # the earliest `now` for which the observation counts (observation time minus clock skew)
    valid_until: datetime  # the latest `now` for which the observation counts (observation time plus the age limit)
    live_suitable: bool
    authoritative: bool    # the verifier came from the repository's own registry, not a caller-built one


_IDENTITIES: dict[str, list[IssuedIdentity]] = {}
_REPORTS: dict[str, None] = {}


def identity_key(*, lab_id: str, observed_network: str, observed_containers: Iterable[str],
                 compose_sha256: str | None, lab_state_id: str | None, established_by: str,
                 established_independently: bool) -> str:
    material = {
        "lab_id": lab_id, "network": observed_network, "containers": sorted(observed_containers),
        "compose": compose_sha256, "state": lab_state_id, "by": established_by,
        "independent": established_independently,
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _bounded_put(store: dict, key: str, value: object) -> None:
    store[key] = value
    while len(store) > _MAX_RECORDS:
        store.pop(next(iter(store)))


def issue_identity(key: str, record: IssuedIdentity) -> None:
    """Called only by `validation.lab_identity.assess_lab_identity`. The key does not include the observation time
    (the evidence object does not carry it), so one key can have several issuance records, each with its own validity
    window; the evidence is valid at `now` if ANY record's window covers it."""
    records = _IDENTITIES.get(key, [])
    if record not in records:
        records = [*records, record][-_MAX_PER_KEY:]
    _bounded_put(_IDENTITIES, key, records)


def lookup_identities(key: str) -> tuple[IssuedIdentity, ...]:
    return tuple(_IDENTITIES.get(key, ()))


def report_key(report_dump: object) -> str:
    """Digest of a report's canonical JSON form (`ValidationReport.model_dump(mode="json")`)."""
    return hashlib.sha256(json.dumps(report_dump, sort_keys=True).encode("utf-8")).hexdigest()


def issue_report(key: str) -> None:
    """Called only by `validation.runner`."""
    _bounded_put(_REPORTS, key, None)


def is_report_issued(key: str) -> bool:
    return key in _REPORTS
