"""
Process-local, thread-safe, provider-aware tokens-per-minute (TPM) accounting for language-model calls.

WHY THIS EXISTS
---------------
Groq rejected AutoShield requests twice (both recorded in `PROJECT_STATUS.md`): an HTTP 413 "Request too
large" and an HTTP 429 `tokens per minute (TPM): Limit 8000`. A per-request size guard (`llm.token_budget`)
stops one oversized prompt, but it cannot stop SEVERAL in-budget requests in the same minute from exceeding a
per-minute quota. The dashboard fires a narrative call plus one grounded answer and one plan per finding, so a
burst can trip the TPM ceiling even when each prompt fits on its own.

This ledger tracks the estimated input-plus-reserved-output tokens that have been committed to a provider
quota inside a sliding time window, and refuses a new request LOCALLY (before anything is sent) when it would
not fit the remaining allowance. The failure is then free, explicit and labelled `request_sent=False`.

WHAT IT IS NOT
--------------
* It is NOT a guarantee against a provider 429. The remote quota is shared with every other client and worker
  and its true remaining value is not visible here; see "LIMITS" below.
* The ~8,000 number is the recorded Groq on-demand ceiling for ONE configuration, not a universal fact and not
  a model context window. It is configurable and its provenance is documented at the call site.
* Token counts fed in are an approximation (see `llm.token_budget`), reconciled with the provider's own usage
  figure after a call when the provider returns one.

LIMITS (process-local accounting)
---------------------------------
* Accounting is per Python process. Separate workers, machines or the provider's other clients are invisible,
  so the real remaining quota can be smaller than this ledger believes.
* Provider-SDK retries happen inside one `generate()` call and are opaque here, so a single reservation can
  correspond to more than one actual provider request. Keep the SDK retry count low (the dashboard uses 0) if
  exact accounting matters.
* There is no waiting or queuing: a request that does not fit fails immediately with an actionable error
  rather than blocking, so there is never an unbounded wait or retry loop.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Callable

from llm.token_budget import RequestNotSentError

# The recorded Groq on-demand ceiling that motivated this (PROJECT_STATUS): 8,000 tokens per minute. A
# documented default for that one configuration, configurable per account/model, NOT a universal limit.
RECORDED_GROQ_TPM = 8000
DEFAULT_WINDOW_SECONDS = 60.0
# A reservation is held for at most this long before it is treated as expired and purged, so a caller that
# forgets to commit or release cannot leak the quota forever. It is a small multiple of the window.
_MAX_RESERVATION_AGE_FACTOR = 2.0
# Hard cap on live entries per bucket so a long-running process cannot grow without bound even under churn.
_MAX_ENTRIES_PER_BUCKET = 100_000


class RateBudgetError(ValueError):
    """A misconfiguration (non-positive limit/window, or a request larger than the whole quota can ever be)."""


class RateBudgetExceededError(RequestNotSentError):
    """
    Raised to refuse a request whose tokens will not fit the remaining per-minute budget. NOTHING was sent
    (`request_sent` is False, inherited). The message carries only counts and is safe to log.
    """

    def __init__(self, needed_tokens: int, remaining_tokens: int, limit_tokens: int,
                 window_seconds: float, detail: str) -> None:
        self.needed_tokens = needed_tokens
        self.remaining_tokens = remaining_tokens
        self.limit_tokens = limit_tokens
        self.window_seconds = window_seconds
        super().__init__(detail)


@dataclass
class _Entry:
    """One accounted request: `tokens` against the quota, stamped at monotonic time `at`."""

    tokens: int
    at: float


class Reservation:
    """
    A handle for one accounted request. Exactly one terminal action must follow a successful reserve:

    * `commit(actual_tokens)` - the request was (or may have been) processed by the provider; keep it counted,
      reconciled to the provider's own usage figure when one is supplied. Use this after a successful call AND
      after any error once the request has left this process, because provider processing may have occurred.
    * `release()` - the request was refused locally BEFORE transmission (or is definitively known not to have
      been processed); give the tokens back. Never call this once a call may have reached the provider.

    Both are idempotent and safe to call once; a second terminal call is a no-op.
    """

    def __init__(self, ledger: "RateBudget", entry: _Entry) -> None:
        self._ledger = ledger
        self._entry = entry
        self._settled = False

    @property
    def tokens(self) -> int:
        return self._entry.tokens

    @property
    def settled(self) -> bool:
        return self._settled

    def commit(self, actual_tokens: int | None = None) -> None:
        """Keep the reservation counted; replace the estimate with the provider's figure when it is valid."""
        with self._ledger._lock:
            if self._settled:
                return
            self._settled = True
            if isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int) or actual_tokens < 0:
                return  # missing or malformed usage metadata: keep the pre-send estimate, never crash
            self._entry.tokens = actual_tokens

    def release(self) -> None:
        """Give the tokens back (only valid before the request could have been processed)."""
        with self._ledger._lock:
            if self._settled:
                return
            self._settled = True
            self._ledger._drop(self._entry)


class RateBudget:
    """
    A single sliding-window token bucket for ONE provider/account/limit, safe for concurrent use in-process.

    `reserve(tokens)` purges expired entries, and either records and returns a `Reservation` or raises
    `RateBudgetExceededError` when the tokens would push the in-window total past `limit_tokens`. There is no
    waiting: a request either fits now or is refused now.
    """

    def __init__(self, limit_tokens: int, *, window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if isinstance(limit_tokens, bool) or not isinstance(limit_tokens, int) or limit_tokens < 1:
            raise RateBudgetError("limit_tokens must be a positive integer")
        if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
            raise RateBudgetError("window_seconds must be positive")
        self.limit_tokens = limit_tokens
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: list[_Entry] = []

    # -- internal (callers hold the lock) --------------------------------------------------------------

    def _purge(self, now: float) -> None:
        horizon = now - self.window_seconds * _MAX_RESERVATION_AGE_FACTOR
        if self._entries and self._entries[0].at <= horizon:
            self._entries = [e for e in self._entries if e.at > horizon]
        if len(self._entries) > _MAX_ENTRIES_PER_BUCKET:          # defensive memory bound under extreme churn
            self._entries = self._entries[-_MAX_ENTRIES_PER_BUCKET:]

    def _used_locked(self, now: float) -> int:
        window_start = now - self.window_seconds
        return sum(e.tokens for e in self._entries if e.at > window_start)

    def _drop(self, entry: _Entry) -> None:
        try:
            self._entries.remove(entry)
        except ValueError:
            pass

    # -- public ----------------------------------------------------------------------------------------

    def used(self) -> int:
        """Tokens accounted within the current window (estimates, reconciled to provider usage where known)."""
        with self._lock:
            now = self._clock()
            self._purge(now)
            return self._used_locked(now)

    def remaining(self) -> int:
        with self._lock:
            now = self._clock()
            self._purge(now)
            return max(0, self.limit_tokens - self._used_locked(now))

    def reserve(self, tokens: int) -> Reservation:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise RateBudgetError("tokens must be a positive integer")
        if tokens > self.limit_tokens:
            raise RateBudgetExceededError(
                tokens, self.limit_tokens, self.limit_tokens, self.window_seconds,
                f"this single request needs about {tokens} tokens, more than the entire "
                f"{self.limit_tokens}-token per-{self.window_seconds:g}s budget can ever allow; "
                f"no request was sent",
            )
        with self._lock:
            now = self._clock()
            self._purge(now)
            used = self._used_locked(now)
            remaining = self.limit_tokens - used
            if tokens > remaining:
                raise RateBudgetExceededError(
                    tokens, max(0, remaining), self.limit_tokens, self.window_seconds,
                    f"the request needs about {tokens} tokens but only {max(0, remaining)} of the "
                    f"{self.limit_tokens}-token per-{self.window_seconds:g}s budget remain "
                    f"(approximate token counts); no request was sent",
                )
            entry = _Entry(tokens=tokens, at=now)
            self._entries.append(entry)
            return Reservation(self, entry)


# ------------------------------------------------------------------------------------------------------
# Shared registry: calls that use the SAME provider/account/limit coordinate through ONE bucket, so two
# simultaneous requests cannot each independently consume the same remaining budget.
# ------------------------------------------------------------------------------------------------------

_REGISTRY: dict[str, RateBudget] = {}
_REGISTRY_LOCK = threading.Lock()


def account_fingerprint(secret: str) -> str:
    """A short, non-reversible tag for an account key, so different accounts get different buckets WITHOUT
    the secret ever being stored. Never returns or logs the secret itself."""
    if not secret:
        return "none"
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:12]


def budget_key(*, provider: str, limit_tokens: int, window_seconds: float, account: str = "") -> str:
    return f"{provider}:{limit_tokens}:{window_seconds:g}:{account_fingerprint(account)}"


def shared_rate_budget(*, provider: str, limit_tokens: int, window_seconds: float = DEFAULT_WINDOW_SECONDS,
                       account: str = "", clock: Callable[[], float] = time.monotonic) -> RateBudget:
    """
    The shared `RateBudget` for a (provider, account, limit, window) combination, created on first use.

    Independent configurations get independent buckets. The clock is honoured only when the bucket is first
    created (so tests that inject a fake clock must use a distinct limit/window/account to get a fresh bucket,
    or call `reset_rate_budgets()` first).
    """
    key = budget_key(provider=provider, limit_tokens=limit_tokens, window_seconds=window_seconds, account=account)
    with _REGISTRY_LOCK:
        budget = _REGISTRY.get(key)
        if budget is None:
            budget = RateBudget(limit_tokens, window_seconds=window_seconds, clock=clock)
            _REGISTRY[key] = budget
        return budget


def reset_rate_budgets() -> None:
    """Clear every shared bucket. For test isolation only (see tests/conftest.py); not for production use."""
    with _REGISTRY_LOCK:
        _REGISTRY.clear()
