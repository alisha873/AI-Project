"""
Offline tests for the process-local TPM rate budget (`llm.rate_budget`). No network and no provider.

They pin: in-window accounting, exact boundaries, single-request-too-big, reconciliation with the provider's
usage figure (including malformed usage), safe release before send, idempotent settlement, expiry/purging with
an injected clock, concurrency safety, and the shared per-configuration registry.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from llm.rate_budget import (
    RateBudget,
    RateBudgetError,
    RateBudgetExceededError,
    account_fingerprint,
    budget_key,
    reset_rate_budgets,
    shared_rate_budget,
)
from llm.token_budget import RequestNotSentError


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_accounting_tracks_used_and_remaining():
    budget = RateBudget(1000, clock=FakeClock())
    assert budget.used() == 0 and budget.remaining() == 1000
    budget.reserve(300)
    budget.reserve(200)
    assert budget.used() == 500 and budget.remaining() == 500


def test_exact_boundary_fits_and_one_more_token_is_refused():
    budget = RateBudget(1000, clock=FakeClock())
    budget.reserve(1000)                       # exactly the whole budget fits
    assert budget.remaining() == 0
    with pytest.raises(RateBudgetExceededError) as raised:
        budget.reserve(1)
    assert raised.value.request_sent is False
    assert isinstance(raised.value, RequestNotSentError)
    assert raised.value.needed_tokens == 1 and raised.value.remaining_tokens == 0


def test_a_single_request_bigger_than_the_whole_budget_is_refused_before_send():
    budget = RateBudget(500, clock=FakeClock())
    with pytest.raises(RateBudgetExceededError) as raised:
        budget.reserve(501)
    assert raised.value.request_sent is False
    assert "more than the entire" in str(raised.value)
    assert budget.used() == 0                  # nothing was recorded


def test_commit_reconciles_to_the_providers_actual_usage():
    budget = RateBudget(1000, clock=FakeClock())
    reservation = budget.reserve(300)          # estimate
    assert budget.used() == 300
    reservation.commit(actual_tokens=420)      # provider said it really used more
    assert budget.used() == 420
    # A second commit is a no-op (already settled).
    reservation.commit(actual_tokens=10)
    assert budget.used() == 420


@pytest.mark.parametrize("bad_usage", [None, MagicMock(), -5, 3.5, True, "123"])
def test_malformed_or_missing_usage_keeps_the_estimate_and_never_crashes(bad_usage):
    budget = RateBudget(1000, clock=FakeClock())
    reservation = budget.reserve(250)
    reservation.commit(actual_tokens=bad_usage)
    assert budget.used() == 250                 # estimate retained, no exception


def test_release_before_send_returns_the_tokens():
    budget = RateBudget(1000, clock=FakeClock())
    reservation = budget.reserve(400)
    assert budget.used() == 400
    reservation.release()
    assert budget.used() == 0
    # Releasing again, or committing after release, does nothing.
    reservation.release()
    reservation.commit(actual_tokens=400)
    assert budget.used() == 0


def test_commit_then_release_is_a_no_op_so_processed_tokens_are_not_given_back():
    budget = RateBudget(1000, clock=FakeClock())
    reservation = budget.reserve(400)
    reservation.commit()                        # request may have been processed; keep it counted
    reservation.release()                       # must NOT give the tokens back
    assert budget.used() == 400


def test_entries_expire_after_the_window():
    clock = FakeClock()
    budget = RateBudget(1000, window_seconds=60, clock=clock)
    budget.reserve(800)
    assert budget.used() == 800
    clock.advance(61)                           # the reservation has aged out of the 60s window
    assert budget.used() == 0
    budget.reserve(800)                         # so a fresh request fits again
    assert budget.used() == 800


def test_window_boundary_counts_tokens_still_inside_the_window():
    clock = FakeClock()
    budget = RateBudget(1000, window_seconds=60, clock=clock)
    budget.reserve(600)
    clock.advance(59)                           # still inside the 60s window
    assert budget.used() == 600
    with pytest.raises(RateBudgetExceededError):
        budget.reserve(500)                     # 600 + 500 > 1000 while both are in-window


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "100"])
def test_invalid_limit_is_refused(bad):
    with pytest.raises(RateBudgetError):
        RateBudget(bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_invalid_reserve_amount_is_refused(bad):
    budget = RateBudget(1000, clock=FakeClock())
    with pytest.raises(RateBudgetError):
        budget.reserve(bad)


def test_concurrent_reservations_never_exceed_the_limit():
    # 200 threads each try to reserve 10 tokens against a 1000-token budget: at most 100 can succeed and the
    # total committed must never exceed the limit (the lock prevents two from consuming the same remaining).
    budget = RateBudget(1000, window_seconds=10_000, clock=FakeClock())
    granted: list[int] = []
    granted_lock = threading.Lock()
    barrier = threading.Barrier(50)

    def worker() -> None:
        barrier.wait()
        try:
            budget.reserve(10)
            with granted_lock:
                granted.append(10)
        except RateBudgetExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(granted) <= 1000
    assert budget.used() <= 1000
    assert len(granted) == 100            # exactly the budget's worth were granted


def test_shared_registry_coordinates_identical_configs_and_separates_different_ones():
    reset_rate_budgets()
    try:
        a = shared_rate_budget(provider="groq", limit_tokens=8000, account="key-A")
        a_again = shared_rate_budget(provider="groq", limit_tokens=8000, account="key-A")
        assert a is a_again                                  # same config -> one shared bucket
        b = shared_rate_budget(provider="groq", limit_tokens=8000, account="key-B")
        assert b is not a                                    # different account -> separate bucket
        c = shared_rate_budget(provider="groq", limit_tokens=4000, account="key-A")
        assert c is not a                                    # different limit -> separate bucket
    finally:
        reset_rate_budgets()


def test_reset_registry_clears_buckets():
    reset_rate_budgets()
    first = shared_rate_budget(provider="groq", limit_tokens=8000, account="k")
    reset_rate_budgets()
    second = shared_rate_budget(provider="groq", limit_tokens=8000, account="k")
    assert first is not second


def test_account_fingerprint_hides_the_secret():
    secret = "gsk_super_secret_value_1234567890"
    fp = account_fingerprint(secret)
    assert secret not in fp and len(fp) == 12
    assert account_fingerprint(secret) == fp                # stable
    assert account_fingerprint("other") != fp               # distinct
    assert account_fingerprint("") == "none"
    assert secret not in budget_key(provider="groq", limit_tokens=8000, window_seconds=60, account=secret)
