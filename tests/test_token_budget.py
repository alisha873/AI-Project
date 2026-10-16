"""
Offline tests for the pre-send token budget (`llm.token_budget`).

No network, no provider, no tokenizer download. The estimator is a labelled approximation, so the tests
check its documented behaviour (monotonic, rounds up, conservative density), not an exact provider count.
"""

from __future__ import annotations

import pytest

from llm.token_budget import (
    APPROXIMATION_LABEL,
    DEFAULT_CHARS_PER_TOKEN,
    BudgetCheck,
    PromptTooLargeError,
    TokenBudget,
    estimate_tokens,
)


def test_estimate_is_zero_for_empty_and_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1               # ceil(1/3.0)
    assert estimate_tokens("a" * 300) == 100       # 300 / 3.0


def test_estimate_is_monotonic_and_rejects_bad_input():
    assert estimate_tokens("a" * 10) <= estimate_tokens("a" * 11)
    with pytest.raises(TypeError):
        estimate_tokens(123)                       # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimate_tokens("x", chars_per_token=0)


def test_default_density_is_conservative_against_the_recorded_groq_count():
    # Recorded: Groq counted a 25,981-char request as 8,456 tokens (~3.07 chars/token). The default 3.0
    # must NOT under-count that request (under-counting would let an oversized request through).
    assert DEFAULT_CHARS_PER_TOKEN <= 25981 / 8456
    assert estimate_tokens("x" * 25981) >= 8456


def test_budget_reserves_room_for_the_completion():
    budget = TokenBudget(max_total_tokens=8000, reserved_completion_tokens=2000, safety_margin_tokens=200)
    assert budget.max_prompt_tokens == 5800


def test_a_small_prompt_fits_and_a_large_one_does_not():
    budget = TokenBudget(max_total_tokens=1000, reserved_completion_tokens=200)
    small = budget.check("short system", "short user")
    assert isinstance(small, BudgetCheck) and small.fits
    big = budget.check("x" * 9000, "y" * 9000)
    assert not big.fits and big.estimated_prompt_tokens > big.max_prompt_tokens


def test_enforce_raises_before_sending_and_carries_only_counts():
    budget = TokenBudget(max_total_tokens=500, reserved_completion_tokens=100)
    with pytest.raises(PromptTooLargeError) as info:
        budget.enforce("x" * 5000, "y" * 5000)
    exc = info.value
    assert exc.request_sent is False
    assert exc.estimated_prompt_tokens > exc.max_prompt_tokens
    assert "no request was sent" in str(exc)
    # the message is safe to log: it must not contain the prompt text
    assert "xxxx" not in str(exc) and "yyyy" not in str(exc)


def test_enforce_returns_the_check_when_it_fits():
    budget = TokenBudget(max_total_tokens=100000, reserved_completion_tokens=1000)
    result = budget.enforce("a little system prompt", "a little user prompt")
    assert result.fits and result.approximation == APPROXIMATION_LABEL


def test_an_injected_exact_estimator_is_used():
    budget = TokenBudget(max_total_tokens=100, reserved_completion_tokens=0)
    # a fake "exact" tokenizer that says every message is 10 tokens
    check = budget.check("whatever", "whatever", estimator=lambda text: 10)
    assert check.estimated_prompt_tokens == 10 + 10 + 8   # + per-message overhead (4 each)


def test_construction_rejects_impossible_budgets():
    with pytest.raises(ValueError):
        TokenBudget(max_total_tokens=0)
    with pytest.raises(ValueError):
        TokenBudget(max_total_tokens=100, reserved_completion_tokens=100)   # no room for a prompt
    with pytest.raises(ValueError):
        TokenBudget(max_total_tokens=-1)
    with pytest.raises(ValueError):
        TokenBudget(max_total_tokens=True)                                  # type: ignore[arg-type]


def test_as_dict_is_counts_only():
    check = TokenBudget(max_total_tokens=8000, reserved_completion_tokens=2000).check("s", "u")
    data = check.as_dict()
    assert set(data) == {
        "fits", "estimated_prompt_tokens", "max_prompt_tokens", "max_total_tokens",
        "reserved_completion_tokens", "safety_margin_tokens", "token_count_method",
    }
    assert "approx" in data["token_count_method"].lower()
