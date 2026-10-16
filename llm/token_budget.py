"""
Provider-aware, pre-send input budgeting for language-model calls.

WHY THIS EXISTS
---------------
A single AutoShield grounded-answer or remediation request has hit Groq's limits twice (both recorded in
`PROJECT_STATUS.md`): an HTTP 413 "Request too large" (8,456 tokens against an 8,000 on-demand limit) and an
HTTP 429 rate-limit (`tokens per minute (TPM): Limit 8000`). The dashboard can also fire several of these
calls in the same minute (see the amplification note in `PROJECT_STATUS.md`). This module lets a caller
refuse an oversized prompt *before* any request is sent, so the failure is local, explicit and free, and the
room reserved for the completion is accounted for.

TOKEN COUNTING IS AN APPROXIMATION
----------------------------------
No tokenizer for the Groq model (`openai/gpt-oss-20b`) is installed, and downloading one would need network
access, which is out of scope. `estimate_tokens` is therefore a CHARACTER-BASED HEURISTIC, not the provider's
own count, and every result produced through this module is labelled as approximate. The default density
(`DEFAULT_CHARS_PER_TOKEN = 3.0`) is calibrated against the one real measurement we have: Groq counted the
25,981-character request that failed with HTTP 413 as 8,456 tokens, i.e. about 3.07 characters per token for
this JSON-with-identifiers content. 3.0 is slightly below that, so the estimate leans toward OVER-counting
(failing early) rather than under-counting (sending a request that the provider then rejects). A caller that
has a real tokenizer can pass its own `estimator` and get an exact count.

This module performs NO network, file or environment access and knows nothing provider-specific beyond the
numbers the caller supplies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# Calibrated against the recorded Groq count (8,456 tokens for 25,981 characters ~= 3.07 chars/token); 3.0
# is deliberately a little lower so the heuristic over-counts rather than under-counts. NOT a provider fact.
DEFAULT_CHARS_PER_TOKEN = 3.0

# A small fixed per-message framing cost (role wrappers, delimiters). Chat APIs add a handful of tokens per
# message; a few extra estimated tokens only make the guard slightly stricter.
DEFAULT_PER_MESSAGE_OVERHEAD = 4

APPROXIMATION_LABEL = (
    "approximate: character-based heuristic (~3.0 chars/token, calibrated to one recorded Groq count), "
    "NOT the provider's tokenizer"
)

Estimator = Callable[[str], int]


def estimate_tokens(text: str, *, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """
    A conservative, LABELLED approximation of the token count of `text`.

    This is not the provider's tokenizer (see the module docstring). It rounds up and never returns a
    negative number. `chars_per_token` must be positive.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    if not text:
        return 0
    return math.ceil(len(text) / chars_per_token)


class RequestNotSentError(Exception):
    """
    Base class for pre-send refusals: the request was rejected locally and NOTHING reached the provider.

    `request_sent` is always False. Callers (RAG answer, remediation planner, CLIs, dashboard) catch this one
    base type to mark a result `request_sent=False` whatever the specific reason (oversized prompt, or no room
    left in the per-minute rate budget). Messages carry only counts, never prompt text or secrets.
    """

    request_sent = False


class PromptTooLargeError(RequestNotSentError):
    """
    Raised to refuse a prompt whose estimated size exceeds the configured budget BEFORE any request is sent.

    `request_sent` is always False: this error means nothing reached the provider. The message is safe to log
    (it contains only counts, never prompt text or secrets).
    """

    def __init__(self, estimated_prompt_tokens: int, max_prompt_tokens: int, detail: str) -> None:
        self.estimated_prompt_tokens = estimated_prompt_tokens
        self.max_prompt_tokens = max_prompt_tokens
        super().__init__(detail)


@dataclass(frozen=True)
class BudgetCheck:
    """The outcome of checking a prompt against a budget (counts only; no prompt text, no secrets)."""

    fits: bool
    estimated_prompt_tokens: int
    max_prompt_tokens: int
    max_total_tokens: int
    reserved_completion_tokens: int
    safety_margin_tokens: int
    approximation: str = APPROXIMATION_LABEL

    def as_dict(self) -> dict[str, object]:
        return {
            "fits": self.fits,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_total_tokens": self.max_total_tokens,
            "reserved_completion_tokens": self.reserved_completion_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "token_count_method": self.approximation,
        }


@dataclass(frozen=True)
class TokenBudget:
    """
    A pre-send input budget for one request.

    `max_total_tokens` is the whole-request ceiling (prompt + completion); `reserved_completion_tokens` is
    held back for the model's reply, and `safety_margin_tokens` absorbs the heuristic's error. The prompt is
    allowed `max_total_tokens - reserved_completion_tokens - safety_margin_tokens` estimated tokens.
    """

    max_total_tokens: int
    reserved_completion_tokens: int = 0
    safety_margin_tokens: int = 0
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN

    def __post_init__(self) -> None:
        for name in ("max_total_tokens", "reserved_completion_tokens", "safety_margin_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an integer of at least 0")
        if self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be at least 1")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        if self.max_prompt_tokens < 1:
            raise ValueError(
                f"the budget leaves no room for a prompt: max_total_tokens ({self.max_total_tokens}) minus "
                f"reserved_completion_tokens ({self.reserved_completion_tokens}) minus safety_margin_tokens "
                f"({self.safety_margin_tokens}) must be at least 1"
            )

    @property
    def max_prompt_tokens(self) -> int:
        return self.max_total_tokens - self.reserved_completion_tokens - self.safety_margin_tokens

    def estimate_prompt_tokens(self, system_prompt: str, user_prompt: str, *, estimator: Estimator | None = None) -> int:
        """Estimated prompt tokens for the two messages, including a small per-message framing cost."""
        count = estimator or (lambda text: estimate_tokens(text, chars_per_token=self.chars_per_token))
        return count(system_prompt) + count(user_prompt) + 2 * DEFAULT_PER_MESSAGE_OVERHEAD

    def check(self, system_prompt: str, user_prompt: str, *, estimator: Estimator | None = None) -> BudgetCheck:
        estimated = self.estimate_prompt_tokens(system_prompt, user_prompt, estimator=estimator)
        return BudgetCheck(
            fits=estimated <= self.max_prompt_tokens,
            estimated_prompt_tokens=estimated,
            max_prompt_tokens=self.max_prompt_tokens,
            max_total_tokens=self.max_total_tokens,
            reserved_completion_tokens=self.reserved_completion_tokens,
            safety_margin_tokens=self.safety_margin_tokens,
        )

    def enforce(self, system_prompt: str, user_prompt: str, *, estimator: Estimator | None = None) -> BudgetCheck:
        """
        Return the check if the prompt fits; otherwise raise `PromptTooLargeError` (no request is sent).

        The error message carries only counts, so it is safe to print and store.
        """
        result = self.check(system_prompt, user_prompt, estimator=estimator)
        if not result.fits:
            raise PromptTooLargeError(
                result.estimated_prompt_tokens,
                result.max_prompt_tokens,
                f"the estimated prompt is {result.estimated_prompt_tokens} tokens, above the "
                f"{result.max_prompt_tokens}-token budget "
                f"(total {result.max_total_tokens}, reserved for completion {result.reserved_completion_tokens}, "
                f"safety margin {result.safety_margin_tokens}; {APPROXIMATION_LABEL}); no request was sent",
            )
        return result
