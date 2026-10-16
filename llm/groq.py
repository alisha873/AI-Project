from __future__ import annotations

from groq import Groq

from llm.base import LLMClient
from llm.rate_budget import RateBudget, RECORDED_GROQ_TPM, shared_rate_budget
from llm.settings import (
    LLMSettings,
    resolve_max_completion_tokens,
    resolve_max_retries,
    resolve_request_timeout,
    resolve_token_budget,
    resolve_tpm_limit,
)
from llm.token_budget import TokenBudget, estimate_tokens

# Default-ON per-minute token budget for Groq (env LLM_TPM_LIMIT overrides; "0" disables). The number is the
# recorded Groq on-demand ceiling (PROJECT_STATUS), configurable per account/model; NOT a universal fact.
DEFAULT_GROQ_TPM = RECORDED_GROQ_TPM
# Output tokens assumed per request for rate accounting when no explicit completion limit is set. A modest
# allowance so input + output are both counted against the per-minute budget; the ledger is reconciled to the
# provider's reported usage after the call.
DEFAULT_RESERVED_COMPLETION_TOKENS = 512


class GroqLLMClient(LLMClient):
    """
    Groq-backed implementation of the provider-neutral LLMClient.

    Completion-token limit: by default none is sent, so the provider's own default applies (unchanged
    behaviour). A limit can be set with the `max_completion_tokens` argument or LLM_MAX_COMPLETION_TOKENS
    and is sent as the SDK's `max_completion_tokens`. It is bounded (see llm.settings) and never clamped
    or adjusted by this client; a provider error is raised to the caller unchanged.

    Per-minute token budget (TPM): ON BY DEFAULT. Before each call the client reserves the estimated input
    plus a modest expected-output allowance against a shared per-(account, limit) budget (default
    DEFAULT_GROQ_TPM, the recorded Groq on-demand ceiling; override with LLM_TPM_LIMIT, disable with
    LLM_TPM_LIMIT=0). A request that would not fit the remaining per-minute allowance raises
    RateBudgetExceededError BEFORE anything is sent (request_sent stays False). After a call the reservation is
    reconciled to the provider's reported usage. This is process-local accounting and is not a guarantee
    against a provider 429 (see llm.rate_budget). It never waits or loops.

    Retries (SDK behaviour, not this class's): this class adds no retry loop, but it builds `Groq(api_key=...)`
    with the SDK's default `max_retries` (2, see `groq.DEFAULT_MAX_RETRIES`). In the installed SDK
    (`groq/_base_client.py`) a call that raises a timeout or connection error, or receives HTTP 408, 409, 429
    or 5xx (or a response header `x-should-retry: true`), is re-sent automatically, after the `retry-after`
    delay when that is 1-60 s and otherwise after a 0.5-8 s exponential backoff. So ONE `generate()` call can
    reach the provider up to three times, and each re-sent request counts against provider rate limits.
    HTTP 413 and 400 are not in that list (a 413 is re-sent only if the server sends `x-should-retry: true`;
    that is not known). Callers that must bound the provider request count exactly cannot rely on counting
    `generate()` calls.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_completion_tokens: int | None = None,
        max_retries: int | None = None,
        token_budget: int | None = None,
        tpm_limit: int | None = None,
        rate_budget: RateBudget | None = None,
        request_timeout: float | None = None,
    ) -> None:
        settings = LLMSettings()

        # Key resolution: explicit argument, then GROQ_API_KEY, then LLM_API_KEY
        # (both read from the environment or the git-ignored local .env file).
        resolved_key = api_key or settings.groq_api_key or settings.api_key
        if not resolved_key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY (or LLM_API_KEY) "
                "in the local .env file or the environment."
            )

        self.model = model or settings.model
        self.max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens, settings.max_completion_tokens
        )
        # SDK retries: None keeps the SDK default (2) and the unchanged construction path; a configured
        # value (0 disables re-sends) is passed to the SDK so one generate() reaches the provider 1 + n times.
        self.max_retries = resolve_max_retries(max_retries, settings.max_retries)

        # Per-request timeout: None keeps the SDK's own default timeout and the unchanged construction path; a
        # configured value (env LLM_REQUEST_TIMEOUT) is passed to the SDK so an operator can bound the deadline.
        self.request_timeout = resolve_request_timeout(request_timeout, settings.request_timeout)

        # Optional pre-send input budget: when a total-token budget is configured, refuse an oversized prompt
        # before any request is sent, reserving room for the configured completion limit. Default: no budget
        # (unchanged behaviour). See llm.token_budget for the approximation used.
        resolved_budget = resolve_token_budget(token_budget, settings.token_budget)
        self.token_budget = (
            TokenBudget(
                max_total_tokens=resolved_budget,
                reserved_completion_tokens=self.max_completion_tokens or 0,
            )
            if resolved_budget is not None
            else None
        )

        # Per-minute (TPM) budget: ON by default (DEFAULT_GROQ_TPM), overridable with LLM_TPM_LIMIT, disabled
        # with LLM_TPM_LIMIT=0 or an injected rate_budget. The bucket is SHARED per (provider, account, limit)
        # so several calls in the same minute coordinate instead of each assuming the whole quota is free.
        resolved_tpm = resolve_tpm_limit(tpm_limit, settings.tpm_limit)
        effective_tpm = DEFAULT_GROQ_TPM if resolved_tpm is None else resolved_tpm
        self.tpm_limit = effective_tpm or None   # 0 -> disabled -> None
        if rate_budget is not None:
            self.rate_budget: RateBudget | None = rate_budget
        elif self.tpm_limit is not None:
            self.rate_budget = shared_rate_budget(
                provider="groq", limit_tokens=self.tpm_limit, account=resolved_key
            )
        else:
            self.rate_budget = None

        # Pass max_retries / timeout only when configured, so the default path stays Groq(api_key=...) exactly.
        client_kwargs: dict[str, object] = {"api_key": resolved_key}
        if self.max_retries is not None:
            client_kwargs["max_retries"] = self.max_retries
        if self.request_timeout is not None:
            client_kwargs["timeout"] = self.request_timeout
        self.client = Groq(**client_kwargs)

    @staticmethod
    def _make_strict_schema(schema: dict) -> dict:
        """Prepare a Pydantic JSON Schema for strict structured output."""
        schema = dict(schema)

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

            properties = schema.get("properties", {})
            schema["required"] = list(properties.keys())

            schema["properties"] = {
                key: GroqLLMClient._make_strict_schema(value)
                if isinstance(value, dict)
                else value
                for key, value in properties.items()
            }

        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = GroqLLMClient._make_strict_schema(
                schema["items"]
            )

        for key in ("anyOf", "oneOf", "allOf"):
            if key in schema:
                schema[key] = [
                    GroqLLMClient._make_strict_schema(item)
                    if isinstance(item, dict)
                    else item
                    for item in schema[key]
                ]

        return schema


    def _reserved_output_tokens(self) -> int:
        """
        Output tokens to count against the per-minute budget for ONE request.

        This is a modest EXPECTED-output allowance, not the configured completion CAP: a large cap (e.g. 8192)
        is the most the model MAY return, not what it usually does, and reserving the whole cap would make a
        cap that is itself near the per-minute quota refuse every request (the recorded 8192-vs-8000
        incompatibility). So we reserve `DEFAULT_RESERVED_COMPLETION_TOKENS`, but never more than the cap when a
        smaller one is set (the model cannot exceed it), and reconcile to the provider's reported usage after
        the call. The cap itself is still enforced by the provider and by the opt-in per-request TokenBudget.
        """
        if self.max_completion_tokens is not None:
            return min(self.max_completion_tokens, DEFAULT_RESERVED_COMPLETION_TOKENS)
        return DEFAULT_RESERVED_COMPLETION_TOKENS

    def _estimate_request_tokens(self, system_prompt: str, user_prompt: str) -> int:
        """Approximate input (system + user) plus reserved output tokens for rate accounting."""
        return estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + self._reserved_output_tokens()

    @staticmethod
    def _usage_total_tokens(response: object) -> int | None:
        """The provider's own total-token count if it is present and a real integer, else None (so the ledger
        keeps the estimate). Defensive against missing or malformed usage metadata."""
        usage = getattr(response, "usage", None)
        total = getattr(usage, "total_tokens", None)
        return total if isinstance(total, int) and not isinstance(total, bool) and total >= 0 else None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> str:
        # Pre-send size guard: raises PromptTooLargeError (request_sent False) before any SDK call when a
        # per-request budget is configured and the estimated prompt exceeds it. No-op when none is configured.
        if self.token_budget is not None:
            self.token_budget.enforce(system_prompt, user_prompt)

        # Pre-send per-minute (TPM) guard: raises RateBudgetExceededError (request_sent False) when this
        # request would not fit the remaining per-minute budget. No request is sent in that case.
        reservation = None
        if self.rate_budget is not None:
            reservation = self.rate_budget.reserve(
                self._estimate_request_tokens(system_prompt, user_prompt)
            )

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = self.max_completion_tokens

        if response_format is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "evidence_extraction",
                    "schema": self._make_strict_schema(response_format),
                    "strict": True,
                },
            }

        try:
            response = self.client.chat.completions.create(**kwargs)
        except BaseException:
            # The request has left this process; the provider may have processed it (and SDK retries may have
            # sent it more than once), so KEEP the tokens counted rather than releasing them. The provider
            # error itself is re-raised UNCHANGED for the caller's error contract.
            if reservation is not None:
                reservation.commit()
            raise

        # Success: reconcile the reservation to the provider's own usage figure when it is available.
        if reservation is not None:
            reservation.commit(self._usage_total_tokens(response))

        return response.choices[0].message.content or ""