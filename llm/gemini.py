"""Gemini-backed implementation of the provider-neutral LLMClient.

Uses the Google Gen AI Python SDK Interactions API (`client.interactions.create`), which is the
documented interface for `gemini-3.5-flash-lite`. This is not the OpenAI/Groq chat-completions
shape: system text is `system_instruction`, the user text is `input`, structured output is
`response_format`, and token counts come back on `usage.total_tokens`.

The model id defaults to `gemini-3.5-flash-lite`. Override it with the `model` argument or
`GEMINI_MODEL`. `LLM_MODEL` is the Groq client's setting and is not read here.

The API key is `GEMINI_API_KEY` only. A missing key raises `GeminiConfigurationError`. This
client does not read `GROQ_API_KEY` or `LLM_API_KEY` and does not fall back to Groq.

Incident prompts are sent with `store=false` so the Interactions API does not keep a server-side
copy (the API's default is to store the interaction). Nothing in this module is executed as a
remediation or a live validation probe.
"""

from __future__ import annotations

import re

from google import genai
from google.genai import types

from llm.base import LLMClient
from llm.rate_budget import RateBudget, shared_rate_budget
from llm.settings import (
    LLMSettings,
    resolve_max_completion_tokens,
    resolve_max_retries,
    resolve_request_timeout,
    resolve_token_budget,
    resolve_tpm_limit,
)
from llm.token_budget import RequestNotSentError, TokenBudget, estimate_tokens

# Exact model id from the Gemini model documentation. Not a Groq model name.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

# Expected output allowance for per-minute accounting when a caller has configured a TPM limit.
# This is not a Gemini quota. No quota is assumed unless LLM_TPM_LIMIT (or an injected budget) sets one.
_RESERVED_COMPLETION_TOKENS = 512

# Documented structured-output keywords the API ignores rather than enforces. They stay in the
# schema sent to Gemini; application validation remains responsible for them.
_UNENFORCED_SCHEMA_KEYWORDS = frozenset({
    "minLength", "maxLength", "minProperties", "maxProperties",
    "uniqueItems", "contains", "minContains", "maxContains",
    "min_length", "max_length", "min_properties", "max_properties",
    "unique_items", "min_contains", "max_contains",
})

# Not part of the documented JSON Schema subset. `default` is provider metadata, not a constraint
# the application validators rely on the provider to enforce (Pydantic still applies defaults).
_OMITTED_SCHEMA_KEYWORDS = frozenset({"default"})

# Schema combinators the structured-output guide does not list. These are refused before a request
# is sent rather than deleted, so a constraint is never dropped silently.
_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset({
    "oneOf", "allOf", "not", "if", "then", "else",
    "patternProperties", "dependentRequired", "dependentSchemas",
})


class GeminiConfigurationError(ValueError):
    """The Gemini client is selected but its key or schema cannot be used. Nothing was sent."""


class GeminiSchemaError(RequestNotSentError):
    """The response schema uses a construct Gemini's structured output does not document."""


class GeminiResponseError(RuntimeError):
    """The provider responded, but the body was not usable text."""


class GeminiProviderError(RuntimeError):
    """A provider or transport failure. `kind` is timeout, rate_limit, transient, client, connection, or provider."""

    def __init__(self, kind: str, message: str, *, status_code: int | None = None) -> None:
        self.kind = kind
        self.status_code = status_code
        super().__init__(message)


def prepare_provider_schema(schema: dict) -> tuple[dict, dict[str, list[str]]]:
    """
    Copy a JSON schema for Gemini structured output.

    Returns the schema to send and a note of keywords that were omitted (`default`) or that the
    provider documents as unenforced (length and uniqueness constraints, which are kept in the
    schema because the API strips them itself). The input dict is not modified. Unsupported
    combinators raise `GeminiSchemaError` before any request.
    """
    if not isinstance(schema, dict):
        raise GeminiSchemaError("response_format must be a JSON schema object.")
    omitted: set[str] = set()
    unenforced: set[str] = set()
    unsupported: set[str] = set()

    def walk(node: object) -> object:
        if isinstance(node, dict):
            copied: dict = {}
            for key, value in node.items():
                if key in _OMITTED_SCHEMA_KEYWORDS:
                    omitted.add(key)
                    continue
                if key in _UNSUPPORTED_SCHEMA_KEYWORDS:
                    unsupported.add(key)
                elif key in _UNENFORCED_SCHEMA_KEYWORDS:
                    unenforced.add(key)
                copied[key] = walk(value)
            return copied
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    prepared = walk(schema)
    if unsupported:
        raise GeminiSchemaError(
            "Gemini structured output does not document "
            + ", ".join(sorted(unsupported))
            + ". The schema was not sent, and those constraints were not dropped."
        )
    notes = {
        "omitted": sorted(omitted),
        "unenforced_by_provider": sorted(unenforced),
    }
    return prepared, notes


def _redact(text: str, secret: str) -> str:
    """Remove credential material from a provider message. Truncate so logs stay small."""
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(x-goog-api-key[\"'\s:=]+)[^\s\"']+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[REDACTED]", text)
    return text[:800]


class GeminiLLMClient(LLMClient):
    """
    Gemini Interactions client.

    Per-minute accounting is off unless `LLM_TPM_LIMIT` (or `tpm_limit` / `rate_budget`) sets a
    positive limit. Groq's recorded 8,000 tokens/minute ceiling is not applied.

    Retries follow the installed SDK (`google-genai` 2.33): omitting retry options sends the
    request once. `LLM_MAX_RETRIES` is the number of extra submissions. Those retries are only for
    HTTP 408, 429, 5xx and transient transport errors. HTTP 400 is not retried. This class adds
    no retry loop of its own.
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
        temperature: float | None = None,
    ) -> None:
        settings = LLMSettings()
        resolved_key = api_key or settings.gemini_api_key
        if not resolved_key:
            raise GeminiConfigurationError(
                "No Gemini API key found. Set GEMINI_API_KEY in the local .env file or the environment."
            )

        configured_model = settings.gemini_model.strip()
        self.model = (model or configured_model or DEFAULT_GEMINI_MODEL).strip()
        if not self.model:
            raise GeminiConfigurationError(
                f"Gemini model id is empty. Set GEMINI_MODEL or use {DEFAULT_GEMINI_MODEL}."
            )

        self.max_completion_tokens = resolve_max_completion_tokens(
            max_completion_tokens, settings.max_completion_tokens
        )
        self.max_retries = resolve_max_retries(max_retries, settings.max_retries)
        self.request_timeout = resolve_request_timeout(request_timeout, settings.request_timeout)
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))):
            raise GeminiConfigurationError("temperature must be a number.")
        self.temperature = None if temperature is None else float(temperature)
        self._api_key = resolved_key

        resolved_budget = resolve_token_budget(token_budget, settings.token_budget)
        self.token_budget = (
            TokenBudget(
                max_total_tokens=resolved_budget,
                reserved_completion_tokens=self.max_completion_tokens or 0,
            )
            if resolved_budget is not None
            else None
        )

        # None or 0: do not invent a Gemini quota. A positive LLM_TPM_LIMIT is the caller's verified limit.
        resolved_tpm = resolve_tpm_limit(tpm_limit, settings.tpm_limit)
        self.tpm_limit = resolved_tpm if resolved_tpm else None
        if rate_budget is not None:
            self.rate_budget: RateBudget | None = rate_budget
        elif self.tpm_limit is not None:
            self.rate_budget = shared_rate_budget(
                provider="gemini", limit_tokens=self.tpm_limit, account=resolved_key
            )
        else:
            self.rate_budget = None

        http_options = None
        if self.max_retries is not None:
            # `attempts` includes the first call. Extra retries stay on the SDK's transient status list.
            http_options = types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=self.max_retries + 1)
            )
        self.client = genai.Client(api_key=resolved_key, http_options=http_options)
        self.last_schema_notes: dict[str, list[str]] = {"omitted": [], "unenforced_by_provider": []}

    def _reserved_output_tokens(self) -> int:
        if self.max_completion_tokens is not None:
            return min(self.max_completion_tokens, _RESERVED_COMPLETION_TOKENS)
        return _RESERVED_COMPLETION_TOKENS

    def _estimate_request_tokens(self, system_prompt: str, user_prompt: str) -> int:
        return estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + self._reserved_output_tokens()

    @staticmethod
    def _usage_total_tokens(interaction: object) -> int | None:
        """Provider total when it is a non-negative integer. Missing or malformed usage keeps the estimate."""
        usage = getattr(interaction, "usage", None)
        total = getattr(usage, "total_tokens", None)
        if total is None and isinstance(usage, dict):
            total = usage.get("total_tokens")
        if isinstance(total, bool) or not isinstance(total, int) or total < 0:
            return None
        return total

    def _provider_error(self, exc: Exception) -> GeminiProviderError:
        code = getattr(exc, "code", None)
        if not isinstance(code, int) or isinstance(code, bool):
            code = getattr(exc, "status_code", None)
        status_code = code if isinstance(code, int) and not isinstance(code, bool) else None
        name = type(exc).__name__
        if status_code == 408 or "Timeout" in name:
            kind = "timeout"
        elif status_code == 429 or "RateLimit" in name or "RESOURCE_EXHAUSTED" in str(exc):
            kind = "rate_limit"
        elif status_code is not None and 500 <= status_code <= 599:
            kind = "transient"
        elif status_code is not None and 400 <= status_code <= 499:
            kind = "client"
        elif "Connect" in name or "Network" in name:
            kind = "connection"
        else:
            kind = "provider"
        detail = _redact(str(exc), self._api_key)
        label = f"HTTP {status_code}" if status_code is not None else name
        return GeminiProviderError(kind, f"Gemini {kind} error ({label}): {detail}", status_code=status_code)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> str:
        if self.token_budget is not None:
            self.token_budget.enforce(system_prompt, user_prompt)

        schema_notes = {"omitted": [], "unenforced_by_provider": []}
        prepared_schema = None
        if response_format is not None:
            prepared_schema, schema_notes = prepare_provider_schema(response_format)
        self.last_schema_notes = schema_notes

        reservation = None
        if self.rate_budget is not None:
            reservation = self.rate_budget.reserve(
                self._estimate_request_tokens(system_prompt, user_prompt)
            )

        kwargs: dict[str, object] = {
            "model": self.model,
            "input": user_prompt,
            "system_instruction": system_prompt,
            "store": False,
        }
        generation_config: dict[str, object] = {}
        if self.max_completion_tokens is not None:
            generation_config["max_output_tokens"] = self.max_completion_tokens
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if generation_config:
            kwargs["generation_config"] = generation_config
        if prepared_schema is not None:
            kwargs["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": prepared_schema,
            }
        if self.request_timeout is not None:
            kwargs["timeout"] = self.request_timeout

        try:
            interaction = self.client.interactions.create(**kwargs)
        except Exception as exc:
            if reservation is not None:
                reservation.commit()
            raise self._provider_error(exc) from None

        if reservation is not None:
            reservation.commit(self._usage_total_tokens(interaction))

        text = getattr(interaction, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            status = getattr(interaction, "status", None)
            raise GeminiResponseError(
                "Gemini returned no text output "
                f"(status={status!r}). The response was not accepted."
            )
        return text
