from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Upper bound for a configured completion-token limit. This is a safety bound chosen here, NOT a
# documented limit of any Groq model: a provider that supports less will reject the request itself.
MAX_COMPLETION_TOKENS_LIMIT = 32768

# Upper bound for a configured pre-send token budget (env LLM_TOKEN_BUDGET). A safety bound chosen here,
# not a provider fact. The recorded Groq on-demand ceiling that motivated this was 8,000 tokens/minute.
MAX_TOKEN_BUDGET_LIMIT = 1_000_000

# Upper bound for configured SDK retries (env LLM_MAX_RETRIES). The Groq SDK default is 2; 0 disables the
# SDK's own re-sends so one generate() call reaches the provider at most once.
MAX_RETRIES_LIMIT = 10

# Upper bound for a configured per-minute token budget (env LLM_TPM_LIMIT). A safety bound chosen here, not a
# provider fact. The recorded Groq on-demand ceiling that motivated the default was 8,000 tokens/minute.
MAX_TPM_LIMIT = 10_000_000

# Upper bound for a configured per-request provider timeout in seconds (env LLM_REQUEST_TIMEOUT). A safety
# bound chosen here, not a provider fact; blank keeps the SDK's own default timeout.
MAX_REQUEST_TIMEOUT_SECONDS = 600.0


def resolve_max_completion_tokens(explicit: int | None, configured: str = "") -> int | None:
    """
    The completion-token limit to send, or None to send none (the provider's own default applies).

    An explicit argument wins over the configured text (`LLM_MAX_COMPLETION_TOKENS`); a blank
    configured value means "not set". Anything that is not a whole number from 1 to
    MAX_COMPLETION_TOKENS_LIMIT is refused with a clear error rather than silently clamped.
    """
    if explicit is not None:
        value: object = explicit
    elif configured and configured.strip():
        text = configured.strip()
        if not (text.isascii() and text.isdigit()):
            raise ValueError(
                f"LLM_MAX_COMPLETION_TOKENS must be a whole number from 1 to {MAX_COMPLETION_TOKENS_LIMIT}, "
                f"got {text[:20]!r}."
            )
        value = int(text)
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_COMPLETION_TOKENS_LIMIT:
        raise ValueError(
            f"max_completion_tokens must be a whole number from 1 to {MAX_COMPLETION_TOKENS_LIMIT}, "
            f"got {value!r}."
        )
    return value


def _resolve_bounded_int(explicit: int | None, configured: str, *, name: str, low: int, high: int) -> int | None:
    """
    Shared resolver for the optional bounded integer settings (token budget, SDK retries).

    An explicit argument wins over the configured text; a blank configured value means "not set" and returns
    None. Anything that is not a whole number within [low, high] is refused with a clear error rather than
    silently clamped. `low` may be 0 (retries) or 1 (token budget).
    """
    if explicit is not None:
        value: object = explicit
    elif configured and configured.strip():
        text = configured.strip()
        if not (text.isascii() and text.isdigit()):
            raise ValueError(f"{name} must be a whole number from {low} to {high}, got {text[:20]!r}.")
        value = int(text)
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}, got {value!r}.")
    return value


def resolve_token_budget(explicit: int | None, configured: str = "") -> int | None:
    """The pre-send token budget to enforce (env LLM_TOKEN_BUDGET), or None to enforce none."""
    return _resolve_bounded_int(explicit, configured, name="LLM_TOKEN_BUDGET", low=1, high=MAX_TOKEN_BUDGET_LIMIT)


def resolve_max_retries(explicit: int | None, configured: str = "") -> int | None:
    """The SDK retry count to request (env LLM_MAX_RETRIES), or None to keep the SDK default."""
    return _resolve_bounded_int(explicit, configured, name="LLM_MAX_RETRIES", low=0, high=MAX_RETRIES_LIMIT)


def resolve_request_timeout(explicit: float | None, configured: str = "") -> float | None:
    """
    The per-request provider timeout in seconds to pass to the SDK (env LLM_REQUEST_TIMEOUT), or None to keep
    the SDK's own default.

    An explicit argument wins over the configured text; a blank configured value means "not set". The value
    may be fractional. Anything that is not a positive number up to MAX_REQUEST_TIMEOUT_SECONDS is refused with
    a clear error rather than silently clamped, so an operator can bound an operation's deadline deliberately.
    """
    if explicit is not None:
        value: object = explicit
    elif configured and configured.strip():
        text = configured.strip()
        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"LLM_REQUEST_TIMEOUT must be a positive number of seconds up to "
                f"{MAX_REQUEST_TIMEOUT_SECONDS:g}, got {text[:20]!r}."
            )
    else:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
        0 < float(value) <= MAX_REQUEST_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"request_timeout must be a positive number of seconds up to {MAX_REQUEST_TIMEOUT_SECONDS:g}, "
            f"got {value!r}."
        )
    return float(value)


def resolve_tpm_limit(explicit: int | None, configured: str = "") -> int | None:
    """
    The per-minute token budget to enforce (env LLM_TPM_LIMIT).

    Returns None when NOTHING is configured (the client then applies its documented default-on limit), 0 when
    explicitly disabled (`LLM_TPM_LIMIT=0`), or a positive integer up to MAX_TPM_LIMIT. Anything else is refused
    with a clear error rather than silently clamped.
    """
    return _resolve_bounded_int(explicit, configured, name="LLM_TPM_LIMIT", low=0, high=MAX_TPM_LIMIT)


class LLMSettings(BaseSettings):
    provider: str = ""
    api_key: str = ""
    model: str = "openai/gpt-oss-20b"
    # Optional completion-token limit (env LLM_MAX_COMPLETION_TOKENS). Kept as text so that a blank line
    # in .env means "not set"; it is validated by resolve_max_completion_tokens().
    max_completion_tokens: str = ""
    # Optional pre-send token budget (env LLM_TOKEN_BUDGET) and SDK retry count (env LLM_MAX_RETRIES). Both
    # kept as text so a blank value means "not set"; validated by resolve_token_budget / resolve_max_retries.
    token_budget: str = ""
    max_retries: str = ""
    # Optional per-minute token budget (env LLM_TPM_LIMIT). Blank = use the client's default-on limit; "0"
    # disables it. Validated by resolve_tpm_limit.
    tpm_limit: str = ""
    # Optional per-request provider timeout in seconds (env LLM_REQUEST_TIMEOUT). Blank = keep the SDK's own
    # default timeout. Validated by resolve_request_timeout.
    request_timeout: str = ""
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
    )
    # Gemini Developer API key. This is not a fallback for GROQ_API_KEY or LLM_API_KEY.
    gemini_api_key: str = Field(
        default="",
        validation_alias="GEMINI_API_KEY",
    )
    # Optional Gemini model override. Blank means the Gemini client uses gemini-3.5-flash-lite.
    # LLM_MODEL is the Groq client's model and is intentionally not reused here.
    gemini_model: str = Field(
        default="",
        validation_alias="GEMINI_MODEL",
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )
