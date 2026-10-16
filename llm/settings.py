from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Upper bound for a configured completion-token limit. This is a safety bound chosen here, NOT a
# documented limit of any Groq model: a provider that supports less will reject the request itself.
MAX_COMPLETION_TOKENS_LIMIT = 32768


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


class LLMSettings(BaseSettings):
    provider: str = ""
    api_key: str = ""
    model: str = "openai/gpt-oss-20b"
    # Optional completion-token limit (env LLM_MAX_COMPLETION_TOKENS). Kept as text so that a blank line
    # in .env means "not set"; it is validated by resolve_max_completion_tokens().
    max_completion_tokens: str = ""
    groq_api_key: str = Field(
        default="",
        validation_alias="GROQ_API_KEY",
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )
