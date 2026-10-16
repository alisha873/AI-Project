from __future__ import annotations

from groq import Groq

from llm.base import LLMClient
from llm.settings import LLMSettings, resolve_max_completion_tokens


class GroqLLMClient(LLMClient):
    """
    Groq-backed implementation of the provider-neutral LLMClient.

    Completion-token limit: by default none is sent, so the provider's own default applies (unchanged
    behaviour). A limit can be set with the `max_completion_tokens` argument or LLM_MAX_COMPLETION_TOKENS
    and is sent as the SDK's `max_completion_tokens`. It is bounded (see llm.settings) and never clamped
    or adjusted by this client; a provider error is raised to the caller unchanged.

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
        self.client = Groq(
            api_key=resolved_key,
        )

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


    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> str:
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

        response = self.client.chat.completions.create(**kwargs)

        return response.choices[0].message.content or ""