from __future__ import annotations

from groq import Groq

from llm.base import LLMClient
from llm.settings import LLMSettings


class GroqLLMClient(LLMClient):
    """
    Groq-backed implementation of the provider-neutral LLMClient.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = LLMSettings()

        self.model = model or settings.model
        self.client = Groq(
            api_key=api_key or settings.api_key,
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