from __future__ import annotations

import requests

from llm.base import LLMClient
from llm.token_budget import TokenBudget


class OllamaLLMClient(LLMClient):
    """
    Ollama-backed implementation of the provider-neutral LLMClient.

    `token_budget` is optional and off by default (Ollama is local, so there is no rate limit or cost), but
    the pre-send guard is provider-neutral: when a budget is supplied an oversized prompt is refused before
    any request is sent.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        token_budget: TokenBudget | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token_budget = token_budget

    def generate(
    self,
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
    ) -> str:
        if self.token_budget is not None:
            self.token_budget.enforce(system_prompt, user_prompt)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
        }

        if response_format is not None:
            payload["format"] = "json"

        print("OLLAMA MODEL:", self.model)
        print("OLLAMA URL:", f"{self.base_url}/api/chat")
        print("OLLAMA FORMAT:", payload.get("format"))

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        response_payload = response.json()

        return response_payload["message"]["content"]