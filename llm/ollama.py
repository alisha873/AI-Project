from __future__ import annotations

import requests

from llm.base import LLMClient


class OllamaLLMClient(LLMClient):
    """
    Ollama-backed implementation of the provider-neutral LLMClient.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
    self,
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
    ) -> str:
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