from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """
    Provider-neutral interface for language-model calls.
    """

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a text response from the configured model.

        response_format may request a structured response from
        providers that support it.
        """
        raise NotImplementedError