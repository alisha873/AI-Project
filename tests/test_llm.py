from llm.base import LLMClient
from llm.settings import LLMSettings


class DummyLLMClient(LLMClient):
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict | None = None,
    ) -> str:
        return "dummy response"


def test_llm_client_contract():
    client = DummyLLMClient()

    result = client.generate(
        system_prompt="system",
        user_prompt="user",
    )

    assert result == "dummy response"


def test_llm_settings_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    settings = LLMSettings(_env_file=None)

    assert settings.provider == ""
    assert settings.api_key == ""


def test_llm_settings_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "test-provider")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    settings = LLMSettings()

    assert settings.provider == "test-provider"
    assert settings.api_key == "test-key"