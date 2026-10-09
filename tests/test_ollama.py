from llm.ollama import OllamaLLMClient


class DummyResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            "message": {
                "content": "AutoShield local LLM ready"
            }
        }


def test_ollama_client_generate(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(
        "llm.ollama.requests.post",
        fake_post,
    )

    client = OllamaLLMClient()

    response_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
            },
        },
        "required": ["status"],
    }

    result = client.generate(
        system_prompt="You are AutoShield.",
        user_prompt="Test the provider.",
        response_format=response_schema,
    )

    assert captured["json"]["format"] == "json"
    assert result == "AutoShield local LLM ready"

    assert captured["url"] == (
        "http://localhost:11434/api/chat"
    )

    assert captured["json"]["model"] == "qwen2.5:3b"
    assert captured["json"]["stream"] is False

    assert captured["json"]["messages"] == [
        {
            "role": "system",
            "content": "You are AutoShield.",
        },
        {
            "role": "user",
            "content": "Test the provider.",
        },
    ]