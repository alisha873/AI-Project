from __future__ import annotations

from unittest.mock import MagicMock

from llm.groq import GroqLLMClient


def test_groq_client_generate_with_json_schema(monkeypatch):
    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"status": "ok"}'
    )

    mock_client.chat.completions.create.return_value = (
        mock_response
    )

    monkeypatch.setattr(
        "llm.groq.Groq",
        lambda api_key: mock_client,
    )

    client = GroqLLMClient(
        model="openai/gpt-oss-20b",
        api_key="test-key",
    )

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
        user_prompt="Return the status.",
        response_format=response_schema,
    )

    assert result == '{"status": "ok"}'

    mock_client.chat.completions.create.assert_called_once()

    call_kwargs = (
        mock_client.chat.completions.create.call_args.kwargs
    )

    assert call_kwargs["model"] == "openai/gpt-oss-20b"

    assert call_kwargs["messages"] == [
        {
            "role": "system",
            "content": "You are AutoShield.",
        },
        {
            "role": "user",
            "content": "Return the status.",
        },
    ]

    expected_schema = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
        },
    },
    "required": ["status"],
    "additionalProperties": False,
}

    assert call_kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "evidence_extraction",
            "schema": expected_schema,
            "strict": True,
        },
    }


def test_groq_client_generate_without_response_format(
    monkeypatch,
):
    mock_client = MagicMock()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        "plain response"
    )

    mock_client.chat.completions.create.return_value = (
        mock_response
    )

    monkeypatch.setattr(
        "llm.groq.Groq",
        lambda api_key: mock_client,
    )

    client = GroqLLMClient(
        model="openai/gpt-oss-20b",
        api_key="test-key",
    )

    result = client.generate(
        system_prompt="You are AutoShield.",
        user_prompt="Say hello.",
    )

    assert result == "plain response"

    call_kwargs = (
        mock_client.chat.completions.create.call_args.kwargs
    )

    assert "response_format" not in call_kwargs

def test_make_strict_schema_handles_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "details": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                },
                "required": ["status"],
            },
        },
        "required": ["details"],
    }

    result = GroqLLMClient._make_strict_schema(schema)

    assert result["additionalProperties"] is False
    assert result["required"] == ["details"]

    nested = result["properties"]["details"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["status"]

    # The helper must not modify the original schema.
    assert "additionalProperties" not in schema