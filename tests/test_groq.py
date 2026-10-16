from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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

def _isolate_settings(monkeypatch, tmp_path):
    """Run in an empty directory with no key variables, so a real local .env is never read."""
    monkeypatch.chdir(tmp_path)
    for name in ("GROQ_API_KEY", "LLM_API_KEY", "LLM_MODEL", "LLM_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


def _capture_key(monkeypatch):
    seen = {}

    def fake_groq(api_key):
        seen["api_key"] = api_key
        return MagicMock()

    monkeypatch.setattr("llm.groq.Groq", fake_groq)
    return seen


def test_groq_client_reads_groq_api_key_from_environment(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    seen = _capture_key(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    GroqLLMClient()
    assert seen["api_key"] == "fake-groq-key"


def test_groq_client_falls_back_to_llm_api_key(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    seen = _capture_key(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "fake-llm-key")
    GroqLLMClient()
    assert seen["api_key"] == "fake-llm-key"


def test_groq_client_explicit_key_wins(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    seen = _capture_key(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    GroqLLMClient(api_key="explicit-key")
    assert seen["api_key"] == "explicit-key"


def test_groq_client_without_any_key_fails_clearly(monkeypatch, tmp_path):
    import pytest

    _isolate_settings(monkeypatch, tmp_path)
    _capture_key(monkeypatch)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqLLMClient()


# ------------------------------------------------------------
# Completion-token limit (Phase 12 blocker fix). All mocked: no network, no real key.
# ------------------------------------------------------------

def _recording_client(monkeypatch, tmp_path, content='{"ok": true}', **client_kwargs):
    """A GroqLLMClient whose SDK object records the request kwargs (no request is ever made)."""
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.delenv("LLM_MAX_COMPLETION_TOKENS", raising=False)
    sdk = MagicMock()
    sdk.chat.completions.create.return_value.choices[0].message.content = content
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    client = GroqLLMClient(api_key="test-key", **client_kwargs)
    return client, sdk.chat.completions.create


def test_no_completion_limit_is_sent_by_default(monkeypatch, tmp_path):
    client, create = _recording_client(monkeypatch, tmp_path)
    assert client.max_completion_tokens is None
    client.generate("system", "user", {"type": "object", "properties": {}})
    kwargs = create.call_args.kwargs
    assert "max_completion_tokens" not in kwargs and "max_tokens" not in kwargs
    assert set(kwargs) == {"model", "messages", "response_format"}   # nothing else changed for existing callers


def test_an_explicit_completion_limit_is_sent_as_max_completion_tokens(monkeypatch, tmp_path):
    client, create = _recording_client(monkeypatch, tmp_path, max_completion_tokens=4096)
    client.generate("system", "user")
    kwargs = create.call_args.kwargs
    assert kwargs["max_completion_tokens"] == 4096 and "max_tokens" not in kwargs   # max_tokens is deprecated


def test_the_limit_can_be_configured_through_the_environment(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", " 6000 ")
    sdk = MagicMock()
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    client = GroqLLMClient(api_key="test-key")
    assert client.max_completion_tokens == 6000
    client.generate("system", "user")
    assert sdk.chat.completions.create.call_args.kwargs["max_completion_tokens"] == 6000


def test_an_explicit_limit_wins_over_the_environment(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "6000")
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    assert GroqLLMClient(api_key="test-key", max_completion_tokens=1234).max_completion_tokens == 1234


def test_a_blank_environment_value_means_not_set(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "   ")
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    assert GroqLLMClient(api_key="test-key").max_completion_tokens is None


def test_the_limit_is_read_from_a_local_env_file(monkeypatch, tmp_path):
    _isolate_settings(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("LLM_MAX_COMPLETION_TOKENS=2048\n", encoding="utf-8")
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    assert GroqLLMClient(api_key="test-key").max_completion_tokens == 2048


def test_the_limit_is_bounded_at_both_ends(monkeypatch, tmp_path):
    from llm.settings import MAX_COMPLETION_TOKENS_LIMIT

    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    assert GroqLLMClient(api_key="k", max_completion_tokens=1).max_completion_tokens == 1
    assert GroqLLMClient(api_key="k", max_completion_tokens=MAX_COMPLETION_TOKENS_LIMIT).max_completion_tokens == MAX_COMPLETION_TOKENS_LIMIT


@pytest.mark.parametrize("bad", [0, -1, 32769, 10**9, True, 2.5, "100"])
def test_an_invalid_explicit_limit_is_refused_not_clamped(monkeypatch, tmp_path, bad):
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    with pytest.raises(ValueError, match="max_completion_tokens"):
        GroqLLMClient(api_key="k", max_completion_tokens=bad)


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "1.5", "99999", "1e3", "12 34", "\u00b2"])
def test_an_invalid_environment_limit_is_refused_not_clamped(monkeypatch, tmp_path, bad):
    _isolate_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", bad)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    with pytest.raises(ValueError, match="LLM_MAX_COMPLETION_TOKENS|max_completion_tokens"):
        GroqLLMClient(api_key="k")


def test_the_limit_changes_neither_the_model_nor_the_schema(monkeypatch, tmp_path):
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    plain, plain_create = _recording_client(monkeypatch, tmp_path, model="openai/gpt-oss-20b")
    plain.generate("s", "u", schema)
    limited, limited_create = _recording_client(monkeypatch, tmp_path, model="openai/gpt-oss-20b", max_completion_tokens=3000)
    limited.generate("s", "u", schema)
    expected = dict(plain_create.call_args.kwargs, max_completion_tokens=3000)
    assert limited_create.call_args.kwargs == expected          # the only difference is the limit
    assert limited_create.call_args.kwargs["response_format"]["json_schema"]["strict"] is True


def test_provider_errors_are_raised_unchanged_and_never_retried(monkeypatch, tmp_path):
    groq = pytest.importorskip("groq")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    # the two 400 bodies seen from the provider (shape only; no incident data)
    bodies = [
        {"error": {"message": "max completion tokens reached before generating a valid document", "type": "invalid_request_error"}},
        {"error": {"message": "Failed to validate JSON.", "type": "invalid_request_error",
                   "code": "json_validate_failed", "failed_generation": ""}},
    ]
    for body in bodies:
        client, create = _recording_client(monkeypatch, tmp_path, max_completion_tokens=2000)
        error = groq.BadRequestError("Error code: 400", response=httpx.Response(400, request=request), body=body)
        create.side_effect = error
        with pytest.raises(groq.BadRequestError) as raised:
            client.generate("system", "user", {"type": "object", "properties": {}})
        assert raised.value is error                    # not wrapped, not translated
        assert create.call_count == 1                   # no retry


def test_response_parsing_returns_the_message_text_or_an_empty_string(monkeypatch, tmp_path):
    client, _ = _recording_client(monkeypatch, tmp_path, content='{"a": 1}')
    assert client.generate("s", "u") == '{"a": 1}'
    client, _ = _recording_client(monkeypatch, tmp_path, content=None)
    assert client.generate("s", "u") == ""              # a missing message body is an empty string, not a crash


def test_the_response_text_is_not_trimmed_or_altered(monkeypatch, tmp_path):
    partial = '{"proposals": [{"title": "cut off mid-'
    client, _ = _recording_client(monkeypatch, tmp_path, content=partial, max_completion_tokens=50)
    assert client.generate("s", "u") == partial        # a truncated reply stays visible so validation rejects it


def test_settings_default_has_no_limit(monkeypatch):
    from llm.settings import LLMSettings, resolve_max_completion_tokens

    monkeypatch.delenv("LLM_MAX_COMPLETION_TOKENS", raising=False)
    settings = LLMSettings(_env_file=None)
    assert settings.max_completion_tokens == ""
    assert resolve_max_completion_tokens(None, settings.max_completion_tokens) is None


def test_the_sdk_client_keeps_its_default_retries_which_the_docs_describe(monkeypatch, tmp_path):
    """
    Pins the behaviour documented in `GroqLLMClient`: no retry loop of our own, but the SDK client is built
    with its default `max_retries` (2), so one generate() call can reach the provider up to three times.
    The real SDK class is only constructed (fake key, no request is sent). If this fails, the retry
    documentation in llm/groq.py, rag/answer.py, remediation/planner.py and PROJECT_STATUS.md is out of date.
    """
    import groq

    _isolate_settings(monkeypatch, tmp_path)
    client = GroqLLMClient(api_key="fake-key-not-real")
    assert groq.DEFAULT_MAX_RETRIES == 2
    assert client.client.max_retries == groq.DEFAULT_MAX_RETRIES
