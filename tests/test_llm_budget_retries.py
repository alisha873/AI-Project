"""
Offline tests for the configurable token budget and SDK retry count on the Groq client.

The real Groq SDK is never contacted: `llm.groq.Groq` is monkeypatched. These tests pin that
- the default construction path is unchanged (Groq is built with only api_key, so existing callers and
  their monkeypatched lambdas keep working),
- a configured retry count is passed through,
- a configured token budget refuses an oversized prompt BEFORE any SDK call is made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm.settings import resolve_max_retries, resolve_request_timeout, resolve_token_budget
from llm.token_budget import PromptTooLargeError


def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GROQ_API_KEY", "LLM_API_KEY", "LLM_MODEL", "LLM_PROVIDER",
                 "LLM_MAX_COMPLETION_TOKENS", "LLM_MAX_RETRIES", "LLM_TOKEN_BUDGET",
                 "LLM_TPM_LIMIT", "LLM_REQUEST_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


# -------------------------------------------------------------- settings resolvers

@pytest.mark.parametrize("configured, expected", [("", None), ("   ", None), ("0", 0), ("2", 2)])
def test_resolve_max_retries(configured, expected):
    assert resolve_max_retries(None, configured) == expected


@pytest.mark.parametrize("configured, expected", [("", None), ("8000", 8000)])
def test_resolve_token_budget(configured, expected):
    assert resolve_token_budget(None, configured) == expected


@pytest.mark.parametrize("bad", ["-1", "1.5", "abc", "99999999999"])
def test_bad_values_are_refused_not_clamped(bad):
    with pytest.raises(ValueError):
        resolve_max_retries(None, bad) if bad != "99999999999" else resolve_max_retries(None, bad)
    with pytest.raises(ValueError):
        resolve_token_budget(None, bad)


def test_explicit_wins_over_configured():
    assert resolve_max_retries(1, "9") == 1
    assert resolve_token_budget(7000, "8000") == 7000


@pytest.mark.parametrize("configured, expected", [("", None), ("   ", None), ("30", 30.0), ("0.5", 0.5)])
def test_resolve_request_timeout(configured, expected):
    assert resolve_request_timeout(None, configured) == expected


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "100000"])
def test_a_bad_request_timeout_is_refused_not_clamped(bad):
    with pytest.raises(ValueError):
        resolve_request_timeout(None, bad)


# -------------------------------------------------------------- Groq construction

def test_the_default_path_builds_groq_with_only_api_key(monkeypatch, tmp_path):
    from llm.groq import GroqLLMClient

    _isolate(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: seen.setdefault("api_key", api_key) or MagicMock())
    client = GroqLLMClient(api_key="k")
    assert seen["api_key"] == "k"
    assert client.max_retries is None and client.token_budget is None
    assert client.request_timeout is None  # no timeout is passed, so the SDK default applies (lambda takes api_key only)


def test_a_configured_timeout_is_passed_to_the_sdk(monkeypatch, tmp_path):
    from llm.groq import GroqLLMClient

    _isolate(monkeypatch, tmp_path)
    seen = {}

    def fake_groq(api_key, timeout):
        seen["api_key"], seen["timeout"] = api_key, timeout
        return MagicMock()

    monkeypatch.setattr("llm.groq.Groq", fake_groq)
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "30")
    client = GroqLLMClient(api_key="k")
    assert client.request_timeout == 30.0 and seen["timeout"] == 30.0


def test_a_configured_retry_count_is_passed_to_the_sdk(monkeypatch, tmp_path):
    from llm.groq import GroqLLMClient

    _isolate(monkeypatch, tmp_path)
    seen = {}

    def fake_groq(api_key, max_retries):
        seen["api_key"], seen["max_retries"] = api_key, max_retries
        return MagicMock()

    monkeypatch.setattr("llm.groq.Groq", fake_groq)
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    client = GroqLLMClient(api_key="k")
    assert client.max_retries == 0 and seen["max_retries"] == 0


def test_a_configured_budget_reserves_the_completion_limit(monkeypatch, tmp_path):
    from llm.groq import GroqLLMClient

    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    monkeypatch.setenv("LLM_TOKEN_BUDGET", "8000")
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2000")
    client = GroqLLMClient(api_key="k")
    assert client.token_budget is not None
    assert client.token_budget.max_total_tokens == 8000
    assert client.token_budget.reserved_completion_tokens == 2000


# -------------------------------------------------------------- pre-send guard

def test_an_oversized_prompt_is_refused_before_any_sdk_call(monkeypatch, tmp_path):
    from llm.groq import GroqLLMClient

    _isolate(monkeypatch, tmp_path)
    sdk = MagicMock()
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    monkeypatch.setenv("LLM_TOKEN_BUDGET", "500")
    client = GroqLLMClient(api_key="k")
    with pytest.raises(PromptTooLargeError) as info:
        client.generate("x" * 5000, "y" * 5000, {"type": "object", "properties": {}})
    assert info.value.request_sent is False
    sdk.chat.completions.create.assert_not_called()         # nothing reached the provider


def test_a_prompt_within_budget_is_sent(monkeypatch, tmp_path):
    from llm.groq import GroqLLMClient

    _isolate(monkeypatch, tmp_path)
    sdk = MagicMock()
    sdk.chat.completions.create.return_value.choices[0].message.content = '{"ok": true}'
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    monkeypatch.setenv("LLM_TOKEN_BUDGET", "100000")
    client = GroqLLMClient(api_key="k")
    assert client.generate("small system", "small user") == '{"ok": true}'
    sdk.chat.completions.create.assert_called_once()
