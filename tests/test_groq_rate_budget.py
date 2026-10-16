"""
Offline tests for the default-on per-minute (TPM) budget wired into `GroqLLMClient`, and its propagation into
the RAG answer and remediation planner layers. The Groq SDK is always monkeypatched; no network, no real key.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm.groq import DEFAULT_GROQ_TPM, GroqLLMClient
from llm.rate_budget import RateBudget, RateBudgetExceededError
from llm.token_budget import RequestNotSentError


def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("GROQ_API_KEY", "LLM_API_KEY", "LLM_MODEL", "LLM_PROVIDER",
                 "LLM_MAX_COMPLETION_TOKENS", "LLM_TOKEN_BUDGET", "LLM_TPM_LIMIT", "LLM_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)


def _usage(total_tokens):
    response = MagicMock()
    response.choices[0].message.content = '{"ok": true}'
    response.usage.total_tokens = total_tokens
    return response


def test_tpm_budget_is_on_by_default(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    client = GroqLLMClient(api_key="test-key")
    assert client.tpm_limit == DEFAULT_GROQ_TPM
    assert client.rate_budget is not None


def test_tpm_budget_can_be_disabled_with_zero(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_TPM_LIMIT", "0")
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    client = GroqLLMClient(api_key="test-key")
    assert client.tpm_limit is None and client.rate_budget is None


def test_tpm_limit_can_be_configured(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("LLM_TPM_LIMIT", "1234")
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    assert GroqLLMClient(api_key="test-key").tpm_limit == 1234


def test_an_oversized_request_is_refused_before_any_sdk_call(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    sdk = MagicMock()
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    tiny = RateBudget(50)                       # 50-token budget; input + reserved output far exceeds it
    client = GroqLLMClient(api_key="test-key", rate_budget=tiny)
    with pytest.raises(RateBudgetExceededError) as raised:
        client.generate("system prompt", "user prompt")
    assert raised.value.request_sent is False
    assert isinstance(raised.value, RequestNotSentError)
    sdk.chat.completions.create.assert_not_called()      # nothing was sent
    assert tiny.used() == 0                               # the reservation was released


def test_a_successful_call_reconciles_to_provider_usage(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = _usage(total_tokens=777)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    budget = RateBudget(100_000)
    client = GroqLLMClient(api_key="test-key", rate_budget=budget)
    client.generate("system", "user")
    assert budget.used() == 777                           # estimate replaced by the provider's real figure


def test_malformed_usage_keeps_the_estimate(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    sdk = MagicMock()
    sdk.chat.completions.create.return_value = MagicMock()   # usage.total_tokens is a MagicMock, not an int
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    budget = RateBudget(100_000)
    client = GroqLLMClient(api_key="test-key", rate_budget=budget)
    client.generate("system", "user")
    assert budget.used() > 0                               # the estimate was kept, no crash


def test_a_provider_error_keeps_the_tokens_counted_and_reraises(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    sdk = MagicMock()
    boom = RuntimeError("boom after send")
    sdk.chat.completions.create.side_effect = boom
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: sdk)
    budget = RateBudget(100_000)
    client = GroqLLMClient(api_key="test-key", rate_budget=budget)
    with pytest.raises(RuntimeError) as raised:
        client.generate("system", "user")
    assert raised.value is boom                            # unchanged
    assert budget.used() > 0                               # committed, not released (may have been processed)


def test_estimate_counts_input_and_reserved_output(monkeypatch, tmp_path):
    from llm.token_budget import estimate_tokens

    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    client = GroqLLMClient(api_key="test-key")
    system, user = "a" * 300, "b" * 300
    expected = estimate_tokens(system) + estimate_tokens(user) + client._reserved_output_tokens()
    assert client._estimate_request_tokens(system, user) == expected
    assert expected > client._reserved_output_tokens()    # input is counted on top of the output reserve


def test_reserved_output_never_exceeds_a_small_completion_cap(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("llm.groq.Groq", lambda api_key: MagicMock())
    # A large cap is NOT reserved in full (that would make the 8192-vs-8000 config refuse everything)...
    big = GroqLLMClient(api_key="test-key", max_completion_tokens=8192)
    assert big._reserved_output_tokens() < 8192
    # ...but a cap smaller than the default reserve IS respected (the model cannot exceed it).
    small = GroqLLMClient(api_key="test-key", max_completion_tokens=100)
    assert small._reserved_output_tokens() == 100


# -- propagation into the RAG and remediation layers -----------------------------------------------------

def test_rag_answer_reports_a_rate_rejection_as_request_not_sent():
    from rag.answer import generate_grounded_answer
    from tests.rag_helpers import FakeLLM, build_context

    context = build_context()
    exc = RateBudgetExceededError(9000, 10, 8000, 60, "no per-minute budget left; no request was sent")
    result = generate_grounded_answer(context, FakeLLM(exc))
    assert result.status == "llm_error"
    assert result.request_sent is False
    assert "No request was sent" in result.message
    assert result.to_dict()["request_sent"] is False


def test_planner_reports_a_rate_rejection_with_its_own_error_kind():
    from remediation.planner import plan_remediation
    from tests.remediation_helpers import make_input
    from tests.rag_helpers import FakeLLM

    exc = RateBudgetExceededError(9000, 10, 8000, 60, "no per-minute budget left; no request was sent")
    result = plan_remediation(make_input(), FakeLLM(exc))
    assert result.status == "llm_error"
    assert result.request_sent is False
    assert result.error_kind == "rate_budget_exceeded"
    assert result.to_dict()["request_sent"] is False
