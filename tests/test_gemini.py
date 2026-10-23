"""Offline tests for the Gemini client and for provider selection on each LLM-backed stage.

The Google Gen AI SDK client is monkeypatched. No network call and no real key.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from llm.gemini import (
    DEFAULT_GEMINI_MODEL,
    GeminiConfigurationError,
    GeminiProviderError,
    GeminiResponseError,
    GeminiSchemaError,
    GeminiLLMClient,
    prepare_provider_schema,
)


def _isolate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in (
        "GEMINI_API_KEY", "GEMINI_MODEL", "GROQ_API_KEY", "LLM_API_KEY", "LLM_MODEL",
        "LLM_PROVIDER", "LLM_TPM_LIMIT", "LLM_MAX_RETRIES", "LLM_TOKEN_BUDGET",
        "LLM_MAX_COMPLETION_TOKENS", "LLM_REQUEST_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def _install_sdk(monkeypatch, interaction=None, create_side_effect=None):
    seen: dict = {}
    sdk = MagicMock()
    if interaction is None:
        interaction = MagicMock()
        interaction.output_text = '{"status": "ok"}'
        interaction.status = "completed"
        interaction.usage.total_tokens = 11
    if create_side_effect is not None:
        sdk.interactions.create.side_effect = create_side_effect
    else:
        sdk.interactions.create.return_value = interaction

    def fake_client(*, api_key, http_options=None):
        seen["api_key"] = api_key
        seen["http_options"] = http_options
        seen["sdk"] = sdk
        return sdk

    monkeypatch.setattr("llm.gemini.genai.Client", fake_client)
    return seen


def test_missing_key_is_rejected_and_groq_key_is_not_a_fallback(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-should-not-be-used")
    monkeypatch.setenv("LLM_API_KEY", "llm-secret-should-not-be-used")
    with pytest.raises(GeminiConfigurationError, match="GEMINI_API_KEY"):
        GeminiLLMClient()


def test_default_model_is_the_gemini_flash_lite_id_not_the_groq_model(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _install_sdk(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    client = GeminiLLMClient()
    assert client.model == DEFAULT_GEMINI_MODEL == "gemini-3.5-flash-lite"
    assert client.tpm_limit is None
    assert client.rate_budget is None


def test_gemini_model_and_explicit_argument_override(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _install_sdk(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    assert GeminiLLMClient().model == "gemini-3.5-flash-lite"
    assert GeminiLLMClient(model="gemini-3.5-flash-lite").model == "gemini-3.5-flash-lite"


def test_generate_maps_prompt_schema_timeout_and_usage(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    seen = _install_sdk(monkeypatch)
    budget = MagicMock()
    reservation = MagicMock()
    budget.reserve.return_value = reservation
    client = GeminiLLMClient(
        api_key="test-key",
        max_completion_tokens=1800,
        request_timeout=90,
        temperature=0.1,
        rate_budget=budget,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
    }
    text = client.generate("system text", "user text", response_format=schema)
    assert text == '{"status": "ok"}'
    kwargs = seen["sdk"].interactions.create.call_args.kwargs
    assert kwargs["model"] == "gemini-3.5-flash-lite"
    assert kwargs["system_instruction"] == "system text"
    assert kwargs["input"] == "user text"
    assert kwargs["store"] is False
    assert kwargs["timeout"] == 90
    assert kwargs["generation_config"] == {"max_output_tokens": 1800, "temperature": 0.1}
    assert kwargs["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }
    reservation.commit.assert_called_once_with(11)
    assert seen["api_key"] == "test-key"


def test_answer_and_plan_schemas_are_sent_unchanged(monkeypatch, tmp_path):
    from rag.prompt import answer_json_schema
    from remediation.prompt import plan_json_schema

    for schema in (answer_json_schema(), plan_json_schema()):
        prepared, notes = prepare_provider_schema(schema)
        assert prepared == schema
        assert notes == {"omitted": [], "unenforced_by_provider": []}


def test_default_keyword_is_omitted_from_the_provider_schema_only():
    from evidence.schema import EvidenceExtraction

    original = EvidenceExtraction.model_json_schema()
    prepared, notes = prepare_provider_schema(original)
    assert "default" in json.dumps(original)
    assert "default" not in json.dumps(prepared)
    assert notes["omitted"] == ["default"]
    assert original["required"] == ["incident_id"]
    assert prepared["properties"]["confidence"]["minimum"] == 0.0
    assert prepared["properties"]["confidence"]["maximum"] == 1.0


def test_unsupported_schema_combinators_are_refused_before_sending(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    seen = _install_sdk(monkeypatch)
    client = GeminiLLMClient(api_key="test-key")
    with pytest.raises(GeminiSchemaError, match="oneOf"):
        client.generate("s", "u", response_format={"oneOf": [{"type": "string"}]})
    seen["sdk"].interactions.create.assert_not_called()


def test_empty_output_is_an_explicit_error(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    interaction = MagicMock()
    interaction.output_text = "  "
    interaction.status = "completed"
    interaction.usage.total_tokens = 4
    _install_sdk(monkeypatch, interaction=interaction)
    budget = MagicMock()
    reservation = MagicMock()
    budget.reserve.return_value = reservation
    client = GeminiLLMClient(api_key="test-key", rate_budget=budget)
    with pytest.raises(GeminiResponseError, match="no text output"):
        client.generate("s", "u")
    reservation.commit.assert_called_once_with(4)


def test_missing_usage_keeps_the_estimate(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    interaction = MagicMock()
    interaction.output_text = "ok"
    interaction.usage = None
    _install_sdk(monkeypatch, interaction=interaction)
    budget = MagicMock()
    reservation = MagicMock()
    budget.reserve.return_value = reservation
    client = GeminiLLMClient(api_key="test-key", rate_budget=budget)
    assert client.generate("s", "u") == "ok"
    reservation.commit.assert_called_once_with(None)


def test_provider_errors_are_classified_and_redacted(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    class Rate(Exception):
        code = 429

        def __str__(self) -> str:
            return "slow down key=super-secret-value-12345"

    class APITimeoutError(Exception):
        pass

    class ClientError(Exception):
        code = 400

        def __str__(self) -> str:
            return "bad schema"

    seen = _install_sdk(monkeypatch, create_side_effect=Rate("x"))
    client = GeminiLLMClient(api_key="super-secret-value-12345")
    budget = MagicMock()
    reservation = MagicMock()
    budget.reserve.return_value = reservation
    client.rate_budget = budget
    with pytest.raises(GeminiProviderError) as rate_error:
        client.generate("s", "u")
    assert rate_error.value.kind == "rate_limit"
    assert rate_error.value.status_code == 429
    assert "super-secret-value-12345" not in str(rate_error.value)
    reservation.commit.assert_called_once_with()
    assert seen["sdk"].interactions.create.call_count == 1

    seen["sdk"].interactions.create.side_effect = APITimeoutError("deadline")
    with pytest.raises(GeminiProviderError) as timeout_error:
        client.generate("s", "u")
    assert timeout_error.value.kind == "timeout"

    seen["sdk"].interactions.create.side_effect = ClientError("bad")
    with pytest.raises(GeminiProviderError) as client_error:
        client.generate("s", "u")
    assert client_error.value.kind == "client"
    assert seen["sdk"].interactions.create.call_count == 3


def test_retry_count_maps_to_sdk_attempts_and_blank_does_not_retry(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    seen = _install_sdk(monkeypatch)
    GeminiLLMClient(api_key="test-key")
    assert seen["http_options"] is None

    GeminiLLMClient(api_key="test-key", max_retries=0)
    assert seen["http_options"].retry_options.attempts == 1

    GeminiLLMClient(api_key="test-key", max_retries=2)
    assert seen["http_options"].retry_options.attempts == 3


def test_configured_tpm_limit_is_used_and_zero_disables_it(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _install_sdk(monkeypatch)
    monkeypatch.setenv("LLM_TPM_LIMIT", "0")
    assert GeminiLLMClient(api_key="test-key").tpm_limit is None
    monkeypatch.setenv("LLM_TPM_LIMIT", "100000")
    client = GeminiLLMClient(api_key="test-key")
    assert client.tpm_limit == 100000
    assert client.rate_budget is not None


def test_hosted_stages_select_gemini(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _install_sdk(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    from scripts.run_evidence_extraction import make_evidence_llm_client
    from scripts.run_grounded_answer import make_llm_client
    from scripts.run_remediation_plan import make_llm_client as remediation_client

    assert remediation_client is make_llm_client
    grounded = make_llm_client("gemini", None)
    evidence = make_evidence_llm_client()
    assert type(grounded).__name__ == "GeminiLLMClient"
    assert type(evidence).__name__ == "GeminiLLMClient"
    assert grounded.model == evidence.model == "gemini-3.5-flash-lite"
    with pytest.raises(ValueError, match="GROQ_API_KEY is not used"):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        make_evidence_llm_client()


def _mini_incident(tmp_path):
    analysis = tmp_path / "incident_001" / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "incident_reconstruction.json").write_text(
        json.dumps({"incident_id": "incident_001", "timeline": [], "claims": []}),
        encoding="utf-8",
    )
    (analysis / "root_cause_analysis.json").write_text(
        json.dumps({"incident_id": "incident_001", "findings": []}),
        encoding="utf-8",
    )
    return tmp_path / "incident_001"


def test_narrative_refuses_a_groq_provider_without_calling_a_model(monkeypatch, tmp_path, capsys):
    incident = _mini_incident(tmp_path)
    monkeypatch.setattr(
        "scripts.run_llm_incident_synthesis.read_dotenv",
        lambda path: {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "super-secret-value-12345"},
    )
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    from scripts.run_llm_incident_synthesis import main

    monkeypatch.setattr(
        "sys.argv",
        ["run_llm_incident_synthesis", "incident_001", "--incident-dir", str(incident)],
    )
    code = main()
    assert code == 1
    err = capsys.readouterr().err
    assert "does not call Groq" in err
    assert "super-secret-value-12345" not in err


def test_narrative_requires_the_gemini_key(monkeypatch, tmp_path, capsys):
    incident = _mini_incident(tmp_path)
    monkeypatch.setattr(
        "scripts.run_llm_incident_synthesis.read_dotenv",
        lambda path: {"LLM_PROVIDER": "gemini"},
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from scripts.run_llm_incident_synthesis import main

    monkeypatch.setattr(
        "sys.argv",
        ["run_llm_incident_synthesis", "incident_001", "--incident-dir", str(incident)],
    )
    assert main() == 1
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_narrative_call_maps_schema_and_the_script_records_gemini(monkeypatch, tmp_path):
    from scripts import run_llm_incident_synthesis as synthesis

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def generate(self, system_prompt, user_prompt, response_format=None):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            captured["schema"] = response_format
            return json.dumps({"schema_version": "1.0", "incident_id": "incident_001"})

    monkeypatch.setattr("llm.gemini.GeminiLLMClient", FakeClient)
    narrative = synthesis.call_gemini({"incident_id": "incident_001"}, "test-key", DEFAULT_GEMINI_MODEL, 90)
    assert narrative["incident_id"] == "incident_001"
    assert captured["init"]["model"] == "gemini-3.5-flash-lite"
    assert captured["init"]["temperature"] == 0.1
    assert captured["init"]["max_completion_tokens"] == 1800
    assert captured["init"]["request_timeout"] == 90.0
    assert captured["schema"] == synthesis.NARRATIVE_RESPONSE_SCHEMA
    assert "untrusted evidence" in captured["system"]
    assert "test-key" not in captured["system"]
    assert "test-key" not in captured["user"]

    incident = _mini_incident(tmp_path)
    monkeypatch.setattr(
        "scripts.run_llm_incident_synthesis.read_dotenv",
        lambda path: {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test-key"},
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(
        "scripts.run_llm_incident_synthesis.call_gemini",
        lambda payload, api_key, model, timeout: {"schema_version": "1.0", "incident_id": "incident_001"},
    )
    monkeypatch.setattr("scripts.run_llm_incident_synthesis.validate_narrative", lambda narrative, payload: [])
    monkeypatch.setattr(
        "sys.argv",
        ["run_llm_incident_synthesis", "incident_001", "--incident-dir", str(incident)],
    )
    assert synthesis.main() == 0
    written = json.loads((incident / "analysis" / "llm_incident_narrative.json").read_text(encoding="utf-8"))
    assert written["provider"] == "gemini"
    assert written["model"] == "gemini-3.5-flash-lite"
    assert "test-key" not in json.dumps(written)


def test_narrative_rejects_malformed_json(monkeypatch):
    from scripts.run_llm_incident_synthesis import call_gemini

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, system_prompt, user_prompt, response_format=None):
            return "not json"

    monkeypatch.setattr("llm.gemini.GeminiLLMClient", FakeClient)
    with pytest.raises(RuntimeError, match="not valid JSON"):
        call_gemini({}, "k", DEFAULT_GEMINI_MODEL, 30)


def test_grounded_answer_cli_reports_a_missing_gemini_key(monkeypatch, tmp_path, capsys):
    _isolate(monkeypatch, tmp_path)
    incident = tmp_path / "incident_001"
    incident.mkdir()
    from scripts.run_grounded_answer import main

    assert main(["incident_001", "--incident-dir", str(incident), "--provider", "gemini"]) == 1
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err
    assert "groq" not in err.lower()
