"""
Offline tests for the dashboard orchestrator's configuration and command construction (`api.main`).

These test only the pure, synchronous helpers and the static STAGES/command shape. No server is started, no
subprocess is launched, no model is called and no secret is read or printed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import api.main as m

_AUTOSHIELD_VARS = (
    "AUTOSHIELD_LLM_PROVIDER", "AUTOSHIELD_LLM_MAX_RETRIES",
    "AUTOSHIELD_LLM_TOKEN_BUDGET", "AUTOSHIELD_LLM_TPM_LIMIT", "AUTOSHIELD_MAX_PASSAGE_CHARS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _AUTOSHIELD_VARS + ("LLM_MAX_RETRIES", "LLM_TOKEN_BUDGET", "LLM_TPM_LIMIT"):
        monkeypatch.delenv(name, raising=False)


# -------------------------------------------------------------- stage shape / dedup

def test_the_rag_stage_is_context_only_and_the_remediation_stage_owns_the_model_call():
    stages = {s["id"]: s for s in m.STAGES}
    rag = stages["rag"]["command"]
    rem = stages["remediation"]["command"]
    # rag stage never calls the model: provider is hard-coded none, not templated
    assert "--provider" in rag and rag[rag.index("--provider") + 1] == "none"
    assert "{provider}" not in rag
    # remediation stage produces BOTH reports in one run, so the grounded answer is not generated twice
    assert "--rag-output" in rem and "{rag_output}" in rem
    assert "--output" in rem and "{remediation_output}" in rem
    assert "{provider}" in rem
    validation = stages["validation"]["command"]
    assert "--live" not in validation
    assert "--output" in validation and "{validation_output}" in validation
    judge = stages["judge"]["command"]
    assert "--live" not in judge and "--provider" not in judge and "{provider}" not in judge
    assert "--report" in judge and "{validation_output}" in judge
    assert "--output" in judge and "{judge_output}" in judge


def test_stages_endpoint_hides_the_raw_command():
    payload = m.stages()
    assert [s["id"] for s in payload["stages"]] == [
        "evidence", "narrative", "rag", "remediation", "validation", "judge",
    ]
    assert "judge_decision.json" in m.ALLOWED_ARTIFACTS
    assert all("command" not in s for s in payload["stages"])


# -------------------------------------------------------------- configuration

def test_default_config_is_safe_and_bounds_requests(monkeypatch):
    config = m._dashboard_llm_config()
    assert config["provider"] == "none"               # no model by default
    assert config["max_retries"] == "0"               # dashboard disables SDK re-sends by default
    assert config["token_budget"] == "" and config["max_passage_chars"] == ""
    assert config["tpm_limit"] == ""                  # unset -> the client's default-on per-minute budget


def test_an_invalid_provider_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("AUTOSHIELD_LLM_PROVIDER", "openai")
    assert m._dashboard_llm_config()["provider"] == "none"


def test_configured_values_are_read(monkeypatch):
    monkeypatch.setenv("AUTOSHIELD_LLM_PROVIDER", "groq")
    monkeypatch.setenv("AUTOSHIELD_LLM_TOKEN_BUDGET", "8000")
    monkeypatch.setenv("AUTOSHIELD_LLM_TPM_LIMIT", "16000")
    monkeypatch.setenv("AUTOSHIELD_MAX_PASSAGE_CHARS", "600")
    config = m._dashboard_llm_config()
    assert config == {"provider": "groq", "max_retries": "0", "token_budget": "8000",
                      "tpm_limit": "16000", "max_passage_chars": "600"}


def test_subprocess_env_maps_the_knobs_and_adds_no_secret(monkeypatch):
    monkeypatch.setenv("AUTOSHIELD_LLM_TOKEN_BUDGET", "8000")
    monkeypatch.setenv("AUTOSHIELD_LLM_TPM_LIMIT", "16000")
    env = m._subprocess_env(m._dashboard_llm_config())
    assert env["LLM_MAX_RETRIES"] == "0" and env["LLM_TOKEN_BUDGET"] == "8000"
    assert env["LLM_TPM_LIMIT"] == "16000"
    assert env["PYTHONPATH"]
    # the helper never invents a credential
    assert "GROQ_API_KEY" not in {k for k in env if k not in __import__("os").environ}


def test_blank_retries_keeps_the_library_default(monkeypatch):
    monkeypatch.setenv("AUTOSHIELD_LLM_MAX_RETRIES", "")
    env = m._subprocess_env(m._dashboard_llm_config())
    assert "LLM_MAX_RETRIES" not in env            # nothing forced -> GroqLLMClient keeps the SDK default (2)


# -------------------------------------------------------------- actionable hints

def test_oversize_output_gets_an_oversize_hint():
    hint = m._actionable_hint("No request was sent: the estimated prompt is 9000 tokens above the budget")
    assert hint and "budget" in hint.lower()


def test_rate_limit_output_gets_a_rate_limit_hint():
    hint = m._actionable_hint("RateLimitError ... tokens per minute ... Limit 8000")
    assert hint and "rate" in hint.lower()


def test_ordinary_output_gets_no_hint():
    assert m._actionable_hint("report written to analysis/rag_report.json") is None


def test_the_dashboard_page_shows_judge_provenance_and_does_not_invent_a_live_finding():
    text = Path(__file__).resolve().parents[1].joinpath("dashboard", "index.html").read_text(encoding="utf-8")
    assert "function judgePanel" in text
    assert "judge_decision.json" in text
    assert "evidence_provenance" in text
    assert "not displayed as a live security finding" in text
    assert "Nothing is fabricated here." in text
