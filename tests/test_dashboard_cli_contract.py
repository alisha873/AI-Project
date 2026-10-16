"""
Contract tests for what the dashboard backend relies on from the two CLIs it launches.

AUDIT CONTEXT (read this first): the dashboard backend (`api/main.py`) is NOT in this branch; it exists only
on `main` (commit f482794). It builds no RAG context itself. Its "rag" and "remediation" stages start
`python -m scripts.run_grounded_answer <incident> --provider <p> --output <file>` and
`python -m scripts.run_remediation_plan <incident> --provider <p> --output <file>` as subprocesses and pass no
`--max-documents`, `--max-chunks-per-document` or `--model`. So the dashboard's effective retrieval and context
limits ARE the CLI defaults pinned here.

These tests cover only this side of that contract. The dashboard itself (HTTP routes, the subprocess call, the
browser page) has NO automated test in this repository, and nothing here starts a server or calls a model.
"""

from __future__ import annotations

from rag.context import DEFAULT_MAX_CHUNKS_PER_DOCUMENT, DEFAULT_MAX_DOCUMENTS, ContextLimits
from retrieval.contract import DEFAULT_LIMIT
from scripts import run_grounded_answer, run_remediation_plan

# The argument shape used by the dashboard for its "rag" and "remediation" stages (provider and output
# path are substituted at run time; the incident id is validated by the dashboard as incident_NNN).
DASHBOARD_ARGV = ["incident_001", "--provider", "groq", "--output", "analysis/report.json"]


def test_the_dashboard_argument_shape_is_accepted_by_both_clis():
    for module in (run_grounded_answer, run_remediation_plan):
        args = module._parser().parse_args(DASHBOARD_ARGV)
        assert args.incident_id == "incident_001"
        assert args.provider == "groq"
        assert str(args.output).replace("\\", "/") == "analysis/report.json"
        assert args.model is None                       # the dashboard never overrides the model


def test_the_cli_defaults_the_dashboard_inherits_are_the_grounded_answer_defaults():
    expected = ContextLimits()
    assert (expected.max_documents, expected.max_chunks_per_document) == (10, 1)
    assert expected.max_documents == DEFAULT_MAX_DOCUMENTS == DEFAULT_LIMIT
    assert expected.max_chunks_per_document == DEFAULT_MAX_CHUNKS_PER_DOCUMENT
    for module in (run_grounded_answer, run_remediation_plan):
        args = module._parser().parse_args(DASHBOARD_ARGV)
        assert (args.max_documents, args.max_chunks_per_document) == (10, 1), module.__name__


def test_the_default_provider_without_the_flag_is_no_model_call():
    # The dashboard falls back to --provider none when AUTOSHIELD_LLM_PROVIDER is unset or invalid.
    for module in (run_grounded_answer, run_remediation_plan):
        assert module._parser().parse_args(["incident_001"]).provider == "none"
        assert module._parser().parse_args(["incident_001", "--provider", "none"]).provider == "none"


def test_the_cli_offers_no_retriever_option_so_the_dashboard_cannot_select_another_one():
    for module in (run_grounded_answer, run_remediation_plan):
        options = {opt for action in module._parser()._actions for opt in action.option_strings}
        assert not any(word in opt for opt in options for word in ("retriever", "semantic", "rrf", "embed", "fusion"))
