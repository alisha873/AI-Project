"""
Phase 12 tests: the schema contract, the priority rule and the structural safety properties of the package
(no network, no process execution, no file writes, no experimental retrievers, no second Groq client).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import remediation
from remediation import contract
from remediation.contract import (
    PRIORITY_RULE,
    PlanDraft,
    ProposalDraft,
    RemediationPlan,
    RemediationProposal,
)
from remediation.priority import priority_basis, rank_proposals
from remediation.prompt import plan_json_schema
from tests.remediation_helpers import good_proposal

PACKAGE = Path(remediation.__file__).parent
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_remediation_plan.py"


def draft(**overrides) -> ProposalDraft:
    return ProposalDraft.model_validate(good_proposal(**overrides))


# ------------------------------------------------------------
# schema consistency
# ------------------------------------------------------------

def test_the_json_schema_lists_exactly_the_draft_fields():
    schema = plan_json_schema()
    assert list(schema["properties"]) == list(PlanDraft.model_fields)
    proposal = schema["properties"]["proposals"]["items"]
    assert list(proposal["properties"]) == list(ProposalDraft.model_fields)
    for name, model in (("affected_area", contract.AffectedArea), ("severity", contract.SeverityAssessment),
                        ("validation", contract.ValidationProcedure), ("confidence", contract.ConfidenceAssessment)):
        assert list(proposal["properties"][name]["properties"]) == list(model.model_fields)
    steps = proposal["properties"]["implementation_steps"]["items"]
    assert list(steps["properties"]) == list(contract.ImplementationStep.model_fields)


def test_every_object_in_the_schema_requires_all_keys_and_forbids_extras():
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False and node["required"] == list(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(plan_json_schema())


def test_schema_enums_match_the_contract_vocabulary():
    properties = plan_json_schema()["properties"]["proposals"]["items"]["properties"]
    assert properties["severity"]["properties"]["level"]["enum"] == list(contract.LEVELS)
    assert properties["disruption"]["enum"] == list(contract.LEVELS)
    assert properties["reversibility"]["enum"] == list(contract.REVERSIBILITIES)
    assert properties["affected_area"]["properties"]["state"]["enum"] == list(contract.AREA_STATES)
    assert properties["confidence"]["properties"]["level"]["enum"] == list(contract.CONFIDENCE_LEVELS)
    assert plan_json_schema()["properties"]["outcome"]["enum"] == list(contract.OUTCOMES)


def test_derived_fields_exist_only_on_the_output_model():
    derived = set(RemediationProposal.model_fields) - set(ProposalDraft.model_fields)
    assert derived == {"proposal_id", "priority_rank", "priority_basis", "linked_hypotheses", "references",
                       "status", "executed", "human_review_required"}
    assert not derived & set(plan_json_schema()["properties"]["proposals"]["items"]["properties"])


def test_the_contract_is_versioned_and_documents_its_meanings():
    assert contract.PLAN_SCHEMA_VERSION == "1.0" and RemediationPlan.model_fields["schema_version"].default == "1.0"
    assert "not a probability" in contract.CONFIDENCE_MEANING and "not calibrated" in contract.CONFIDENCE_MEANING
    first = contract.PRIORITY_CRITERIA[0].lower()
    assert "ordinal" in first and "not a risk score" in first
    assert contract.PRIORITY_RULE in contract.PRIORITY_CRITERIA


# ------------------------------------------------------------
# strictness
# ------------------------------------------------------------

def test_the_draft_models_do_not_coerce_types():
    with pytest.raises(Exception):
        draft(configuration_dependent="true")
    with pytest.raises(Exception):
        draft(title=123)
    with pytest.raises(Exception):
        draft(unexpected=1)  # unknown keys are errors


def test_drafts_are_immutable():
    proposal = draft()
    with pytest.raises(Exception):
        proposal.title = "changed"  # type: ignore[misc]


# ------------------------------------------------------------
# priority rule
# ------------------------------------------------------------

def order(*drafts: ProposalDraft) -> list[int]:
    return [index for index, _rank, _basis in rank_proposals(list(drafts))]


def test_evidence_linkage_outranks_severity():
    linked_low = draft(severity={"level": "low", "rationale": "r"})
    general_high = draft(title="General", observed_evidence_ids=[], severity={"level": "high", "rationale": "r"})
    assert order(general_high, linked_low) == [1, 0]
    assert priority_basis(general_high).evidence_linkage == "general_hardening"
    assert priority_basis(linked_low).evidence_linkage == "incident_linked"


def test_severity_then_disruption_then_reversibility_then_model_order():
    def make(severity="medium", disruption="low", reversibility="reversible"):
        return draft(severity={"level": severity, "rationale": "r"}, disruption=disruption,
                     reversibility=reversibility, preconditions=["human review"])

    assert order(make("low"), make("high"), make("medium")) == [1, 2, 0]
    assert order(make(disruption="high"), make(disruption="low"), make(disruption="unknown")) == [1, 0, 2]
    assert order(make(reversibility="irreversible"), make(reversibility="unknown"),
                 make(reversibility="partially_reversible"), make()) == [3, 2, 1, 0]
    assert order(make(), make(), make()) == [0, 1, 2]                      # ties keep the model's order


def test_ranks_start_at_one_and_are_dense():
    ranks = [rank for _i, rank, _b in rank_proposals([draft(), draft(title="B"), draft(title="C")])]
    assert ranks == [1, 2, 3] and rank_proposals([]) == []


def test_the_documented_rule_is_the_one_in_every_basis():
    assert priority_basis(draft()).ordering_rule == PRIORITY_RULE


# ------------------------------------------------------------
# structural safety of the whole package (AST scans)
# ------------------------------------------------------------

def _modules():
    return sorted(PACKAGE.glob("*.py")) + [SCRIPT]


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


FORBIDDEN_IMPORT_PREFIXES = (
    "subprocess", "socket", "requests", "urllib", "httpx", "http", "aiohttp", "shutil", "docker", "paramiko",
    "groq", "ollama", "openai", "torch", "numpy", "sentence_transformers", "transformers", "pandas",
    "retrieval.semantic", "retrieval.rrf", "retrieval.embeddings", "retrieval.embeddings_st", "retrieval.vector_cache",
    "retrieval.fts5_v2", "retrieval.comparison", "retrieval.evaluation", "retrieval.evaluation_v2",
    "retrieval.benchmark_v2", "retrieval.benchmark_v3", "retrieval.freeze",
    "evaluation", "graph", "attacks", "validation",
)


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_can_reach_the_network_run_processes_or_use_experimental_retrieval(path):
    for name in _imports(path):
        assert not name.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{path.name} imports {name}"


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_the_planner_package_never_writes_files_or_runs_commands(path):
    # (`str.replace` and `list.remove` are ordinary string/list methods, so they are not listed)
    banned_calls = {"open", "system", "popen", "exec", "eval", "write_text", "write_bytes", "read_text", "unlink",
                    "rmdir", "mkdir", "rename", "chmod", "run", "Popen", "call", "check_output"}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            assert name not in banned_calls, f"{path.name} calls {name}()"


def test_only_the_cli_may_write_and_only_on_request():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("write_text") == 1 and "if args.output" in text
    assert "ground_truth" not in text.replace("`ground_truth/`", "")  # only mentioned in the docstring as not read


def test_the_package_uses_the_existing_llm_abstraction_and_no_second_client():
    planner = (PACKAGE / "planner.py").read_text(encoding="utf-8")
    assert "from llm.base import LLMClient" in planner and "llm_client.generate(" in planner
    assert not any("Groq(" in p.read_text(encoding="utf-8") for p in PACKAGE.glob("*.py"))
    assert not any(name.startswith("llm.groq") for path in PACKAGE.glob("*.py") for name in _imports(path))


def test_no_module_reads_the_answer_key_or_the_incident_directory_directly():
    for path in PACKAGE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
        }
        strings = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
        # (the output field name "ground_truth_used" is allowed; a PATH into ground_truth/ is not)
        assert not any("ground_truth/" in s or "ground_truth.json" in s or s.endswith(".jsonl") for s in strings), path.name


def test_the_package_pins_the_production_retriever_through_phase_11d():
    report = (PACKAGE / "report.py").read_text(encoding="utf-8")
    assert "run_incident_rag" in report and "PRODUCTION_RETRIEVER_NAME" in report


def test_phase_11d_private_helpers_that_phase_12_reuses_still_exist():
    from rag import citations

    for name in ("_CONFIRMATION", "_EVIDENCE_TOKEN_RE", "_HIGH_IMPACT", "_PATH_RE", "_REFERENCE_TOKEN_RE",
                 "_identifiers", "_is_negated"):
        assert hasattr(citations, name), name
    json.dumps({"reused": "rag.citations"})
