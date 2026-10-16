"""
Phase 13A: structural safety of the `validation` package (AST scans).

The package must not be able to reach a network, a process, Docker, a model, a file or the answer key, and it must
not reach into other phases' private internals. Phases 8-12 must not depend on it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import validation

PACKAGE = Path(validation.__file__).parent
ROOT = PACKAGE.parent
MODULES = sorted(PACKAGE.glob("*.py"))

FORBIDDEN_IMPORT_PREFIXES = (
    "subprocess", "socket", "ssl", "requests", "urllib.request", "urllib3", "http", "httpx", "aiohttp", "asyncio",
    "docker", "paramiko", "shutil", "os", "pathlib", "tempfile", "sqlite3", "dotenv",
    "llm", "groq", "ollama", "openai", "torch", "numpy", "pandas", "transformers", "sentence_transformers",
    "retrieval", "rag", "knowledge", "ingestion", "agents", "evidence", "graph", "evaluation", "attacks",
    "scripts", "remediation.grounding", "remediation.prompt", "remediation.report", "remediation.inputs",
    "remediation.configuration", "remediation.priority", "remediation.text_checks",
)
ALLOWED_REMEDIATION_IMPORTS = {"remediation.contract", "remediation.planner", "remediation"}


def imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_can_reach_the_network_a_process_docker_a_model_the_filesystem_or_other_internals(path):
    for name in imports(path):
        assert not name.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{path.name} imports {name}"
        if name.startswith("remediation"):
            assert name in ALLOWED_REMEDIATION_IMPORTS, f"{path.name} imports private Phase 12 module {name}"


def test_the_only_urllib_use_is_pure_string_parsing():
    allowed = {"urllib.parse"}
    for path in MODULES:
        for name in imports(path):
            if name.startswith("urllib"):
                assert name in allowed, f"{path.name} imports {name}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_opens_files_runs_commands_or_resolves_names(path):
    banned = {"open", "system", "popen", "exec", "eval", "compile", "write_text", "write_bytes", "read_text",
              "read_bytes", "unlink", "rmdir", "mkdir", "rename", "chmod", "run", "Popen", "call", "check_output",
              "getaddrinfo", "gethostbyname", "create_connection", "urlopen", "connect", "send", "environ", "getenv",
              "__import__"}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if path.name == "runner.py" and name == "send":  # the runner's one call site hands a request to a transport
                continue
            assert name not in banned, f"{path.name} calls {name}()"


def test_a_transport_is_called_from_exactly_one_place_in_the_runner():
    sites = []
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(function):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "send"):
                    sites.append((path.name, function.name))
    assert sites == [("runner.py", "_send")]


def test_the_live_transport_lock_is_closed_in_this_build():
    from validation import runner
    assert runner.LIVE_TRANSPORT_ENABLED is False


def test_the_orchestrator_never_uses_a_transport():
    tree = ast.parse((PACKAGE / "orchestrator.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "send"]
    assert calls == []
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_validation_report")
    uses = [n for n in ast.walk(function) if isinstance(n, ast.Name) and n.id == "transport"]
    assert len(uses) == 0, "the reserved transport parameter must not be used in Phase 13A"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_module_contains_a_path_into_the_answer_key(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
    }
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings]
    assert not any("ground_truth/" in s or "ground_truth.json" in s or "ground_truth\\" in s for s in strings), path.name


def test_the_scenario_source_path_is_not_in_the_answer_key_directory():
    from validation.scenario import SOURCE_PATH
    assert "ground_truth" not in SOURCE_PATH.split("/")
    assert (ROOT / SOURCE_PATH).is_file()


def test_the_validation_package_does_not_use_ground_truth_in_any_runtime_name():
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if "ground_truth" in name:
                    assert name in {"ground_truth_used", "ground_truth_consulted"}, f"{path.name}: {name}"


def test_earlier_phases_do_not_import_the_validation_package():
    for folder in ("remediation", "rag", "retrieval", "knowledge", "ingestion", "agents", "evidence", "graph", "llm"):
        for path in (ROOT / folder).glob("*.py"):
            for name in imports(path):
                assert not (name == "validation" or name.startswith("validation.")), f"{path} imports {name}"


def test_the_draft_stubs_in_the_evidence_schema_are_untouched_and_unused_by_the_new_package():
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        assert "evidence.schema" not in text
        for stub in ("ValidationCheck", "ValidationResult", "JudgeDecision"):
            assert stub not in text, f"{path.name} mentions the unused draft stub {stub}"
