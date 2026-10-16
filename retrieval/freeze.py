"""
Phase 11 (Milestone 11C) - protection of the frozen final benchmark: pinned hash, guard, preregistration.

Three things live here, deliberately separate from the benchmark loader and the evaluators, so the
development tooling can import the guard WITHOUT importing the v3 loader:

1. `FROZEN_V3_SHA256`: the SHA-256 of `benchmark_11c.json`, recorded in the preflight before any 11C code
   existed. `retrieval.benchmark_v3.load_benchmark_v3` refuses any other file.
2. `refuse_final_benchmark(path)`: the development script calls this on its `--benchmark` path. It refuses a
   file whose hash equals the frozen value, whose JSON format is the v3 format, or whose name is the v3 file.
3. The preregistration record: a JSON file, written BEFORE the final evaluation, describing the exact frozen
   state: the benchmark hash, the corpus fingerprint, the model id/revision/weights and embedding
   configuration, the installed pinned package versions, the cache format and index fingerprint, the
   chunk-to-document rule, retrieval limits, lane filters, RRF constant and tie-breaks, the metrics and label
   sets, and a SHA-256 of every implementation file. `verify_preregistration` compares the record with the
   CURRENT state and returns every difference; the final runner refuses to run if there is any (including a
   changed implementation file or a changed benchmark hash).

What this does and does not protect
-----------------------------------
It makes accidental or casual tuning on v3 fail loudly: an edited benchmark, a changed model, a changed
setting or a changed source file all block the run, and the runner will not overwrite an existing final
report. It cannot stop a determined person from deleting the record and the report and starting over; the
protection is a process guard, not a cryptographic one. The record also stores the git HEAD (informational:
the work is uncommitted, so file hashes are the real identifier) and the freeze time (informational; it is
not part of the compared state).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from retrieval.contract import RetrievalError

FROZEN_V3_SHA256 = "31e26a84813783c235cfa899b02d42a1d5eec006e691f2a1825f21b14c144d10"
FROZEN_V3_FORMAT = "autoshield.retrieval_benchmark.v3"
FROZEN_V3_FILENAME = "benchmark_11c.json"
PREREGISTRATION_FORMAT = "autoshield.preregistration.v1"

# Source files whose bytes define "the implementation". Globs are relative to the repository root.
IMPLEMENTATION_GLOBS = (
    "retrieval/*.py",
    "knowledge/*.py",
    "scripts/run_semantic_comparison.py",
    "scripts/run_semantic_final.py",
    "scripts/download_embedding_model.py",
    "requirements-embeddings.txt",
)

METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "hit_at_k_counts", "full_recall_at_k_counts")
SYSTEMS = (
    "S1 fts5-bm25-v1",
    "S2 fts5-bm25-v2 (top2_sum)",
    "S3 semantic, one ranking over all documents",
    "S4 semantic, lane-specific (CWE-only cases on the weakness lane, CAPEC-only cases on the attack-pattern lane)",
    "S5 RRF (k=60) of the two lane rankings of the same semantic retriever",
)
DECISION_RULES = (
    "Strict labels are primary; strict-plus-lenient is a sensitivity analysis only.",
    "Report counts of questions answered at rank <= k beside every percentage; no significance claims.",
    "Behaviour cases are reported separately and never mixed into Recall or MRR; no abstention claim is made "
    "because no abstention rule was chosen on development data.",
    "Systems that lose to the keyword baselines are reported. No setting is changed after any v3 result is seen.",
)


class FrozenBenchmarkError(RetrievalError):
    """A benchmark file is not the frozen v3 file (its SHA-256 differs from the pinned value)."""


class FinalBenchmarkProtectedError(RetrievalError):
    """The development tooling was pointed at the frozen final benchmark."""


class PreregistrationError(RetrievalError):
    """The preregistration record is missing, malformed, altered or does not match the current state."""


# ============================================================
# GUARD AGAINST ACCIDENTAL USE DURING DEVELOPMENT
# ============================================================

def is_final_benchmark_file(path: str | Path) -> bool:
    """True if the file is (or claims to be) the frozen v3 benchmark. Reads it but never interprets cases."""
    file = Path(path)
    if file.name == FROZEN_V3_FILENAME:
        return True
    try:
        data = file.read_bytes()
    except OSError:
        return False
    if hashlib.sha256(data).hexdigest() == FROZEN_V3_SHA256:
        return True
    head = data[:4096].decode("utf-8", errors="ignore")
    return bool(re.search(r'"format"\s*:\s*"' + re.escape(FROZEN_V3_FORMAT) + '"', head))


def refuse_final_benchmark(path: str | Path) -> None:
    """Raise `FinalBenchmarkProtectedError` if `path` is the frozen final benchmark."""
    if is_final_benchmark_file(path):
        raise FinalBenchmarkProtectedError(
            f"{path} is the frozen final held-out benchmark (v3). It may only be run once, by "
            "scripts/run_semantic_final.py, after the configuration is frozen and preregistered."
        )


# ============================================================
# IMPLEMENTATION FINGERPRINT
# ============================================================

def implementation_hashes(repo_root: str | Path) -> dict[str, str]:
    """SHA-256 of every implementation file (see `IMPLEMENTATION_GLOBS`), keyed by repo-relative POSIX path."""
    root = Path(repo_root)
    hashes: dict[str, str] = {}
    for pattern in IMPLEMENTATION_GLOBS:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(hashes.items()))


def fingerprint_of(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def git_head(repo_root: str | Path) -> str | None:
    """The current git HEAD commit id, or None when git is unavailable (informational only)."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = done.stdout.strip()
    return head if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def check_pinned_requirements(requirements_file: str | Path) -> tuple[dict[str, str], list[str]]:
    """
    Compare every `name==version` pin with the installed version.

    Returns (installed versions by pinned name, problems). A missing package or a different version is a problem.
    """
    installed: dict[str, str] = {}
    problems: list[str] = []
    for line in Path(requirements_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, pinned = line.partition("==")
        if not sep:
            problems.append(f"requirement {line!r} is not an exact == pin")
            continue
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed")
            continue
        installed[name] = version
        if version != pinned:
            problems.append(f"{name} is {version}, pinned {pinned}")
    return dict(sorted(installed.items())), problems


# ============================================================
# PREREGISTRATION RECORD
# ============================================================

def build_frozen_state(
    *,
    benchmark_sha256: str,
    corpus_fingerprint: str,
    document_count: int,
    chunk_count: int,
    chunker: Mapping[str, Any],
    embedding_identity: Mapping[str, Any],
    embedding_fingerprint: str,
    runtime_versions: Mapping[str, str],
    installed_pins: Mapping[str, str],
    cache_format: str,
    cache_index_fingerprint: str,
    retrieval_settings: Mapping[str, Any],
    implementation: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the deterministic state that the preregistration freezes (no timestamps, no paths)."""
    return {
        "benchmark": {"format": FROZEN_V3_FORMAT, "sha256": benchmark_sha256},
        "corpus": {"fingerprint": corpus_fingerprint, "documents": document_count, "chunks": chunk_count,
                   "chunker": dict(chunker)},
        "embedding": {"identity": dict(embedding_identity), "fingerprint": embedding_fingerprint,
                      "runtime_versions": dict(runtime_versions)},
        "dependencies": {"requirements_file": "requirements-embeddings.txt", "installed_pins": dict(installed_pins)},
        "cache": {"format": cache_format, "index_fingerprint": cache_index_fingerprint},
        "retrieval": dict(retrieval_settings),
        "evaluation": {
            "metrics": list(METRICS), "systems": list(SYSTEMS), "decision_rules": list(DECISION_RULES),
            "label_sets": ["strict (primary)", "strict_plus_lenient (sensitivity)"],
        },
        "implementation": {"files": dict(implementation), "fingerprint": fingerprint_of(dict(implementation))},
    }


def build_preregistration(frozen: Mapping[str, Any], *, frozen_at_utc: str, head: str | None) -> dict[str, Any]:
    """The record to write to disk: the frozen state, its hash, and informational metadata."""
    body = {
        "format": PREREGISTRATION_FORMAT,
        "frozen_at_utc": frozen_at_utc,
        "git_head_informational": head,
        "note": (
            "Written before the final evaluation. The 'frozen' block is compared field by field with the "
            "live state; any difference blocks the run. The v3 labels are single-pass AI labels, not blind, "
            "not independently reviewed. A 23-document, hand-curated corpus and 30 AI-labelled questions; "
            "results do not show generalisation."
        ),
        "frozen": dict(frozen),
        "frozen_sha256": fingerprint_of(dict(frozen)),
    }
    return {**body, "record_sha256": fingerprint_of(body)}


def _differences(recorded: Any, current: Any, path: str = "frozen") -> list[str]:
    if isinstance(recorded, dict) and isinstance(current, dict):
        problems: list[str] = []
        for key in sorted(set(recorded) | set(current)):
            if key not in recorded:
                problems.append(f"{path}.{key}: not in the record")
            elif key not in current:
                problems.append(f"{path}.{key}: missing from the current state")
            else:
                problems.extend(_differences(recorded[key], current[key], f"{path}.{key}"))
        return problems
    return [] if recorded == current else [f"{path}: recorded {recorded!r}, current {current!r}"]


def verify_preregistration(record: object, current_frozen: Mapping[str, Any]) -> list[str]:
    """Every reason the record cannot authorise a run now (empty list = the record matches the live state)."""
    if not isinstance(record, dict):
        return ["the preregistration is not a JSON object"]
    expected_keys = {"format", "frozen_at_utc", "git_head_informational", "note", "frozen", "frozen_sha256",
                     "record_sha256"}
    if set(record) != expected_keys:
        return [f"the preregistration has unexpected keys: {sorted(set(record) ^ expected_keys)}"]
    if record["format"] != PREREGISTRATION_FORMAT:
        return [f"unknown preregistration format {record['format']!r}"]
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    problems: list[str] = []
    if record["record_sha256"] != fingerprint_of(body):
        problems.append("record_sha256 does not match the record (the file was edited after it was written)")
    frozen = record["frozen"]
    if not isinstance(frozen, dict) or record["frozen_sha256"] != fingerprint_of(frozen):
        problems.append("frozen_sha256 does not match the frozen block")
        return problems
    if frozen.get("benchmark", {}).get("sha256") != FROZEN_V3_SHA256:
        problems.append("the recorded benchmark SHA-256 is not the pinned frozen v3 value")
    problems.extend(_differences(frozen, dict(current_frozen)))
    return problems


def load_preregistration(path: str | Path) -> dict[str, Any]:
    """Read the record (strict JSON); raises `PreregistrationError` if it is missing or unreadable."""
    file = Path(path)
    if not file.is_file():
        raise PreregistrationError(f"no preregistration record at {file}; freeze the configuration first")
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreregistrationError(f"cannot read the preregistration {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise PreregistrationError("the preregistration is not a JSON object")
    return data


def require_valid_preregistration(path: str | Path, current_frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Load and verify the record against the live state; raise `PreregistrationError` listing every problem."""
    record = load_preregistration(path)
    problems = verify_preregistration(record, current_frozen)
    if problems:
        raise PreregistrationError("the preregistration does not match the current state:\n  " + "\n  ".join(problems))
    return record
