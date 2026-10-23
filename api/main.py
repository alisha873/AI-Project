"""Minimal live orchestration API for the AutoShield dashboard.

Run from repository root:
    python -m pip install fastapi uvicorn
    python -m uvicorn api.main:app --reload

This orchestrator calls existing scripts as subprocesses. It does not fabricate
stage results. Stages requiring unavailable modules/data/config will report errors.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------------------
# Dashboard LLM configuration (documented, consistent across the stages it launches)
# --------------------------------------------------------------------------------------------------
# The dashboard runs the SAME committed CLIs as a developer would; it adds no RAG logic of its own.
# These knobs make the multi-stage run consistent and bound the number of provider requests. They are
# read once, from the environment, and are logged (without secrets) at the start of a run.
#
# * AUTOSHIELD_LLM_PROVIDER     none | gemini | groq | ollama
#                               (default gemini: narrative synthesis and the remediation stage call Gemini.
#                               The RAG stage stays --provider none and does not call a model. groq is an
#                               explicit override only; an unknown value falls back to none, never to groq.)
# * AUTOSHIELD_LLM_MAX_RETRIES  SDK re-send count for one call (default "0" here: the dashboard can fire
#                               several calls per minute, so its subprocesses disable the SDK's own
#                               re-sends to avoid multiplying provider requests. The library default is
#                               still 2; this only affects dashboard-launched stages. Set to "" to keep
#                               the library default.)
# * AUTOSHIELD_LLM_TOKEN_BUDGET pre-send PER-REQUEST token budget (default unset: no per-request size cap
#                               beyond the TPM budget below). Set e.g. "8000" to also refuse an oversized
#                               single prompt before sending.
# * AUTOSHIELD_LLM_TPM_LIMIT    per-MINUTE token budget forwarded as LLM_TPM_LIMIT (default unset). The Gemini
#                               client does not assume Groq's recorded 8,000 tokens/minute ceiling. Set a
#                               number only for a limit verified on the configured account, or "0" to disable.
# * AUTOSHIELD_MAX_PASSAGE_CHARS shorten each reference passage in the prompt (default unset: full text).


def _dashboard_llm_config() -> dict[str, str]:
    provider = os.environ.get("AUTOSHIELD_LLM_PROVIDER", "gemini").strip().lower()
    if provider not in {"none", "gemini", "groq", "ollama"}:
        provider = "none"
    return {
        "provider": provider,
        # Default "0" re-sends for dashboard stages (documented above); AUTOSHIELD_LLM_MAX_RETRIES="" keeps
        # the library default by passing nothing through.
        "max_retries": os.environ.get("AUTOSHIELD_LLM_MAX_RETRIES", "0").strip(),
        "token_budget": os.environ.get("AUTOSHIELD_LLM_TOKEN_BUDGET", "").strip(),
        "tpm_limit": os.environ.get("AUTOSHIELD_LLM_TPM_LIMIT", "").strip(),
        "max_passage_chars": os.environ.get("AUTOSHIELD_MAX_PASSAGE_CHARS", "").strip(),
    }


def _subprocess_env(config: dict[str, str]) -> dict[str, str]:
    """Environment for a launched CLI: the parent env plus the documented LLM_* knobs (no secrets added)."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    if config["max_retries"]:
        env["LLM_MAX_RETRIES"] = config["max_retries"]
    if config["token_budget"]:
        env["LLM_TOKEN_BUDGET"] = config["token_budget"]
    if config["tpm_limit"]:
        env["LLM_TPM_LIMIT"] = config["tpm_limit"]
    return env


# Phrases in a stage's output that deserve a clearer, actionable hint in the event stream.
_RATE_LIMIT_HINT = re.compile(r"rate.?limit|RateLimitError|tokens per minute|TPM|HTTP 429|code.?429", re.IGNORECASE)
_OVERSIZE_HINT = re.compile(r"no request was sent|prompt .*budget|request too large|HTTP 413|code.?413", re.IGNORECASE)


def _actionable_hint(text: str) -> str | None:
    if _OVERSIZE_HINT.search(text):
        return ("The prompt exceeded the token budget and was refused before sending. Lower "
                "AUTOSHIELD_MAX_PASSAGE_CHARS or --max-documents, or raise AUTOSHIELD_LLM_TOKEN_BUDGET if the "
                "provider allows a larger request.")
    if _RATE_LIMIT_HINT.search(text):
        return ("The provider rate-limited the request (tokens-per-minute). Reduce the prompt size "
                "(AUTOSHIELD_MAX_PASSAGE_CHARS / --max-documents), wait for the window to reset, or upgrade "
                "the provider tier. SDK re-sends are disabled by default for the dashboard to avoid "
                "multiplying requests.")
    return None


app = FastAPI(title="AutoShield Live Pipeline", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

RUNS: dict[str, dict[str, Any]] = {}
STAGES = [
    {"id": "evidence", "label": "Evidence normalization", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_incident_analysis", "{incident_id}"],
     "description": "Normalize and load evidence, then build deterministic reconstruction outputs."},
    {"id": "narrative", "label": "LLM narrative synthesis", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_llm_incident_synthesis", "{incident_id}"],
     "description": "Generate an evidence-grounded narrative from the saved deterministic outputs."},
    # RAG stage: context/retrieval preview only (provider forced to none). The grounded ANSWER is generated
    # once, in the remediation stage below, which reuses it for planning and also writes the RAG report. This
    # removes the previous duplicate Phase 11D model call (the rag stage and remediation stage each ran it).
    {"id": "rag", "label": "RAG knowledge retrieval (context preview)", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_grounded_answer", "{incident_id}", "--provider", "none", "--output", "{rag_output}"],
     "description": "Build and save the FTS5 v1 retrieval context (no model call). The grounded answer is "
                    "produced once in the remediation stage and written to the same RAG report."},
    {"id": "remediation", "label": "Grounded answer + remediation planning", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_remediation_plan", "{incident_id}", "--provider", "{provider}",
                 "--rag-output", "{rag_output}", "--output", "{remediation_output}"],
     "description": "Run the committed grounded-answer + advisory remediation planner once, writing both the "
                    "RAG report and the remediation plan (no duplicate grounded-answer call)."},
    {"id": "validation", "label": "Controlled validation (dry run)", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_controlled_validation", "{incident_id}",
                 "--output", "{validation_output}"],
     "description": "Offline Phase 13 dry run: writes a blocked or not_validated report. No --live flag, "
                    "so zero network requests. A genuine live run is a separate, approval-gated CLI invocation."},
    {"id": "judge", "label": "Judge (offline evidence assessment)", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_judge", "--report", "{validation_output}",
                 "--output", "{judge_output}"],
     "description": "Offline Phase 14 judge: reads the saved validation report and writes a decision. "
                    "No model, no network, and no live probe. A dry-run report cannot become a live finding."},
]

# Artifact files the dashboard may return. The judge file is the Phase 14 assessment, not a fabricated demo.
ALLOWED_ARTIFACTS = (
    "incident_reconstruction.json",
    "root_cause_analysis.json",
    "llm_incident_narrative.json",
    "rag_report.json",
    "remediation_plan.json",
    "validation_report.json",
    "judge_decision.json",
)

class StartRequest(BaseModel):
    incident_id: str = Field(default="incident_001", pattern=r"^incident_[0-9]{3}$")

def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()

def incident_path(incident_id: str) -> Path | None:
    candidates = [
        ROOT / incident_id, ROOT / "incidents" / incident_id,
        ROOT / "data" / "incidents" / incident_id,
        ROOT / "infrastructure" / "incidents" / incident_id,
    ]
    found = [p for p in candidates if p.is_dir()]
    return found[0] if len(found) == 1 else None



@app.get("/api/shap")
def shap_explanation():
    """Serve the latest locally generated GCN SHAP explanation."""
    path = ROOT / "evaluation" / "results" / "gnn_shap_explanation.json"
    if not path.is_file():
        raise HTTPException(
            404,
            "No SHAP explanation found. Run python -m evaluation.run_gnn_shap first."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"Could not read SHAP result: {exc}") from exc

    if not isinstance(payload.get("contributions"), list):
        raise HTTPException(500, "SHAP result has no contributions list.")
    return payload


@app.get("/api/bnn-risk")
def bnn_risk_results():
    """Return saved Bayesian risk-proxy experiment results."""
    import json

    path = ROOT / "evaluation" / "results" / "bnn_risk_results.json"
    if not path.is_file():
        return {
            "available": False,
            "message": "Run python -m evaluation.run_bnn_risk first."
        }
    return {
        "available": True,
        "result": json.loads(path.read_text(encoding="utf-8"))
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AutoShield Live Pipeline", "repo_root": str(ROOT)}

@app.get("/api/stages")
def stages():
    return {"stages": [{k: v for k, v in s.items() if k != "command"} for s in STAGES]}

@app.get("/api/incidents")
def incidents():
    paths = []
    for base in (ROOT, ROOT / "incidents", ROOT / "data" / "incidents", ROOT / "infrastructure" / "incidents"):
        if base.is_dir():
            paths.extend(p.name for p in base.iterdir() if p.is_dir() and re.fullmatch(r"incident_[0-9]{3}", p.name))
    return {"incidents": sorted(set(paths))}


@app.get("/api/incidents/{incident_id}/artifacts")
def incident_artifacts(incident_id: str):
    """Return allowlisted JSON artifacts for the selected incident."""
    if not re.fullmatch(r"incident_[0-9]{3}", incident_id):
        raise HTTPException(400, "Invalid incident ID")
    incident_dir = incident_path(incident_id)
    if incident_dir is None:
        raise HTTPException(404, "Incident directory not found or ambiguous")
    analysis_dir = incident_dir / "analysis"
    allowed = ALLOWED_ARTIFACTS
    artifacts = {}
    for name in allowed:
        path = analysis_dir / name
        if path.is_file():
            try:
                artifacts[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                artifacts[name] = {"_read_error": str(exc)}
    return {"incident_id": incident_id, "artifacts": artifacts}

@app.post("/api/runs")
def start_run(body: StartRequest):
    incident_dir = incident_path(body.incident_id)
    if incident_dir is None:
        raise HTTPException(404, f"Could not uniquely locate {body.incident_id} under the supported incident folders.")
    run_id = uuid.uuid4().hex[:12]
    RUNS[run_id] = {
        "run_id": run_id, "incident_id": body.incident_id, "status": "queued",
        "created_at": timestamp(), "events": [], "current_stage": None,
        "llm_provider": os.environ.get("AUTOSHIELD_LLM_PROVIDER", "gemini"),
    }
    return {"run_id": run_id, "incident_id": body.incident_id, "status": "queued"}

@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Unknown run ID")
    return run

@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str):
    run = RUNS.get(run_id)
    if not run:
        raise HTTPException(404, "Unknown run ID")

    async def emit(event: dict[str, Any]):
        return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    async def generator():
        run["status"] = "running"
        config = _dashboard_llm_config()
        run["llm_provider"] = config["provider"]
        yield await emit({"type": "run_started", "run_id": run_id, "incident_id": run["incident_id"],
                          "llm_config": {k: v for k, v in config.items()}})   # counts/flags only, no secrets
        env = _subprocess_env(config)
        for stage in STAGES:
            run["current_stage"] = stage["id"]
            event = {"type": "stage_started", "stage": stage["id"], "label": stage["label"], "at": timestamp()}
            run["events"].append(event)
            yield await emit(event)
            if stage["kind"] == "pending":
                event = {"type": "stage_pending", "stage": stage["id"], "label": stage["label"], "message": stage["description"], "at": timestamp()}
                run["events"].append(event)
                yield await emit(event)
                continue

            incident_dir = incident_path(run["incident_id"])
            assert incident_dir is not None
            output_map = {
                "provider": config["provider"],
                "rag_output": str(incident_dir / "analysis" / "rag_report.json"),
                "remediation_output": str(incident_dir / "analysis" / "remediation_plan.json"),
                "validation_output": str(incident_dir / "analysis" / "validation_report.json"),
                "judge_output": str(incident_dir / "analysis" / "judge_decision.json"),
            }
            cmd = [part.format(incident_id=run["incident_id"], **output_map) for part in stage["command"]]
            # Apply the shared passage-compression setting to the two RAG-bearing stages.
            if config["max_passage_chars"] and stage["id"] in {"rag", "remediation"}:
                cmd += ["--max-passage-chars", config["max_passage_chars"]]
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, env=env
            )
            assert proc.stdout is not None
            stage_hint: str | None = None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                message = line.decode("utf-8", errors="replace").rstrip()
                item = {"type": "stage_log", "stage": stage["id"], "message": message, "at": timestamp()}
                run["events"].append(item)
                yield await emit(item)
                stage_hint = stage_hint or _actionable_hint(message)
            code = await proc.wait()
            event = {"type": "stage_completed" if code == 0 else "stage_failed",
                     "stage": stage["id"], "exit_code": code, "at": timestamp(),
                     "message": "Process exited successfully." if code == 0 else f"Process exited with code {code}."}
            if code != 0 and stage_hint:
                event["advice"] = stage_hint
            run["events"].append(event)
            yield await emit(event)
            if code != 0:
                # Fail fast: don't imply downstream stages ran on valid outputs.
                run["status"] = "failed"
                run["current_stage"] = stage["id"]
                failure = {"type": "run_failed", "stage": stage["id"], "message": event["message"]}
                if stage_hint:
                    failure["advice"] = stage_hint
                yield await emit(failure)
                return
        run["status"] = "completed_with_pending_stages"
        run["current_stage"] = None
        yield await emit({"type": "run_finished", "status": run["status"], "at": timestamp()})
    return StreamingResponse(generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
