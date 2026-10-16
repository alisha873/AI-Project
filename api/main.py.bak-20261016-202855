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
    {"id": "rag", "label": "RAG knowledge retrieval", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_grounded_answer", "{incident_id}", "--provider", "{provider}", "--output", "{rag_output}"],
     "description": "Run the committed RAG retrieval/grounded-answer pipeline and save its report."},
    {"id": "remediation", "label": "Remediation planning", "kind": "script",
     "command": [sys.executable, "-m", "scripts.run_remediation_plan", "{incident_id}", "--provider", "{provider}", "--output", "{remediation_output}"],
     "description": "Run the committed advisory remediation planner and save its report."},
    {"id": "validation", "label": "Controlled validation / Judge / SHAP / risk", "kind": "pending",
     "description": "Not launched: implementation entry points are not present in the uploaded archives."},
]

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
    allowed = ("incident_reconstruction.json", "root_cause_analysis.json", "llm_incident_narrative.json", "rag_report.json", "remediation_plan.json")
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
        "llm_provider": os.environ.get("AUTOSHIELD_LLM_PROVIDER", "none"),
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
        yield await emit({"type": "run_started", "run_id": run_id, "incident_id": run["incident_id"]})
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

            provider = os.environ.get("AUTOSHIELD_LLM_PROVIDER", "none").strip().lower()
            if provider not in {"none", "groq", "ollama"}:
                provider = "none"
            incident_dir = incident_path(run["incident_id"])
            assert incident_dir is not None
            output_map = {
                "provider": provider,
                "rag_output": str(incident_dir / "analysis" / "rag_report.json"),
                "remediation_output": str(incident_dir / "analysis" / "remediation_plan.json"),
            }
            cmd = [part.format(incident_id=run["incident_id"], **output_map) for part in stage["command"]]
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(ROOT)}
            )
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                message = line.decode("utf-8", errors="replace").rstrip()
                item = {"type": "stage_log", "stage": stage["id"], "message": message, "at": timestamp()}
                run["events"].append(item)
                yield await emit(item)
            code = await proc.wait()
            event = {"type": "stage_completed" if code == 0 else "stage_failed",
                     "stage": stage["id"], "exit_code": code, "at": timestamp(),
                     "message": "Process exited successfully." if code == 0 else f"Process exited with code {code}."}
            run["events"].append(event)
            yield await emit(event)
            if code != 0:
                # Fail fast: don't imply downstream stages ran on valid outputs.
                run["status"] = "failed"
                run["current_stage"] = stage["id"]
                yield await emit({"type": "run_failed", "stage": stage["id"], "message": event["message"]})
                return
        run["status"] = "completed_with_pending_stages"
        run["current_stage"] = None
        yield await emit({"type": "run_finished", "status": run["status"], "at": timestamp()})
    return StreamingResponse(generator(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
