"""
Shared helpers for the Phase 11D grounded-answer tests (not a test module).

Everything here is SYNTHETIC test data: the Phase 8/9 dictionaries copy the SHAPE
of `incident_001/analysis/*.json` (field names, placeholder strings, evidence IDs)
and the knowledge-base documents are invented text with made-up identifiers
(`CWE-1`, `CAPEC-1`). Nothing here is security knowledge or a real corpus, and
no language model is called: `FakeLLM` returns whatever the test scripts.
"""

from __future__ import annotations

import copy
import json

from llm.base import LLMClient
from rag.context import GroundedContext
from rag.pipeline import build_contexts_from_analysis
from tests.retrieval_helpers import make_doc, make_store

INCIDENT = "incident_001"

HYPOTHESIS = {
    "status": "hypothesis_not_confirmed",
    "root_cause": "Not confirmed from the available evidence.",
    "affected_component": "Undetermined from request evidence alone.",
    "vulnerability_type": "potential_sql_injection (CWE-89 hypothesis)",
    "supporting_evidence_ids": ["application-4", "nginx-19"],
    "confidence": 0.55,
    "rationale": (
        "One or more normalized events contain SQL-injection-like request syntax. This supports an "
        "injection-attempt hypothesis but does not establish vulnerable server-side query construction, "
        "successful exploitation, or database access."
    ),
    "evidence_needed_to_confirm": [
        "application source or query-construction evidence showing unsafe SQL handling",
        "correlated application/database logs demonstrating the query was executed",
    ],
}


def root_cause_doc(*findings, incident_id: str = INCIDENT) -> dict:
    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "agent": "phase_9_root_cause",
        "findings": [copy.deepcopy(f) for f in (findings or (HYPOTHESIS,))],
        "validation": {"ground_truth_used": False},
    }


def reconstruction_doc(incident_id: str = INCIDENT) -> dict:
    def request(path: str, upstream: str | None) -> dict:
        return {
            "method": "GET",
            "path": path,
            "query": "message=%27%20OR%20%271%27%3D%271",
            "status": 200,
            "upstream": upstream,
            "upstream_status": "200" if upstream else None,
            "request_id": None,
        }

    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "timeline": [
            {
                "timestamp": "2026-10-09T11:07:46.240160Z",
                "event_id": "application-4",
                "source": "application",
                "event_type": "http_request",
                "action": "request",
                "request_context": request("/search", None),
            },
            {
                "timestamp": "2026-10-09T11:07:46.241152Z",
                "event_id": "nginx-19",
                "source": "nginx",
                "event_type": "http_request",
                "action": "request",
                "request_context": request("/api/search", "172.19.0.6:8000"),
            },
        ],
        "claims": [
            {
                "claim_id": "claim_001",
                "statement": "Event application-4 contains SQL-injection-like syntax in /search.",
                "evidence_ids": ["application-4"],
                "request_context": request("/search", None),
                "limitation": "A request pattern alone does not prove successful exploitation.",
            },
            {
                "claim_id": "claim_002",
                "statement": "Event nginx-19 contains SQL-injection-like syntax in /api/search.",
                "evidence_ids": ["nginx-19"],
                "request_context": request("/api/search", "172.19.0.6:8000"),
                "limitation": "A request pattern alone does not prove successful exploitation.",
            },
        ],
    }


def synthetic_documents() -> list:
    return [
        make_doc(
            "CWE-1",
            "Query Weakness",
            "sql injection: the application builds a database query from request text without "
            "neutralizing special characters, so an attacker can change the query.",
        ),
        make_doc(
            "CAPEC-1",
            "Injection Attack",
            "an attacker sends sql injection syntax in a request to alter a database query; see CWE-1.",
            source="capec",
            doc_type="attack_pattern",
        ),
        make_doc("CWE-2", "Filesystem Weakness", "filesystem path escapes the allowed folder"),
    ]


def synthetic_store():
    return make_store(*synthetic_documents())


def build_context(store=None, *, root_cause=None, reconstruction="default", **kwargs) -> GroundedContext:
    """One context for the hypothesis finding on a synthetic store (caller closes the store)."""
    own = store is None
    store = store or synthetic_store()
    try:
        rc = root_cause or root_cause_doc()
        rec = reconstruction_doc() if reconstruction == "default" else reconstruction
        built = build_contexts_from_analysis(rc, rec, store, **kwargs)
        assert len(built.contexts) == 1
        return built.contexts[0]
    finally:
        if own:
            store.close()


class FakeLLM(LLMClient):
    """Returns scripted responses in order (the last repeats) and records every call."""

    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.model = "fake-model"

    def generate(self, system_prompt, user_prompt, response_format=None) -> str:
        self.calls.append(
            {"system": system_prompt, "user": user_prompt, "response_format": response_format}
        )
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response


def good_answer(context: GroundedContext) -> dict:
    """A fully valid answer for `context` (uses its first reference, if any)."""
    refs = context.reference_ids
    answer = {
        "schema_version": "1.0",
        "incident_id": context.incident_id,
        "observed_evidence": [
            {
                "statement": "GET /search with status 200 was recorded.",
                "evidence_ids": ["application-4"],
            }
        ],
        "hypotheses": [
            {
                "hypothesis_id": "H1",
                "statement": "The requests may be an SQL injection attempt; this is not confirmed.",
                "status": "hypothesis_only",
                "evidence_ids": ["application-4", "nginx-19"],
                "reference_ids": list(refs[:1]),
            }
        ],
        "reference_guidance": [],
        "reference_relevance": [{"reference_id": ref, "assessment": "relevant"} for ref in refs],
        "evidence_gaps": ["No application or database log shows the query being executed."],
        "limitations": ["Reference passages are general knowledge, not proof about this incident."],
    }
    if refs:
        answer["reference_guidance"] = [
            {
                "statement": "Building a database query from request text without neutralizing it is a weakness.",
                "reference_ids": [refs[0]],
            }
        ]
    return answer


def as_json(answer: dict) -> str:
    return json.dumps(answer)
