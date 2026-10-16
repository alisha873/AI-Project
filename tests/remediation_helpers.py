"""
Shared helpers for the Phase 12 remediation-planner tests (not a test module).

Everything here is SYNTHETIC test data. It reuses the Phase 11D helpers (`tests.rag_helpers`): the Phase 8/9
dictionaries copy the SHAPE of `incident_001/analysis/*.json` and the knowledge-base documents are invented
text with made-up identifiers (`CWE-1`, `CAPEC-1`, `CWE-3`). The configuration facts are invented too.
No language model is called: `FakeLLM` returns whatever a test scripts, and nothing touches a network.
"""

from __future__ import annotations

import copy
import json

from rag.answer import GroundedAnswerResult, generate_grounded_answer
from rag.context import GroundedContext
from remediation.configuration import ConfigurationContext, build_configuration_context
from remediation.inputs import PlannerInput, build_planner_input
from tests.rag_helpers import (
    FakeLLM,
    as_json,
    build_context,
    good_answer,
    reconstruction_doc,
    synthetic_documents,
    synthetic_store,
)
from tests.retrieval_helpers import make_doc, make_store

__all__ = [
    "FakeLLM", "as_json", "build_context", "good_answer", "reconstruction_doc", "synthetic_store",
    "RELEVANT_A", "RELEVANT_B", "OFF_TOPIC", "three_reference_context", "answered", "make_input",
    "reconstruction_with_configuration", "configuration", "good_plan", "good_proposal", "plan_json",
]

RELEVANT_A = "cwe:CWE-1#0000"
RELEVANT_B = "capec:CAPEC-1#0000"
OFF_TOPIC = "cwe:CWE-3#0000"


def reconstruction_with_configuration(*extra_edges: dict) -> dict:
    """The synthetic reconstruction plus configured-only relationships in the Phase 8 shape."""
    doc = reconstruction_doc()
    doc["graph_relationships"] = {
        "observed": [],
        "configured_only": [
            {
                "source": "container:test-api",
                "source_name": "test-api",
                "target": "ip:172.19.0.6",
                "target_name": "172.19.0.6",
                "relationship": "connects_to",
                "basis": "configured",
                "evidence_ids": ["docker-container-network-1-test_net"],
                "source_artifacts": ["docker/network_after_attack.json"],
                "attributes": {"relationship_basis": "docker_network_membership"},
            },
            {
                "source": "container:test-api",
                "source_name": "test-api",
                "target": "process:test-api:uvicorn",
                "target_name": "uvicorn",
                "relationship": "executes",
                "basis": "configured",
                "evidence_ids": [],
                "source_artifacts": ["docker/containers_after_attack.json"],
                "attributes": {"relationship_basis": "configured_startup"},
            },
            *extra_edges,
        ],
    }
    return doc


def configuration() -> ConfigurationContext:
    return build_configuration_context(reconstruction_with_configuration())


def three_reference_context() -> GroundedContext:
    """A context with two on-topic synthetic passages and one off-topic one that still matches the query."""
    off_topic = make_doc(
        "CWE-3",
        "Parser Weakness",
        "xml parser entity expansion exhausts memory; sql injection is mentioned here only in passing.",
    )
    store = make_store(*synthetic_documents(), off_topic)
    try:
        context = build_context(store)
    finally:
        store.close()
    assert set(context.reference_ids) == {RELEVANT_A, RELEVANT_B, OFF_TOPIC}, context.reference_ids
    return context


def answered(context: GroundedContext, labels: dict[str, str] | None = None) -> GroundedAnswerResult:
    """A real accepted Phase 11D result, produced through `generate_grounded_answer` with a fake model.

    `labels` overrides the relevance labels (default: every passage relevant). Passages the answer cites are
    always relevant; override only the others.
    """
    answer = good_answer(context)
    if labels:
        answer = copy.deepcopy(answer)
        for entry in answer["reference_relevance"]:
            entry["assessment"] = labels.get(entry["reference_id"], entry["assessment"])
    result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
    assert result.accepted, result.validation
    return result


def make_input(
    context: GroundedContext | None = None,
    labels: dict[str, str] | None = None,
    config: ConfigurationContext | None = None,
    *,
    with_answer: bool = True,
) -> PlannerInput:
    context = context or three_reference_context()
    result = answered(context, labels or {OFF_TOPIC: "not_relevant"}) if with_answer else None
    return build_planner_input(context, result, config if config is not None else configuration())


def good_proposal(**overrides) -> dict:
    """One valid proposal (incident-linked, cites the relevant on-topic passage CWE-1)."""
    proposal = {
        "title": "Neutralize request text before it reaches the database query",
        "description": "Change how the search handler builds its database query so that request text can not change the query structure.",
        "problem_addressed": "Requests to /search and /api/search contain SQL-injection-like syntax; whether the query handling is unsafe is not confirmed.",
        "affected_area": {"state": "identified", "description": "The search request path /search seen in event application-4."},
        "observed_evidence_ids": ["application-4", "nginx-19"],
        "hypothesis_ids": ["H1"],
        "configuration_fact_ids": [],
        "reference_ids": [RELEVANT_A],
        "rationale": "The events show requests with injection-like syntax and the cited passage describes building queries from request text as a weakness; this supports reviewing the query handling, not a conclusion that it is vulnerable.",
        "reference_support_limits": "The passage is general knowledge about a weakness class; it does not show that this application builds queries unsafely.",
        "severity": {"level": "high", "rationale": "Injection weaknesses can expose data if present; presence here is unconfirmed."},
        "expected_security_effect": "If the query handling is unsafe, this change is intended to reduce the chance that request text alters a query.",
        "operational_side_effects": ["Searches containing unusual characters may behave differently and must be regression tested."],
        "implementation_steps": [
            {"text": "Build database queries so that request text can not change their structure.", "origin": "cited_reference"},
            {"text": "Review the handler code for other places where request text reaches a query.", "origin": "general_practice"},
        ],
        "configuration_dependent": False,
        "disruption": "medium",
        "reversibility": "reversible",
        "preconditions": ["A human reviewer approves the change and a rollback path exists."],
        "dependencies": ["Access to the application source, which the supplied evidence does not include."],
        "unresolved_questions": ["How does the handler build its query today?"],
        "validation": {
            "procedure": "In the isolated lab, replay the original search requests after the change and compare the responses.",
            "expected_result": "The injection-like requests no longer alter query behaviour; this would hold only for the requests tested.",
        },
        "confidence": {"level": "medium", "rationale": "Two events support the hypothesis and one passage states the mitigation; the hypothesis is unconfirmed."},
    }
    proposal.update(overrides)
    return proposal


def _second_proposal() -> dict:
    return good_proposal(
        title="Add input validation for the search parameter",
        description="Reject or normalize unexpected characters in the search parameter before processing.",
        problem_addressed="The search parameter in the observed requests carries unusual characters.",
        affected_area={"state": "identified", "description": "The search endpoint /api/search seen in event nginx-19."},
        observed_evidence_ids=["nginx-19"],
        severity={"level": "medium", "rationale": "Validation reduces exposure only as a secondary layer."},
        disruption="low",
        preconditions=[],
        implementation_steps=[{"text": "Validate the search parameter against an allow-list of expected characters.", "origin": "general_practice"}],
        confidence={"level": "low", "rationale": "The step is general practice, not stated by the cited passage."},
        validation={"procedure": "Replay a small set of valid and invalid search values in the isolated lab.", "expected_result": "Invalid values are rejected and valid values still work."},
    )


def _third_proposal() -> dict:
    return good_proposal(
        title="Review logging so query execution can be correlated with requests",
        description="Make sure application and database logs can later show whether a query was executed for a given request.",
        problem_addressed="The current evidence cannot show whether a query was executed.",
        affected_area={"state": "unknown", "description": "The logging components are not described by the supplied material."},
        observed_evidence_ids=[],
        hypothesis_ids=[],
        reference_ids=[RELEVANT_B],
        severity={"level": "low", "rationale": "Improves later investigation; does not itself reduce exposure."},
        disruption="low",
        preconditions=[],
        implementation_steps=[{"text": "Record a request identifier in application and database logs.", "origin": "general_practice"}],
        configuration_dependent=True,
        confidence={"level": "low", "rationale": "General practice only."},
        validation={"procedure": "Generate a test request in the lab and trace it through the logs.", "expected_result": "The request can be matched across logs."},
    )


def good_plan(planner_input: PlannerInput | None = None, *proposals: dict) -> dict:
    """A valid plan answer with three proposals returned in a deliberately non-priority order."""
    incident = (planner_input or make_input()).context.incident_id
    return {
        "schema_version": "1.0",
        "incident_id": incident,
        "outcome": "proposals",
        "insufficient_reason": "",
        "proposals": list(proposals) or [_third_proposal(), good_proposal(), _second_proposal()],
        "unresolved_questions": ["Which component builds the query is not known."],
        "limitations": ["Proposals rest on an unconfirmed hypothesis and a small reference corpus."],
    }


def plan_json(plan: dict) -> str:
    return json.dumps(plan)
