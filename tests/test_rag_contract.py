"""
Phase 11D contract tests: the ids the PROMPT shows, the ids the VALIDATOR accepts and what a model that gets them
wrong looks like. Mocked answers only (no network).

Why they exist: a live answer was rejected with 18 `unknown_reference_id`, 10 `missing_assessment` and
1 `unsupported_identifier`. These tests pin the contract (no mismatch was found between prompt, context and
validator), reproduce those failure CATEGORIES deterministically, and show that validation was not loosened.
They do not show why the live model produced those ids; that cause is unconfirmed.
"""

from __future__ import annotations

import collections
import json

import pytest

from rag.citations import validate_answer
from rag.prompt import RELEVANCE_ASSESSMENTS, SYSTEM_PROMPT, answer_json_schema, build_user_prompt
from rag.report import run_incident_rag
from remediation.prompt import model_view
from tests.rag_helpers import (
    FakeLLM,
    as_json,
    build_context,
    good_answer,
    root_cause_doc,
    synthetic_store,
)
from tests.remediation_helpers import make_input, reconstruction_with_configuration


@pytest.fixture()
def context():
    return build_context()


def codes(answer, context) -> collections.Counter:
    return collections.Counter(validate_answer(answer, context).codes)


def mangled(answer: dict, transform) -> dict:
    """The same answer with every reference id passed through `transform`."""
    out = json.loads(json.dumps(answer))
    for entry in out["reference_relevance"]:
        entry["reference_id"] = transform(entry["reference_id"])
    for hypothesis in out["hypotheses"]:
        hypothesis["reference_ids"] = [transform(r) for r in hypothesis["reference_ids"]]
    for item in out["reference_guidance"]:
        item["reference_ids"] = [transform(r) for r in item["reference_ids"]]
    return out


# ------------------------------------------------------------
# the ids shown in the prompt are exactly the ids the validator accepts
# ------------------------------------------------------------

def test_the_prompt_lists_exactly_the_reference_ids_the_validator_accepts(context):
    view = json.loads(build_user_prompt(context))["context"]
    shown = [p["reference_id"] for p in view["reference_passages"]]
    assert shown == view["reference_ids_to_assess"] == list(context.reference_ids) and len(shown) >= 2
    for reference_id in shown:                                   # each one is accepted as cited and as assessed
        answer = good_answer(context)
        answer["reference_relevance"] = [{"reference_id": r, "assessment": "relevant"} for r in shown]
        answer["hypotheses"][0]["reference_ids"] = [reference_id]
        answer["reference_guidance"] = [{"statement": "A general statement.", "reference_ids": [reference_id]}]
        assert validate_answer(answer, context).passed, reference_id


def test_the_other_id_like_passage_fields_are_not_accepted_as_reference_ids(context):
    view = json.loads(build_user_prompt(context))["context"]
    for passage in view["reference_passages"]:
        assert passage["document_id"] != passage["reference_id"] and passage["source_id"] != passage["reference_id"]
        assert passage["reference_id"].startswith(passage["document_id"] + "#")
    base = good_answer(context)
    by_id = {r.reference_id: r for r in context.references}
    for label, transform in (
        ("document_id", lambda ref: by_id[ref].doc_id),
        ("source_id", lambda ref: by_id[ref].source_id),
        ("id without the #NNNN suffix", lambda ref: ref.split("#")[0]),
    ):
        result = validate_answer(mangled(base, transform), context)
        assert not result.passed, label
        assert "unknown_reference_id" in result.codes and "missing_assessment" in result.codes, label


# ------------------------------------------------------------
# the live failure categories, reproduced with mocked answers
# ------------------------------------------------------------

def test_wrong_reference_ids_produce_exactly_the_live_failure_shape(context):
    base = good_answer(context)
    passages = len(context.references)
    hypothesis_refs = sum(len(h["reference_ids"]) for h in base["hypotheses"])
    guidance_refs = sum(len(g["reference_ids"]) for g in base["reference_guidance"])
    result = validate_answer(mangled(base, lambda ref: ref.split("#")[0]), context)
    counted = collections.Counter(result.codes)
    # one unknown id per assessment entry, one per citation, and every passage left unassessed
    assert counted["unknown_reference_id"] == passages + hypothesis_refs + guidance_refs
    assert counted["missing_assessment"] == passages
    # the order of the codes: relevance problems first, then the citation problems
    order = list(result.codes)
    assert order[:passages] == ["unknown_reference_id"] * passages
    assert order[passages:2 * passages] == ["missing_assessment"] * passages


def test_a_statement_that_names_an_identifier_it_cannot_support_is_still_flagged(context):
    answer = mangled(good_answer(context), lambda ref: ref.split("#")[0])   # the citations no longer resolve
    answer["reference_guidance"][0]["statement"] = "CWE-1 describes building a query from request text."
    assert "unsupported_identifier" in codes(answer, context)


# ------------------------------------------------------------
# validation is not weakened
# ------------------------------------------------------------

def test_every_passage_must_be_assessed_and_each_omission_is_reported(context):
    answer = good_answer(context)
    answer["reference_relevance"] = answer["reference_relevance"][:-1]
    answer["hypotheses"][0]["reference_ids"] = answer["hypotheses"][0]["reference_ids"][:1]
    counted = codes(answer, context)
    assert counted["missing_assessment"] == 1 and sum(counted.values()) == 1
    answer["reference_relevance"] = []
    assert codes(answer, context)["missing_assessment"] == len(context.references)


def test_an_invented_reference_id_is_rejected_even_when_everything_else_is_valid(context):
    answer = good_answer(context)
    answer["reference_guidance"][0]["reference_ids"] = ["cwe:CWE-999#0000"]
    assert codes(answer, context)["unknown_reference_id"] == 1


def test_an_assessment_twice_or_with_a_bad_label_is_still_rejected(context):
    answer = good_answer(context)
    answer["reference_relevance"].append(dict(answer["reference_relevance"][0]))
    assert codes(answer, context)["duplicate_assessment"] == 1
    answer = good_answer(context)
    answer["reference_relevance"][0]["assessment"] = "maybe"
    assert "invalid_value" in codes(answer, context)
    answer = good_answer(context)
    answer["reference_relevance"][0]["extra"] = "x"
    assert "unknown_key" in codes(answer, context)


def test_a_fully_assessed_answer_with_every_permitted_label_passes(context):
    assert validate_answer(good_answer(context), context).passed
    answer = good_answer(context)
    labels = [RELEVANCE_ASSESSMENTS[(i + 1) % 3] for i in range(len(context.references))]
    for entry, label in zip(answer["reference_relevance"], labels):
        entry["assessment"] = label
    answer["reference_guidance"] = []
    answer["hypotheses"][0]["reference_ids"] = []
    assert validate_answer(answer, context).passed


def test_the_schema_and_the_permitted_labels_are_unchanged():
    assert RELEVANCE_ASSESSMENTS == ("relevant", "not_relevant", "unclear")
    schema = answer_json_schema()
    item = schema["properties"]["reference_relevance"]["items"]
    assert item["required"] == ["reference_id", "assessment"] and item["additionalProperties"] is False
    assert item["properties"]["assessment"]["enum"] == list(RELEVANCE_ASSESSMENTS)
    assert "reference_ids_to_assess" not in schema["properties"]      # the list is input, not part of the answer


# ------------------------------------------------------------
# the prompt states the requirements unambiguously
# ------------------------------------------------------------

def test_the_system_prompt_states_the_identifier_and_assessment_requirements():
    for phrase in (
        "EXACTLY, character for character",
        "Never shorten, reconstruct",
        "document_id and source_id",
        "are NOT",
        "reference_ids_to_assess",
        'ends\n  with "#" and four digits',
        "Never omit a passage",
        "exactly the keys reference_id and assessment",
        "exactly these keys and no others",
        "Never mix them",
    ):
        assert phrase in SYSTEM_PROMPT, phrase


def test_the_prompt_contains_no_concrete_example_id_that_a_model_could_copy(context):
    for reference in context.references:
        assert reference.reference_id not in SYSTEM_PROMPT


def test_the_planner_never_sees_the_phase_11d_assessment_list():
    """The planner must not learn the ids of excluded passages, so the 11D-only key is not forwarded."""
    planner_input = make_input()
    view = model_view(planner_input)
    assert "reference_ids_to_assess" not in view
    assert {p["reference_id"] for p in view["reference_passages"]} == {r.reference_id for r in planner_input.usable_references}


# ------------------------------------------------------------
# the standalone Phase 11D report already keeps the rejected answer
# ------------------------------------------------------------

def test_the_phase_11d_report_keeps_a_rejected_answer_and_leaves_accepted_ones_compatible(tmp_path):
    incident = tmp_path / "incident_001"
    (incident / "analysis").mkdir(parents=True)
    (incident / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc()), encoding="utf-8")
    (incident / "analysis" / "incident_reconstruction.json").write_text(
        json.dumps(reconstruction_with_configuration()), encoding="utf-8")
    context = build_context()
    bad = as_json(mangled(good_answer(context), lambda ref: ref.split("#")[0]))
    with synthetic_store() as store:
        rejected = run_incident_rag(incident, store, FakeLLM(bad))
        accepted = run_incident_rag(incident, store, FakeLLM(as_json(good_answer(context))))
    (entry,) = rejected.to_dict()["answers"]
    assert entry["status"] == "rejected_citations" and entry["answer"] is None and entry["raw_response"] == bad
    (ok,) = accepted.to_dict()["answers"]
    assert ok["status"] == "answered" and ok["answer"] == good_answer(context)
    assert set(ok) == set(entry) == {"context", "status", "message", "answer", "validation", "raw_response", "llm"}
