"""
Phase 12 tests: configuration context and deterministic planner-input preparation.

Synthetic data only (see tests/remediation_helpers.py); a mocked model; no network.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pytest

from rag.answer import GroundedAnswerResult
from remediation.configuration import (
    UNAVAILABLE,
    ConfigurationError,
    ConfigurationFactInput,
    build_configuration_context,
)
from remediation.inputs import (
    ANSWER_NOT_VALIDATED,
    NO_OBSERVED_EVIDENCE,
    NO_REFERENCE_MATERIAL,
    NO_RELEVANT_REFERENCE,
    RELEVANCE_NOT_ASSESSED,
    PlannerInputError,
    build_planner_input,
)
from remediation.prompt import build_user_prompt, model_view
from tests.remediation_helpers import (
    OFF_TOPIC,
    RELEVANT_A,
    RELEVANT_B,
    FakeLLM,
    answered,
    as_json,
    build_context,
    configuration,
    good_answer,
    make_input,
    reconstruction_with_configuration,
    three_reference_context,
)
from tests.rag_helpers import reconstruction_doc


def reason_codes(planner_input) -> list[str]:
    return [reason.code for reason in planner_input.insufficiency]


# ------------------------------------------------------------
# configuration context
# ------------------------------------------------------------

def test_configuration_comes_only_from_configured_edges_with_stable_ids():
    first = configuration()
    second = build_configuration_context(copy.deepcopy(reconstruction_with_configuration()))
    assert first.status == "available" and first == second
    assert first.fact_ids == ("CFG-1", "CFG-2")
    fact = first.facts[0]
    assert (fact.subject, fact.attribute, fact.basis) == ("container:test-api", "connects_to", "configured")
    assert fact.source_artifacts == ("docker/network_after_attack.json",)
    assert not first.conflicts


def test_edges_that_do_not_declare_themselves_configured_are_not_trusted():
    doc = reconstruction_with_configuration(
        {"source": "container:x", "relationship": "connects_to", "target": "ip:1.2.3.4", "basis": "observed"},
        {"source": "container:x", "relationship": "", "target": "ip:1.2.3.4", "basis": "configured"},
        "not even an object",
    )
    context = build_configuration_context(doc)
    assert context.fact_ids == ("CFG-1", "CFG-2")  # the three bad entries add nothing
    assert any("3 configured-relationship entries were skipped" in note for note in context.notes)


def test_missing_configuration_is_reported_as_unavailable_not_invented():
    for reconstruction in (None, reconstruction_doc(), {"graph_relationships": {}}, {"graph_relationships": {"configured_only": []}}):
        context = build_configuration_context(reconstruction)
        assert context.status == "unavailable" and context.facts == () and context.notes
    assert UNAVAILABLE.status == "unavailable"


def test_conflicting_single_valued_facts_are_withheld_and_reported():
    context = build_configuration_context(
        reconstruction_with_configuration(),
        (
            ConfigurationFactInput("container:test-api", "runs_as_user", "root", ("a.json",), single_valued=True),
            ConfigurationFactInput("container:test-api", "runs_as_user", "appuser", ("b.json",), single_valued=True),
        ),
    )
    assert context.status == "conflicting"
    assert len(context.conflicts) == 1
    assert set(context.conflicts[0].values) == {"root", "appuser"}
    conflict_ids = set(context.conflicting_fact_ids)
    assert len(conflict_ids) == 2
    assert conflict_ids.isdisjoint(context.fact_ids)       # neither side is offered as a fact
    assert len(context.facts) == 2                          # the unrelated Phase 8 facts remain


def test_multi_valued_relationships_are_never_treated_as_conflicts():
    doc = reconstruction_with_configuration(
        {"source": "container:test-api", "relationship": "connects_to", "target": "ip:10.0.0.9", "basis": "configured"}
    )
    context = build_configuration_context(doc)
    assert context.status == "available" and not context.conflicts


def test_identical_facts_merge_and_agreeing_single_valued_facts_do_not_conflict():
    same = (ConfigurationFactInput("container:test-api", "image", "app:1", single_valued=True),) * 2
    context = build_configuration_context(None, same)
    assert context.status == "available" and len(context.facts) == 1


@pytest.mark.parametrize("bad", [
    {"graph_relationships": "nope"},
    {"graph_relationships": {"configured_only": "nope"}},
    {"graph_relationships": {"ground_truth": {}}},
])
def test_malformed_configuration_input_fails_loudly(bad):
    with pytest.raises(ConfigurationError):
        build_configuration_context(bad)


def test_invalid_extra_facts_are_refused():
    with pytest.raises(ConfigurationError):
        ConfigurationFactInput("", "a", "b")
    with pytest.raises(ConfigurationError):
        build_configuration_context(None, ("not a fact input",))  # type: ignore[arg-type]


# ------------------------------------------------------------
# reference filtering by the validated Phase 11D relevance labels
# ------------------------------------------------------------

def test_only_passages_labelled_relevant_are_usable_and_not_relevant_is_excluded():
    planner_input = make_input()
    assert planner_input.ready and planner_input.relevance_source == "validated_phase_11d_answer"
    assert {ref.reference_id for ref in planner_input.usable_references} == {RELEVANT_A, RELEVANT_B}
    assert planner_input.excluded_references == ((OFF_TOPIC, "not_relevant"),)


def test_unclear_passages_are_excluded_too():
    planner_input = make_input(labels={OFF_TOPIC: "unclear"})
    assert (OFF_TOPIC, "unclear") in planner_input.excluded_references
    assert OFF_TOPIC not in {ref.reference_id for ref in planner_input.usable_references}


def test_the_model_never_sees_an_excluded_passage():
    planner_input = make_input()
    view = model_view(planner_input)
    shown = {p["reference_id"] for p in view["reference_passages"]}
    assert shown == {RELEVANT_A, RELEVANT_B} and view["withheld_reference_count"] == 1
    prompt = build_user_prompt(planner_input)
    assert OFF_TOPIC not in prompt and "xml parser entity expansion" not in prompt
    assert json.loads(prompt)["context"]["configuration"]["status"] == "available"


def test_usable_passages_keep_document_and_chunk_provenance():
    planner_input = make_input()
    ref = {r.reference_id: r for r in planner_input.usable_references}[RELEVANT_A]
    assert (ref.doc_id, ref.chunk_id, ref.reference_id) == ("cwe:CWE-1", RELEVANT_A, RELEVANT_A)
    assert ref.source == "cwe" and ref.source_version == "test-1" and len(ref.chunk_sha256) == 64
    assert ref.provenance["origin_kind"] == "bundled_fixture" and ref.basis == "general_reference_knowledge"


def test_hypotheses_stay_hypotheses_in_the_prepared_input():
    summary = make_input().summary()
    assert [(h.hypothesis_id, h.status, h.is_confirmed, h.basis) for h in summary.hypotheses] == [
        ("H1", "hypothesis_not_confirmed", False, "inferred_hypothesis")
    ]
    assert summary.observed_evidence_ids == ["application-4", "nginx-19"]
    assert OFF_TOPIC not in summary.usable_reference_ids  # references are never listed as evidence
    assert set(summary.observed_evidence_ids).isdisjoint(summary.usable_reference_ids)


# ------------------------------------------------------------
# sufficiency gate
# ------------------------------------------------------------

def test_no_answer_means_relevance_is_not_assessed_and_nothing_is_usable():
    planner_input = make_input(with_answer=False)
    assert not planner_input.ready and reason_codes(planner_input) == [RELEVANCE_NOT_ASSESSED]
    assert planner_input.usable_references == () and planner_input.relevance_source == "unavailable"
    assert {label for _, label in planner_input.excluded_references} == {"unassessed"}


@pytest.mark.parametrize("response, status", [
    ("not json", "rejected_invalid_response"),
    (RuntimeError("down"), "llm_error"),
])
def test_a_rejected_or_failed_answer_makes_the_input_insufficient(response, status):
    context = three_reference_context()
    from rag.answer import generate_grounded_answer

    result = generate_grounded_answer(context, FakeLLM(response))
    assert result.status == status
    planner_input = build_planner_input(context, result, configuration())
    assert reason_codes(planner_input) == [ANSWER_NOT_VALIDATED] and planner_input.usable_references == ()


def test_a_hand_built_answered_result_with_invalid_content_is_not_trusted():
    context = three_reference_context()
    answer = good_answer(context)
    answer["reference_relevance"] = [{"reference_id": r, "assessment": "relevant"} for r in context.reference_ids[:1]]  # incomplete
    forged = GroundedAnswerResult(context, "answered", "forged", answer=answer)
    planner_input = build_planner_input(context, forged, configuration())
    assert reason_codes(planner_input) == [ANSWER_NOT_VALIDATED] and planner_input.usable_references == ()


def test_an_answer_for_another_context_is_refused():
    other = build_context()
    with pytest.raises(PlannerInputError):
        build_planner_input(three_reference_context(), answered(other), configuration())


def test_all_passages_not_relevant_gives_no_relevant_reference():
    context = three_reference_context()
    answer = good_answer(context)
    # the Phase 11D validator forbids citing a not_relevant passage, so mark all and drop the citations
    answer["reference_relevance"] = [{"reference_id": r, "assessment": "not_relevant"} for r in context.reference_ids]
    answer["hypotheses"][0]["reference_ids"] = []
    answer["reference_guidance"] = []
    from rag.answer import generate_grounded_answer

    result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
    assert result.accepted, result.validation
    planner_input = build_planner_input(context, result, configuration())
    assert reason_codes(planner_input) == [NO_RELEVANT_REFERENCE] and not planner_input.ready
    assert planner_input.usable_references == ()


def test_empty_retrieval_gives_no_reference_material():
    from tests.retrieval_helpers import make_doc, make_store

    store = make_store(make_doc("CWE-9", "Unrelated", "completely different vocabulary about gardening"))
    try:
        context = build_context(store)
    finally:
        store.close()
    assert context.references == ()
    planner_input = build_planner_input(context, None, configuration())
    assert reason_codes(planner_input) == [NO_REFERENCE_MATERIAL]


def test_missing_incident_evidence_is_insufficient():
    context = build_context(reconstruction=None)
    assert context.observed_evidence == ()
    planner_input = build_planner_input(context, None, configuration())
    assert NO_OBSERVED_EVIDENCE in reason_codes(planner_input) and not planner_input.ready


def test_missing_or_conflicting_configuration_does_not_block_planning():
    for config in (UNAVAILABLE, build_configuration_context(None, (
        ConfigurationFactInput("c", "image", "a", single_valued=True),
        ConfigurationFactInput("c", "image", "b", single_valued=True),
    ))):
        planner_input = make_input(config=config)
        assert planner_input.ready
        assert planner_input.summary().configuration_status == config.status


def test_wrong_argument_types_are_refused():
    with pytest.raises(PlannerInputError):
        build_planner_input("context", None)  # type: ignore[arg-type]
    with pytest.raises(PlannerInputError):
        build_planner_input(three_reference_context(), "answer")  # type: ignore[arg-type]
    with pytest.raises(PlannerInputError):
        build_planner_input(three_reference_context(), None, "config")  # type: ignore[arg-type]


def test_planner_input_is_immutable():
    planner_input = make_input()
    with pytest.raises(dataclasses.FrozenInstanceError):
        planner_input.insufficiency = ()  # type: ignore[misc]
