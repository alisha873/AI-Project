"""
Tests for Phase 11D steps 4-5: grounded answer generation with MOCKED language-model responses.

No real model, no network. `FakeLLM` (tests/rag_helpers.py) returns scripted text, so these tests show
how the pipeline reacts to good, empty, invalid and unsupported responses; they say nothing about
how any real model behaves.
"""

from __future__ import annotations

import copy
import json

import pytest

from rag.answer import describe_llm, generate_grounded_answer
from rag.citations import validate_answer
from rag.context import ContextLimits
from rag.pipeline import build_contexts_for_incident
from rag.prompt import (
    ANSWER_KEYS,
    SYSTEM_PROMPT,
    answer_json_schema,
    build_user_prompt,
    context_for_prompt,
)
from rag.report import REPORT_CAVEATS, run_incident_rag
from tests.rag_helpers import (
    FakeLLM,
    as_json,
    build_context,
    good_answer,
    reconstruction_doc,
    root_cause_doc,
    synthetic_store,
)
from tests.retrieval_helpers import INCIDENT_001, make_doc, make_store

from tests.test_rag_context import needs_real_data, real_store  # noqa: F401  (fixture re-use)


@pytest.fixture()
def context():
    return build_context()


def empty_retrieval_context():
    with make_store(make_doc("CWE-9", "Unrelated", "zebra giraffe savanna")) as store:
        return build_context(store)


# ------------------------------------------------------------------ the happy path
class TestAnswered:
    def test_a_valid_mocked_answer_is_accepted(self, context):
        llm = FakeLLM(as_json(good_answer(context)))
        result = generate_grounded_answer(context, llm)
        assert result.status == "answered" and result.accepted
        assert result.answer == good_answer(context) and result.validation.passed
        assert result.llm == "FakeLLM(fake-model)" == describe_llm(llm)
        assert len(llm.calls) == 1

    def test_the_client_receives_the_system_prompt_user_prompt_and_schema(self, context):
        llm = FakeLLM(as_json(good_answer(context)))
        generate_grounded_answer(context, llm)
        (call,) = llm.calls
        assert call["system"] == SYSTEM_PROMPT
        assert call["user"] == build_user_prompt(context)
        assert call["response_format"] == answer_json_schema()

    def test_the_user_prompt_separates_the_three_kinds_and_keeps_retrieved_text_verbatim(self, context):
        message = json.loads(build_user_prompt(context))["context"]
        assert set(message) == {"incident_id", "observed_evidence", "unresolved_evidence_ids",
                                "hypotheses", "reference_ids_to_assess", "reference_passages", "retrieval"}
        assert message["reference_ids_to_assess"] == list(context.reference_ids) != []
        assert message["hypotheses"][0]["is_confirmed"] is False
        assert message["hypotheses"][0]["status"] == "hypothesis_not_confirmed"
        assert {e["evidence_id"] for e in message["observed_evidence"]} == {"application-4", "nginx-19"}
        for passage, ref in zip(message["reference_passages"], context.references):
            assert passage["reference_id"] == ref.chunk_id and passage["text"] == ref.text
            assert passage["document_id"] == ref.doc_id and passage["source_id"] == ref.source_id
        assert message["retrieval"]["method"] == "fts5-bm25-v1" and message["retrieval"]["keyword_search"] is True

    def test_the_prompt_is_deterministic_and_contains_no_answer_key_or_narrative(self, context):
        assert build_user_prompt(context) == build_user_prompt(build_context())
        text = (SYSTEM_PROMPT + build_user_prompt(context)).lower()
        assert "ground_truth" not in text and "narrative" not in text and "benchmark" not in text

    def test_the_system_prompt_states_the_grounding_rules(self):
        for phrase in ("untrusted data", "observed_evidence", "hypotheses", "reference_guidance",
                       "not_relevant", "Do not fill the gap with outside knowledge",
                       "never call it confirmed", "Never invent",
                       'assessment for it is exactly "relevant"',
                       "supported by the retrieved text of the relevant passages",
                       'use the words "does not establish" or',
                       'Do not rewrite that denial as "without establishing"'):
            assert phrase in SYSTEM_PROMPT, phrase

    def test_a_model_may_mark_everything_irrelevant_and_give_no_guidance(self, context):
        answer = good_answer(context)
        for entry in answer["reference_relevance"]:
            entry["assessment"] = "not_relevant"
        answer["reference_guidance"] = []
        answer["hypotheses"][0]["reference_ids"] = []
        answer["limitations"] = ["None of the retrieved passages addresses this incident."]
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "answered" and result.answer["reference_guidance"] == []

    def test_insufficient_evidence_hypothesis_without_citations_is_a_valid_answer(self, context):
        answer = good_answer(context)
        answer["hypotheses"][0].update(status="insufficient_evidence", evidence_ids=[], reference_ids=[])
        answer["observed_evidence"] = []
        assert generate_grounded_answer(context, FakeLLM(as_json(answer))).status == "answered"


# ------------------------------------------------------------------ no support: the model is not called
class TestNothingToGroundOn:
    def test_empty_retrieval_never_calls_the_model_and_invents_nothing(self):
        context = empty_retrieval_context()
        llm = FakeLLM(as_json(good_answer(context)))
        result = generate_grounded_answer(context, llm)
        assert result.status == "not_generated_no_reference_material"
        assert result.answer is None and result.validation is None and llm.calls == []
        assert "no support was invented" in result.message
        assert result.context.references == () and result.context.observed_evidence  # incident side intact

    def test_missing_incident_evidence_never_calls_the_model(self):
        context = build_context(reconstruction=None)
        assert context.references and context.observed_evidence == ()
        llm = FakeLLM(as_json(good_answer(context)))
        result = generate_grounded_answer(context, llm)
        assert result.status == "not_generated_no_incident_evidence" and llm.calls == [] and result.answer is None

    def test_a_model_that_ignores_empty_retrieval_would_be_rejected_by_validation(self):
        context = empty_retrieval_context()
        answer = good_answer(context)
        answer["reference_guidance"] = [{"statement": "Use parameterized queries.", "reference_ids": ["cwe:CWE-89#0000"]}]
        validation = validate_answer(answer, context)
        assert not validation.passed and {"no_reference_material", "unknown_reference_id"} <= set(validation.codes)


# ------------------------------------------------------------------ rejected responses
class TestRejected:
    @pytest.mark.parametrize("response", ["", "   ", "not json", "[1, 2]", '"text"', "```json\n{}\n```",
                                          "Here is the answer: {}"])
    def test_a_response_that_is_not_a_json_object(self, context, response):
        result = generate_grounded_answer(context, FakeLLM(response))
        assert result.status == "rejected_invalid_response" and result.answer is None and not result.accepted

    def test_fabricated_citations_are_rejected_and_listed(self, context):
        answer = good_answer(context)
        answer["observed_evidence"][0]["evidence_ids"] = ["nginx-404"]
        answer["reference_guidance"][0]["reference_ids"] = ["cwe:CWE-79#0000"]
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "rejected_citations" and result.answer is None
        assert {"unknown_evidence_id", "unknown_reference_id"} <= set(result.validation.codes)
        assert result.raw_response == as_json(answer) and not result.accepted

    def test_an_unsupported_success_claim_is_rejected(self, context):
        answer = good_answer(context)
        answer["observed_evidence"][0]["statement"] = "The attacker successfully exploited GET /search."
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "rejected_citations" and "high_impact_claim" in result.validation.codes

    def test_presenting_the_hypothesis_as_confirmed_is_rejected(self, context):
        answer = good_answer(context)
        answer["hypotheses"][0]["statement"] = "The SQL injection vulnerability is confirmed."
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "rejected_citations" and "confirmation_claim" in result.validation.codes

    def test_an_answer_citing_another_incident_is_rejected(self, context):
        answer = good_answer(context)
        answer["incident_id"] = "incident_002"
        assert generate_grounded_answer(context, FakeLLM(as_json(answer))).status == "rejected_citations"

    def test_there_is_no_silent_retry_or_repair(self, context):
        llm = FakeLLM("not json", as_json(good_answer(context)))
        result = generate_grounded_answer(context, llm)
        assert result.status == "rejected_invalid_response" and len(llm.calls) == 1

    def test_guidance_that_cites_a_not_relevant_reference_is_rejected(self, context):
        answer = good_answer(context)
        cited = answer["reference_guidance"][0]["reference_ids"][0]
        for entry in answer["reference_relevance"]:
            if entry["reference_id"] == cited:
                entry["assessment"] = "not_relevant"
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "rejected_citations" and result.answer is None
        assert result.validation is not None and not result.validation.passed
        assert "not_relevant_reference_cited" in result.validation.codes
        assert any(cited in issue.message for issue in result.validation.issues)

    def test_a_valid_answer_cites_only_relevant_references(self, context):
        answer = good_answer(context)
        relevant = {
            entry["reference_id"]
            for entry in answer["reference_relevance"]
            if entry["assessment"] == "relevant"
        }
        cited = {ref for item in answer["reference_guidance"] for ref in item["reference_ids"]}
        assert cited and cited <= relevant
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "answered" and result.validation.passed and result.validation.issues == ()
        assert result.answer["reference_guidance"][0]["reference_ids"] == answer["reference_guidance"][0]["reference_ids"]

    def test_an_invalid_answer_stays_rejected_when_a_later_response_would_pass(self, context):
        invalid = good_answer(context)
        cited = invalid["reference_guidance"][0]["reference_ids"][0]
        for entry in invalid["reference_relevance"]:
            if entry["reference_id"] == cited:
                entry["assessment"] = "not_relevant"
        llm = FakeLLM(as_json(invalid), as_json(good_answer(context)))
        result = generate_grounded_answer(context, llm)
        assert len(llm.calls) == 1
        assert result.status == "rejected_citations" and result.answer is None
        assert result.validation is not None and not result.validation.passed
        assert "not_relevant_reference_cited" in result.validation.codes

    def test_a_provider_failure_is_reported_with_a_short_message_only(self, context):
        secret = "Authorization: Bearer sk-SECRET " + "x" * 2000
        result = generate_grounded_answer(context, FakeLLM(RuntimeError(secret)))
        assert result.status == "llm_error" and result.answer is None and result.raw_response is None
        assert "RuntimeError" in result.message and len(result.message) < 500

    def test_wrong_argument_types_raise(self, context):
        with pytest.raises(TypeError):
            generate_grounded_answer("context", FakeLLM("{}"))
        with pytest.raises(TypeError):
            generate_grounded_answer(context, object())


# ------------------------------------------------------------------ schema
def _objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


class TestResponseSchema:
    def test_top_level_keys_match_the_validated_keys(self):
        assert tuple(answer_json_schema()["properties"]) == ANSWER_KEYS

    def test_every_object_is_closed_with_all_properties_required_and_no_refs(self):
        schema = answer_json_schema()
        assert "$ref" not in json.dumps(schema) and "$defs" not in json.dumps(schema)
        for obj in _objects(schema):
            assert obj["additionalProperties"] is False and obj["required"] == list(obj["properties"])

    def test_the_groq_client_strict_conversion_leaves_it_unchanged(self):
        groq_module = pytest.importorskip("llm.groq")
        schema = answer_json_schema()
        assert groq_module.GroqLLMClient._make_strict_schema(copy.deepcopy(schema)) == schema

    def test_a_valid_answer_conforms_to_the_schema_shape(self, context):
        answer = good_answer(context)
        assert tuple(answer) == ANSWER_KEYS
        for key, spec in answer_json_schema()["properties"].items():
            assert (spec["type"] == "array") == isinstance(answer[key], list), key


# ------------------------------------------------------------------ report
def write_incident(tmp_path):
    incident = tmp_path / "incident_001"
    (incident / "analysis").mkdir(parents=True, exist_ok=True)
    (incident / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc()), encoding="utf-8")
    (incident / "analysis" / "incident_reconstruction.json").write_text(
        json.dumps(reconstruction_doc()), encoding="utf-8")
    return incident


class TestIncidentReport:
    def test_context_only_run_has_no_answers_and_names_the_retriever(self, tmp_path):
        with synthetic_store() as store:
            report = run_incident_rag(write_incident(tmp_path), store)
        assert report.results == () and report.generation_requested is False
        assert report.retrievers == ("fts5-bm25-v1",) and report.uses_only_production_retriever
        data = json.loads(json.dumps(report.to_dict()))
        assert data["retrieval_methods_used"] == ["fts5-bm25-v1"] and data["ground_truth_used"] is False
        assert data["answers"] == [] and data["caveats"] == list(REPORT_CAVEATS)

    def test_run_with_a_mocked_model(self, tmp_path):
        with synthetic_store() as store:
            context = build_contexts_for_incident(write_incident(tmp_path), store).contexts[0]
            llm = FakeLLM(as_json(good_answer(context)))
            report = run_incident_rag(write_incident(tmp_path), store, llm)
        assert [r.status for r in report.results] == ["answered"] and len(llm.calls) == 1
        data = json.loads(json.dumps(report.to_dict()))
        assert data["answers"][0]["status"] == "answered" and data["generation_requested"] is True

    def test_the_report_is_deterministic(self, tmp_path):
        with synthetic_store() as store:
            first = run_incident_rag(write_incident(tmp_path), store)
            second = run_incident_rag(write_incident(tmp_path), store)
        assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(second.to_dict(), sort_keys=True)

    def test_a_non_production_retriever_is_reported_as_such(self, tmp_path):
        import dataclasses

        from retrieval.fts5 import Fts5Retriever

        class Experimental:
            name = "experimental-x"

            def __init__(self, inner):
                self.inner = inner

            def retrieve(self, query):
                return dataclasses.replace(self.inner.retrieve(query), retriever=self.name)

        with synthetic_store() as store:
            report = run_incident_rag(write_incident(tmp_path), store, retriever=Experimental(Fts5Retriever(store)))
        assert report.retrievers == ("experimental-x",) and not report.uses_only_production_retriever
        assert report.to_dict()["uses_only_production_retriever"] is False

    def test_the_context_view_for_the_prompt_omits_provenance_hashes(self, context):
        text = json.dumps(context_for_prompt(context))
        assert context.references[0].provenance["artifact_sha256"] not in text
        assert context.references[0].chunk_sha256 not in text

    def test_context_limits_flow_through_the_report(self, tmp_path):
        with synthetic_store() as store:
            report = run_incident_rag(write_incident(tmp_path), store, limits=ContextLimits(max_documents=1))
        assert len(report.contexts[0].references) == 1 and report.contexts[0].omitted_documents >= 1


# ------------------------------------------------------------------ the real incident_001 path (mocked model)
@needs_real_data
class TestRealIncident001WithMockedModel:
    """Real Phase 8/9 files and the real MITRE subset; the model is a script, not a real model."""

    def test_end_to_end_with_a_mocked_valid_answer(self, real_store):  # noqa: F811
        context = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
        llm = FakeLLM(as_json(good_answer(context)))
        report = run_incident_rag(INCIDENT_001, real_store, llm)
        (result,) = report.results
        assert report.uses_only_production_retriever and report.retrievers == ("fts5-bm25-v1",)
        assert result.status == "answered" and result.answer["hypotheses"][0]["hypothesis_id"] == "H1"
        assert result.context.hypotheses[0].is_confirmed is False
        sent = json.loads(llm.calls[0]["user"])["context"]
        assert {p["reference_id"] for p in sent["reference_passages"]} == set(result.context.reference_ids)

    def test_a_citation_to_a_chunk_that_was_not_retrieved_is_rejected(self, real_store):  # noqa: F811
        context = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
        answer = good_answer(context)
        answer["reference_guidance"][0]["reference_ids"] = ["cwe:CWE-79#0000"]
        result = generate_grounded_answer(context, FakeLLM(as_json(answer)))
        assert result.status == "rejected_citations" and "unknown_reference_id" in result.validation.codes

    def test_real_references_that_are_off_topic_can_be_marked_not_relevant(self, real_store):  # noqa: F811
        # The real keyword result includes records that do not fit a SQL-injection hypothesis
        # (for example the server-side request forgery pattern). A model may say so; the validator then
        # forbids using them.
        context = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
        answer = good_answer(context)
        off_topic = next((r.reference_id for r in context.references if "Request Forgery" in r.title), None)
        assert off_topic is not None, "expected the real keyword result to include an off-topic record"
        next(e for e in answer["reference_relevance"] if e["reference_id"] == off_topic)["assessment"] = "not_relevant"
        assert validate_answer(answer, context).passed
        answer["reference_guidance"][0]["reference_ids"] = [off_topic]
        assert "not_relevant_reference_cited" in validate_answer(answer, context).codes
