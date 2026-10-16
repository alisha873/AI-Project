"""
Tests for Phase 11D step 5a: deterministic citation validation (`rag.citations`).

Every case starts from a known-good answer on a synthetic context and breaks exactly one thing,
so a failure names the rule that regressed. No language model; no network.
"""

from __future__ import annotations

import copy

import pytest

from rag.citations import (
    MAX_STATEMENT_CHARS,
    InvalidAnswerError,
    parse_answer_json,
    validate_answer,
)
from tests.rag_helpers import (
    HYPOTHESIS,
    build_context,
    good_answer,
    root_cause_doc,
)
from tests.retrieval_helpers import make_doc, make_store


@pytest.fixture(scope="module")
def context():
    ctx = build_context()
    assert {"cwe:CWE-1", "capec:CAPEC-1"} <= {r.doc_id for r in ctx.references}
    return ctx


def ref_of(context, doc_id: str) -> str:
    return next(r.reference_id for r in context.references if r.doc_id == doc_id)


def codes(answer, context) -> set[str]:
    return set(validate_answer(answer, context).codes)


def broken(context, mutate) -> dict:
    answer = copy.deepcopy(good_answer(context))
    mutate(answer)
    return answer


def test_a_valid_answer_passes(context):
    result = validate_answer(good_answer(context), context)
    assert result.passed and result.issues == () and "does not prove" in result.note


def test_a_valid_answer_without_any_reference_material_passes():
    with make_store(make_doc("CWE-9", "Unrelated", "zebra giraffe savanna")) as store:
        empty = build_context(store)
    answer = good_answer(empty)
    assert answer["reference_guidance"] == [] and answer["reference_relevance"] == []
    assert validate_answer(answer, empty).passed


def test_all_issues_are_reported_not_only_the_first(context):
    answer = broken(context, lambda a: (a["observed_evidence"][0].update(evidence_ids=["nginx-99"]),
                                        a["reference_guidance"][0].update(reference_ids=["cwe:CWE-9#0000"])))
    assert {"unknown_evidence_id", "unknown_reference_id"} <= codes(answer, context)


# ------------------------------------------------------------------ structure
class TestStructure:
    def test_not_an_object(self, context):
        assert codes([], context) == {"invalid_structure"}

    @pytest.mark.parametrize("key", ["observed_evidence", "hypotheses", "reference_guidance",
                                     "reference_relevance", "evidence_gaps", "limitations", "incident_id"])
    def test_a_missing_top_level_key(self, context, key):
        answer = broken(context, lambda a: a.pop(key))
        assert "missing_key" in codes(answer, context)

    def test_an_unexpected_top_level_key(self, context):
        assert "unknown_key" in codes(broken(context, lambda a: a.update(root_cause_confirmed=True)), context)

    def test_an_unexpected_item_key_such_as_references_inside_observed_evidence(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(reference_ids=[ref_of(context, "cwe:CWE-1")]))
        assert "unknown_key" in codes(answer, context)

    def test_wrong_incident_and_schema_version(self, context):
        assert "wrong_incident" in codes(broken(context, lambda a: a.update(incident_id="incident_002")), context)
        assert "invalid_value" in codes(broken(context, lambda a: a.update(schema_version="2.0")), context)

    @pytest.mark.parametrize("key", ["observed_evidence", "hypotheses", "reference_guidance",
                                     "reference_relevance", "evidence_gaps", "limitations"])
    def test_sections_must_be_arrays(self, context, key):
        assert "invalid_structure" in codes(broken(context, lambda a: a.update({key: "text"})), context)

    def test_empty_overlong_and_non_string_statements(self, context):
        for value in ("", "   ", "x" * (MAX_STATEMENT_CHARS + 1), 7):
            answer = broken(context, lambda a: a["observed_evidence"][0].update(statement=value))
            assert codes(answer, context) & {"invalid_structure", "statement_too_long"}, repr(value)[:20]

    def test_citation_lists_must_be_arrays_of_strings(self, context):
        for value in ("application-4", [1], [""], None):
            answer = broken(context, lambda a: a["observed_evidence"][0].update(evidence_ids=value))
            assert "invalid_structure" in codes(answer, context), value


class TestParseAnswerJson:
    def test_parses_an_object(self):
        assert parse_answer_json(' {"a": 1}\n') == {"a": 1}

    @pytest.mark.parametrize("text", ["", "   ", "not json", "[1]", '"x"', "```json\n{}\n```", None])
    def test_rejects_everything_else(self, text):
        with pytest.raises(InvalidAnswerError):
            parse_answer_json(text)


# ------------------------------------------------------------------ citations
class TestEvidenceCitations:
    def test_unknown_evidence_id(self, context):
        assert "unknown_evidence_id" in codes(
            broken(context, lambda a: a["observed_evidence"][0].update(evidence_ids=["nginx-99"])), context)

    def test_observed_statement_without_citation(self, context):
        assert "missing_citation" in codes(
            broken(context, lambda a: a["observed_evidence"][0].update(evidence_ids=[])), context)

    def test_a_reference_id_cannot_be_cited_as_evidence(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(evidence_ids=[ref_of(context, "cwe:CWE-1")]))
        assert "category_mixing" in codes(answer, context)

    def test_an_evidence_id_without_a_phase_8_description_is_not_citable(self):
        finding = {**HYPOTHESIS, "supporting_evidence_ids": ["application-4", "ghost-99"]}
        ctx = build_context(root_cause=root_cause_doc(finding))
        assert ctx.evidence_ids == ("application-4",) and ctx.unresolved_evidence_ids == ("ghost-99",)
        answer = good_answer(ctx)
        answer["hypotheses"][0]["evidence_ids"] = ["application-4"]  # nginx-19 is not part of this finding
        assert validate_answer(answer, ctx).passed
        answer["observed_evidence"][0]["evidence_ids"] = ["ghost-99"]
        assert "unknown_evidence_id" in codes(answer, ctx)

    def test_ids_mentioned_in_prose_must_be_cited_by_the_same_item(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(
            statement="GET /search was recorded in application-4 and nginx-19."))
        assert codes(answer, context) == {"prose_citation_mismatch"}

    def test_unknown_ids_mentioned_in_prose(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(
            statement="GET /search was recorded in application-4 and docker-container-4."))
        assert "unknown_evidence_id" in codes(answer, context)

    def test_hyphenated_words_are_not_mistaken_for_event_ids(self, context):
        answer = broken(context, lambda a: a["limitations"].append("An auth-related or api-style log is absent."))
        assert validate_answer(answer, context).passed


class TestReferenceCitations:
    def test_unknown_reference_id(self, context):
        answer = broken(context, lambda a: a["reference_guidance"][0].update(reference_ids=["cwe:CWE-9#0000"]))
        assert "unknown_reference_id" in codes(answer, context)

    def test_guidance_without_citation(self, context):
        answer = broken(context, lambda a: a["reference_guidance"][0].update(reference_ids=[]))
        assert "missing_citation" in codes(answer, context)

    def test_incident_evidence_cannot_be_cited_as_a_reference(self, context):
        answer = broken(context, lambda a: a["reference_guidance"][0].update(reference_ids=["nginx-19"]))
        assert "category_mixing" in codes(answer, context)

    def test_guidance_must_not_talk_about_incident_events(self, context):
        answer = broken(context, lambda a: a["reference_guidance"][0].update(
            statement="This weakness is visible in nginx-19."))
        assert "prose_citation_mismatch" in codes(answer, context) or "unknown_evidence_id" in codes(answer, context)

    def test_chunk_ids_in_prose_must_be_cited_by_the_same_item(self, context):
        cited = good_answer(context)["reference_guidance"][0]["reference_ids"][0]
        other = next(r for r in context.reference_ids if r != cited)
        answer = broken(context, lambda a: a["reference_guidance"][0].update(statement=f"See also {other}."))
        assert "prose_citation_mismatch" in codes(answer, context)

    def test_a_fabricated_chunk_id_in_prose(self, context):
        answer = broken(context, lambda a: a["reference_guidance"][0].update(statement="See cwe:CWE-77#0004."))
        assert "unknown_reference_id" in codes(answer, context)

    def test_a_hypothesis_may_cite_both_kinds(self, context):
        assert validate_answer(good_answer(context), context).passed
        assert good_answer(context)["hypotheses"][0]["reference_ids"]


class TestHypotheses:
    def test_each_supplied_hypothesis_must_be_addressed(self, context):
        assert "missing_hypothesis" in codes(broken(context, lambda a: a.update(hypotheses=[])), context)

    def test_unknown_and_duplicate_hypothesis_ids(self, context):
        assert "unknown_hypothesis_id" in codes(
            broken(context, lambda a: a["hypotheses"][0].update(hypothesis_id="H9")), context)
        duplicate = broken(context, lambda a: a["hypotheses"].append(copy.deepcopy(a["hypotheses"][0])))
        assert "duplicate_hypothesis" in codes(duplicate, context)

    def test_hypothesis_only_requires_evidence(self, context):
        answer = broken(context, lambda a: a["hypotheses"][0].update(evidence_ids=[]))
        assert "missing_citation" in codes(answer, context)

    def test_insufficient_evidence_may_have_no_evidence_ids(self, context):
        answer = broken(context, lambda a: a["hypotheses"][0].update(evidence_ids=[], status="insufficient_evidence"))
        assert validate_answer(answer, context).passed

    @pytest.mark.parametrize("status", ["confirmed", "likely", "", None])
    def test_status_must_be_an_allowed_value(self, context, status):
        assert "invalid_value" in codes(broken(context, lambda a: a["hypotheses"][0].update(status=status)), context)


# ------------------------------------------------------------------ relevance
class TestReferenceRelevance:
    def test_every_retrieved_passage_must_be_assessed(self, context):
        answer = broken(context, lambda a: a["reference_relevance"].pop())
        assert "missing_assessment" in codes(answer, context)

    def test_duplicate_unknown_and_invalid_assessments(self, context):
        dup = broken(context, lambda a: a["reference_relevance"].append(copy.deepcopy(a["reference_relevance"][0])))
        assert "duplicate_assessment" in codes(dup, context)
        unknown = broken(context, lambda a: a["reference_relevance"].append(
            {"reference_id": "cwe:CWE-9#0000", "assessment": "relevant"}))
        assert "unknown_reference_id" in codes(unknown, context)
        bad = broken(context, lambda a: a["reference_relevance"][0].update(assessment="perfect"))
        assert "invalid_value" in codes(bad, context)

    def test_a_not_relevant_passage_may_not_be_cited(self, context):
        cited = good_answer(context)["reference_guidance"][0]["reference_ids"][0]

        def mark(answer):
            for entry in answer["reference_relevance"]:
                if entry["reference_id"] == cited:
                    entry["assessment"] = "not_relevant"

        assert "not_relevant_reference_cited" in codes(broken(context, mark), context)

    def test_irrelevant_retrieval_is_handled_by_empty_guidance(self, context):
        def all_not_relevant(answer):
            for entry in answer["reference_relevance"]:
                entry["assessment"] = "not_relevant"
            answer["reference_guidance"] = []
            answer["hypotheses"][0]["reference_ids"] = []
            answer["limitations"].append("The retrieved passages do not address this incident.")

        assert validate_answer(broken(context, all_not_relevant), context).passed

    def test_no_retrieval_means_no_guidance(self):
        with make_store(make_doc("CWE-9", "Unrelated", "zebra giraffe savanna")) as store:
            empty = build_context(store)
        answer = good_answer(empty)
        answer["reference_guidance"] = [{"statement": "SQL injection is bad.", "reference_ids": ["cwe:CWE-89#0000"]}]
        assert {"no_reference_material", "unknown_reference_id"} <= codes(answer, empty)
        answer = good_answer(empty)
        answer["hypotheses"][0]["reference_ids"] = ["cwe:CWE-89#0000"]
        assert "unknown_reference_id" in codes(answer, empty)


# ------------------------------------------------------------------ identifiers
class TestIdentifiers:
    def test_observed_evidence_may_not_name_knowledge_base_identifiers(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(statement="GET /search shows CWE-89."))
        assert "unsupported_identifier" in codes(answer, context)

    def test_an_identifier_the_hypothesis_itself_states_is_allowed_in_a_hypothesis(self, context):
        answer = broken(context, lambda a: a["hypotheses"][0].update(
            statement="Phase 9 names CWE-89 as a hypothesis; this is not confirmed."))
        assert validate_answer(answer, context).passed

    def test_an_invented_identifier_is_rejected_everywhere(self, context):
        for section, index in (("hypotheses", 0), ("reference_guidance", 0)):
            answer = broken(context, lambda a: a[section][index].update(statement="This is CWE-999."))
            assert "unsupported_identifier" in codes(answer, context), section
        answer = broken(context, lambda a: a["limitations"].append("See also CAPEC-12345."))
        assert "unsupported_identifier" in codes(answer, context)

    def test_guidance_may_name_the_cited_passage_and_identifiers_inside_it(self, context):
        capec = ref_of(context, "capec:CAPEC-1")  # its text says "see CWE-1"
        answer = broken(context, lambda a: (
            a["reference_guidance"][0].update(reference_ids=[capec],
                                              statement="CAPEC-1 relates to CWE-1 according to its text.")))
        assert validate_answer(answer, context).passed

    def test_guidance_may_not_name_an_identifier_of_a_passage_it_does_not_cite(self, context):
        cwe = ref_of(context, "cwe:CWE-1")
        answer = broken(context, lambda a: a["reference_guidance"][0].update(
            reference_ids=[cwe], statement="CAPEC-1 describes how this is attacked."))
        assert "unsupported_identifier" in codes(answer, context)

    def test_identifiers_are_case_insensitive(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(statement="GET /search, see cwe-89."))
        assert "unsupported_identifier" in codes(answer, context)


# ------------------------------------------------------------------ facts
class TestHttpFacts:
    @pytest.mark.parametrize("statement", [
        "POST /search returned status 200.",
        "GET /search returned HTTP 404.",
        "GET /admin returned status 200.",
        "GET /api/search returned status 200.",  # that path belongs to nginx-19, which is not cited here
    ])
    def test_mismatches_with_the_cited_request_are_rejected(self, context, statement):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(statement=statement))
        assert "http_fact_mismatch" in codes(answer, context)

    def test_matching_facts_pass_including_the_second_event(self, context):
        answer = broken(context, lambda a: a["observed_evidence"].append(
            {"statement": "GET /api/search with HTTP 200 was recorded by the proxy.", "evidence_ids": ["nginx-19"]}))
        assert validate_answer(answer, context).passed

    def test_ordinary_words_and_slashes_are_not_http_facts(self, context):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(
            statement="An attacker may get data; the TCP/IP stack and/or a proxy were not examined, 2/3 of GET /search."))
        assert validate_answer(answer, context).passed

    def test_hypotheses_are_fact_checked_against_their_cited_events_too(self, context):
        answer = broken(context, lambda a: a["hypotheses"][0].update(
            statement="A DELETE request may be an injection attempt; this is not confirmed."))
        assert "http_fact_mismatch" in codes(answer, context)


class TestImpactAndConfirmation:
    @pytest.mark.parametrize("statement", [
        "The database was accessed through GET /search.",
        "GET /search was a successful exploitation.",
        "The attacker successfully exploited the application.",
        "Data was exfiltrated via GET /search.",
        "The server was compromised after GET /search.",
        "There is no doubt that the server was compromised.",
    ])
    def test_positive_impact_claims_are_rejected(self, context, statement):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(statement=statement))
        assert "high_impact_claim" in codes(answer, context)

    @pytest.mark.parametrize("statement", [
        "GET /search is recorded; there is no evidence of successful exploitation.",
        "GET /search does not establish that the database was accessed.",
        "GET /search is recorded; the server was not compromised according to any event shown here.",
        "GET /search is recorded; successful exploitation is not confirmed.",
    ])
    def test_negated_wording_is_allowed(self, context, statement):
        answer = broken(context, lambda a: a["observed_evidence"][0].update(statement=statement))
        assert "high_impact_claim" not in codes(answer, context)

    def test_impact_claims_are_rejected_in_every_section(self, context):
        text = "The attacker gained admin access."
        for section in ("hypotheses", "reference_guidance"):
            answer = broken(context, lambda a: a[section][0].update(statement=text))
            assert "high_impact_claim" in codes(answer, context), section
        assert "high_impact_claim" in codes(broken(context, lambda a: a["limitations"].append(text)), context)

    @pytest.mark.parametrize("statement", [
        "The SQL injection vulnerability is confirmed.",
        "This is a confirmed vulnerability in /search.",
        "The root cause was confirmed by the logs.",
        "The evidence confirms that the attack succeeded.",
    ])
    def test_a_hypothesis_cannot_be_presented_as_confirmed(self, context, statement):
        answer = broken(context, lambda a: a["hypotheses"][0].update(statement=statement))
        assert "confirmation_claim" in codes(answer, context)

    @pytest.mark.parametrize("statement", [
        "The root cause is not confirmed.",
        "The SQL injection vulnerability is unconfirmed.",
        "There is no confirmed vulnerability.",
        "The hypothesis cannot confirm the vulnerability.",
    ])
    def test_hedged_wording_is_allowed(self, context, statement):
        answer = broken(context, lambda a: a["hypotheses"][0].update(statement=statement))
        assert "confirmation_claim" not in codes(answer, context)

    def test_when_phase_9_confirmed_the_finding_confirmation_wording_is_allowed(self):
        ctx = build_context(root_cause=root_cause_doc({**HYPOTHESIS, "status": "confirmed"}))
        assert ctx.hypotheses[0].is_confirmed
        answer = good_answer(ctx)
        answer["hypotheses"][0]["statement"] = "The SQL injection vulnerability is confirmed by Phase 9."
        assert validate_answer(answer, ctx).passed


def impact_issues(context, text: str) -> list[str]:
    """The high_impact_claim / confirmation_claim messages produced by `text` as a limitation."""
    answer = broken(context, lambda a: a["limitations"].append(text))
    return [issue.message for issue in validate_answer(answer, context).issues
            if issue.code in ("high_impact_claim", "confirmation_claim")]


class TestSentenceScopedNegation:
    """Regression: cautious, explicitly negated wording is accepted; affirmative claims stay rejected."""

    @pytest.mark.parametrize("text", [
        # the reported sentence and its direct variations
        "The evidence does not establish successful exploitation.",
        "Successful exploitation has not been established by the available evidence.",
        "Successful exploitation has not been confirmed.",
        "Successful exploitation was not established by the available evidence.",
        "Successful exploitation is not yet confirmed.",
        "The evidence does not currently establish successful exploitation.",
        "The available evidence cannot establish successful exploitation.",
        "The available evidence does not establish that successful exploitation occurred.",
        "There is no evidence of successful exploitation.",
        # a negating verb over a list (the wording the Phase 9 rationale itself uses)
        "This supports an injection-attempt hypothesis but does not establish vulnerable server-side "
        "query construction, successful exploitation, or database access.",
        "The evidence does not establish that the request reached the database or that successful "
        "exploitation occurred.",
        "There is no evidence of data exfiltration, privilege escalation, or successful exploitation.",
        # a negated claim followed by an unrelated sentence that is itself cautious
        "The evidence does not establish successful exploitation. Further logs are needed.",
    ])
    def test_cautious_negated_wording_is_accepted(self, context, text):
        assert impact_issues(context, text) == [], text

    @pytest.mark.parametrize("text", [
        "Successful exploitation occurred.",
        "The attacker successfully exploited the application.",
        "The application was successfully exploited through the search endpoint.",
        "The evidence shows that successful exploitation occurred.",
        "Successful exploitation has been established by the available evidence.",
        "The evidence establishes successful exploitation.",
        "The evidence does establish successful exploitation.",
        "Data was exfiltrated.",
    ])
    def test_affirmative_unsupported_claims_are_still_rejected(self, context, text):
        assert impact_issues(context, text), text

    @pytest.mark.parametrize("text, claim", [
        # a separate sentence
        ("The evidence does not establish successful exploitation. The attacker successfully exploited "
         "the application.", "successfully exploited"),
        ("Successful exploitation has not been established. The server was compromised.", "server was compromised"),
        ("There is no evidence of data exfiltration. Data was exfiltrated.", "Data was exfiltrated"),
        # a separate clause in the same sentence
        ("The evidence does not establish successful exploitation, but the server was compromised.",
         "server was compromised"),
        ("Successful exploitation has not been established, however data was exfiltrated.", "data was exfiltrated"),
        ("The evidence does not establish the injection, and the server was compromised.", "server was compromised"),
        ("The evidence does not establish the injection, or the server was compromised.", "server was compromised"),
        # separated by sentence-ending punctuation other than a full stop
        ("The evidence does not establish successful exploitation; the server was compromised.",
         "server was compromised"),
        ("There is no evidence of exploitation: the database was accessed.", "database was accessed"),
        ("The evidence does not establish this!\nThe server was compromised.", "server was compromised"),
        # a comma splice: the list item is really a new clause
        ("The evidence does not establish the injection, successful exploitation occurred.",
         "successful exploitation"),
    ])
    def test_a_negation_in_one_sentence_or_clause_cannot_excuse_another_claim(self, context, text, claim):
        messages = impact_issues(context, text)
        assert len(messages) == 1 and claim in messages[0], (text, messages)

    def test_the_negated_claim_is_not_reported_only_the_affirmative_one(self, context):
        text = ("The evidence does not establish successful exploitation. "
                "The attacker successfully exploited the application.")
        (message,) = impact_issues(context, text)
        assert "successfully exploited" in message and "successful exploitation" not in message

    def test_a_negated_claim_before_a_confirmation_claim_in_another_sentence(self, context):
        text = "There is no confirmed vulnerability. The SQL injection vulnerability is confirmed."
        answer = broken(context, lambda a: a["hypotheses"][0].update(statement=text))
        assert "confirmation_claim" in codes(answer, context)

    def test_the_reported_sentence_passes_in_every_free_text_section(self, context):
        text = "The evidence does not establish successful exploitation."
        for section in ("observed_evidence", "hypotheses", "reference_guidance"):
            answer = broken(context, lambda a: a[section][0].update(statement=text))
            assert "high_impact_claim" not in codes(answer, context), section
        assert validate_answer(broken(context, lambda a: a["limitations"].append(text)), context).passed
        assert validate_answer(broken(context, lambda a: a["evidence_gaps"].append(text)), context).passed


class TestFreeTextSections:
    def test_gaps_and_limitations_must_not_invent_ids_or_identifiers(self, context):
        for key in ("evidence_gaps", "limitations"):
            assert "unknown_evidence_id" in codes(broken(context, lambda a: a[key].append("Check nginx-404 as well.")), context)
            assert "unsupported_identifier" in codes(broken(context, lambda a: a[key].append("Compare CWE-77.")), context)

    def test_gaps_may_use_known_ids_and_hedged_statements(self, context):
        answer = broken(context, lambda a: a["evidence_gaps"].append(
            "nginx-19 and application-4 do not show that successful exploitation occurred."))
        assert validate_answer(answer, context).passed
