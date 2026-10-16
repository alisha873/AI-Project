"""
Offline tests for the grounded-answer pre-send budget handling and reference-passage compression.

No network and no real model: `FakeLLM` returns scripted responses or raises scripted exceptions. The
compression is checked on the model-facing prompt; the stored context (and therefore validation) keeps the
full passage text.
"""

from __future__ import annotations

import json

from llm.token_budget import PromptTooLargeError
from rag.answer import generate_grounded_answer
from rag.prompt import build_user_prompt, context_for_prompt
from tests.rag_helpers import FakeLLM, as_json, build_context, good_answer


# -------------------------------------------------------------- request_sent / budget

def test_a_prompt_too_large_error_reports_no_request_sent():
    context = build_context()
    result = generate_grounded_answer(context, FakeLLM(PromptTooLargeError(9000, 7000, "too big; no request was sent")))
    assert result.status == "llm_error"
    assert result.request_sent is False
    assert "No request was sent" in result.message
    assert result.to_dict()["request_sent"] is False


def test_a_provider_error_after_sending_reports_request_sent_true():
    context = build_context()
    result = generate_grounded_answer(context, FakeLLM(RuntimeError("429 rate limited")))
    assert result.status == "llm_error" and result.request_sent is True


def test_an_accepted_answer_reports_request_sent_true():
    context = build_context()
    result = generate_grounded_answer(context, FakeLLM(as_json(good_answer(context))))
    assert result.accepted and result.request_sent is True and result.to_dict()["request_sent"] is True


# -------------------------------------------------------------- compression preserves provenance

def test_compression_shortens_only_text_and_keeps_every_identifier():
    context = build_context()
    view = context_for_prompt(context, max_characters_per_passage=200)
    for passage, ref in zip(view["reference_passages"], context.references):
        assert passage["reference_id"] == ref.reference_id         # id preserved exactly
        assert passage["document_id"] == ref.doc_id
        assert passage["source_id"] == ref.source_id and passage["title"] == ref.title
        if len(ref.text) > 200:
            assert passage["text_truncated"] is True
            assert passage["text_full_chars"] == len(ref.text)
            assert "shortened" in passage["text"]                  # explicit, not silent
            assert len(passage["text"]) < len(ref.text) + 200


def test_compression_is_off_by_default():
    context = build_context()
    view = context_for_prompt(context)
    for passage, ref in zip(view["reference_passages"], context.references):
        assert passage["text"] == ref.text
        assert "text_truncated" not in passage


def test_compression_never_touches_evidence_or_hypotheses():
    context = build_context()
    plain = context_for_prompt(context)
    compressed = context_for_prompt(context, max_characters_per_passage=200)
    assert plain["observed_evidence"] == compressed["observed_evidence"]
    assert plain["hypotheses"] == compressed["hypotheses"]


def test_compression_flows_through_generation_and_validation_still_passes():
    from tests.retrieval_helpers import make_doc, make_store

    long_text = ("sql injection: the application builds a database query from request text without "
                 "neutralizing special characters, so an attacker can change the query. ") * 4
    store = make_store(
        make_doc("CWE-1", "Query Weakness", long_text),
        make_doc("CAPEC-1", "Injection Attack", long_text, source="capec", doc_type="attack_pattern"),
    )
    try:
        context = build_context(store)
        assert any(len(ref.text) > 200 for ref in context.references)
        llm = FakeLLM(as_json(good_answer(context)))
        result = generate_grounded_answer(context, llm, max_characters_per_passage=200)
        # the model actually received the shortened passages
        sent = json.loads(llm.calls[0]["user"])
        assert any(p.get("text_truncated") for p in sent["context"]["reference_passages"])
        # validation runs on the stored full context, so the good answer is still accepted
        assert result.accepted and result.validation.passed
    finally:
        store.close()


def test_an_unreasonably_small_cap_is_rejected():
    import pytest

    context = build_context()
    with pytest.raises(ValueError):
        build_user_prompt(context, max_characters_per_passage=10)
