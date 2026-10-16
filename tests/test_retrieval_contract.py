"""Tests for the retrieval contract and the FTS5 backend (synthetic documents, offline)."""

from __future__ import annotations

import typing

import pytest

from knowledge.schema import DocType
from knowledge.store import InvalidSearchQueryError, KnowledgeStore
from retrieval.contract import (
    MAX_LIMIT,
    VALID_DOC_TYPES,
    ChunkMatch,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
    Retriever,
)
from retrieval.fts5 import STOPWORDS, Fts5Retriever, compile_fts_expression
from tests.retrieval_helpers import make_doc, make_store, small_corpus


def _item(rank: int, doc_id: str = "cwe:CWE-1") -> RetrievedItem:
    chunk = ChunkMatch(f"{doc_id}#0000", 0, 0, 5, "alpha", 1.0)
    return RetrievedItem(rank, doc_id, "cwe", doc_id.split(":")[1], "weakness", "T", None, 1.0, (chunk,))


class TestQueryValidation:
    def test_valid_query_has_defaults(self):
        query = RetrievalQuery("some text")
        assert query.limit == 10 and query.phrases == () and query.doc_types is None

    @pytest.mark.parametrize("text", ["", "   ", "\n\t"])
    def test_empty_text_is_rejected(self, text):
        with pytest.raises(InvalidRetrievalQueryError):
            RetrievalQuery(text)

    def test_non_string_and_oversized_text_is_rejected(self):
        with pytest.raises(InvalidRetrievalQueryError):
            RetrievalQuery(None)  # type: ignore[arg-type]
        with pytest.raises(InvalidRetrievalQueryError):
            RetrievalQuery("x" * 4001)

    @pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1, True, "5", 2.5])
    def test_invalid_limits_are_rejected(self, limit):
        with pytest.raises(InvalidRetrievalQueryError):
            RetrievalQuery("text", limit=limit)  # type: ignore[arg-type]

    def test_limit_bounds_are_accepted(self):
        assert RetrievalQuery("t", limit=1).limit == 1
        assert RetrievalQuery("t", limit=MAX_LIMIT).limit == MAX_LIMIT

    def test_doc_types_are_validated_and_normalized(self):
        assert RetrievalQuery("t", doc_types=("weakness", "attack_pattern", "weakness")).doc_types == (
            "attack_pattern",
            "weakness",
        )
        for bad in (("nonsense",), (), "weakness"):
            with pytest.raises(InvalidRetrievalQueryError):
                RetrievalQuery("t", doc_types=bad)  # type: ignore[arg-type]

    def test_phrases_are_validated(self):
        assert RetrievalQuery("t", phrases=["a b"]).phrases == ("a b",)  # list is normalized to a tuple
        for bad in ("a phrase", ("",), (" ",), (1,), tuple("p%d" % i for i in range(17))):
            with pytest.raises(InvalidRetrievalQueryError):
                RetrievalQuery("t", phrases=bad)  # type: ignore[arg-type]

    def test_valid_doc_types_match_the_knowledge_schema(self):
        assert VALID_DOC_TYPES == set(typing.get_args(DocType))


class TestResultInvariants:
    def test_empty_result_has_status_no_results(self):
        result = RetrievalResult(RetrievalQuery("t"), "x", (), "no_results")
        assert result.doc_ids == () and result.status == "no_results"

    def test_status_must_match_items(self):
        with pytest.raises(ValueError):
            RetrievalResult(RetrievalQuery("t"), "x", (), "ok")
        with pytest.raises(ValueError):
            RetrievalResult(RetrievalQuery("t"), "x", (_item(1),), "no_results")

    def test_ranks_must_be_contiguous_from_one(self):
        with pytest.raises(ValueError):
            RetrievalResult(RetrievalQuery("t"), "x", (_item(2),), "ok")

    def test_a_document_may_appear_only_once(self):
        with pytest.raises(ValueError):
            RetrievalResult(RetrievalQuery("t"), "x", (_item(1), _item(2)), "ok")

    def test_result_cannot_exceed_the_query_limit(self):
        items = (_item(1, "cwe:CWE-1"), _item(2, "cwe:CWE-2"))
        with pytest.raises(ValueError):
            RetrievalResult(RetrievalQuery("t", limit=1), "x", items, "ok")

    def test_item_chunks_must_belong_to_the_item(self):
        chunk = ChunkMatch("cwe:CWE-9#0000", 0, 0, 5, "alpha", 1.0)
        with pytest.raises(ValueError):
            RetrievedItem(1, "cwe:CWE-1", "cwe", "CWE-1", "weakness", "T", None, 1.0, (chunk,))
        with pytest.raises(ValueError):
            RetrievedItem(1, "cwe:CWE-1", "cwe", "CWE-1", "weakness", "T", None, 1.0, ())

    def test_fts5_retriever_satisfies_the_protocol(self):
        with KnowledgeStore(":memory:") as store:
            assert isinstance(Fts5Retriever(store), Retriever)


class TestCompileExpression:
    def expr(self, text, **kw):
        return compile_fts_expression(RetrievalQuery(text, **kw))

    def test_words_are_quoted_lowercased_and_joined_with_or(self):
        assert self.expr("Database Query").expression == '"database" OR "query"'

    def test_identifiers_become_quoted_phrases_first(self):
        compiled = self.expr("injection seen, CWE-89 and capec-66 noted")
        assert compiled.expression.startswith('"CWE-89" OR "CAPEC-66"')
        assert "89" not in compiled.terms[2:] and "cwe" not in compiled.terms

    def test_phrases_come_before_words_and_are_normalized(self):
        compiled = self.expr("alpha beta", phrases=("SQL  Injection!",))
        assert compiled.terms == ("sql injection", "alpha", "beta")

    def test_stopwords_digits_single_letters_and_duplicates_are_dropped(self):
        compiled = self.expr("The the of 42 a x database database and potential hypothesis")
        assert compiled.terms == ("database",)
        assert {"the", "potential", "hypothesis"} <= STOPWORDS

    def test_fts_operators_in_free_text_cannot_break_the_expression(self):
        compiled = self.expr('NOT -bar:baz "unbalanced (x) AND * near/2 db')
        assert "baz" in compiled.terms and compiled.expression.count('"') % 2 == 0
        with make_store(*small_corpus()) as store:  # and the store accepts it
            Fts5Retriever(store).retrieve(RetrievalQuery('NOT -bar:baz "unbalanced (x) AND * near/2 db'))

    def test_nothing_searchable_is_an_error(self):
        for text in ("the of and", "123 456", "!!! ???", "a b"):
            with pytest.raises(InvalidRetrievalQueryError, match="no searchable terms"):
                self.expr(text)

    def test_same_input_gives_the_same_expression(self):
        a = self.expr("alpha beta gamma", phrases=("x y",))
        b = self.expr("alpha beta gamma", phrases=("x y",))
        assert (a.expression, a.terms, a.dropped) == (b.expression, b.terms, b.dropped)

    def test_overlong_expression_drops_lowest_priority_terms_and_reports_them(self):
        words = [f"word{i:03d}xyz" for i in range(200)]
        compiled = self.expr(" ".join(words)[:3999])
        assert len(compiled.expression) <= 1000
        assert compiled.dropped
        assert compiled.terms == tuple(words[: len(compiled.terms)])  # the head is kept, in order
        everything = compiled.terms + compiled.dropped  # nothing vanishes silently
        assert everything[:-1] == tuple(words[: len(everything) - 1])  # (the input's last word is cut mid-way)
        assert not set(compiled.terms) & set(compiled.dropped)


class TestFts5Retriever:
    def test_results_preserve_source_and_document_identity(self):
        with make_store(*small_corpus()) as store:
            result = Fts5Retriever(store).retrieve(RetrievalQuery("database query request"))
            top = result.items[0]
            assert (top.doc_id, top.source, top.source_id, top.doc_type) == (
                "cwe:CWE-1", "cwe", "CWE-1", "weakness"
            )
            assert top.title == "Alpha Weakness" and top.source_url == "https://example.invalid/CWE-1"
            stored = store.get_chunks("cwe:CWE-1")
            for match in top.chunks:
                chunk = stored[match.ordinal]
                assert (match.chunk_id, match.char_start, match.char_end, match.text) == (
                    chunk.chunk_id, chunk.char_start, chunk.char_end, chunk.text
                )

    def test_ranking_is_ordered_ranked_from_one_and_limited(self):
        with make_store(*small_corpus()) as store:
            result = Fts5Retriever(store).retrieve(RetrievalQuery("alpha database", limit=2))
            assert [i.rank for i in result.items] == [1, 2] and len(result.items) == 2
            assert result.items[0].score >= result.items[1].score
            assert result.status == "ok" and not result.possibly_incomplete
            assert result.retriever == "fts5-bm25-v1" and result.backend_query == '"alpha" OR "database"'

    def test_multiple_matching_chunks_collapse_into_one_source_item(self):
        many = make_doc("CWE-50", "Long One", " ".join(["repeated keyword filler words"] * 1200))
        single = make_doc("CWE-51", "Short One", "repeated keyword once")
        with make_store(many, single) as store:
            assert store.count_chunks() > 3  # the long document really has several chunks
            result = Fts5Retriever(store).retrieve(RetrievalQuery("repeated keyword"))
            assert sorted(result.doc_ids) == ["cwe:CWE-50", "cwe:CWE-51"]  # each source once
            long_item = next(i for i in result.items if i.doc_id == "cwe:CWE-50")
            assert len(long_item.chunks) > 1
            assert long_item.score == max(c.score for c in long_item.chunks)  # best chunk decides

    def test_empty_result_is_a_valid_no_results_outcome(self):
        with make_store(*small_corpus()) as store:
            result = Fts5Retriever(store).retrieve(RetrievalQuery("zzzunmatchable"))
            assert result.items == () and result.status == "no_results" and not result.possibly_incomplete

    def test_empty_store_returns_no_results(self):
        with KnowledgeStore(":memory:") as store:
            assert Fts5Retriever(store).retrieve(RetrievalQuery("anything")).status == "no_results"

    def test_doc_type_filter_keeps_only_requested_types(self):
        with make_store(*small_corpus()) as store:
            retriever = Fts5Retriever(store)
            weak = retriever.retrieve(RetrievalQuery("alpha database", doc_types=("weakness",)))
            attack = retriever.retrieve(RetrievalQuery("alpha database", doc_types=("attack_pattern",)))
            assert weak.doc_ids == ("cwe:CWE-1",) and attack.doc_ids == ("capec:CAPEC-1",)
            both = retriever.retrieve(RetrievalQuery("alpha database"))
            assert set(both.doc_ids) == {"cwe:CWE-1", "capec:CAPEC-1"}

    def test_ties_are_broken_by_doc_id_deterministically(self):
        twins = [make_doc("CWE-8", "Twin", "twin text keyword"), make_doc("CWE-7", "Twin", "twin text keyword")]
        with make_store(*twins) as store:
            result = Fts5Retriever(store).retrieve(RetrievalQuery("keyword"))
            assert result.items[0].score == result.items[1].score
            assert result.doc_ids == ("cwe:CWE-7", "cwe:CWE-8")

    def test_repeated_calls_give_identical_results(self):
        with make_store(*small_corpus()) as store:
            retriever = Fts5Retriever(store)
            query = RetrievalQuery("alpha database request attacker", limit=4)
            assert retriever.retrieve(query) == retriever.retrieve(query)

    def test_possibly_incomplete_is_flagged_when_the_chunk_window_is_exhausted(self):
        huge = make_doc("CWE-60", "Huge", " ".join(["windowfill word"] * 20000))
        with make_store(huge) as store:
            assert store.count_chunks() > 100
            result = Fts5Retriever(store).retrieve(RetrievalQuery("windowfill", limit=5))
            assert result.doc_ids == ("cwe:CWE-60",) and result.possibly_incomplete

    def test_complete_when_enough_documents_were_found(self):
        with make_store(*small_corpus()) as store:
            result = Fts5Retriever(store).retrieve(RetrievalQuery("alpha database", limit=1))
            assert not result.possibly_incomplete

    def test_invalid_query_object_and_unsearchable_text_raise_invalid_query(self):
        with make_store(*small_corpus()) as store:
            retriever = Fts5Retriever(store)
            with pytest.raises(InvalidRetrievalQueryError):
                retriever.retrieve("alpha")  # type: ignore[arg-type]
            with pytest.raises(InvalidRetrievalQueryError):
                retriever.retrieve(RetrievalQuery("the of and"))

    def test_store_query_rejection_maps_to_invalid_query(self, monkeypatch):
        with make_store(*small_corpus()) as store:
            def reject(*args, **kwargs):
                raise InvalidSearchQueryError("bad syntax")
            monkeypatch.setattr(store, "search", reject)
            with pytest.raises(InvalidRetrievalQueryError, match="bad syntax"):
                Fts5Retriever(store).retrieve(RetrievalQuery("alpha"))

    def test_store_failure_maps_to_backend_error_with_cause(self):
        store = make_store(*small_corpus())
        retriever = Fts5Retriever(store)
        store.close()
        with pytest.raises(RetrievalBackendError) as info:
            retriever.retrieve(RetrievalQuery("alpha"))
        assert info.value.__cause__ is not None

    def test_constructor_requires_a_knowledge_store(self):
        with pytest.raises(TypeError):
            Fts5Retriever(object())  # type: ignore[arg-type]
