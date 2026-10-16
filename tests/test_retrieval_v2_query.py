"""
Milestone 11B tests: hardened keyword query handling (`fts5_v2`) and lane definitions.

Offline and deterministic. Documents are SYNTHETIC test text (see
`tests/retrieval_helpers.py`); real-archive tests live in
`tests/test_retrieval_v2_evaluation.py` and skip with an explicit reason when the
archives are absent.
"""

from __future__ import annotations

import pytest

from knowledge.store import MAX_QUERY_LENGTH
from retrieval import fts5_v2, lanes
from retrieval.contract import (
    LOW_INFORMATION_PREFIX,
    InvalidRetrievalQueryError,
    RetrievalQuery,
    RetrievalResult,
)
from retrieval.fts5 import Fts5Retriever
from retrieval.fts5_v2 import Fts5RetrieverV2, compile_fts_expression_v2, normalize_query_text
from tests.retrieval_helpers import make_doc, make_store, small_corpus


def compile_(text, **kw):
    return compile_fts_expression_v2(RetrievalQuery(text=text, **kw))


def notes_of(text, **kw):
    return compile_(text, **kw).notes


# ---------------------------------------------------------------- normalisation

class TestNormalisation:
    def test_plain_text_is_unchanged_and_has_no_notes(self):
        assert normalize_query_text("database query text") == ("database query text", [])

    @pytest.mark.parametrize("dash", ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "\uff0d"])
    def test_every_dash_variant_becomes_an_ascii_hyphen(self, dash):
        text, notes = normalize_query_text(f"CWE{dash}79")
        assert text == "CWE-79" and notes

    def test_full_width_letters_are_folded_by_nfkc(self):
        text, notes = normalize_query_text("\uff33\uff31\uff2c")
        assert text == "SQL" and any("NFKC" in n for n in notes)

    def test_soft_hyphen_is_removed_and_zero_width_space_separates(self):
        assert normalize_query_text("sql\u00adinjection")[0] == "sqlinjection"
        assert normalize_query_text("sql\u200binjection")[0] == "sql injection"


# ---------------------------------------------------------------- identifiers

class TestIdentifiers:
    @pytest.mark.parametrize("text", ["CWE-89", "cwe-89", "CWE 89", "CWE_89", "CWE\u201189"])
    def test_identifier_spellings_compile_to_the_canonical_form(self, text):
        assert compile_(text).terms == ("CWE-89",)

    def test_identifier_is_one_phrase_not_a_minus_operator(self):
        assert compile_("fix CWE-89").expression == '"CWE-89" OR "fix"'

    def test_several_identifiers_are_all_kept_and_ranked_first(self):
        compiled = compile_("compare CAPEC-66 with CWE-89")
        assert compiled.terms[:2] == ("CWE-89", "CAPEC-66") or compiled.terms[:2] == ("CAPEC-66", "CWE-89")
        assert set(compiled.terms[:2]) == {"CWE-89", "CAPEC-66"}

    def test_rewritten_identifier_is_reported_in_notes(self):
        assert any("'cwe-89'" in n and "'CWE-89'" in n for n in notes_of("cwe-89"))

    def test_canonical_identifier_needs_no_note(self):
        assert notes_of("CWE-89") == ()

    def test_glued_identifier_is_not_guessed(self):
        # "CWE89" is a different token; the compiler must not invent an identifier.
        assert compile_("CWE89 injection").terms == ("cwe89", "injection")

    def test_identifier_query_is_not_low_information(self):
        compiled = compile_("CWE-89")
        assert compiled.low_information is False
        assert not any(LOW_INFORMATION_PREFIX in n for n in compiled.notes)

    def test_repeated_identifier_is_kept_once(self):
        assert compile_("CWE-89 CWE-89 cwe 89").terms == ("CWE-89",)


# ---------------------------------------------------------------- quotes and punctuation

class TestQuotesAndPunctuation:
    def test_double_quoted_span_becomes_an_exact_phrase_ranked_before_its_words(self):
        # The phrase has priority; its words follow as ordinary terms (all OR-ed).
        assert compile_('the "sql injection" attack').terms == ("sql injection", "sql", "injection", "attack")

    def test_curly_quoted_span_becomes_an_exact_phrase_ranked_before_its_words(self):
        assert compile_("the \u201csql injection\u201d attack").terms == ("sql injection", "sql", "injection", "attack")

    def test_quoting_is_a_boost_not_a_rank_one_guarantee(self):
        # Documented limitation: the phrase is OR-ed with the words, so a document with the
        # words spread across chunks can still outrank one that holds the exact phrase.
        assert " OR " in compile_('"sql injection"').expression

    def test_unbalanced_quote_is_ignored_and_reported(self):
        compiled = compile_('sql "injection attack')
        assert compiled.terms == ("sql", "injection", "attack")
        assert any("unbalanced" in n for n in compiled.notes)

    def test_quoted_span_with_no_words_is_reported(self):
        assert any("no letters or digits" in n for n in notes_of('sql "!!!" injection'))

    def test_hyphenated_compound_becomes_an_adjacent_phrase_plus_words(self):
        terms = compile_("server-side request").terms
        assert terms[0] == "server side" and set(terms) == {"server side", "request", "server", "side"}

    def test_all_digit_compound_is_not_a_phrase(self):
        assert "10 20" not in compile_("range 10-20 values").terms

    @pytest.mark.parametrize("text", ["sql OR injection", "sql AND injection", "sql NOT injection", "inj*", "NEAR(sql injection)",
                                      "col:sql", "sql ^injection", "(sql", "sql)", "sql -injection", "a'b", "sql'; DROP TABLE x;--"])
    def test_fts5_operators_are_data_never_syntax(self, text):
        store = make_store(*small_corpus())
        with store:
            result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text=text + " database"))
        assert isinstance(result, RetrievalResult)

    def test_punctuation_only_query_raises_the_typed_error(self):
        with pytest.raises(InvalidRetrievalQueryError):
            compile_("!!! ??? ...")

    def test_only_stop_words_raises_the_typed_error(self):
        with pytest.raises(InvalidRetrievalQueryError):
            compile_("the and of it")

    def test_ignored_words_are_reported_not_silently_lost(self):
        assert any("ignored low-content word" in n and "the" in n for n in notes_of("the database"))

    def test_whitespace_only_query_is_refused_by_the_query_object(self):
        with pytest.raises(InvalidRetrievalQueryError):
            RetrievalQuery(text="   ")


# ---------------------------------------------------------------- repeated, low-information, Unicode, limits

class TestOtherQueryShapes:
    def test_repeated_words_are_collapsed(self):
        assert compile_("injection injection INJECTION").terms == ("injection",)

    def test_one_word_query_runs_but_is_flagged_low_information(self):
        compiled = compile_("attack")
        assert compiled.low_information is True
        assert any(n.startswith(LOW_INFORMATION_PREFIX) for n in compiled.notes)

    def test_one_repeated_word_is_still_low_information(self):
        assert compile_("attack attack attack").low_information is True

    def test_two_distinct_words_are_not_flagged(self):
        assert compile_("attack surface").low_information is False

    def test_unicode_letters_are_searchable(self):
        assert compile_("\u0438\u043d\u044a\u0435\u043a\u0446\u0438\u044f \u0434\u0430\u043d\u043d\u044b\u0445").terms

    @pytest.mark.parametrize("text", ["\U0001f600 injection attack", "sql\u0000injection attack"])
    def test_emoji_and_control_characters_do_not_crash(self, text):
        assert "injection" in compile_(text).terms

    def test_long_query_is_cut_from_the_end_and_reported(self):
        words = [f"word{i:04d}" for i in range(400)]
        compiled = compile_("CWE-89 " + " ".join(words)[:3900])
        assert len(compiled.expression) <= MAX_QUERY_LENGTH
        assert compiled.terms[0] == "CWE-89" and compiled.dropped
        assert any("cut" in n for n in compiled.notes)

    def test_phrases_from_the_query_object_are_included(self):
        assert "sql injection" in compile_("attack", phrases=("SQL-Injection",)).terms or \
            "sql injection" in compile_("attack", phrases=("sql injection",)).terms

    def test_compilation_is_deterministic(self):
        a, b = compile_('Fix "sql injection" CWE-89'), compile_('Fix "sql injection" CWE-89')
        assert (a.expression, a.terms, a.notes) == (b.expression, b.terms, b.notes)


# ---------------------------------------------------------------- the retriever

IDENT_DOCS = [
    make_doc("CWE-10", "Input Handling", "database query built from request text. see CWE-77 for related issues"),
    make_doc("CWE-77", "Command Handling", "shell command built from request text with metacharacters"),
    make_doc("CAPEC-10", "Crafted Request", "attacker sends crafted request text to the database", source="capec",
             doc_type="attack_pattern"),
    make_doc("CAPEC-77", "Command Abuse", "attacker injects shell command into request text", source="capec",
             doc_type="attack_pattern"),
]


@pytest.fixture()
def store():
    with make_store(*IDENT_DOCS) as s:
        yield s


class TestRetriever:
    def test_name_reflects_the_aggregation(self, store):
        assert Fts5RetrieverV2(store).name == "fts5-bm25-v2"
        assert Fts5RetrieverV2(store, aggregation="best_chunk").name == "fts5-bm25-v2-bestchunk"

    def test_unknown_aggregation_and_wrong_store_type_are_refused(self, store):
        with pytest.raises(ValueError):
            Fts5RetrieverV2(store, aggregation="mean")
        with pytest.raises(TypeError):
            Fts5RetrieverV2(object())

    def test_non_query_argument_is_a_typed_error(self, store):
        with pytest.raises(InvalidRetrievalQueryError):
            Fts5RetrieverV2(store).retrieve("database")

    def test_result_is_source_level_with_provenance_and_chunks(self, store):
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="database request text"))
        assert result.retriever == "fts5-bm25-v2" and result.status == "ok"
        assert len(set(result.doc_ids)) == len(result.doc_ids)
        for item in result.items:
            assert item.doc_id == f"{item.source}:{item.source_id}"
            assert store.has_document(item.doc_id)
            assert item.chunks and all(c.chunk_id.startswith(item.doc_id + "#") for c in item.chunks)
            assert item.source_url and item.title

    def test_chunk_scores_are_positive_and_item_score_is_their_aggregate(self, store):
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="database request text"))
        for item in result.items:
            scores = sorted((c.score for c in item.chunks), reverse=True)
            assert all(s > 0 for s in scores)
            assert item.score == pytest.approx(sum(scores[:2]))

    def test_best_chunk_aggregation_scores_by_the_top_chunk_only(self, store):
        result = Fts5RetrieverV2(store, aggregation="best_chunk").retrieve(RetrievalQuery(text="database request text"))
        for item in result.items:
            assert item.score == pytest.approx(max(c.score for c in item.chunks))

    def test_results_are_ordered_by_score_then_doc_id(self, store):
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="request text"))
        keys = [(-i.score, i.doc_id) for i in result.items]
        assert keys == sorted(keys) and [i.rank for i in result.items] == list(range(1, len(keys) + 1))

    def test_repeated_calls_are_identical(self, store):
        retriever = Fts5RetrieverV2(store)
        query = RetrievalQuery(text='"crafted request" database CWE-77')
        assert retriever.retrieve(query) == retriever.retrieve(query)

    def test_hyphenated_identifier_matches_text_that_contains_it(self, store):
        # Keyword search sees only indexed text (title/content), not the doc_id metadata.
        # In this synthetic corpus only CWE-10's text contains the string "CWE-77".
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="CWE-77", doc_types=("weakness",)))
        assert result.doc_ids == ("cwe:CWE-10",)

    def test_identifier_query_does_not_treat_the_hyphen_as_a_minus(self, store):
        # With a raw FTS5 string, "-77" would be a column filter / NOT operator; v2 must just search.
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="CWE-77"))
        assert result.status == "ok" and result.backend_query == '"CWE-77"'

    def test_no_match_gives_no_results_not_an_error(self, store):
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="zzzzzz yyyyyy"))
        assert result.status == "no_results" and result.items == ()

    def test_unsearchable_query_raises_not_returns_empty(self, store):
        with pytest.raises(InvalidRetrievalQueryError):
            Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="the and of"))

    def test_notes_travel_in_the_result_contract(self, store):
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="cwe-77 and the request"))
        assert any("'CWE-77'" in n for n in result.query_notes)
        assert any("ignored low-content" in n for n in result.query_notes)

    def test_dropped_terms_are_reported_in_the_result(self, store):
        words = " ".join(f"word{i:04d}" for i in range(300))
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text=words[:3900]))
        assert result.dropped_terms and any("cut" in n for n in result.query_notes)

    def test_low_information_query_still_runs_and_is_flagged(self, store):
        result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="attacker"))
        assert result.status == "ok"
        assert any(n.startswith(LOW_INFORMATION_PREFIX) for n in result.query_notes)

    def test_v1_result_has_no_notes_and_is_unchanged_in_shape(self, store):
        result = Fts5Retriever(store).retrieve(RetrievalQuery(text="database request text"))
        assert result.retriever == "fts5-bm25-v1" and result.query_notes == ()


# ---------------------------------------------------------------- lane filtering

class TestLaneFiltering:
    QUERY = "request text database command"

    def run(self, store, lane_names, limit=10):
        query = RetrievalQuery(text=self.QUERY, limit=limit,
                               doc_types=tuple(t for n in lane_names for t in lanes.lane_by_name(n).doc_types))
        return Fts5RetrieverV2(store).retrieve(query)

    def test_weakness_lane_never_returns_attack_patterns(self, store):
        result = self.run(store, ["weakness"])
        assert result.items and {i.doc_type for i in result.items} == {"weakness"}

    def test_attack_pattern_lane_never_returns_weaknesses(self, store):
        result = self.run(store, ["attack_pattern"])
        assert result.items and {i.doc_type for i in result.items} == {"attack_pattern"}

    def test_both_lanes_return_each_lane_without_duplicates(self, store):
        result = self.run(store, ["weakness", "attack_pattern"])
        assert {i.doc_type for i in result.items} == {"weakness", "attack_pattern"}
        assert len(set(result.doc_ids)) == len(result.doc_ids)

    def test_lane_results_are_the_unfiltered_ranking_restricted_to_that_lane(self, store):
        everything = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text=self.QUERY, limit=10))
        for name in ("weakness", "attack_pattern"):
            doc_type = lanes.lane_by_name(name).doc_types[0]
            expected = [i.doc_id for i in everything.items if i.doc_type == doc_type]
            assert list(self.run(store, [name]).doc_ids) == expected

    def test_lane_query_keeps_everything_but_the_doc_types(self):
        base = RetrievalQuery(text="x y", phrases=("x y",), limit=7)
        for lane in lanes.LANES:
            derived = lanes.lane_query(base, lane)
            assert (derived.text, derived.phrases, derived.limit) == ("x y", ("x y",), 7)
            assert derived.doc_types == lane.doc_types

    def test_limit_applies_per_lane_query(self, store):
        assert len(self.run(store, ["weakness"], limit=1).items) == 1


# ---------------------------------------------------------------- lane definitions and RRF specification

class TestLaneDefinitions:
    def test_two_disjoint_lanes_cover_both_source_types(self):
        names = [l.name for l in lanes.LANES]
        assert names == ["weakness", "attack_pattern"]
        types = [t for l in lanes.LANES for t in l.doc_types]
        assert len(types) == len(set(types)) and set(types) == {"weakness", "attack_pattern"}

    def test_lane_lookup(self):
        assert lanes.lane_for_doc_type("weakness") is lanes.WEAKNESS_LANE
        assert lanes.lane_for_doc_type("attack_pattern") is lanes.ATTACK_PATTERN_LANE
        assert lanes.lane_for_doc_type("nonsense") is None
        with pytest.raises(InvalidRetrievalQueryError):
            lanes.lane_by_name("nonsense")

    def test_a_lane_with_an_unknown_doc_type_or_no_name_is_refused(self):
        with pytest.raises(ValueError):
            lanes.Lane("bad", ("not_a_type",), "x")
        with pytest.raises(ValueError):
            lanes.Lane("", ("weakness",), "x")

    def test_rrf_is_specified_as_constants_only_not_implemented(self):
        assert lanes.RRF_DEFAULT_K == 60 and lanes.RRF_MIN_K == 1
        public = {name for name in dir(lanes) if not name.startswith("_")}
        assert not {name for name in public if "fuse" in name.lower() or name.lower().startswith("rrf_")} - {
            "RRF_DEFAULT_K", "RRF_MIN_K"}

    def test_embedding_requirements_are_recorded_without_choosing_a_model(self):
        keys = {k for k, _ in lanes.EMBEDDING_MODEL_REQUIREMENTS}
        assert {"local_execution", "licence", "dimensions", "reproducibility", "resources", "scope", "evaluation"} <= keys

    def test_documented_tie_break_and_decisions_are_in_the_module_docstring(self):
        doc = lanes.__doc__
        for phrase in ("Reciprocal Rank Fusion", "tie", "provenance", "CVE", "fine-tun"):
            assert phrase.lower() in doc.lower()


def test_v2_module_does_not_import_embedding_or_network_libraries():
    import ast
    import inspect

    forbidden = {"numpy", "torch", "sentence_transformers", "transformers", "faiss", "sklearn", "requests",
                 "urllib", "http", "socket", "openai"}
    for module in (fts5_v2, lanes):
        tree = ast.parse(inspect.getsource(module))
        imported = {n.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for n in
                    (a.name for a in node.names)}
        imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                     if isinstance(node, ast.ImportFrom) and node.module}
        assert not imported & forbidden, (module.__name__, imported & forbidden)
