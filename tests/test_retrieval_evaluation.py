"""
Tests for `retrieval.evaluation` (metrics, benchmark loader, harness).

Metric tests use hand-calculated examples. Harness tests use (a) a scripted
fake retriever, to prove the numbers come from what the retriever returned,
and (b) the real `Fts5Retriever` on small synthetic stores. Real-corpus tests
use the pinned MITRE archives and are skipped, with an explicit reason, when
they are absent. No expected benchmark SCORE is hard-coded anywhere.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingestion.mitre_corpus import build_corpus_documents, ingest_corpus_documents, load_manifest
from knowledge.store import KnowledgeStore
from retrieval import evaluation
from retrieval.contract import (
    ChunkMatch,
    InvalidRetrievalQueryError,
    RetrievalBackendError,
    RetrievalQuery,
    RetrievalResult,
    RetrievedItem,
)
from retrieval.evaluation import (
    BENCHMARK_FORMAT,
    Benchmark,
    BenchmarkCase,
    BenchmarkError,
    UnknownLabelError,
    corpus_fingerprint,
    dedupe_ranking,
    evaluate_benchmark,
    first_relevant_rank,
    load_benchmark,
    recall_at_k,
    reciprocal_rank,
)
from retrieval.fts5 import Fts5Retriever
from tests.retrieval_helpers import (
    ARCHIVE_SKIP_REASON,
    BENCHMARK_PATH,
    MANIFEST_PATH,
    REAL_CAPEC_ZIP,
    REAL_CWE_ZIP,
    archives_present,
    make_doc,
    make_store,
)

needs_real_archives = pytest.mark.skipif(not archives_present(), reason=ARCHIVE_SKIP_REASON)


# ---------------------------------------------------------------- helpers

def case(case_id="C1", labels=("cwe:CWE-1",), question="database query text", style="descriptive",
         task="task", rationale="because"):
    return BenchmarkCase(
        case_id=case_id,
        question=question,
        task=task,
        relevant_doc_ids=tuple(labels),
        rationale=rationale,
        query_style=style,
    )


def benchmark(*cases):
    return Benchmark("b-test", "description", "protocol", {}, tuple(cases))


class ScriptedRetriever:
    """Returns pre-scripted ranked doc IDs for a question; records how it was called."""

    name = "scripted"

    def __init__(self, script):
        self.script = script
        self.calls = []

    def retrieve(self, query):
        self.calls.append(query)
        ids = self.script[query.text]
        items = tuple(
            RetrievedItem(
                rank, doc_id, doc_id.split(":")[0], doc_id.split(":")[1], "weakness", "T", None, 1.0,
                (ChunkMatch(f"{doc_id}#0000", 0, 0, 1, "x", 1.0),),
            )
            for rank, doc_id in enumerate(ids, start=1)
        )
        return RetrievalResult(query, self.name, items, "ok" if items else "no_results")


def valid_file_data():
    return {
        "format": BENCHMARK_FORMAT,
        "benchmark_id": "b",
        "description": "d",
        "protocol": "p",
        "corpus": {"note": "x"},
        "cases": [
            {"case_id": "C1", "query_style": "descriptive", "question": "database query text",
             "task": "t", "relevant_doc_ids": ["cwe:CWE-1"], "rationale": "r"},
        ],
    }


def write_benchmark(tmp_path, data):
    path = tmp_path / "bench.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------- metrics

class TestMetricsByHand:
    RANKING = ["a", "b", "c", "d"]
    RELEVANT = ["b", "d", "z"]  # "z" is never returned

    def test_recall_at_k_hand_calculated(self):
        assert recall_at_k(self.RANKING, self.RELEVANT, 1) == 0.0
        assert recall_at_k(self.RANKING, self.RELEVANT, 2) == pytest.approx(1 / 3)  # b
        assert recall_at_k(self.RANKING, self.RELEVANT, 3) == pytest.approx(1 / 3)
        assert recall_at_k(self.RANKING, self.RELEVANT, 4) == pytest.approx(2 / 3)  # b, d
        assert recall_at_k(self.RANKING, self.RELEVANT, 100) == pytest.approx(2 / 3)  # z never appears

    def test_reciprocal_rank_hand_calculated(self):
        assert reciprocal_rank(self.RANKING, self.RELEVANT) == pytest.approx(1 / 2)  # b is at rank 2
        assert reciprocal_rank(["b", "a"], ["b"]) == 1.0
        assert reciprocal_rank(["x", "y", "z", "q", "b"], ["b"]) == pytest.approx(1 / 5)
        assert reciprocal_rank(["x", "y"], ["b"]) == 0.0

    def test_first_relevant_rank(self):
        assert first_relevant_rank(self.RANKING, self.RELEVANT) == 2
        assert first_relevant_rank(["x"], ["b"]) is None

    def test_mean_reciprocal_rank_and_recall_over_two_cases_by_hand(self):
        report = evaluate_benchmark(
            benchmark(case("C1", ["cwe:B"], "q one"), case("C2", ["cwe:A", "cwe:C"], "q two")),
            ScriptedRetriever({"q one": ["cwe:B", "cwe:A"], "q two": ["cwe:X", "cwe:Y", "cwe:A"]}),
            corpus_doc_ids=["cwe:A", "cwe:B", "cwe:C", "cwe:X", "cwe:Y"],
            ks=(1, 3), retrieve_limit=3,
        )
        # Case 1: first relevant at rank 1 -> RR 1; Recall@1 = Recall@3 = 1.
        # Case 2: cwe:A at rank 3 -> RR 1/3; cwe:C never returned -> Recall@1 = 0, Recall@3 = 1/2.
        assert report.mrr() == pytest.approx((1 + 1 / 3) / 2)
        assert report.mean_recall_at_k() == {1: pytest.approx(0.5), 3: pytest.approx(0.75)}

    def test_source_level_deduplication_counts_each_source_once(self):
        ranking = ["a", "a", "b", "a", "c"]  # chunk-level hits: a matched three times
        assert dedupe_ranking(ranking) == ["a", "b", "c"]
        assert first_relevant_rank(ranking, ["c"]) == 3  # not 5
        assert reciprocal_rank(ranking, ["c"]) == pytest.approx(1 / 3)
        assert recall_at_k(ranking, ["c"], 2) == 0.0 and recall_at_k(ranking, ["c"], 3) == 1.0
        assert recall_at_k(["a", "a", "a"], ["a", "b"], 3) == 0.5  # one relevant source, not three

    def test_empty_ranking_scores_zero_not_undefined(self):
        assert recall_at_k([], ["a"], 5) == 0.0
        assert reciprocal_rank([], ["a"]) == 0.0
        assert first_relevant_rank([], ["a"]) is None

    def test_no_relevant_labels_is_undefined_not_zero(self):
        assert recall_at_k(["a"], [], 3) is None
        assert reciprocal_rank(["a"], []) is None

    def test_duplicate_labels_and_bad_k_are_rejected(self):
        with pytest.raises(BenchmarkError, match="duplicate"):
            recall_at_k(["a"], ["a", "a"], 1)
        with pytest.raises(BenchmarkError):
            reciprocal_rank(["a"], ["b", "b"])
        for bad in (0, -1, True, "3", 2.0):
            with pytest.raises(ValueError):
                recall_at_k(["a"], ["a"], bad)  # type: ignore[arg-type]

    def test_ties_do_not_create_fractional_ranks(self):
        # Two documents the retriever scored identically still occupy distinct positions.
        report = evaluate_benchmark(
            benchmark(case("C1", ["cwe:B"], "q")),
            ScriptedRetriever({"q": ["cwe:A", "cwe:B"]}),
            corpus_doc_ids=["cwe:A", "cwe:B"], ks=(1, 2), retrieve_limit=2,
        )
        result = report.case_results[0]
        assert result.first_relevant_rank == 2 and result.reciprocal_rank == 0.5


# ---------------------------------------------------------------- cases and loader

class TestBenchmarkCaseValidation:
    def test_valid_case(self):
        assert case().relevant_doc_ids == ("cwe:CWE-1",)

    @pytest.mark.parametrize("bad_id", ["", " x", "a b", "-x", "x" * 65])
    def test_bad_case_id(self, bad_id):
        with pytest.raises(BenchmarkError, match="case_id"):
            case(bad_id)

    @pytest.mark.parametrize("field", ["question", "task", "rationale"])
    def test_blank_text_fields_are_rejected(self, field):
        with pytest.raises(BenchmarkError, match=field):
            case(**{field: "  "})

    def test_invalid_query_style(self):
        with pytest.raises(BenchmarkError, match="query_style"):
            case(style="other")

    def test_empty_labels_are_rejected(self):
        with pytest.raises(BenchmarkError, match="at least one"):
            case(labels=())

    def test_duplicate_labels_are_rejected_with_the_case_named(self):
        with pytest.raises(BenchmarkError, match="C1.*duplicate"):
            case(labels=("cwe:CWE-1", "cwe:CWE-1"))

    @pytest.mark.parametrize("label", ["CWE-89", "nosuchsource:CWE-1", "cwe:", "cwe:CWE 1", ":x", "cwe:a#b"])
    def test_invalid_relevance_labels_are_rejected_clearly(self, label):
        with pytest.raises(BenchmarkError, match="invalid document ID"):
            case(labels=(label,))

    @pytest.mark.parametrize("question", [
        "Suspected CWE-89 in the search endpoint",
        "see cwe-89", "pattern CAPEC-66 observed", "matches CVE-2021-44228",
    ])
    def test_questions_may_not_contain_answer_identifiers(self, question):
        with pytest.raises(BenchmarkError, match="identifier"):
            case(question=question)


class TestLoader:
    def test_valid_file_loads(self, tmp_path):
        loaded = load_benchmark(write_benchmark(tmp_path, valid_file_data()))
        assert loaded.benchmark_id == "b" and [c.case_id for c in loaded.cases] == ["C1"]

    @pytest.mark.parametrize(
        "change",
        [
            lambda d: d.update(format="other"),
            lambda d: d.update(extra=1),
            lambda d: d.pop("protocol"),
            lambda d: d.update(cases=[]),
            lambda d: d.update(cases={}),
            lambda d: d["cases"][0].update(expected=["cwe:CWE-1"]),
            lambda d: d["cases"][0].pop("rationale"),
            lambda d: d["cases"][0].update(relevant_doc_ids="cwe:CWE-1"),
            lambda d: d["cases"].append(copy.deepcopy(d["cases"][0])),
            lambda d: d.update(description=" "),
        ],
        ids=["format", "unknown-key", "missing-key", "empty-cases", "cases-not-list",
             "case-unknown-key", "case-missing-key", "labels-not-list", "duplicate-case-id",
             "blank-description"],
    )
    def test_structural_errors_are_rejected(self, tmp_path, change):
        data = valid_file_data()
        change(data)
        with pytest.raises(BenchmarkError):
            load_benchmark(write_benchmark(tmp_path, data))

    def test_duplicate_json_keys_unreadable_and_invalid_json_are_errors(self, tmp_path):
        with pytest.raises(BenchmarkError, match="duplicate key"):
            load_benchmark(write_benchmark(tmp_path, '{"format": "a", "format": "b"}'))
        with pytest.raises(BenchmarkError, match="not valid JSON"):
            load_benchmark(write_benchmark(tmp_path, "{nope"))
        with pytest.raises(BenchmarkError, match="cannot read"):
            load_benchmark(tmp_path / "missing.json")

    def test_a_case_error_names_the_case(self, tmp_path):
        data = valid_file_data()
        data["cases"][0]["relevant_doc_ids"] = []
        with pytest.raises(BenchmarkError, match="C1"):
            load_benchmark(write_benchmark(tmp_path, data))


# ---------------------------------------------------------------- harness

class TestHarness:
    def test_unknown_labels_are_rejected_before_any_retrieval(self):
        retriever = ScriptedRetriever({"q": ["cwe:A"]})
        with pytest.raises(UnknownLabelError, match=r"C1.*cwe:MISSING"):
            evaluate_benchmark(benchmark(case("C1", ["cwe:MISSING"], "q")), retriever,
                               corpus_doc_ids=["cwe:A"], ks=(1,), retrieve_limit=1)
        assert retriever.calls == []

    def test_metrics_follow_the_retrievers_actual_output(self):
        bench = benchmark(case("C1", ["cwe:A"], "q"))
        corpus = ["cwe:A", "cwe:B"]
        hit = evaluate_benchmark(bench, ScriptedRetriever({"q": ["cwe:A", "cwe:B"]}),
                                 corpus_doc_ids=corpus, ks=(1,), retrieve_limit=2)
        miss = evaluate_benchmark(bench, ScriptedRetriever({"q": ["cwe:B"]}),
                                  corpus_doc_ids=corpus, ks=(1,), retrieve_limit=2)
        assert hit.mrr() == 1.0 and hit.mean_recall_at_k() == {1: 1.0}
        assert miss.mrr() == 0.0 and miss.mean_recall_at_k() == {1: 0.0}
        assert hit.case_results[0].retrieved_doc_ids == ("cwe:A", "cwe:B")

    def test_the_question_is_the_only_text_given_to_the_retriever(self):
        retriever = ScriptedRetriever({"the question": ["cwe:A"]})
        evaluate_benchmark(benchmark(case("C1", ["cwe:A"], "the question", rationale="SECRET-RATIONALE",
                                          task="SECRET-TASK")),
                           retriever, corpus_doc_ids=["cwe:A"], ks=(1,), retrieve_limit=5)
        assert len(retriever.calls) == 1
        sent = retriever.calls[0]
        assert sent == RetrievalQuery(text="the question", limit=5)
        assert "cwe:A" not in sent.text and not sent.phrases and sent.doc_types is None

    def test_empty_results_score_zero_and_are_counted(self):
        report = evaluate_benchmark(benchmark(case("C1", ["cwe:A"], "q")), ScriptedRetriever({"q": []}),
                                    corpus_doc_ids=["cwe:A"], ks=(1, 2), retrieve_limit=2)
        result = report.case_results[0]
        assert result.status == "no_results" and result.first_relevant_rank is None
        assert result.reciprocal_rank == 0.0 and dict(result.recall_at_k) == {1: 0.0, 2: 0.0}
        assert report.num_no_results == 1 and report.num_errors == 0 and report.mrr() == 0.0

    def test_unrunnable_queries_score_zero_are_flagged_and_stay_in_the_means(self):
        with make_store(make_doc("CWE-1", "Alpha", "alpha database")) as store:
            report = evaluate_benchmark(
                benchmark(case("C1", ["cwe:CWE-1"], "alpha database"), case("C2", ["cwe:CWE-1"], "the of and")),
                Fts5Retriever(store), corpus_doc_ids=store.list_doc_ids(), ks=(1,), retrieve_limit=3)
        ok, failed = report.case_results
        assert ok.status == "ok" and failed.status == "error" and "no searchable terms" in failed.error
        assert failed.reciprocal_rank == 0.0 and report.num_errors == 1
        assert report.mrr() == 0.5  # the failure is averaged in, not excluded

    def test_unexpected_exceptions_are_not_swallowed(self):
        class Broken:
            name = "broken"
            def retrieve(self, query):
                raise RuntimeError("bug")
        with pytest.raises(RuntimeError):
            evaluate_benchmark(benchmark(case("C1", ["cwe:A"], "q")), Broken(),
                               corpus_doc_ids=["cwe:A"], ks=(1,), retrieve_limit=1)

    def test_backend_errors_are_recorded_as_case_errors(self):
        class Failing:
            name = "failing"
            def retrieve(self, query):
                raise RetrievalBackendError("db down")
        report = evaluate_benchmark(benchmark(case("C1", ["cwe:A"], "q")), Failing(),
                                    corpus_doc_ids=["cwe:A"], ks=(1,), retrieve_limit=1)
        assert report.case_results[0].status == "error" and "db down" in report.case_results[0].error

    def test_parameter_validation(self):
        bench, retriever = benchmark(case("C1", ["cwe:A"], "q")), ScriptedRetriever({"q": []})
        for ks, limit in [((), 5), ((0,), 5), ((True,), 5), ((11,), 10), ((1,), 0), ((1,), True)]:
            with pytest.raises(ValueError):
                evaluate_benchmark(bench, retriever, corpus_doc_ids=["cwe:A"], ks=ks, retrieve_limit=limit)

    def test_a_retriever_that_repeats_sources_is_not_double_counted(self):
        # A chunk-level retriever might return the same source several times. The harness must
        # still count each source once and rank by distinct sources.
        class ChunkLevel:
            name = "chunk-level"

            def retrieve(self, query):
                return SimpleNamespace(
                    doc_ids=("cwe:A", "cwe:A", "cwe:A", "cwe:B"), status="ok", possibly_incomplete=False)

        report = evaluate_benchmark(benchmark(case("C1", ["cwe:B"], "q")), ChunkLevel(),
                                    corpus_doc_ids=["cwe:A", "cwe:B"], ks=(1, 2), retrieve_limit=4)
        result = report.case_results[0]
        assert result.retrieved_doc_ids == ("cwe:A", "cwe:B")
        assert result.first_relevant_rank == 2 and result.reciprocal_rank == 0.5

    def test_real_fts5_multichunk_document_does_not_crowd_out_other_sources(self):
        long_doc = make_doc("CWE-50", "Long", " ".join(["floodterm filler"] * 1500))
        other = make_doc("CWE-51", "Other", "floodterm appears once here")
        with make_store(long_doc, other) as store:
            assert store.count_chunks() > 5
            report = evaluate_benchmark(
                benchmark(case("C1", ["cwe:CWE-51"], "floodterm")), Fts5Retriever(store),
                corpus_doc_ids=store.list_doc_ids(), ks=(1, 2), retrieve_limit=5)
        result = report.case_results[0]
        assert sorted(result.retrieved_doc_ids) == ["cwe:CWE-50", "cwe:CWE-51"]
        assert result.first_relevant_rank == 2  # a second source, not rank > 2 after chunk repeats

    def test_real_fts5_metrics_change_when_the_corpus_changes(self):
        bench = benchmark(case("C1", ["cwe:CWE-1"], "alpha database"))
        with make_store(make_doc("CWE-1", "Alpha", "alpha database"), make_doc("CWE-2", "Beta", "beta")) as store:
            before = evaluate_benchmark(bench, Fts5Retriever(store), corpus_doc_ids=["cwe:CWE-1"],
                                        ks=(1,), retrieve_limit=3)
            store.delete_document("cwe:CWE-1")
            after = evaluate_benchmark(bench, Fts5Retriever(store), corpus_doc_ids=["cwe:CWE-1"],
                                       ks=(1,), retrieve_limit=3)
        assert before.mrr() == 1.0 and after.mrr() == 0.0 and after.case_results[0].status == "no_results"


class TestReports:
    def make_report(self):
        docs = [make_doc("CWE-1", "Alpha", "alpha database query"),
                make_doc("CWE-2", "Beta", "beta filesystem path"),
                make_doc("CAPEC-1", "Gamma", "gamma attacker request", source="capec", doc_type="attack_pattern")]
        bench = benchmark(
            case("C1", ["cwe:CWE-1"], "alpha database", "terminology"),
            case("C2", ["cwe:CWE-2", "capec:CAPEC-1"], "filesystem attacker", "descriptive"),
            case("C3", ["cwe:CWE-1"], "nothingmatches", "descriptive"),
        )
        with make_store(*docs) as store:
            return evaluate_benchmark(bench, Fts5Retriever(store), corpus_doc_ids=store.list_doc_ids(),
                                      ks=(1, 3), retrieve_limit=3, fingerprint=corpus_fingerprint(store))

    def test_report_is_deterministic(self):
        a, b = self.make_report(), self.make_report()
        assert a == b and a.to_json() == b.to_json() and a.render_text() == b.render_text()
        assert json.loads(a.to_json()) == a.to_dict()

    def test_report_contents_and_breakdown_by_style(self):
        data = self.make_report().to_dict()
        assert data["num_cases"] == 3 and data["num_no_results"] == 1 and data["num_errors"] == 0
        assert data["overall"]["cases"] == 3
        assert data["by_query_style"]["terminology"]["cases"] == 1
        assert data["by_query_style"]["descriptive"]["cases"] == 2
        assert data["by_query_style"]["terminology"]["mrr"] == 1.0
        assert all(isinstance(c["case_id"], str) for c in data["cases"])
        assert "generated" not in json.dumps(data) and "timestamp" not in json.dumps(data)

    def test_text_rendering_mentions_counts_and_every_case(self):
        text = self.make_report().render_text()
        for needle in ("3 cases", "no-result cases: 1", "R@1=", "MRR=", "C1", "C2", "C3", "terminology"):
            assert needle in text

    def test_corpus_fingerprint_is_stable_and_sensitive_to_content(self):
        with make_store(make_doc("CWE-1", "A", "alpha"), make_doc("CWE-2", "B", "beta")) as one:
            first = corpus_fingerprint(one)
            assert first == corpus_fingerprint(one) and len(first) == 64
        with make_store(make_doc("CWE-2", "B", "beta"), make_doc("CWE-1", "A", "alpha")) as two:
            assert corpus_fingerprint(two) == first  # insertion order does not matter
        with make_store(make_doc("CWE-1", "A", "alpha changed"), make_doc("CWE-2", "B", "beta")) as three:
            assert corpus_fingerprint(three) != first


# ---------------------------------------------------------------- the fixed benchmark file

class TestFixedBenchmarkFile:
    def test_file_loads_and_is_well_formed(self):
        bench = load_benchmark(BENCHMARK_PATH)
        ids = [c.case_id for c in bench.cases]
        assert len(ids) == len(set(ids)) >= 20
        assert {c.query_style for c in bench.cases} == {"terminology", "descriptive"}

    def test_every_label_is_a_record_pinned_in_the_manifest(self):
        manifest = load_manifest(MANIFEST_PATH)
        pinned = {f"cwe:CWE-{s.record_id}" for s in manifest.cwe.selection} | {
            f"capec:CAPEC-{s.record_id}" for s in manifest.capec.selection}
        bench = load_benchmark(BENCHMARK_PATH)
        used = {label for c in bench.cases for label in c.relevant_doc_ids}
        assert used <= pinned and len(pinned) == 23

    def test_questions_never_contain_identifiers_or_label_ids(self):
        for c in load_benchmark(BENCHMARK_PATH).cases:
            lowered = c.question.lower()
            assert "cwe-" not in lowered and "capec-" not in lowered and "cve-" not in lowered
            for label in c.relevant_doc_ids:
                assert label.split(":")[1].lower() not in lowered

    def test_protocol_documents_the_limits(self):
        bench = load_benchmark(BENCHMARK_PATH)
        for phrase in ("NOT blind", "manually curated", "BEFORE running any retrieval", "no negative"):
            assert phrase.lower() in (bench.protocol + bench.description).lower()

    def test_evaluation_module_does_not_use_the_runtime_query_builder(self):
        tree = ast.parse(Path(evaluation.__file__).read_text(encoding="utf-8"))
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert "retrieval.query_builder" not in imported

    def test_smoke_questions_and_benchmark_are_distinct_sets(self):
        # The old smoke checks live in tests/test_mitre_corpus.py (RETRIEVAL_CHECKS). Do not import
        # that module; compare to its source text instead.
        smoke = (Path(__file__).parent / "test_mitre_corpus.py").read_text(encoding="utf-8")
        for c in load_benchmark(BENCHMARK_PATH).cases:
            assert c.question not in smoke


# ---------------------------------------------------------------- real corpus (skippable)

@pytest.fixture(scope="module")
def real_store():
    manifest = load_manifest(MANIFEST_PATH)
    documents = build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
    store = KnowledgeStore(":memory:")
    ingest_corpus_documents(manifest, documents, store=store)
    yield store
    store.close()


@needs_real_archives
class TestRealCorpusBaseline:
    def run(self, store, **kw):
        return evaluate_benchmark(
            load_benchmark(BENCHMARK_PATH), Fts5Retriever(store), corpus_doc_ids=store.list_doc_ids(),
            fingerprint=corpus_fingerprint(store), **kw)

    def test_baseline_runs_on_the_real_corpus_without_errors(self, real_store):
        report = self.run(real_store)
        assert report.corpus_doc_count == 23 and report.num_cases == 22
        assert report.num_errors == 0 and report.num_possibly_incomplete == 0

    def test_metrics_are_valid_and_recall_never_decreases_with_k(self, real_store):
        report = self.run(real_store)
        recalls = [report.mean_recall_at_k()[k] for k in report.ks]
        assert all(0.0 <= r <= 1.0 for r in recalls) and recalls == sorted(recalls)
        assert 0.0 <= report.mrr() <= 1.0

    def test_reported_metrics_match_an_independent_recomputation_from_raw_retrieval(self, real_store):
        report = self.run(real_store)
        retriever = Fts5Retriever(real_store)
        reciprocal_ranks, recalls_at_5 = [], []
        for c in load_benchmark(BENCHMARK_PATH).cases:
            ranked = list(retriever.retrieve(RetrievalQuery(text=c.question, limit=10)).doc_ids)
            found = [i for i, doc_id in enumerate(ranked, start=1) if doc_id in c.relevant_doc_ids]
            reciprocal_ranks.append(1 / found[0] if found else 0.0)
            recalls_at_5.append(len(set(ranked[:5]) & set(c.relevant_doc_ids)) / len(c.relevant_doc_ids))
        assert report.mrr() == pytest.approx(sum(reciprocal_ranks) / len(reciprocal_ranks))
        assert report.mean_recall_at_k()[5] == pytest.approx(sum(recalls_at_5) / len(recalls_at_5))

    def test_real_results_preserve_identity_and_are_source_level(self, real_store):
        retriever = Fts5Retriever(real_store)
        for c in load_benchmark(BENCHMARK_PATH).cases:
            result = retriever.retrieve(RetrievalQuery(text=c.question, limit=10))
            assert len(set(result.doc_ids)) == len(result.doc_ids)
            for item in result.items:
                assert real_store.has_document(item.doc_id)
                assert item.doc_id == f"{item.source}:{item.source_id}"
                assert all(m.chunk_id.startswith(item.doc_id + "#") for m in item.chunks)

    def test_real_baseline_is_deterministic(self, real_store):
        first, second = self.run(real_store), self.run(real_store)
        assert first.to_json() == second.to_json()
        assert first.corpus_fingerprint == corpus_fingerprint(real_store)

    def test_a_fresh_database_gives_the_same_report(self, real_store):
        manifest = load_manifest(MANIFEST_PATH)
        documents = build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
        with KnowledgeStore(":memory:") as other:
            ingest_corpus_documents(manifest, documents, store=other)
            assert self.run(other).to_json() == self.run(real_store).to_json()

    def test_incident_001_hypothesis_query_runs_end_to_end(self, real_store):
        from retrieval.query_builder import build_queries_for_incident
        from tests.retrieval_helpers import INCIDENT_001

        if not (INCIDENT_001 / "analysis" / "root_cause_analysis.json").is_file():
            pytest.skip("incident_001/analysis artifacts are not present; end-to-end query NOT tested")
        built = build_queries_for_incident(INCIDENT_001).queries[0]
        result = Fts5Retriever(real_store).retrieve(built.query)
        assert result.status == "ok" and built.is_confirmed is False
        assert all(real_store.has_document(doc_id) for doc_id in result.doc_ids)