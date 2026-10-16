"""
Milestone 11B tests: the grouped evaluation harness, the paired comparison, and the
real-corpus runs (which skip with an explicit reason when the archives are absent).

Metrics are checked against hand calculations; no number here is copied from a run.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ingestion.mitre_corpus import build_corpus_documents, ingest_corpus_documents, load_manifest
from knowledge.store import KnowledgeStore
from retrieval.benchmark_v2 import (
    BenchmarkCaseV2,
    BenchmarkV2,
    LabelProvenance,
    ReviewRecord,
    load_benchmark_v2,
)
from retrieval.contract import (
    LOW_INFORMATION_PREFIX,
    InvalidRetrievalQueryError,
    RetrievalQuery,
)
from retrieval.evaluation import BenchmarkError, UnknownLabelError, corpus_fingerprint
from retrieval.evaluation_v2 import (
    CAVEATS,
    CorpusMismatchError,
    compare_reports,
    evaluate_v2,
    random_expected_recall_at_k,
    random_expected_reciprocal_rank,
)
from retrieval.fts5 import Fts5Retriever
from retrieval.fts5_v2 import Fts5RetrieverV2
from tests.retrieval_helpers import (
    ARCHIVE_SKIP_REASON,
    MANIFEST_PATH,
    REAL_CAPEC_ZIP,
    REAL_CWE_ZIP,
    archives_present,
    make_doc,
    make_store,
)
from tests.test_retrieval_benchmark_v2 import V2_PATH

needs_real_archives = pytest.mark.skipif(not archives_present(), reason=ARCHIVE_SKIP_REASON)

FP = "a" * 64
CORPUS = [f"cwe:CWE-{n}" for n in range(1, 10)]


# ---------------------------------------------------------------- helpers

def prov(author, reviews=(), status="author_only_awaiting_review"):
    return LabelProvenance("author", tuple(author), tuple(reviews), None, status)


def case(case_id, labels, *, split="development", category="terminology", style="descriptive",
         question=None, expectation="relevant_documents", reviews=(), author=None):
    return BenchmarkCaseV2(
        case_id, split, "v2", category, style, question or f"question {case_id}", "task", expectation,
        tuple(labels), "because", prov(labels if author is None else author, reviews),
    )


def bench(*cases, fingerprint=FP):
    return BenchmarkV2("b2", "b1", "d", "p", {"corpus_fingerprint": fingerprint}, tuple(cases))


class Scripted:
    """Returns pre-scripted doc IDs per question (duplicates allowed) and records every query."""

    name = "scripted"

    def __init__(self, script, notes=None):
        self.script, self.notes, self.calls = script, notes or {}, []

    def retrieve(self, query):
        self.calls.append(query)
        outcome = self.script[query.text]
        if isinstance(outcome, Exception):
            raise outcome
        items = tuple(SimpleNamespace(doc_id=d, score=1.0 / rank) for rank, d in enumerate(outcome, start=1))
        return SimpleNamespace(
            doc_ids=tuple(outcome), items=items, status="ok" if items else "no_results",
            possibly_incomplete=False, query_notes=tuple(self.notes.get(query.text, ())),
        )


def run(benchmark, script, **kw):
    kw.setdefault("corpus_doc_ids", CORPUS)
    return evaluate_v2(benchmark, Scripted(script), **kw)


# ---------------------------------------------------------------- hand-calculated metrics

class TestMetricsByHand:
    # Q1: relevant {2}; ranking 1,2,3     -> rank 2: R@1=0,   R@3=1,   R@5=1, RR=0.5
    # Q2: relevant {1,9}; ranking 1,4,5   -> rank 1: R@1=0.5, R@3=0.5, R@5=0.5, RR=1.0, 9 missing
    # Q3: relevant {7}; no results        -> R@k=0, RR=0, 7 missing
    # Means: R@1 = (0+.5+0)/3, R@3 = (1+.5+0)/3, R@5 = (1+.5+0)/3, MRR = (.5+1+0)/3
    BENCH = bench(case("Q1", ["cwe:CWE-2"]), case("Q2", ["cwe:CWE-1", "cwe:CWE-9"]), case("Q3", ["cwe:CWE-7"]))
    SCRIPT = {"question Q1": ["cwe:CWE-1", "cwe:CWE-2", "cwe:CWE-3"],
              "question Q2": ["cwe:CWE-1", "cwe:CWE-4", "cwe:CWE-5"], "question Q3": []}

    def report(self):
        return run(self.BENCH, self.SCRIPT)

    def test_aggregate_metrics(self):
        overall = self.report().to_dict()["overall"]
        assert overall["mean_recall_at_k"] == {"1": pytest.approx(0.5 / 3, abs=1e-6), "3": pytest.approx(1.5 / 3, abs=1e-6),
                                               "5": pytest.approx(1.5 / 3, abs=1e-6)}
        assert overall["mrr"] == pytest.approx(1.5 / 3, abs=1e-6)
        assert overall["cases"] == 3 and overall["num_no_results"] == 1 and overall["num_with_missing_relevant"] == 2

    def test_per_case_values_and_missing_documents(self):
        by_id = {r.case_id: r for r in self.report().scored}
        assert (by_id["Q1"].first_relevant_rank, by_id["Q1"].reciprocal_rank) == (2, 0.5)
        assert (by_id["Q1"].recall(1), by_id["Q1"].recall(3)) == (0.0, 1.0)
        assert by_id["Q2"].recall(1) == 0.5 and by_id["Q2"].missing_doc_ids == ("cwe:CWE-9",)
        assert by_id["Q3"].status == "no_results" and by_id["Q3"].missing_doc_ids == ("cwe:CWE-7",)
        assert by_id["Q3"].reciprocal_rank == 0.0 and by_id["Q3"].first_relevant_rank is None

    def test_duplicate_ids_in_a_ranking_count_once(self):
        b = bench(case("Q1", ["cwe:CWE-2", "cwe:CWE-3"]))
        r = run(b, {"question Q1": ["cwe:CWE-1", "cwe:CWE-1", "cwe:CWE-2", "cwe:CWE-2", "cwe:CWE-3"]}).scored[0]
        assert r.retrieved_doc_ids == ("cwe:CWE-1", "cwe:CWE-2", "cwe:CWE-3")
        assert (r.first_relevant_rank, r.recall(3)) == (2, 1.0) and r.recall(1) == 0.0

    def test_group_summaries_split_the_same_cases(self):
        b = bench(case("Q1", ["cwe:CWE-2"], category="terminology"),
                  case("Q2", ["cwe:CWE-1", "cwe:CWE-9"], category="forensic_observation", split="heldout"))
        data = run(b, {"question Q1": ["cwe:CWE-2"], "question Q2": ["cwe:CWE-1"]}).to_dict()
        assert data["by_category"]["terminology"]["mrr"] == 1.0
        assert data["by_split"]["heldout"]["mean_recall_at_k"]["1"] == 0.5
        assert data["by_label_multiplicity"]["single_relevant"]["cases"] == 1
        assert data["by_label_multiplicity"]["multiple_relevant"]["cases"] == 1
        assert data["overall"]["small_group"] is True

    def test_recall_at_10_is_reported_separately_as_a_diagnostic(self):
        data = self.report().to_dict()["overall"]
        assert "10" not in data["mean_recall_at_k"] and "10" in data["diagnostic_mean_recall_at_k"]
        assert "R@10" in self.report().render_text() and "diag" in self.report().render_text()


class TestRandomComparator:
    def test_expected_recall_at_k(self):
        assert random_expected_recall_at_k(3, 1) == pytest.approx(1 / 3)
        assert random_expected_recall_at_k(23, 10) == pytest.approx(10 / 23)
        assert random_expected_recall_at_k(3, 5) == 1.0

    def test_expected_reciprocal_rank_hand_values(self):
        # n=3, one relevant: (1 + 1/2 + 1/3)/3 = 11/18.  n=3, two relevant: (2/3)*1 + (1/3)*(1/2)... = 5/6.
        assert random_expected_reciprocal_rank(3, 1) == pytest.approx(11 / 18)
        assert random_expected_reciprocal_rank(3, 2) == pytest.approx(5 / 6)
        assert random_expected_reciprocal_rank(3, 3) == pytest.approx(1.0)


# ---------------------------------------------------------------- harness behaviour

class TestHarness:
    def test_only_the_question_text_reaches_the_retriever(self):
        b = bench(case("Q1", ["cwe:CWE-2"], question="a plain question"))
        retriever = Scripted({"a plain question": ["cwe:CWE-2"]})
        evaluate_v2(b, retriever, corpus_doc_ids=CORPUS)
        (query,) = retriever.calls
        assert (query.text, query.phrases, query.doc_types) == ("a plain question", (), None)

    def test_unknown_label_is_refused_before_any_retrieval(self):
        b = bench(case("Q1", ["cwe:CWE-2"]), case("Q2", ["cwe:CWE-404"]))
        retriever = Scripted({})
        with pytest.raises(UnknownLabelError):
            evaluate_v2(b, retriever, corpus_doc_ids=CORPUS)
        assert retriever.calls == []

    def test_corpus_fingerprint_mismatch_is_refused_unless_allowed(self):
        b = bench(case("Q1", ["cwe:CWE-2"]))
        with pytest.raises(CorpusMismatchError):
            run(b, {}, fingerprint="b" * 64)
        assert run(b, {"question Q1": ["cwe:CWE-2"]}, fingerprint="b" * 64, allow_corpus_mismatch=True).scored
        assert run(b, {"question Q1": ["cwe:CWE-2"]}, fingerprint=FP).corpus_fingerprint == FP

    def test_retrieval_error_is_recorded_as_a_failed_case_not_a_crash(self):
        b = bench(case("Q1", ["cwe:CWE-2"]))
        r = run(b, {"question Q1": InvalidRetrievalQueryError("nothing searchable")}).scored[0]
        assert r.status == "error" and "nothing searchable" in r.error and r.reciprocal_rank == 0.0

    def test_split_selection(self):
        b = bench(case("D", ["cwe:CWE-2"]), case("H", ["cwe:CWE-2"], split="heldout"))
        script = {"question D": ["cwe:CWE-2"], "question H": ["cwe:CWE-2"]}
        assert [r.case_id for r in run(b, script, split="heldout").scored] == ["H"]
        assert [r.case_id for r in run(b, script, split="development").scored] == ["D"]
        assert len(run(b, script).scored) == 2

    @pytest.mark.parametrize("kw", [
        {"split": "validation"}, {"label_mode": "loose"}, {"ks": ()}, {"ks": (0,)}, {"ks": (True,)},
        {"retrieve_limit": 0}, {"ks": (1, 20), "retrieve_limit": 10},
    ])
    def test_invalid_options_are_refused(self, kw):
        with pytest.raises(ValueError):
            run(bench(case("Q1", ["cwe:CWE-2"])), {"question Q1": []}, **kw)

    def test_final_and_agreed_label_modes(self):
        reviewer = ReviewRecord("r", "ai_subagent", "blind", ("cwe:CWE-1",), "")
        b = bench(case("Q1", ["cwe:CWE-1", "cwe:CWE-2"], reviews=[reviewer]),
                  case("Q2", ["cwe:CWE-3"], reviews=[ReviewRecord("r", "ai_subagent", "blind", ("cwe:CWE-4",), "")]))
        script = {"question Q1": ["cwe:CWE-1"], "question Q2": ["cwe:CWE-3"]}
        final = run(b, script, label_mode="final")
        agreed = run(b, script, label_mode="agreed")
        assert {r.case_id: r.relevant_doc_ids for r in final.scored} == {
            "Q1": ("cwe:CWE-1", "cwe:CWE-2"), "Q2": ("cwe:CWE-3",)}
        assert {r.case_id: r.relevant_doc_ids for r in agreed.scored} == {"Q1": ("cwe:CWE-1",)}
        assert agreed.excluded_case_ids == ("Q2",)  # no label survives unanimous agreement
        assert final.scored[0].recall(1) == 0.5 and agreed.scored[0].recall(1) == 1.0

    def test_ties_keep_the_retriever_order(self):
        # The harness never reorders: it scores exactly the ranking the retriever returned.
        b = bench(case("Q1", ["cwe:CWE-2"]))
        assert run(b, {"question Q1": ["cwe:CWE-3", "cwe:CWE-2"]}).scored[0].first_relevant_rank == 2

    def test_real_retriever_breaks_score_ties_by_doc_id(self):
        docs = [make_doc(f"CWE-{n}", "Same Title", "identical words about database queries") for n in (3, 1, 2)]
        with make_store(*docs) as store:
            result = Fts5RetrieverV2(store).retrieve(RetrievalQuery(text="identical database"))
        assert list(result.doc_ids) == ["cwe:CWE-1", "cwe:CWE-2", "cwe:CWE-3"]


class TestBehaviourCases:
    def beh(self, case_id, category, expectation):
        return case(case_id, [], category=category, expectation=expectation, style="short")

    def test_handled_by_typed_error_no_results_or_a_low_information_flag(self):
        b = bench(self.beh("L1", "low_information", "low_information"), self.beh("L2", "low_information", "low_information"),
                  self.beh("U1", "unanswerable", "no_relevant_documents"), self.beh("L3", "low_information",
                                                                                     "low_information"))
        report = run(b, {"question L1": InvalidRetrievalQueryError("x"), "question L2": ["cwe:CWE-1"],
                         "question U1": [], "question L3": ["cwe:CWE-1"]},
                     )
        by_id = {r.case_id: r for r in report.behaviour}
        assert by_id["L1"].handled and "typed error" in by_id["L1"].reason
        assert not by_id["L2"].handled and "without a low-information flag" in by_id["L2"].reason
        assert by_id["U1"].handled and by_id["U1"].num_returned == 0
        assert not report.scored  # behaviour cases never enter recall/MRR

    def test_low_information_note_counts_as_handled(self):
        b = bench(self.beh("L1", "low_information", "low_information"))
        report = evaluate_v2(b, Scripted({"question L1": ["cwe:CWE-1"]},
                                         notes={"question L1": [f"{LOW_INFORMATION_PREFIX} only one word"]}),
                             corpus_doc_ids=CORPUS)
        assert report.behaviour[0].handled and report.behaviour[0].top_doc_id == "cwe:CWE-1"

    def test_behaviour_summary_counts(self):
        b = bench(self.beh("U1", "unanswerable", "no_relevant_documents"), self.beh("U2", "unanswerable",
                                                                                     "no_relevant_documents"))
        data = run(b, {"question U1": [], "question U2": ["cwe:CWE-1"]}).to_dict()
        assert data["behaviour_summary"]["no_relevant_documents"] == {"cases": 2, "handled": 1, "not_handled": 1}


# ---------------------------------------------------------------- reports

class TestReports:
    def make(self):
        b = bench(case("Q1", ["cwe:CWE-2"]), case("L1", [], category="low_information",
                                                  expectation="low_information", style="short"))
        return run(b, {"question Q1": ["cwe:CWE-2"], "question L1": []}, benchmark_sha256="c" * 64)

    def test_json_is_deterministic_and_has_no_timestamps(self):
        first, second = self.make().to_json(), self.make().to_json()
        assert first == second and first.endswith("\n")
        lowered = first.lower()
        assert not any(word in lowered for word in ("timestamp", "generated_at", "datetime", "time_utc"))
        assert json.loads(first)["benchmark_sha256"] == "c" * 64

    def test_report_carries_corpus_version_and_caveats(self):
        data = json.loads(self.make().to_json())
        assert data["benchmark_id"] == "b2" and data["corpus_doc_count"] == len(CORPUS)
        assert data["caveats"] == list(CAVEATS) and any("No human reviewer" in c for c in data["caveats"])

    def test_text_report_mentions_small_groups_and_caveats(self):
        text = self.make().render_text()
        assert "indicative only" in text and "caveat:" in text and "random-ranking" in text

    def test_text_report_makes_no_significance_claim(self):
        text = self.make().render_text().lower()
        assert "significant" not in text.replace("no significance test", "")


class TestComparison:
    B = bench(case("Q1", ["cwe:CWE-2"]), case("Q2", ["cwe:CWE-2"]), case("Q3", ["cwe:CWE-2"]), case("Q4", ["cwe:CWE-2"]))
    OLD = {"question Q1": ["cwe:CWE-1", "cwe:CWE-2"], "question Q2": ["cwe:CWE-2"],
           "question Q3": ["cwe:CWE-3"], "question Q4": ["cwe:CWE-2"]}
    NEW = {"question Q1": ["cwe:CWE-2"], "question Q2": ["cwe:CWE-1", "cwe:CWE-2"],
           "question Q3": ["cwe:CWE-2"], "question Q4": ["cwe:CWE-2"]}

    def test_counts_and_deltas(self):
        old, new = run(self.B, self.OLD), run(self.B, self.NEW)
        result = compare_reports(old, new)
        outcomes = {c["case_id"]: c["outcome"] for c in result["changes"]}
        assert outcomes == {"Q1": "improved", "Q2": "worsened", "Q3": "improved", "Q4": "unchanged"}
        assert result["counts"] == {"improved": 2, "worsened": 1, "unchanged": 1}
        # MRR old = (.5+1+0+1)/4 = 0.625; new = (1+.5+1+1)/4 = 0.875
        assert result["delta_mrr"] == pytest.approx(0.25)
        assert result["delta_mean_recall_at_k"]["1"] == pytest.approx(0.5 - 0.25)
        assert "no significance" in result["caveat"]

    def test_comparison_refuses_different_conditions(self):
        old = run(self.B, self.OLD)
        with pytest.raises(BenchmarkError):
            compare_reports(old, run(self.B, self.NEW, label_mode="agreed"))
        with pytest.raises(BenchmarkError):
            compare_reports(old, run(self.B, self.NEW, split="development", ks=(1, 3)))
        other = bench(case("Q1", ["cwe:CWE-2"]))
        with pytest.raises(BenchmarkError):
            compare_reports(old, run(other, {"question Q1": ["cwe:CWE-2"]}))


# ---------------------------------------------------------------- real corpus (skippable)

@pytest.fixture(scope="module")
def real_store():
    manifest = load_manifest(MANIFEST_PATH)
    documents = build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
    store = KnowledgeStore(":memory:")
    ingest_corpus_documents(manifest, documents, store=store)
    yield store
    store.close()


def real_run(store, retriever, **kw):
    return evaluate_v2(load_benchmark_v2(V2_PATH), retriever, corpus_doc_ids=store.list_doc_ids(),
                       fingerprint=corpus_fingerprint(store), **kw)


@needs_real_archives
class TestRealCorpus:
    def test_real_corpus_matches_the_fingerprint_recorded_in_the_benchmark(self, real_store):
        assert corpus_fingerprint(real_store) == load_benchmark_v2(V2_PATH).corpus_fingerprint
        assert len(real_store.list_doc_ids()) == 23

    @pytest.mark.parametrize("make", [Fts5Retriever, Fts5RetrieverV2,
                                      lambda s: Fts5RetrieverV2(s, aggregation="best_chunk")])
    def test_every_retriever_runs_every_case_without_errors(self, real_store, make):
        report = real_run(real_store, make(real_store))
        assert len(report.scored) == 49 and report.excluded_case_ids == ()
        assert all(r.status in ("ok", "no_results") for r in report.scored)
        assert report.to_dict()["overall"]["num_errors"] == 0

    def test_metrics_are_valid_and_recall_does_not_decrease_with_k(self, real_store):
        overall = real_run(real_store, Fts5RetrieverV2(real_store)).to_dict()["overall"]
        recalls = [overall["mean_recall_at_k"][k] for k in ("1", "3", "5")] + [overall["diagnostic_mean_recall_at_k"]["10"]]
        assert recalls == sorted(recalls) and all(0 <= r <= 1 for r in recalls) and 0 <= overall["mrr"] <= 1

    def test_reported_metrics_match_an_independent_recomputation(self, real_store):
        retriever = Fts5RetrieverV2(real_store)
        bench_ = load_benchmark_v2(V2_PATH)
        rr, r5 = [], []
        for c in bench_.select(scored=True):
            ranked = list(retriever.retrieve(RetrievalQuery(text=c.question, limit=10)).doc_ids)
            hits = [i for i, d in enumerate(ranked, 1) if d in c.relevant_doc_ids]
            rr.append(1 / hits[0] if hits else 0.0)
            r5.append(len(set(ranked[:5]) & set(c.relevant_doc_ids)) / len(c.relevant_doc_ids))
        overall = real_run(real_store, retriever).to_dict()["overall"]
        assert overall["mrr"] == pytest.approx(sum(rr) / len(rr), abs=1e-6)
        assert overall["mean_recall_at_k"]["5"] == pytest.approx(sum(r5) / len(r5), abs=1e-6)

    def test_real_run_is_byte_identical_across_runs_and_fresh_databases(self, real_store):
        first = real_run(real_store, Fts5RetrieverV2(real_store)).to_json()
        assert first == real_run(real_store, Fts5RetrieverV2(real_store)).to_json()
        manifest = load_manifest(MANIFEST_PATH)
        documents = build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
        with KnowledgeStore(":memory:") as other:
            ingest_corpus_documents(manifest, documents, store=other)
            assert first == real_run(other, Fts5RetrieverV2(other)).to_json()

    def test_v1_and_v2_are_comparable_on_the_same_corpus_and_questions(self, real_store):
        old, new = real_run(real_store, Fts5Retriever(real_store)), real_run(real_store, Fts5RetrieverV2(real_store))
        result = compare_reports(old, new)
        assert sum(result["counts"].values()) == 49
        assert old.corpus_fingerprint == new.corpus_fingerprint and old.retriever != new.retriever

    def test_lane_filtering_on_the_real_corpus(self, real_store):
        retriever = Fts5RetrieverV2(real_store)
        for c in load_benchmark_v2(V2_PATH).select(scored=True)[:15]:
            for doc_type in ("weakness", "attack_pattern"):
                result = retriever.retrieve(RetrievalQuery(text=c.question, limit=10, doc_types=(doc_type,)))
                assert {i.doc_type for i in result.items} <= {doc_type}
            both = retriever.retrieve(RetrievalQuery(text=c.question, limit=10,
                                                     doc_types=("weakness", "attack_pattern")))
            assert len(set(both.doc_ids)) == len(both.doc_ids)

    def test_real_results_preserve_identity_and_chunks(self, real_store):
        retriever = Fts5RetrieverV2(real_store)
        for c in load_benchmark_v2(V2_PATH).select(scored=True)[:20]:
            for item in retriever.retrieve(RetrievalQuery(text=c.question, limit=10)).items:
                assert real_store.has_document(item.doc_id) and item.doc_id == f"{item.source}:{item.source_id}"
                assert item.chunks and all(m.chunk_id.startswith(item.doc_id + "#") for m in item.chunks)
