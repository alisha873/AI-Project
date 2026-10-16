"""
Milestone 11B tests: benchmark v2 file format, the frozen benchmark files, and the
separation of labels from runtime code. Offline and deterministic.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from retrieval import benchmark_v2, fts5, fts5_v2, lanes, query_builder
from retrieval.benchmark_v2 import (
    BENCHMARK_FORMAT_V2,
    SPLITS,
    load_benchmark_v2,
)
from retrieval.contract import RetrievalQuery
from retrieval.evaluation import BenchmarkError, load_benchmark
from retrieval.query_builder import ForbiddenQueryInputError, build_retrieval_queries
from tests.retrieval_helpers import BENCHMARK_PATH, REPO_ROOT

V2_PATH = REPO_ROOT / "data" / "retrieval_benchmark" / "retrieval_benchmark_v2.json"
SHEET_PATH = REPO_ROOT / "data" / "retrieval_benchmark" / "annotation_sheet_v2.json"
GUIDE_PATH = REPO_ROOT / "data" / "retrieval_benchmark" / "ANNOTATION_GUIDE_v2.md"

# Pinned on purpose: these files are frozen. An intentional change must update the pin
# AND be explained in PROJECT_STATUS.md; a silent change must fail here.
V1_SHA256 = "f3d0a0697c4e5028a5c7f14bd92e5a24fe4ba606f71505e1363664ba69aa86c3"
V2_SHA256 = "bba76432dd087c98e08dbd89772cc607f92759803fb0decd958390947b9947b5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- synthetic files

def case_scored(**over):
    case = {
        "case_id": "C1", "split": "development", "origin": "v2", "category": "terminology",
        "query_style": "descriptive", "question": "database query text", "task": "t",
        "expectation": "relevant_documents", "relevant_doc_ids": ["cwe:CWE-1"], "rationale": "r",
        "label_provenance": {
            "authored_by": "author", "author_labels": ["cwe:CWE-1"], "reviews": [],
            "adjudication": None, "review_status": "author_only_awaiting_review",
        },
    }
    case.update(over)
    return case


def case_behaviour(**over):
    case = {
        "case_id": "C2", "split": "heldout", "origin": "v2", "category": "low_information",
        "query_style": "short", "question": "attack", "task": "t", "expectation": "low_information",
        "relevant_doc_ids": [], "rationale": "r",
        "label_provenance": {
            "authored_by": "author", "author_labels": [], "reviews": [], "adjudication": None,
            "review_status": "not_applicable_no_labels",
        },
    }
    case.update(over)
    return case


def valid_data():
    return {
        "format": BENCHMARK_FORMAT_V2, "benchmark_id": "b2", "supersedes": "b1", "description": "d",
        "protocol": "p", "annotation_guide": "g.md",
        "corpus": {"cwe_version": "x", "capec_version": "y", "is_complete_catalogue": False,
                   "document_count": 4, "manifest": "m", "corpus_fingerprint": "a" * 64},
        "splits": {"development": "dev", "heldout": "held"},
        "review_summary": {"human_review": "not_performed", "ai_review_passes": 0, "note": "n",
                           "items_reviewed": 0, "exact_set_agreement": 0, "adjudicated": 0,
                           "adjudicated_case_ids": []},
        "cases": [case_scored(), case_behaviour()],
    }


def write(tmp_path, data):
    path = tmp_path / "b.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def review(labels, reviewer_type="ai_subagent"):
    return {"reviewer_id": "r1", "reviewer_type": reviewer_type, "method": "blind", "labels": labels, "note": ""}


class TestLoader:
    def test_valid_file_loads(self, tmp_path):
        bench = load_benchmark_v2(write(tmp_path, valid_data()))
        assert [c.case_id for c in bench.cases] == ["C1", "C2"]
        assert bench.corpus_fingerprint == "a" * 64
        assert [c.case_id for c in bench.select(scored=True)] == ["C1"]
        assert [c.case_id for c in bench.select(split="heldout")] == ["C2"]

    def test_unreadable_and_malformed_files_raise_the_typed_error(self, tmp_path):
        with pytest.raises(BenchmarkError):
            load_benchmark_v2(tmp_path / "missing.json")
        with pytest.raises(BenchmarkError):
            load_benchmark_v2(write(tmp_path, "{not json"))

    def test_duplicate_json_keys_are_refused(self, tmp_path):
        text = json.dumps(valid_data())[:-1] + ', "format": "again"}'
        with pytest.raises(BenchmarkError, match="duplicate key"):
            load_benchmark_v2(write(tmp_path, text))

    @pytest.mark.parametrize("mutate, message", [
        (lambda d: d.update(format="autoshield.retrieval_benchmark.v1"), "format"),
        (lambda d: d.update(extra=1), "unknown"),
        (lambda d: d.pop("protocol"), "missing"),
        (lambda d: d["corpus"].update(corpus_fingerprint="xyz"), "fingerprint"),
        (lambda d: d.update(cases=[]), "non-empty"),
        (lambda d: d["cases"].append(case_scored()), "duplicate case_id"),
        (lambda d: d["cases"][0].update(split="validation"), "split"),
        (lambda d: d["cases"][0].update(category="made_up"), "category"),
        (lambda d: d["cases"][0].update(question="  "), "question"),
        (lambda d: d["cases"][0].update(origin="v1", split="heldout"), "development"),
        (lambda d: d["cases"][0].update(relevant_doc_ids=[]), "relevant"),
        (lambda d: d["cases"][0].update(relevant_doc_ids=["cwe:CWE-1", "cwe:CWE-1"]), "duplicate"),
        (lambda d: d["cases"][0].update(relevant_doc_ids=["not a doc id"]), "document ID"),
        (lambda d: d["cases"][1].update(relevant_doc_ids=["cwe:CWE-1"]), "behaviour case"),
        (lambda d: d["cases"][1].update(category="terminology"), "does not fit"),
        (lambda d: d["cases"][0].update(category="low_information"), "does not fit"),
    ])
    def test_invalid_structure_is_refused(self, tmp_path, mutate, message):
        data = valid_data()
        mutate(data)
        with pytest.raises(BenchmarkError, match=message):
            load_benchmark_v2(write(tmp_path, data))

    def test_identifier_in_a_question_is_only_allowed_for_identifier_lookup(self, tmp_path):
        data = valid_data()
        data["cases"][0]["question"] = "what is CWE-1 about"
        with pytest.raises(BenchmarkError, match="identifier_lookup"):
            load_benchmark_v2(write(tmp_path, data))
        data["cases"][0].update(category="identifier_lookup", query_style="identifier")
        assert load_benchmark_v2(write(tmp_path, data)).cases[0].category == "identifier_lookup"

    def test_identifier_lookup_must_name_only_labelled_records(self, tmp_path):
        data = valid_data()
        data["cases"][0].update(category="identifier_lookup", question="what is CWE-2 about")
        with pytest.raises(BenchmarkError, match="labelled"):
            load_benchmark_v2(write(tmp_path, data))

    def test_labels_must_agree_with_recorded_provenance(self, tmp_path):
        data = valid_data()
        data["cases"][0]["label_provenance"]["author_labels"] = ["cwe:CWE-2"]
        with pytest.raises(BenchmarkError, match="contradict"):
            load_benchmark_v2(write(tmp_path, data))

    def test_human_status_requires_a_recorded_human_reviewer(self, tmp_path):
        data = valid_data()
        prov = data["cases"][0]["label_provenance"]
        prov.update(review_status="human_review_agreed", reviews=[review(["cwe:CWE-1"], "ai_subagent")])
        data["review_summary"].update(items_reviewed=1, exact_set_agreement=1)
        with pytest.raises(BenchmarkError, match="human reviewer"):
            load_benchmark_v2(write(tmp_path, data))

    def test_ai_review_cannot_be_declared_human_in_the_summary(self, tmp_path):
        data = valid_data()
        data["cases"][0]["label_provenance"].update(review_status="ai_review_agreed", reviews=[review(["cwe:CWE-1"])])
        data["review_summary"].update(items_reviewed=1, exact_set_agreement=1)
        assert load_benchmark_v2(write(tmp_path, data)).cases[0].provenance.review_status == "ai_review_agreed"
        data["review_summary"]["human_review"] = "performed"
        with pytest.raises(BenchmarkError, match="contradicts"):
            load_benchmark_v2(write(tmp_path, data))

    def test_agreed_status_with_a_differing_review_is_refused(self, tmp_path):
        data = valid_data()
        data["cases"][0]["label_provenance"].update(review_status="ai_review_agreed", reviews=[review(["cwe:CWE-9"])])
        data["review_summary"].update(items_reviewed=1, exact_set_agreement=1)
        with pytest.raises(BenchmarkError, match="equal the author labels"):
            load_benchmark_v2(write(tmp_path, data))

    def test_adjudication_must_account_for_the_disagreement(self, tmp_path):
        data = valid_data()
        adjudication = {
            "decision": "keep", "reason": "r", "rationale": "r", "final_labels": ["cwe:CWE-1"],
            "contested_doc_ids": ["cwe:CWE-9"], "adjudicated_by": "author", "independent": False,
            "awaiting_human_adjudication": True,
        }
        data["cases"][0]["label_provenance"].update(
            review_status="ai_review_adjudicated", reviews=[review(["cwe:CWE-1", "cwe:CWE-9"])],
            adjudication=adjudication)
        data["review_summary"].update(items_reviewed=1, exact_set_agreement=0, adjudicated=1,
                                      adjudicated_case_ids=["C1"])
        assert load_benchmark_v2(write(tmp_path, data)).cases[0].contested_doc_ids == ("cwe:CWE-9",)
        adjudication["contested_doc_ids"] = []
        with pytest.raises(BenchmarkError, match="contested"):
            load_benchmark_v2(write(tmp_path, data))

    def test_review_summary_counts_must_match_the_cases(self, tmp_path):
        data = valid_data()
        data["review_summary"]["items_reviewed"] = 5
        with pytest.raises(BenchmarkError, match="review_summary"):
            load_benchmark_v2(write(tmp_path, data))

    def test_agreed_labels_are_the_author_reviewer_intersection(self, tmp_path):
        data = valid_data()
        data["cases"][0].update(relevant_doc_ids=["cwe:CWE-1", "cwe:CWE-2"])
        prov = data["cases"][0]["label_provenance"]
        prov.update(author_labels=["cwe:CWE-1", "cwe:CWE-2"], review_status="ai_review_adjudicated",
                    reviews=[review(["cwe:CWE-1"])],
                    adjudication={"decision": "keep", "reason": "r", "rationale": "r",
                                  "final_labels": ["cwe:CWE-1", "cwe:CWE-2"], "contested_doc_ids": ["cwe:CWE-2"],
                                  "adjudicated_by": "author", "independent": False,
                                  "awaiting_human_adjudication": True})
        data["review_summary"].update(items_reviewed=1, adjudicated=1, adjudicated_case_ids=["C1"])
        case = load_benchmark_v2(write(tmp_path, data)).cases[0]
        assert case.agreed_doc_ids == ("cwe:CWE-1",) and case.relevant_doc_ids == ("cwe:CWE-1", "cwe:CWE-2")


# ---------------------------------------------------------------- the frozen files

class TestFrozenFiles:
    def test_v1_benchmark_file_is_unchanged(self):
        assert sha256(BENCHMARK_PATH) == V1_SHA256

    def test_v2_benchmark_file_is_frozen(self):
        assert sha256(V2_PATH) == V2_SHA256

    def test_v1_still_loads_with_the_v1_loader(self):
        assert len(load_benchmark(BENCHMARK_PATH).cases) == 22

    def test_v2_loads_and_is_versioned_as_the_successor_of_v1(self):
        bench = load_benchmark_v2(V2_PATH)
        assert bench.benchmark_id.endswith("-v2") and bench.supersedes == load_benchmark(BENCHMARK_PATH).benchmark_id

    def test_size_and_composition(self):
        bench = load_benchmark_v2(V2_PATH)
        assert 40 <= len(bench.cases) <= 60
        assert len({c.question for c in bench.cases}) == len(bench.cases)  # distinct questions
        assert len(bench.select(scored=True)) >= 40
        categories = {c.category for c in bench.cases}
        assert {"forensic_observation", "terminology", "attack_technique", "mitigation_detection",
                "disambiguation", "identifier_lookup", "low_information", "unanswerable"} <= categories

    def test_development_and_heldout_are_disjoint_and_both_populated(self):
        bench = load_benchmark_v2(V2_PATH)
        dev, held = bench.select(split="development"), bench.select(split="heldout")
        assert dev and held and {c.split for c in bench.cases} == set(SPLITS)
        assert not {c.case_id for c in dev} & {c.case_id for c in held}
        assert not {c.question for c in dev} & {c.question for c in held}
        assert len(bench.select(split="heldout", scored=True)) >= 15

    def test_every_v1_case_is_carried_in_development_with_its_labels_not_silently_changed(self):
        v1 = {c.case_id: c for c in load_benchmark(BENCHMARK_PATH).cases}
        v2 = {c.case_id: c for c in load_benchmark_v2(V2_PATH).cases}
        assert set(v1) <= set(v2)
        for case_id, old in v1.items():
            new = v2[case_id]
            assert new.origin == "v1" and new.split == "development" and new.question == old.question
            assert set(new.provenance.author_labels) == set(old.relevant_doc_ids)
            if set(new.relevant_doc_ids) != set(old.relevant_doc_ids):  # a change must be an explicit adjudication
                assert new.provenance.review_status.endswith("_adjudicated")
                assert new.provenance.adjudication is not None

    def test_review_status_is_honest_about_the_absence_of_human_review(self):
        bench = load_benchmark_v2(V2_PATH)
        reviewers = [r for c in bench.cases for r in c.provenance.reviews]
        assert reviewers and not any(r.is_human for r in reviewers)
        assert not any(c.provenance.review_status.startswith("human") for c in bench.cases)
        raw = json.loads(V2_PATH.read_text(encoding="utf-8"))
        assert raw["review_summary"]["human_review"] == "not_performed"
        adjudicated = [c for c in bench.cases if c.provenance.adjudication]
        assert adjudicated and all(not c.provenance.adjudication.independent for c in adjudicated)
        assert all(c.provenance.adjudication.awaiting_human_adjudication for c in adjudicated)

    def test_every_scored_case_has_provenance_and_rationale(self):
        for case in load_benchmark_v2(V2_PATH).select(scored=True):
            assert case.rationale.strip() and case.provenance.authored_by and case.provenance.author_labels
            assert case.provenance.review_status in benchmark_v2.REVIEW_STATUSES

    def test_annotation_sheet_has_questions_only_and_no_labels(self):
        sheet = json.loads(SHEET_PATH.read_text(encoding="utf-8"))
        keys, reviewer_fields = set(), []

        def walk(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    keys.add(key)
                    if key == "reviewer_relevant_doc_ids":
                        reviewer_fields.append(value)
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(sheet)
        assert not {"relevant_doc_ids", "author_labels", "rationale", "label_provenance"} & keys
        assert reviewer_fields and all(value is None for value in reviewer_fields)  # blank for the reviewer
        assert GUIDE_PATH.is_file()
        questions = {c.question for c in load_benchmark_v2(V2_PATH).cases if c.provenance.reviews}
        assert all(q in json.dumps(sheet, ensure_ascii=False) for q in questions)  # every reviewed question is on it

    def test_labels_do_not_appear_in_the_questions_of_non_lookup_cases(self):
        for case in load_benchmark_v2(V2_PATH).cases:
            if case.category != "identifier_lookup":
                lowered = case.question.lower()
                assert not any(label.partition(":")[2].lower() in lowered for label in case.relevant_doc_ids)


# ---------------------------------------------------------------- labels never reach the runtime path

class TestLabelsStayOutOfRuntime:
    @pytest.mark.parametrize("module", [fts5, fts5_v2, lanes, query_builder])
    def test_runtime_modules_do_not_import_benchmark_or_evaluation_code(self, module):
        tree = ast.parse(inspect.getsource(module))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names)
        assert not {n for n in names if "evaluation" in n or "benchmark" in n}, (module.__name__, names)

    def test_a_query_object_has_no_place_for_labels(self):
        fields = set(RetrievalQuery.__dataclass_fields__)
        assert fields == {"text", "phrases", "limit", "doc_types"}

    def test_the_production_query_builder_refuses_answer_keys(self):
        with pytest.raises(ForbiddenQueryInputError):
            build_retrieval_queries({"incident_id": "x", "relevant_doc_ids": ["cwe:CWE-89"], "findings": []})

    def test_a_deep_copy_of_the_benchmark_cannot_be_used_to_build_a_query(self):
        case = copy.deepcopy(load_benchmark_v2(V2_PATH).cases[0])
        with pytest.raises(ForbiddenQueryInputError):
            build_retrieval_queries({"incident_id": "x", "relevant_doc_ids": list(case.relevant_doc_ids),
                                     "findings": []})
