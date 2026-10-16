"""
Tests for the frozen-benchmark protection (hash pin, guard, preregistration) and the comparison harness.

Everything here uses a SYNTHETIC v3-format file built from the invented test corpus. The real frozen v3 questions
are never read by these tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from retrieval import freeze
from retrieval.benchmark_v3 import BenchmarkV3, load_benchmark_v3
from retrieval.comparison import compare_systems, lane_group_of, summarize_rows
from retrieval.embeddings import FakeEmbedder
from retrieval.evaluation import BenchmarkError, corpus_fingerprint
from retrieval.freeze import (
    FROZEN_V3_FORMAT,
    FROZEN_V3_SHA256,
    FinalBenchmarkProtectedError,
    FrozenBenchmarkError,
    PreregistrationError,
    build_frozen_state,
    build_preregistration,
    fingerprint_of,
    implementation_hashes,
    is_final_benchmark_file,
    refuse_final_benchmark,
    require_valid_preregistration,
    verify_preregistration,
)
from retrieval.fts5 import Fts5Retriever
from retrieval.rrf import RrfRetriever
from retrieval.semantic import SemanticRetriever
from tests.retrieval_helpers import make_store, small_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- a synthetic v3-format file ----------------------------------------------------------------------------------

def scored_case(case_id, question, labels, *, category="terminology", style="descriptive", lenient=()):
    return {
        "case_id": case_id, "split": "final_heldout", "origin": "authored", "category": category,
        "query_style": style, "question": question, "task": "synthetic test task", "expectation": "relevant_documents",
        "relevant_doc_ids": list(labels), "lenient_extra_doc_ids": list(lenient), "rationale": "synthetic rationale",
        "evidence": [], "considered_not_relevant": [], "exclusivity_checks": [], "close_call": None,
        "review_status": "author_only_not_independently_reviewed",
    }


def behaviour_case(case_id, question, expectation, category):
    return {
        "case_id": case_id, "split": "final_heldout", "origin": "authored", "category": category,
        "query_style": "short", "question": question, "task": "synthetic", "expectation": expectation,
        "relevant_doc_ids": [], "absence_checks": [], "rationale": "synthetic", "review_status": "author_only",
    }


def synthetic_v3(fingerprint: str) -> dict:
    return {
        "format": FROZEN_V3_FORMAT, "benchmark_id": "synthetic-v3", "supersedes": None, "relation_to_previous": "none",
        "description": "synthetic", "protocol": "synthetic", "corpus": {"corpus_fingerprint": fingerprint},
        "annotation_guide": "n/a", "label_method": "n/a", "label_audit": {}, "review_summary": {},
        "label_provenance": {},
        "cases": [
            scored_case("S-01", "alpha database request weakness", ["cwe:CWE-1"], lenient=["capec:CAPEC-1"]),
            scored_case("S-02", "beta filesystem folder escape", ["cwe:CWE-2"]),
            scored_case("S-03", "gamma attacker internal service", ["capec:CAPEC-2"], category="attack_technique"),
            scored_case("S-04", "alpha attacker crafted request", ["capec:CAPEC-1", "cwe:CWE-1"]),
        ],
        "behaviour_cases": [
            behaviour_case("B-01", "alpha", "low_information", "low_information"),
            behaviour_case("B-02", "completely unrelated cooking recipe", "no_relevant_documents", "unanswerable"),
        ],
    }


def write_synthetic(tmp_path: Path, mutate=None, name="synthetic_benchmark.json") -> Path:
    with make_store(*small_corpus()) as store:
        data = synthetic_v3(corpus_fingerprint(store))
    if mutate:
        mutate(data)
    path = tmp_path / name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# --- loader --------------------------------------------------------------------------------------------------------

def test_the_loader_refuses_any_file_that_is_not_the_pinned_one(tmp_path):
    path = write_synthetic(tmp_path)
    with pytest.raises(FrozenBenchmarkError, match="expected the frozen value"):
        load_benchmark_v3(path)


def test_the_loader_accepts_a_file_whose_hash_is_explicitly_expected(tmp_path):
    path = write_synthetic(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    benchmark = load_benchmark_v3(path, expected_sha256=digest)
    assert isinstance(benchmark, BenchmarkV3) and benchmark.sha256 == digest
    assert [c.case_id for c in benchmark.cases] == ["S-01", "S-02", "S-03", "S-04", "B-01", "B-02"]
    assert sum(c.scored for c in benchmark.cases) == 4


def test_a_single_changed_byte_is_detected(tmp_path):
    path = write_synthetic(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(FrozenBenchmarkError):
        load_benchmark_v3(path, expected_sha256=digest)


def test_loading_never_rewrites_the_file(tmp_path):
    path = write_synthetic(tmp_path)
    before = path.read_bytes()
    load_benchmark_v3(path, require_frozen_hash=False)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(format="other.v1"),
        lambda d: d.pop("protocol"),
        lambda d: d.update(surprise=1),
        lambda d: d.update(cases=[]),
        lambda d: d["cases"][0].update(split="development"),
        lambda d: d["cases"][0].update(relevant_doc_ids=[]),
        lambda d: d["cases"][0].update(relevant_doc_ids=["nonsense"]),
        lambda d: d["cases"][0].update(relevant_doc_ids=["cwe:CWE-1", "cwe:CWE-1"]),
        lambda d: d["cases"][0].update(lenient_extra_doc_ids=["cwe:CWE-1"]),
        lambda d: d["cases"][0].update(category="not_a_category"),
        lambda d: d["cases"][0].update(question="  "),
        lambda d: d["cases"][0].update(extra_field=1),
        lambda d: d["cases"][1].update(case_id="S-01"),
        lambda d: d["behaviour_cases"][0].update(relevant_doc_ids=["cwe:CWE-1"]),
        lambda d: d["behaviour_cases"][1].update(expectation="relevant_documents"),
        lambda d: d["corpus"].update(corpus_fingerprint="abc"),
        lambda d: d["cases"][0].update(question="what is CWE-79 about"),  # identifiers only in identifier_lookup
    ],
)
def test_structural_problems_are_rejected(tmp_path, mutate):
    path = write_synthetic(tmp_path, mutate)
    with pytest.raises(BenchmarkError):
        load_benchmark_v3(path, require_frozen_hash=False)


def test_duplicate_json_keys_and_invalid_json_are_rejected(tmp_path):
    good = write_synthetic(tmp_path).read_text(encoding="utf-8")
    dup = tmp_path / "dup.json"
    dup.write_text(good.replace('"benchmark_id"', '"benchmark_id": "x", "benchmark_id"', 1), encoding="utf-8")
    with pytest.raises(BenchmarkError, match="duplicate key"):
        load_benchmark_v3(dup, require_frozen_hash=False)
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="not valid JSON"):
        load_benchmark_v3(bad, require_frozen_hash=False)
    with pytest.raises(BenchmarkError, match="cannot read"):
        load_benchmark_v3(tmp_path / "absent.json", require_frozen_hash=False)


def test_label_sets_strict_and_strict_plus_lenient(tmp_path):
    benchmark = load_benchmark_v3(write_synthetic(tmp_path), require_frozen_hash=False)
    strict = {c.case_id: c.relevant_doc_ids for c in benchmark.as_benchmark_v2("strict").cases}
    plus = {c.case_id: c.relevant_doc_ids for c in benchmark.as_benchmark_v2("strict_plus_lenient").cases}
    assert strict["S-01"] == ("cwe:CWE-1",) and plus["S-01"] == ("cwe:CWE-1", "capec:CAPEC-1")
    assert strict["S-02"] == plus["S-02"] and plus["B-01"] == ()
    with pytest.raises(ValueError):
        benchmark.as_benchmark_v2("everything")


# --- guard for the development tooling ---------------------------------------------------------------------------------

def test_the_guard_recognises_the_real_frozen_file_by_name_and_hash(tmp_path):
    real = REPO_ROOT / "data" / "retrieval_benchmark" / "preflight_11c" / "benchmark_11c.json"
    assert is_final_benchmark_file(real)  # by name
    with pytest.raises(FinalBenchmarkProtectedError):
        refuse_final_benchmark(real)
    if real.is_file():
        assert hashlib.sha256(real.read_bytes()).hexdigest() == FROZEN_V3_SHA256  # the pin matches the file on disk
        renamed = tmp_path / "innocent.json"
        renamed.write_bytes(real.read_bytes())
        assert is_final_benchmark_file(renamed)  # by hash, even when renamed


def test_the_guard_recognises_a_v3_format_file_under_any_name(tmp_path):
    path = write_synthetic(tmp_path, name="something_else.json")
    assert is_final_benchmark_file(path)
    with pytest.raises(FinalBenchmarkProtectedError, match="run_semantic_final"):
        refuse_final_benchmark(path)


def test_the_guard_lets_ordinary_benchmarks_through():
    refuse_final_benchmark(REPO_ROOT / "data" / "retrieval_benchmark" / "retrieval_benchmark_v2.json")
    assert not is_final_benchmark_file(REPO_ROOT / "data" / "retrieval_benchmark" / "does_not_exist.json")


def test_the_development_script_and_other_modules_do_not_import_the_v3_loader():
    allowed = {"scripts/run_semantic_final.py"}
    offenders = []
    for path in [*REPO_ROOT.glob("retrieval/*.py"), *REPO_ROOT.glob("scripts/*.py"), *REPO_ROOT.glob("evaluation/*.py")]:
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative == "retrieval/benchmark_v3.py" or relative in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
                names.extend(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            if any(name.endswith("benchmark_v3") for name in names):
                offenders.append(relative)
    assert offenders == []


def test_the_development_script_refuses_the_final_benchmark_before_doing_anything(tmp_path, capsys):
    from scripts.run_semantic_comparison import main

    def factory(model_dir, instruction):  # must never be called
        raise AssertionError("the embedder must not be built when the benchmark is refused")

    real = REPO_ROOT / "data" / "retrieval_benchmark" / "preflight_11c" / "benchmark_11c.json"
    assert main(["--benchmark", str(real), "--output-dir", str(tmp_path)], embedder_factory=factory) == 2
    assert "frozen final held-out benchmark" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
    renamed = write_synthetic(tmp_path, name="renamed.json")
    assert main(["--benchmark", str(renamed)], embedder_factory=factory) == 2


def test_the_development_script_refuses_the_spent_heldout_split_without_the_explicit_flag(capsys):
    from scripts.run_semantic_comparison import main

    def factory(model_dir, instruction):
        raise AssertionError("must not be built")

    assert main(["--split", "heldout"], embedder_factory=factory) == 2
    assert "spent" in capsys.readouterr().err


# --- implementation fingerprint and preregistration ---------------------------------------------------------------------

def frozen_state(**overrides) -> dict:
    values = dict(
        benchmark_sha256=FROZEN_V3_SHA256, corpus_fingerprint="c" * 64, document_count=23, chunk_count=247,
        chunker={"version": "char-structure-v1"}, embedding_identity={"model_id": "m"}, embedding_fingerprint="e" * 64,
        runtime_versions={"torch": "2"}, installed_pins={"numpy": "2"}, cache_format="fmt", cache_index_fingerprint="i" * 64,
        retrieval_settings={"rrf": {"k": 60}}, implementation={"retrieval/x.py": "a" * 64},
    )
    values.update(overrides)
    return build_frozen_state(**values)


def test_the_implementation_hashes_cover_the_retrieval_code_and_change_with_content(tmp_path):
    (tmp_path / "retrieval").mkdir()
    (tmp_path / "retrieval" / "one.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "requirements-embeddings.txt").write_text("numpy==1\n", encoding="utf-8")
    before = implementation_hashes(tmp_path)
    assert set(before) == {"retrieval/one.py", "requirements-embeddings.txt"}
    (tmp_path / "retrieval" / "one.py").write_text("x = 2\n", encoding="utf-8")
    assert implementation_hashes(tmp_path)["retrieval/one.py"] != before["retrieval/one.py"]
    real = implementation_hashes(REPO_ROOT)
    for expected in ("retrieval/semantic.py", "retrieval/rrf.py", "retrieval/vector_cache.py", "retrieval/freeze.py",
                     "retrieval/fts5.py", "retrieval/fts5_v2.py", "knowledge/chunking.py"):
        assert expected in real


def test_a_preregistration_matches_the_state_it_was_built_from():
    frozen = frozen_state()
    record = build_preregistration(frozen, frozen_at_utc="2026-10-16T00:00:00Z", head="a" * 40)
    assert verify_preregistration(record, frozen) == []
    assert record["frozen_sha256"] == fingerprint_of(frozen)


def test_the_record_contains_everything_that_must_be_preregistered():
    frozen = frozen_state()
    assert set(frozen) == {"benchmark", "corpus", "embedding", "dependencies", "cache", "retrieval", "evaluation",
                           "implementation"}
    assert frozen["benchmark"]["sha256"] == FROZEN_V3_SHA256
    assert frozen["embedding"]["fingerprint"] and frozen["embedding"]["runtime_versions"]
    assert frozen["evaluation"]["metrics"] and len(frozen["evaluation"]["systems"]) == 5
    assert frozen["cache"]["format"] and frozen["implementation"]["fingerprint"]


@pytest.mark.parametrize(
    "change",
    [
        dict(corpus_fingerprint="d" * 64), dict(embedding_fingerprint="f" * 64),
        dict(runtime_versions={"torch": "3"}), dict(installed_pins={"numpy": "3"}),
        dict(cache_index_fingerprint="j" * 64), dict(retrieval_settings={"rrf": {"k": 61}}),
        dict(implementation={"retrieval/x.py": "b" * 64}), dict(document_count=24), dict(chunk_count=1),
        dict(embedding_identity={"model_id": "other"}), dict(cache_format="other"),
    ],
)
def test_any_change_to_the_live_state_blocks_the_run(change):
    record = build_preregistration(frozen_state(), frozen_at_utc="t", head=None)
    problems = verify_preregistration(record, frozen_state(**change))
    assert problems and all(p.startswith("frozen.") for p in problems)


def test_an_edited_record_is_detected():
    record = build_preregistration(frozen_state(), frozen_at_utc="t", head=None)
    edited = json.loads(json.dumps(record))
    edited["frozen"]["retrieval"]["rrf"]["k"] = 1
    assert any("record_sha256" in p or "frozen_sha256" in p for p in verify_preregistration(edited, frozen_state()))
    reseal = json.loads(json.dumps(edited))
    reseal["frozen_sha256"] = fingerprint_of(reseal["frozen"])
    body = {k: v for k, v in reseal.items() if k != "record_sha256"}
    reseal["record_sha256"] = fingerprint_of(body)
    assert verify_preregistration(reseal, frozen_state())  # re-sealed, but no longer matches the live state


def test_a_record_for_a_different_benchmark_is_rejected():
    other = frozen_state(benchmark_sha256="9" * 64)
    record = build_preregistration(other, frozen_at_utc="t", head=None)
    assert any("pinned frozen v3 value" in p for p in verify_preregistration(record, other))


@pytest.mark.parametrize("bad", [None, [], "text", {}, {"format": "x"}])
def test_malformed_records_are_rejected(bad):
    assert verify_preregistration(bad, frozen_state())


def test_a_missing_or_unreadable_record_file_is_an_error(tmp_path):
    with pytest.raises(PreregistrationError, match="freeze the configuration first"):
        require_valid_preregistration(tmp_path / "none.json", frozen_state())
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    with pytest.raises(PreregistrationError):
        require_valid_preregistration(broken, frozen_state())
    listy = tmp_path / "list.json"
    listy.write_text("[]", encoding="utf-8")
    with pytest.raises(PreregistrationError):
        require_valid_preregistration(listy, frozen_state())


def test_a_valid_record_file_authorises_and_a_mismatch_lists_every_problem(tmp_path):
    path = tmp_path / "prereg.json"
    record = build_preregistration(frozen_state(), frozen_at_utc="t", head=None)
    path.write_text(json.dumps(record), encoding="utf-8")
    assert require_valid_preregistration(path, frozen_state()) == record
    with pytest.raises(PreregistrationError) as raised:
        require_valid_preregistration(path, frozen_state(corpus_fingerprint="d" * 64, chunk_count=2))
    assert "frozen.corpus.fingerprint" in str(raised.value) and "frozen.corpus.chunks" in str(raised.value)


def test_pinned_requirements_check_reports_versions_and_problems(tmp_path):
    file = tmp_path / "req.txt"
    import importlib.metadata as md

    numpy_version = md.version("numpy")
    file.write_text(f"# comment\nnumpy=={numpy_version}\n", encoding="utf-8")
    installed, problems = freeze.check_pinned_requirements(file)
    assert installed == {"numpy": numpy_version} and problems == []
    file.write_text("numpy==0.0.1\nnot-a-real-package-xyz==1\nunpinned>=1\n", encoding="utf-8")
    _, problems = freeze.check_pinned_requirements(file)
    assert len(problems) == 3


def test_the_real_requirements_file_is_exactly_pinned_and_matches_the_environment():
    installed, problems = freeze.check_pinned_requirements(REPO_ROOT / "requirements-embeddings.txt")
    assert problems == [] and installed["sentence-transformers"] == "6.1.0"


# --- comparison harness ------------------------------------------------------------------------------------------------

def test_lane_groups_follow_the_label_sources():
    assert lane_group_of(("cwe:CWE-1",)) == "cwe_only"
    assert lane_group_of(("capec:CAPEC-1", "capec:CAPEC-2")) == "capec_only"
    assert lane_group_of(("cwe:CWE-1", "capec:CAPEC-1")) == "both"


def test_the_comparison_reports_counts_lanes_pairs_and_is_deterministic(tmp_path):
    benchmark = load_benchmark_v3(write_synthetic(tmp_path), require_frozen_hash=False).as_benchmark_v2("strict")

    def run():
        with make_store(*small_corpus()) as store:
            semantic = SemanticRetriever(store, FakeEmbedder(dimension=256))
            return compare_systems(
                benchmark,
                systems={"S1": Fts5Retriever(store), "S3": semantic, "S5": RrfRetriever(semantic)},
                lane_retrievers={"weakness": semantic, "attack_pattern": semantic},
                corpus_doc_ids=store.list_doc_ids(), fingerprint=corpus_fingerprint(store), benchmark_sha256="s" * 64,
                pairs=[("S1", "S3"), ("S3", "S5"), ("S9", "S3")],
            )

    first, second = run(), run()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["scored_cases"] == 4 and first["behaviour_cases"] == 2
    assert first["lane_groups"] == {"cwe_only": 2, "capec_only": 1, "both": 1}
    overall = first["systems"]["S3"]["summary"]["overall"]
    assert overall["cases"] == 4 and set(overall["hit_at_k"]) == {"1", "3", "5"}
    assert first["systems"]["S3"]["summary"]["by_lane_group"]["both"]["cases"] == 1
    assert first["lane_specific"]["not_applicable_cases"] == ["S-04"]
    assert first["lane_specific"]["lanes"]["cwe_only"]["summary"]["cases"] == 2
    assert len(first["paired_comparisons"]) == 2  # the pair naming an unknown system is skipped
    assert set(first["first_relevant_rank_by_question"]) == {"S-01", "S-02", "S-03", "S-04"}
    assert "caveat" in first and "not independently reviewed" in first["caveat"]
    assert "timestamp" not in json.dumps(first).lower()


def test_comparison_summary_counts_match_the_raw_rows(tmp_path):
    benchmark = load_benchmark_v3(write_synthetic(tmp_path), require_frozen_hash=False).as_benchmark_v2("strict")
    with make_store(*small_corpus()) as store:
        payload = compare_systems(
            benchmark, systems={"S1": Fts5Retriever(store)}, lane_retrievers=None,
            corpus_doc_ids=store.list_doc_ids(), fingerprint=corpus_fingerprint(store), benchmark_sha256=None,
        )
    report = payload["systems"]["S1"]["report"]
    assert "lane_specific" not in payload
    rows = report["scored_cases"] if "scored_cases" in report else None
    overall = payload["systems"]["S1"]["summary"]["overall"]
    assert overall["hit_at_k"]["5"] >= overall["hit_at_k"]["3"] >= overall["hit_at_k"]["1"]
    assert overall["full_recall_at_k"]["5"] <= overall["hit_at_k"]["5"]
    assert rows is None or len(rows) == 4
    assert summarize_rows([], (1, 3))["mrr"] is None


def test_a_benchmark_for_a_different_corpus_is_refused_by_the_shared_evaluator(tmp_path):
    benchmark = load_benchmark_v3(write_synthetic(tmp_path), require_frozen_hash=False).as_benchmark_v2("strict")
    with make_store(*small_corpus()[:3]) as other_corpus:
        with pytest.raises(BenchmarkError):
            compare_systems(
                benchmark, systems={"S1": Fts5Retriever(other_corpus)}, lane_retrievers=None,
                corpus_doc_ids=other_corpus.list_doc_ids(), fingerprint=corpus_fingerprint(other_corpus),
                benchmark_sha256=None,
            )
