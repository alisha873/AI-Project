"""
Tests for the 11C command-line scripts: the development comparison, the model download/verify tool, and the
freeze -> run-once lifecycle of the final evaluation (fake embedder, synthetic corpus, synthetic benchmark).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pytest

from retrieval import benchmark_v3, freeze
from retrieval.embeddings import FakeEmbedder
from scripts import download_embedding_model, run_semantic_comparison, run_semantic_final
from tests.retrieval_helpers import ARCHIVE_SKIP_REASON, archives_present, make_store, small_corpus
from tests.test_retrieval_final_protection import write_synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_factory(model_dir, instruction):
    return FakeEmbedder(dimension=256, query_instruction="Represent: " if instruction else "")


def corpus_opener(args):
    return make_store(*small_corpus())


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A tiny repository root, a synthetic 'frozen' benchmark and the paths the final script needs."""
    root = tmp_path / "repo"
    (root / "retrieval").mkdir(parents=True)
    (root / "retrieval" / "impl.py").write_text("VALUE = 1\n", encoding="utf-8")
    numpy_version = importlib.metadata.version("numpy")
    (root / "requirements-embeddings.txt").write_text(f"numpy=={numpy_version}\n", encoding="utf-8")
    benchmark = write_synthetic(tmp_path, name="frozen_synthetic.json")
    digest = hashlib.sha256(benchmark.read_bytes()).hexdigest()
    monkeypatch.setattr(freeze, "FROZEN_V3_SHA256", digest)
    monkeypatch.setattr(benchmark_v3, "FROZEN_V3_SHA256", digest)
    out = tmp_path / "out"
    return {
        "root": root, "benchmark": benchmark, "digest": digest, "prereg": out / "prereg.json", "report": out / "report.json",
        "cache": tmp_path / "cache", "tmp": tmp_path,
    }


def args_for(lab, command, *extra):
    base = [command, "--benchmark", str(lab["benchmark"]), "--cache-dir", str(lab["cache"]),
            "--preregistration", str(lab["prereg"])]
    if command == "run":
        base += ["--report", str(lab["report"])]
    return [*base, *extra]


def call(lab, command, *extra, factory=fake_factory):
    return run_semantic_final.main(
        args_for(lab, command, *extra), embedder_factory=factory, corpus_opener=corpus_opener, repo_root=lab["root"]
    )


# --- freeze ----------------------------------------------------------------------------------------------------------

def test_freeze_writes_a_verifiable_record_without_reading_any_benchmark_case(lab, monkeypatch, capsys):
    def forbidden(*a, **k):
        raise AssertionError("freeze must not parse the benchmark cases")

    monkeypatch.setattr(run_semantic_final, "load_benchmark_v3", forbidden)
    assert call(lab, "freeze") == 0
    record = json.loads(lab["prereg"].read_text(encoding="utf-8"))
    frozen = record["frozen"]
    assert frozen["benchmark"]["sha256"] == lab["digest"]
    assert frozen["corpus"]["documents"] == 4 and frozen["corpus"]["chunks"] >= 4
    assert frozen["embedding"]["fingerprint"] == fake_factory(None, True).config.fingerprint()
    assert frozen["retrieval"]["rrf"]["k"] == 60 and "retrieve_limit" in frozen["retrieval"]
    assert "retrieval/impl.py" in frozen["implementation"]["files"]
    assert frozen["cache"]["index_fingerprint"]
    assert record["frozen_at_utc"].endswith("Z")
    assert "record sha256" in capsys.readouterr().out


def test_freeze_never_overwrites_an_existing_record(lab, capsys):
    assert call(lab, "freeze") == 0
    before = lab["prereg"].read_bytes()
    assert call(lab, "freeze") == 2
    assert lab["prereg"].read_bytes() == before
    assert "never overwritten" in capsys.readouterr().err


def test_freeze_refuses_a_benchmark_that_is_not_the_pinned_file(lab, monkeypatch, capsys):
    monkeypatch.setattr(freeze, "FROZEN_V3_SHA256", "0" * 64)
    assert call(lab, "freeze") == 2
    assert not lab["prereg"].exists()
    assert "expected the frozen value" in capsys.readouterr().err


def test_freeze_refuses_when_installed_packages_differ_from_the_pins(lab, capsys):
    (lab["root"] / "requirements-embeddings.txt").write_text("numpy==0.0.1\n", encoding="utf-8")
    assert call(lab, "freeze") == 2
    assert not lab["prereg"].exists()
    assert "numpy is" in capsys.readouterr().err


# --- run -----------------------------------------------------------------------------------------------------------------

def test_run_without_a_preregistration_is_refused_and_writes_nothing(lab, capsys):
    assert call(lab, "run") == 2
    assert not lab["report"].exists()
    assert "freeze the configuration first" in capsys.readouterr().err


def test_the_full_lifecycle_freeze_run_then_refuse_a_second_run(lab, capsys):
    assert call(lab, "freeze") == 0
    assert call(lab, "run") == 0
    report = json.loads(lab["report"].read_text(encoding="utf-8"))
    record = json.loads(lab["prereg"].read_text(encoding="utf-8"))
    assert report["format"] == run_semantic_final.REPORT_FORMAT
    assert report["preregistration"]["record_sha256"] == record["record_sha256"]
    assert report["benchmark"] == {"id": "synthetic-v3", "sha256": lab["digest"], "scored_cases": 4, "behaviour_cases": 2}
    assert report["primary_label_set"] == "strict" and set(report["label_sets"]) == {"strict", "strict_plus_lenient"}
    for payload in report["label_sets"].values():
        assert set(payload["systems"]) == {"S1_fts5_v1", "S2_fts5_v2", "S3_semantic", "S5_rrf"}
        assert "lane_specific" in payload and len(payload["paired_comparisons"]) == 5
        assert payload["systems"]["S3_semantic"]["summary"]["overall"]["cases"] == 4
    assert any("not independently reviewed" in text for text in report["limitations"])
    assert "timestamp" not in json.dumps(report["label_sets"]).lower()
    out = capsys.readouterr().out
    assert "(primary)" in out and "(sensitivity)" in out

    frozen_bytes = lab["report"].read_bytes()
    assert call(lab, "run") == 2  # run-once guard
    assert lab["report"].read_bytes() == frozen_bytes
    assert "run once" in capsys.readouterr().err


def test_the_final_report_is_created_exclusively_and_never_overwritten(tmp_path):
    target = tmp_path / "r.json"
    run_semantic_final._write_exclusive(target, "one")
    with pytest.raises(FileExistsError):
        run_semantic_final._write_exclusive(target, "two")
    assert target.read_text(encoding="utf-8") == "one"


def test_two_independent_final_runs_produce_identical_reports(lab, tmp_path, monkeypatch):
    assert call(lab, "freeze") == 0
    assert call(lab, "run") == 0
    second = dict(lab, report=tmp_path / "out" / "report2.json")
    assert call(second, "run") == 0
    first_data = json.loads(lab["report"].read_text(encoding="utf-8"))
    second_data = json.loads(second["report"].read_text(encoding="utf-8"))
    assert first_data == second_data


@pytest.mark.parametrize("what", ["implementation", "benchmark", "model", "setting", "record", "dependency"])
def test_any_change_after_the_freeze_blocks_the_final_run(lab, monkeypatch, capsys, what):
    assert call(lab, "freeze") == 0
    factory = fake_factory
    if what == "implementation":
        (lab["root"] / "retrieval" / "impl.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif what == "benchmark":
        lab["benchmark"].write_bytes(lab["benchmark"].read_bytes() + b" ")
    elif what == "model":
        factory = lambda model_dir, instruction: FakeEmbedder(dimension=256, revision="1" * 40)  # noqa: E731
    elif what == "setting":
        original = run_semantic_final.retrieval_settings
        monkeypatch.setattr(run_semantic_final, "retrieval_settings", lambda k: {**original(k), "lane_depth": 7})
    elif what == "record":
        text = lab["prereg"].read_text(encoding="utf-8").replace('"k": 60', '"k": 61')
        lab["prereg"].write_text(text, encoding="utf-8")
    elif what == "dependency":
        (lab["root"] / "requirements-embeddings.txt").write_text("numpy==0.0.1\n", encoding="utf-8")
    assert call(lab, "run", factory=factory) == 2
    assert not lab["report"].exists()
    assert "Refused" in capsys.readouterr().err


def test_a_benchmark_for_another_corpus_is_refused(lab, monkeypatch, capsys):
    # Rebuild the "frozen" file for a different corpus fingerprint, pin it, and freeze with the right corpus.
    other = write_synthetic(lab["tmp"], mutate=lambda d: d["corpus"].update(corpus_fingerprint="a" * 64),
                            name="other_corpus.json")
    digest = hashlib.sha256(other.read_bytes()).hexdigest()
    monkeypatch.setattr(freeze, "FROZEN_V3_SHA256", digest)
    monkeypatch.setattr(benchmark_v3, "FROZEN_V3_SHA256", digest)
    lab = dict(lab, benchmark=other, digest=digest)
    assert call(lab, "freeze") == 0
    assert call(lab, "run") == 2
    assert "different corpus" in capsys.readouterr().err
    assert not lab["report"].exists()


def test_the_final_script_has_no_option_that_changes_a_retrieval_setting():
    parser = run_semantic_final._parser()
    options = {option for sub in parser._subparsers._group_actions[0].choices.values() for action in sub._actions
               for option in action.option_strings}
    assert not {"--rrf-k", "--instruction", "--rebuild-cache", "--label-mode", "--split", "--limit", "--k"} & options


# --- development script (needs the real corpus archives; the embedder is the fake one) -----------------------------------

@pytest.mark.skipif(not archives_present(), reason=ARCHIVE_SKIP_REASON)
def test_the_development_script_runs_end_to_end_deterministically(tmp_path, capsys):
    outputs = []
    for run in ("cold", "warm1", "warm2"):
        out = tmp_path / run
        code = run_semantic_comparison.main(
            ["--instruction", "both", "--cache-dir", str(tmp_path / "cache"), "--output-dir", str(out),
             "--timing-output", str(out / "timing.json")],
            embedder_factory=fake_factory,
        )
        assert code == 0
        outputs.append((out / "dev_development_final_instruction-both.json").read_bytes())
        timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
        assert "evaluation_seconds" in timing
    # The report records what the cache did, so a cold and a warm run differ in that block only (by design);
    # two runs with the same cache state are byte-identical.
    cold, warm1 = (json.loads(data) for data in outputs[:2])
    assert cold["cache"]["instruction_on"]["created"] == 247 and warm1["cache"]["instruction_on"]["reused"] == 247
    cold.pop("cache"), warm1.pop("cache")
    assert cold == warm1
    assert outputs[1] == outputs[2]
    report = json.loads(outputs[2])
    assert report["format"] == run_semantic_comparison.REPORT_FORMAT
    assert report["benchmark"]["split_status"] == "development"
    assert report["corpus"] == {"fingerprint": report["corpus"]["fingerprint"], "documents": 23, "chunks": 247}
    assert set(report["comparison"]["systems"]) >= {"S1_fts5_v1", "S2_fts5_v2", "S3_semantic", "S5_rrf"}
    assert report["ablation_systems"] == ["S3x_semantic_no_instruction", "S5x_rrf_no_instruction"]
    assert "timestamp" not in json.dumps(report).lower() and str(tmp_path) not in json.dumps(report)
    assert report["cache"]["instruction_on"]["total"] == 247
    out = capsys.readouterr().out
    assert "cache[instruction_on]" in out and "created 247" in out and "reused 247" in out


@pytest.mark.skipif(not archives_present(), reason=ARCHIVE_SKIP_REASON)
def test_the_spent_heldout_split_is_labelled_when_explicitly_requested(tmp_path):
    code = run_semantic_comparison.main(
        ["--split", "heldout", "--include-spent-heldout", "--cache-dir", str(tmp_path / "cache"),
         "--output-dir", str(tmp_path / "o")],
        embedder_factory=fake_factory,
    )
    assert code == 0
    report = json.loads((tmp_path / "o" / "dev_heldout_final_instruction-on.json").read_text(encoding="utf-8"))
    assert report["benchmark"]["split_status"] == "observed_spent_heldout"


def test_the_development_script_fails_cleanly_when_the_model_is_missing(tmp_path, capsys):
    def missing(model_dir, instruction):
        from retrieval.embeddings_st import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(tmp_path / "no_such_model")

    if not archives_present():
        pytest.skip(ARCHIVE_SKIP_REASON)
    assert run_semantic_comparison.main(["--cache-dir", str(tmp_path / "c")], embedder_factory=missing) == 1
    assert "does not exist" in capsys.readouterr().err


def test_peak_memory_measurement_returns_a_positive_number_or_none():
    value = run_semantic_comparison.peak_working_set_bytes()
    assert value is None or value > 1_000_000


# --- download / verify tool (no network) -----------------------------------------------------------------------------------

def test_verify_only_fails_cleanly_for_a_missing_or_wrong_model_directory(tmp_path, capsys):
    assert download_embedding_model.main(["--verify-only", "--model-dir", str(tmp_path / "none")]) == 1
    assert "does not exist" in capsys.readouterr().err
    (tmp_path / "model.safetensors").write_bytes(b"not the real weights")
    assert download_embedding_model.main(["--verify-only", "--model-dir", str(tmp_path)]) == 1
    assert "not the pinned model" in capsys.readouterr().err


def test_the_download_tool_only_requests_approved_files():
    source = (REPO_ROOT / "scripts" / "download_embedding_model.py").read_text(encoding="utf-8")
    assert "allow_patterns=list(ALLOWED_DOWNLOAD_PATTERNS)" in source and "revision=BGE_REVISION" in source
    assert 'revision="main"' not in source and "trust_remote_code" not in source
