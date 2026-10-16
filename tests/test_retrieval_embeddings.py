"""Tests for the embedding interface and the approved-model file verification (no network, no model)."""

from __future__ import annotations

import numpy as np
import pytest

from retrieval import embeddings_st
from retrieval.embeddings import (
    FAKE_REVISION,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingOutputError,
    FakeEmbedder,
    canonical_sha256,
    prepare_passages,
    prepare_queries,
    validate_embeddings,
)
from retrieval.embeddings_st import (
    BGE_DIMENSION,
    BGE_MODEL_ID,
    BGE_QUERY_INSTRUCTION,
    BGE_REVISION,
    BGE_WEIGHTS_SHA256,
    ModelFilesError,
    bge_config,
    git_blob_sha1,
    sha256_file,
    verify_model_directory,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def make_config(**overrides) -> EmbeddingConfig:
    values = dict(
        model_id="org/model", model_revision="1" * 40, dimension=4, query_instruction="", passage_prefix="",
        max_seq_length=128, pooling="cls", normalize=True, similarity="cosine",
    )
    values.update(overrides)
    return EmbeddingConfig(**values)


# --- configuration and fingerprint ------------------------------------------------------------

def test_fingerprint_is_deterministic_and_matches_canonical_hash():
    config = make_config()
    assert config.fingerprint() == make_config().fingerprint()
    assert config.fingerprint() == canonical_sha256(config.identity())
    assert len(config.fingerprint()) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"model_revision": "2" * 40},
        {"model_id": "org/other"},
        {"dimension": 8},
        {"query_instruction": "Represent: "},
        {"passage_prefix": "passage: "},
        {"max_seq_length": 256},
        {"pooling": "mean"},
        {"weights_sha256": DIGEST_A},
        {"file_digests": (("vocab.txt", DIGEST_A),)},
    ],
)
def test_every_identity_field_changes_the_fingerprint(change):
    assert make_config(**change).fingerprint() != make_config().fingerprint()


def test_file_digest_order_does_not_matter():
    one = make_config(file_digests=(("a", DIGEST_A), ("b", DIGEST_B)))
    two = make_config(file_digests=(("b", DIGEST_B), ("a", DIGEST_A)))
    assert one.fingerprint() == two.fingerprint()


@pytest.mark.parametrize(
    "bad",
    [
        {"model_id": " "}, {"model_revision": ""}, {"dimension": 0}, {"dimension": True}, {"max_seq_length": -1},
        {"query_instruction": None}, {"pooling": "max"}, {"similarity": "euclid"}, {"normalize": 1},
        {"similarity": "cosine", "normalize": False}, {"dtype": "float64"}, {"weights_sha256": "abc"},
        {"file_digests": (("x", "short"),)},
    ],
)
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(EmbeddingError):
        make_config(**bad)


def test_short_name_marks_missing_instruction():
    assert make_config(model_id="BAAI/bge-small-en-v1.5", query_instruction="x").short_name == "bge-small-en-v1.5@1111111"
    assert make_config(model_id="BAAI/bge-small-en-v1.5").short_name.endswith("+noinstr")


# --- preparing text ---------------------------------------------------------------------------

def test_prefixes_apply_to_the_right_mode_only():
    config = make_config(query_instruction="Q: ", passage_prefix="P: ")
    assert prepare_passages(config, ["  chunk  "]) == ["P:   chunk  "]  # passages are not trimmed
    assert prepare_queries(config, ["  question  "]) == ["Q: question"]


@pytest.mark.parametrize("bad", ["a single string", [""], ["ok", "   "], [None], [3], b"bytes"])
def test_bad_inputs_are_rejected(bad):
    with pytest.raises(EmbeddingInputError):
        prepare_passages(make_config(), bad)
    with pytest.raises(EmbeddingInputError):
        prepare_queries(make_config(), bad)


def test_embedder_applies_the_bge_instruction_to_queries_only():
    embedder = FakeEmbedder(query_instruction=BGE_QUERY_INSTRUCTION)
    embedder.embed_queries(["find sql injection"])
    embedder.embed_passages(["a chunk of text"])
    assert embedder.received_queries == [BGE_QUERY_INSTRUCTION + "find sql injection"]
    assert embedder.received_passages == ["a chunk of text"]


# --- fake embedder and output validation ------------------------------------------------------

def test_fake_embedder_shape_dtype_unit_norm_and_determinism():
    embedder = FakeEmbedder(dimension=16)
    first = embedder.embed_passages(["alpha beta", "gamma"])
    again = FakeEmbedder(dimension=16).embed_passages(["alpha beta", "gamma"])
    assert first.shape == (2, 16) and first.dtype == np.float32
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(first, again)
    assert embedder.config.model_revision == FAKE_REVISION


def test_fake_embedder_similar_texts_are_closer_than_unrelated_ones():
    embedder = FakeEmbedder(dimension=64)
    a, b, c = embedder.embed_passages(["sql injection database query", "database query sql", "banana potassium fruit"])
    assert float(a @ b) > float(a @ c)


def test_empty_input_returns_an_empty_matrix_without_calling_the_model():
    embedder = FakeEmbedder(dimension=8)
    result = embedder.embed_passages([])
    assert result.shape == (0, 8) and result.dtype == np.float32
    assert embedder.encode_calls == 0


@pytest.mark.parametrize(
    "transform",
    [
        lambda m: m[:, :-1],  # wrong dimension
        lambda m: m[:-1],  # wrong row count
        lambda m: np.where(np.arange(m.shape[1]) == 0, np.nan, m),  # NaN
        lambda m: np.where(np.arange(m.shape[1]) == 0, np.inf, m),  # infinity
        lambda m: m * 0.0,  # all zero
        lambda m: m[0],  # 1-D
        lambda m: m.astype(str),  # not numeric
        lambda m: m[None],  # 3-D
    ],
)
def test_malformed_model_output_is_rejected(transform):
    embedder = FakeEmbedder(dimension=8, transform=transform)
    with pytest.raises(EmbeddingOutputError):
        embedder.embed_passages(["one two", "three four"])


def test_the_embedder_renormalises_a_scaled_output_before_checking_unit_length():
    scaled = FakeEmbedder(dimension=8, transform=lambda m: m * 3.0).embed_passages(["one two"])
    assert np.allclose(np.linalg.norm(scaled, axis=1), 1.0, atol=1e-6)


def test_validate_embeddings_can_normalise_first():
    config = make_config(dimension=2)
    result = validate_embeddings(np.array([[3.0, 4.0]]), rows=1, config=config, normalize_first=True)
    assert np.allclose(result, [[0.6, 0.8]])
    with pytest.raises(EmbeddingOutputError):
        validate_embeddings(np.array([[3.0, 4.0]]), rows=1, config=config)


def test_validate_embeddings_rejects_ragged_input():
    with pytest.raises(EmbeddingOutputError):
        validate_embeddings([[1.0, 0.0], [1.0]], rows=2, config=make_config(dimension=2))


# --- model-directory verification (tiny stand-in files and patched pins) -----------------------

@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    names = ["config.json", "vocab.txt", "1_Pooling/config.json"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {name}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"pretend weights")
    monkeypatch.setattr(embeddings_st, "BGE_SMALL_FILES", {n: git_blob_sha1(tmp_path / n) for n in names})
    monkeypatch.setattr(embeddings_st, "BGE_WEIGHTS_SHA256", sha256_file(tmp_path / "model.safetensors"))
    return tmp_path


def test_a_matching_directory_verifies_and_reports_digests(model_dir):
    files = verify_model_directory(model_dir)
    assert files.weights_sha256 == sha256_file(model_dir / "model.safetensors")
    assert [name for name, _ in files.file_digests] == sorted(
        ["1_Pooling/config.json", "config.json", "model.safetensors", "vocab.txt"]
    )


def test_a_missing_directory_is_rejected(tmp_path):
    with pytest.raises(ModelFilesError, match="does not exist"):
        verify_model_directory(tmp_path / "absent")


def test_missing_weights_are_rejected(model_dir):
    (model_dir / "model.safetensors").unlink()
    with pytest.raises(ModelFilesError, match="missing"):
        verify_model_directory(model_dir)


def test_altered_weights_are_rejected(model_dir):
    (model_dir / "model.safetensors").write_bytes(b"tampered weights")
    with pytest.raises(ModelFilesError, match="not the pinned model"):
        verify_model_directory(model_dir)


def test_a_missing_small_file_is_rejected(model_dir):
    (model_dir / "vocab.txt").unlink()
    with pytest.raises(ModelFilesError, match="vocab.txt"):
        verify_model_directory(model_dir)


def test_an_altered_tokenizer_file_is_rejected(model_dir):
    (model_dir / "vocab.txt").write_text("altered", encoding="utf-8")
    with pytest.raises(ModelFilesError, match="blob id"):
        verify_model_directory(model_dir)


@pytest.mark.parametrize("name", ["pytorch_model.bin", "extra.pt", "weights.pkl", "model.onnx", "sub/old.pth", "A.PICKLE"])
def test_pickle_capable_or_unapproved_files_are_rejected(model_dir, name):
    path = model_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    with pytest.raises(ModelFilesError, match="pickle-capable"):
        verify_model_directory(model_dir)


def test_harmless_download_bookkeeping_files_are_allowed(model_dir):
    cache = model_dir / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.metadata").write_text("meta", encoding="utf-8")
    (cache / "model.safetensors.lock").write_text("", encoding="utf-8")
    verify_model_directory(model_dir)


# --- the approved model's pinned values ---------------------------------------------------------

def test_the_pinned_values_are_the_approved_ones():
    assert BGE_MODEL_ID == "BAAI/bge-small-en-v1.5"
    assert BGE_REVISION == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert BGE_WEIGHTS_SHA256 == "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad"
    assert len(BGE_WEIGHTS_SHA256) == 64 and len(BGE_REVISION) == 40
    assert "model.safetensors" in embeddings_st.ALLOWED_DOWNLOAD_PATTERNS
    assert not any(p.endswith(embeddings_st.FORBIDDEN_SUFFIXES) for p in embeddings_st.ALLOWED_DOWNLOAD_PATTERNS)


def test_bge_config_records_the_approved_settings(model_dir):
    config = bge_config(verify_model_directory(model_dir))
    assert (config.model_id, config.model_revision, config.dimension) == (BGE_MODEL_ID, BGE_REVISION, BGE_DIMENSION)
    assert config.query_instruction == BGE_QUERY_INSTRUCTION and config.passage_prefix == ""
    assert (config.pooling, config.normalize, config.similarity, config.max_seq_length) == ("cls", True, "cosine", 512)
    assert config.weights_sha256 is not None
    assert bge_config(verify_model_directory(model_dir), query_instruction="").fingerprint() != config.fingerprint()


def test_the_default_model_directory_is_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv(embeddings_st.MODEL_DIR_ENV, str(tmp_path))
    assert embeddings_st.default_model_dir() == tmp_path
    monkeypatch.delenv(embeddings_st.MODEL_DIR_ENV)
    assert embeddings_st.default_model_dir().name == f"bge-small-en-v1.5-{BGE_REVISION[:12]}"


def test_the_loader_never_enables_remote_code_or_the_network():
    source = (embeddings_st.REPO_ROOT / "retrieval" / "embeddings_st.py").read_text(encoding="utf-8")
    assert "trust_remote_code=False" in source and "local_files_only=True" in source
    assert '"use_safetensors": True' in source
    assert "trust_remote_code=True" not in source
