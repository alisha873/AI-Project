"""Tests for the fingerprinted, atomic, pickle-free embedding cache (synthetic chunks, fake embedder)."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from retrieval import vector_cache
from retrieval.embeddings import FakeEmbedder
from retrieval.semantic import load_chunks
from retrieval.vector_cache import (
    CACHE_FORMAT,
    CacheError,
    VectorCache,
    entry_key,
    index_fingerprint,
)
from tests.retrieval_helpers import make_doc, make_store, small_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def chunks():
    with make_store(*small_corpus()) as store:
        return load_chunks(store)


@pytest.fixture
def cache(tmp_path):
    return VectorCache(tmp_path / "cache")


def manifest_of(cache: VectorCache) -> dict:
    return json.loads(cache.manifest_path.read_text(encoding="utf-8"))


def write_manifest(cache: VectorCache, data: dict) -> None:
    cache.manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def replace_vectors(cache: VectorCache, payload: bytes, *, fix_hash: bool = True) -> None:
    """Overwrite the vectors file (and by default make the manifest hash agree, to test the later checks)."""
    manifest = manifest_of(cache)
    (cache.directory / manifest["vector_file"]).write_bytes(payload)
    if fix_hash:
        manifest["vectors_sha256"] = hashlib.sha256(payload).hexdigest()
        write_manifest(cache, manifest)


def npy_bytes(array, *, allow_pickle: bool = False) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=allow_pickle)
    return buffer.getvalue()


# --- miss, hit, order ---------------------------------------------------------------------------

def test_first_sync_creates_everything_and_writes_the_cache(chunks, cache):
    embedder = FakeEmbedder()
    matrix, report = cache.sync(chunks, embedder)
    assert (report.total, report.created, report.reused, report.invalidated, report.rejected) == (len(chunks), len(chunks), 0, 0, 0)
    assert report.status_before == "missing" and report.written
    assert matrix.shape == (len(chunks), 32) and matrix.dtype == np.float32
    assert cache.manifest_path.is_file()
    assert len(list(cache.directory.glob("vectors-*.npy"))) == 1


def test_second_sync_reuses_every_vector_without_calling_the_embedder(chunks, cache):
    embedder = FakeEmbedder()
    first, _ = cache.sync(chunks, embedder)
    calls = embedder.encode_calls
    second, report = cache.sync(chunks, embedder)
    assert (report.reused, report.created, report.invalidated, report.rejected) == (len(chunks), 0, 0, 0)
    assert report.status_before == "valid" and not report.written
    assert embedder.encode_calls == calls
    assert np.array_equal(first, second)


def test_rows_follow_the_order_of_the_chunks_given(chunks, cache):
    embedder = FakeEmbedder()
    cache.sync(chunks, embedder)
    reversed_chunks = list(reversed(chunks))
    matrix, report = cache.sync(reversed_chunks, embedder)
    assert report.reused == len(chunks)
    assert np.array_equal(matrix, embedder.embed_passages([c.text for c in reversed_chunks]))


def test_cached_vectors_equal_directly_computed_ones(chunks, cache):
    embedder = FakeEmbedder()
    matrix, _ = cache.sync(chunks, embedder)
    assert np.array_equal(matrix, embedder.embed_passages([c.text for c in chunks]))


def test_the_same_inputs_give_byte_identical_cache_files(chunks, tmp_path):
    one, two = VectorCache(tmp_path / "one"), VectorCache(tmp_path / "two")
    one.sync(chunks, FakeEmbedder())
    two.sync(chunks, FakeEmbedder())
    assert one.manifest_path.read_bytes() == two.manifest_path.read_bytes()
    name = manifest_of(one)["vector_file"]
    assert (one.directory / name).read_bytes() == (two.directory / name).read_bytes()


# --- invalidation -------------------------------------------------------------------------------

def test_a_changed_chunk_is_recomputed_and_the_rest_reused(cache):
    docs = small_corpus()
    with make_store(*docs) as store:
        before = load_chunks(store)
    cache.sync(before, FakeEmbedder())
    changed = [docs[0], make_doc("CWE-2", "Beta Weakness", "beta now says something entirely different"), *docs[2:]]
    with make_store(*changed) as store:
        after = load_chunks(store)
    matrix, report = cache.sync(after, FakeEmbedder())
    assert report.status_before == "valid"
    assert report.created == 1 and report.reused == len(after) - 1 and report.invalidated == 1
    assert report.written and "no longer match" in " ".join(report.reasons)
    assert np.array_equal(matrix, FakeEmbedder().embed_passages([c.text for c in after]))
    _, again = cache.sync(after, FakeEmbedder())
    assert again.reused == len(after) and again.created == 0


def test_removed_chunks_are_invalidated_and_the_cache_rewritten(chunks, cache):
    cache.sync(chunks, FakeEmbedder())
    _, report = cache.sync(chunks[:2], FakeEmbedder())
    assert report.reused == 2 and report.created == 0 and report.invalidated == len(chunks) - 2 and report.written
    assert len(manifest_of(cache)["entries"]) == 2


@pytest.mark.parametrize(
    "other",
    [
        dict(revision="1" * 40),
        dict(query_instruction="Represent: "),
        dict(passage_prefix="passage: "),
        dict(dimension=48),
        dict(model_id="autoshield-test/other-model"),
    ],
)
def test_any_embedding_configuration_change_makes_the_cache_stale(chunks, cache, other):
    cache.sync(chunks, FakeEmbedder())
    embedder = FakeEmbedder(**other)
    matrix, report = cache.sync(chunks, embedder)
    assert report.status_before == "stale"
    assert report.reused == 0 and report.created == len(chunks) and report.invalidated == len(chunks)
    assert matrix.shape == (len(chunks), embedder.config.dimension)
    assert manifest_of(cache)["embedding_fingerprint"] == embedder.config.fingerprint()


def test_different_package_versions_make_the_cache_stale(chunks, cache):
    class Newer(FakeEmbedder):
        @property
        def runtime_versions(self):
            return {"fake-embedder": "2"}

    cache.sync(chunks, FakeEmbedder())
    _, report = cache.sync(chunks, Newer())
    assert report.status_before == "stale" and "package versions" in " ".join(report.reasons)
    assert report.created == len(chunks)


def test_a_changed_chunker_version_is_not_reused(chunks, cache):
    cache.sync(chunks, FakeEmbedder())
    bumped = [c.model_copy(update={"chunker_version": c.chunker_version + "-next"}) for c in chunks]
    _, report = cache.sync(bumped, FakeEmbedder())
    assert report.created == len(chunks) and report.reused == 0


def test_entry_keys_depend_on_every_ingredient():
    base = entry_key("f" * 64, "chunk-1", "a" * 64, "v1")
    assert base == entry_key("f" * 64, "chunk-1", "a" * 64, "v1")
    assert len({
        base, entry_key("e" * 64, "chunk-1", "a" * 64, "v1"), entry_key("f" * 64, "chunk-2", "a" * 64, "v1"),
        entry_key("f" * 64, "chunk-1", "b" * 64, "v1"), entry_key("f" * 64, "chunk-1", "a" * 64, "v2"),
    }) == 5
    assert index_fingerprint("f" * 64, ["a", "b"]) != index_fingerprint("f" * 64, ["b", "a"])


# --- rebuild, clear, write=False ------------------------------------------------------------------

def test_rebuild_ignores_existing_contents(chunks, cache):
    cache.sync(chunks, FakeEmbedder())
    embedder = FakeEmbedder()
    _, report = cache.sync(chunks, embedder, rebuild=True)
    assert report.status_before == "rebuild_requested" and report.created == len(chunks) and report.reused == 0
    assert embedder.encode_calls == 1 and report.written


def test_clear_removes_the_cache_so_the_next_sync_starts_fresh(chunks, cache):
    cache.sync(chunks, FakeEmbedder())
    assert cache.clear() == 2  # manifest + one vectors file
    assert not cache.manifest_path.exists()
    _, report = cache.sync(chunks, FakeEmbedder())
    assert report.status_before == "missing" and report.created == len(chunks)
    assert VectorCache(cache.directory / "nowhere").clear() == 0


def test_write_false_computes_but_leaves_no_files(chunks, cache):
    _, report = cache.sync(chunks, FakeEmbedder(), write=False)
    assert report.created == len(chunks) and not report.written
    assert not cache.directory.exists()


def test_no_chunks_and_duplicate_chunks(chunks, cache):
    matrix, report = cache.sync([], FakeEmbedder())
    assert matrix.shape == (0, 32) and report.total == 0 and not report.written
    with pytest.raises(CacheError, match="unique"):
        cache.sync([chunks[0], chunks[0]], FakeEmbedder())


# --- corruption is rejected, never half-used -----------------------------------------------------

def corrupt_cases():
    def garbage_manifest(c): c.manifest_path.write_text("{not json", encoding="utf-8")
    def empty_manifest(c): c.manifest_path.write_text("", encoding="utf-8")
    def list_manifest(c): c.manifest_path.write_text("[]", encoding="utf-8")
    def duplicate_key(c):
        text = c.manifest_path.read_text(encoding="utf-8")
        c.manifest_path.write_text(text.replace('"dtype"', '"dtype": "float32", "dtype"', 1), encoding="utf-8")
    def nan_constant(c):
        text = c.manifest_path.read_text(encoding="utf-8")
        c.manifest_path.write_text(text.replace('"dimension": 32', '"dimension": NaN', 1), encoding="utf-8")
    def extra_key(c): m = manifest_of(c); m["surprise"] = 1; write_manifest(c, m)
    def missing_key(c): m = manifest_of(c); del m["entries"]; write_manifest(c, m)
    def unknown_format(c): m = manifest_of(c); m["cache_format"] = "other.v9"; write_manifest(c, m)
    def traversal(c): m = manifest_of(c); m["vector_file"] = "../outside.npy"; write_manifest(c, m)
    def absolute_path(c): m = manifest_of(c); m["vector_file"] = "C:/windows/vectors-0123456789abcdef.npy"; write_manifest(c, m)
    def bad_dimension(c): m = manifest_of(c); m["dimension"] = True; write_manifest(c, m)
    def bad_dtype(c): m = manifest_of(c); m["dtype"] = "float64"; write_manifest(c, m)
    def bad_hash_field(c): m = manifest_of(c); m["vectors_sha256"] = "xyz"; write_manifest(c, m)
    def entries_not_list(c): m = manifest_of(c); m["entries"] = {}; write_manifest(c, m)
    def entry_extra_key(c): m = manifest_of(c); m["entries"][0]["extra"] = 1; write_manifest(c, m)
    def entry_bad_ordinal(c): m = manifest_of(c); m["entries"][0]["ordinal"] = -1; write_manifest(c, m)
    def entry_key_altered(c): m = manifest_of(c); m["entries"][0]["chunk_sha256"] = "0" * 64; write_manifest(c, m)
    def duplicate_chunk(c): m = manifest_of(c); m["entries"][1] = dict(m["entries"][0]); write_manifest(c, m)
    def index_altered(c): m = manifest_of(c); m["index_fingerprint"] = "0" * 64; write_manifest(c, m)
    def vectors_flipped(c):
        path = c.directory / manifest_of(c)["vector_file"]
        data = bytearray(path.read_bytes()); data[-1] ^= 0xFF; path.write_bytes(bytes(data))
    def vectors_truncated(c):
        path = c.directory / manifest_of(c)["vector_file"]
        path.write_bytes(path.read_bytes()[:-40])
    def vectors_deleted(c): (c.directory / manifest_of(c)["vector_file"]).unlink()
    def vectors_not_npy(c): replace_vectors(c, b"this is not an npy file")
    def vectors_pickle(c): replace_vectors(c, npy_bytes(np.array([{"a": 1}] * 4, dtype=object), allow_pickle=True))
    def vectors_wrong_shape(c): replace_vectors(c, npy_bytes(np.zeros((2, 32), dtype=np.float32)))
    def vectors_wrong_dtype(c): replace_vectors(c, npy_bytes(np.ones((4, 32), dtype=np.float64)))
    def vectors_1d(c): replace_vectors(c, npy_bytes(np.ones(128, dtype=np.float32)))
    def vectors_nan(c):
        m = np.ones((4, 32), dtype=np.float32) / np.sqrt(32); m[0, 0] = np.nan; replace_vectors(c, npy_bytes(m))
    def vectors_inf(c):
        m = np.ones((4, 32), dtype=np.float32) / np.sqrt(32); m[1, 1] = np.inf; replace_vectors(c, npy_bytes(m))
    def vectors_not_unit(c): replace_vectors(c, npy_bytes(np.full((4, 32), 3.0, dtype=np.float32)))
    return {name: fn for name, fn in list(locals().items()) if callable(fn)}


@pytest.mark.parametrize("name", sorted(corrupt_cases()))
def test_every_kind_of_corruption_is_detected_and_rebuilt(chunks, cache, name):
    embedder = FakeEmbedder()
    expected, _ = cache.sync(chunks, embedder)
    corrupt_cases()[name](cache)
    assert cache.read(embedder.config, embedder.runtime_versions).status == "corrupt"
    matrix, report = cache.sync(chunks, FakeEmbedder())
    assert report.status_before == "corrupt" and report.reused == 0 and report.created == len(chunks)
    assert report.written and report.reasons
    assert np.array_equal(matrix, expected)  # a corrupt cache never leaks into the results
    assert cache.read(embedder.config, embedder.runtime_versions).status == "valid"


def test_a_corrupt_cache_reports_how_many_vectors_were_rejected(chunks, cache):
    cache.sync(chunks, FakeEmbedder())
    data = bytearray((cache.directory / manifest_of(cache)["vector_file"]).read_bytes())
    data[-1] ^= 0xFF
    (cache.directory / manifest_of(cache)["vector_file"]).write_bytes(bytes(data))
    _, report = cache.sync(chunks, FakeEmbedder())
    assert report.rejected == len(chunks) and report.invalidated == 0


def test_a_pickle_payload_is_never_loaded(chunks, cache, tmp_path):
    marker = tmp_path / "executed"

    class Boom:
        def __reduce__(self):
            return (marker.write_text, ("code ran",))

    cache.sync(chunks, FakeEmbedder())
    replace_vectors(cache, npy_bytes(np.array([Boom()] * 4, dtype=object), allow_pickle=True))
    state = cache.read(FakeEmbedder().config, FakeEmbedder().runtime_versions)
    assert state.status == "corrupt" and not marker.exists()


def test_the_cache_module_has_no_pickle():
    source = (REPO_ROOT / "retrieval" / "vector_cache.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not ({"pickle", "cPickle", "joblib", "dill", "shelve"} & imported)
    assert "allow_pickle=False" in source and "allow_pickle=True" not in source


# --- atomic writes -------------------------------------------------------------------------------

def test_a_crash_before_the_manifest_commit_leaves_the_old_cache_valid(cache):
    docs = small_corpus()
    with make_store(*docs) as store:
        before = load_chunks(store)
    embedder = FakeEmbedder()
    cache.sync(before, embedder)
    old_manifest = cache.manifest_path.read_bytes()
    with make_store(docs[0], make_doc("CWE-2", "Beta", "beta changed text"), *docs[2:]) as store:
        after = load_chunks(store)

    real = vector_cache._replace

    def crash_on_manifest(src, dst):
        if Path(dst).name == vector_cache.MANIFEST_NAME:
            raise OSError("simulated crash before the commit point")
        return real(src, dst)

    vector_cache._replace = crash_on_manifest
    try:
        with pytest.raises(OSError, match="simulated crash"):
            cache.sync(after, embedder)
    finally:
        vector_cache._replace = real

    assert cache.manifest_path.read_bytes() == old_manifest  # untouched
    assert cache.read(embedder.config, embedder.runtime_versions).status == "valid"
    assert not [p for p in cache.directory.iterdir() if ".tmp-" in p.name]
    _, report = cache.sync(after, embedder)  # the next run recovers and cleans up the orphaned vectors file
    assert report.written
    assert len(list(cache.directory.glob("vectors-*.npy"))) == 1
    assert cache.read(embedder.config, embedder.runtime_versions).status == "valid"


def test_a_crash_while_writing_the_vectors_file_leaves_nothing_behind(chunks, cache):
    real = vector_cache._replace

    def crash(src, dst):
        raise OSError("simulated crash")

    vector_cache._replace = crash
    try:
        with pytest.raises(OSError):
            cache.sync(chunks, FakeEmbedder())
    finally:
        vector_cache._replace = real
    assert not cache.manifest_path.exists()
    assert not [p for p in cache.directory.iterdir() if ".tmp-" in p.name]
    _, report = cache.sync(chunks, FakeEmbedder())
    assert report.status_before == "missing" and report.written


def test_the_manifest_is_plain_json_with_no_timestamps(chunks, cache):
    cache.sync(chunks, FakeEmbedder())
    manifest = manifest_of(cache)
    assert manifest["cache_format"] == CACHE_FORMAT and manifest["dtype"] == "float32" and manifest["normalized"] is True
    text = cache.manifest_path.read_text(encoding="utf-8").lower()
    assert "timestamp" not in text and "created_at" not in text and "_utc" not in text
    assert [e["chunk_id"] for e in manifest["entries"]] == [c.chunk_id for c in chunks]
