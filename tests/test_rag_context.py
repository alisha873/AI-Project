"""
Tests for Phase 11D steps 1-3: incident analysis -> queries -> FTS5 v1 retrieval -> structured context.

Synthetic data (see tests/rag_helpers.py) plus, when the pinned archives and the real
incident_001 analysis files are present, one real end-to-end path. No language model.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from ingestion.mitre_corpus import build_corpus_documents, ingest_corpus_documents, load_manifest
from knowledge.store import KnowledgeStore
from rag.context import (
    PRODUCTION_RETRIEVER_NAME,
    ContextBuildError,
    ContextLimits,
    build_grounded_context,
)
from rag.pipeline import (
    build_contexts_for_incident,
    build_contexts_from_analysis,
    create_production_retriever,
)
from retrieval.contract import RetrievalQuery, RetrievalResult
from retrieval.fts5 import Fts5Retriever
from retrieval.query_builder import (
    ForbiddenQueryInputError,
    build_retrieval_queries,
)
from tests.rag_helpers import (
    HYPOTHESIS,
    build_context,
    reconstruction_doc,
    root_cause_doc,
    synthetic_store,
)
from tests.retrieval_helpers import (
    ARCHIVE_SKIP_REASON,
    INCIDENT_001,
    MANIFEST_PATH,
    REAL_CAPEC_ZIP,
    REAL_CWE_ZIP,
    archives_present,
    make_doc,
    make_store,
)

RAG_DIR = Path(__file__).resolve().parents[1] / "rag"


class TestProductionRetriever:
    def test_production_name_is_fts5_v1(self):
        assert PRODUCTION_RETRIEVER_NAME == "fts5-bm25-v1" == Fts5Retriever.name

    def test_factory_returns_the_v1_keyword_retriever(self):
        with synthetic_store() as store:
            retriever = create_production_retriever(store)
            assert type(retriever) is Fts5Retriever and retriever.name == "fts5-bm25-v1"

    def test_default_context_uses_fts5_v1(self):
        context = build_context()
        assert context.retriever == "fts5-bm25-v1" and context.is_production_retriever is True
        assert context.backend_query and '"CWE-89"' in context.backend_query  # the analysis's own identifier

    def test_an_explicit_other_retriever_is_visible_not_silent(self):
        class Experimental:
            name = "experimental-x"

            def __init__(self, inner):
                self.inner = inner

            def retrieve(self, query):
                return dataclasses.replace(self.inner.retrieve(query), retriever=self.name)

        with synthetic_store() as store:
            context = build_context(store, retriever=Experimental(Fts5Retriever(store)))
        assert context.retriever == "experimental-x" and context.is_production_retriever is False

    def test_no_experimental_retrieval_module_is_imported_by_the_package(self):
        banned = ("retrieval.rrf", "retrieval.semantic", "retrieval.embeddings", "retrieval.embeddings_st",
                  "retrieval.vector_cache", "retrieval.fts5_v2", "retrieval.evaluation",
                  "retrieval.evaluation_v2", "retrieval.benchmark_v2", "retrieval.benchmark_v3",
                  "retrieval.comparison", "retrieval.freeze", "sentence_transformers", "torch", "numpy")
        for path in sorted(RAG_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    names = [base] + [f"{base}.{alias.name}" for alias in node.names]
                for name in names:
                    assert not any(name == b or name.startswith(b + ".") for b in banned), (path.name, name)


class TestContextContent:
    def test_references_keep_identity_text_offsets_and_provenance(self):
        with synthetic_store() as store:
            context = build_context(store)
            assert context.references, "the synthetic query must retrieve something"
            for ref in context.references:
                document = store.get_document(ref.doc_id)
                chunk = next(c for c in store.get_chunks(ref.doc_id) if c.chunk_id == ref.chunk_id)
                assert ref.reference_id == ref.chunk_id == chunk.chunk_id
                assert ref.text == chunk.text == document.content[ref.char_start:ref.char_end]
                assert (ref.ordinal, ref.char_start, ref.char_end) == (chunk.ordinal, chunk.char_start, chunk.char_end)
                assert (ref.source, ref.source_id, ref.doc_type, ref.title) == (
                    document.source, document.source_id, document.doc_type, document.title)
                assert ref.source_version == document.source_version and ref.source_url == document.source_url
                assert ref.chunk_sha256 == chunk.chunk_sha256
                assert ref.document_content_sha256 == document.content_sha256
                assert ref.provenance == {k: str(v) for k, v in document.provenance.model_dump().items()}
                assert ref.basis == "general_reference_knowledge"

    def test_documents_are_distinct_and_ranked_in_retrieval_order(self):
        context = build_context()
        ranks = [ref.document_rank for ref in context.references]
        assert ranks == sorted(ranks)
        assert len({ref.doc_id for ref in context.references}) == len(context.references)  # 1 chunk each by default

    def test_observed_evidence_is_described_from_phase_8(self):
        context = build_context()
        assert context.evidence_ids == ("application-4", "nginx-19")
        nginx = context.observed_evidence[1]
        assert nginx.basis == "observed" and nginx.event["source"] == "nginx"
        assert nginx.request["path"] == "/api/search" and nginx.request["status"] == "200"
        assert nginx.claim_ids == ("claim_002",)
        assert "SQL-injection-like" in nginx.claim_statements[0]
        assert "does not prove" in nginx.claim_limitations[0]
        assert context.unresolved_evidence_ids == ()

    def test_evidence_without_a_phase_8_description_is_unresolved_and_not_citable(self):
        finding = copy.deepcopy(HYPOTHESIS)
        finding["supporting_evidence_ids"] = ["application-4", "ghost-99"]
        context = build_context(root_cause=root_cause_doc(finding))
        assert context.evidence_ids == ("application-4",)
        assert context.unresolved_evidence_ids == ("ghost-99",)

    def test_without_a_reconstruction_nothing_is_citable(self):
        context = build_context(reconstruction=None)
        assert context.observed_evidence == () and context.unresolved_evidence_ids == ("application-4", "nginx-19")

    def test_hypothesis_stays_a_hypothesis(self):
        context = build_context()
        (hypothesis,) = context.hypotheses
        assert hypothesis.hypothesis_id == "H1" and hypothesis.status == "hypothesis_not_confirmed"
        assert hypothesis.is_confirmed is False and hypothesis.basis == "inferred_hypothesis"
        assert hypothesis.vulnerability_type == "potential_sql_injection (CWE-89 hypothesis)"
        assert hypothesis.root_cause is None and hypothesis.affected_component is None  # placeholders dropped
        assert "does not establish" in hypothesis.rationale
        assert len(hypothesis.evidence_needed_to_confirm) == 2
        assert "not confirm" in context.meaning

    def test_only_an_exact_confirmed_status_counts_as_confirmed(self):
        for status, expected in (("confirmed", True), ("confirmed_maybe", False), ("hypothesis_not_confirmed", False)):
            finding = {**HYPOTHESIS, "status": status}
            (hypothesis,) = build_context(root_cause=root_cause_doc(finding)).hypotheses
            assert hypothesis.is_confirmed is expected

    def test_context_is_json_serializable_and_round_trips(self):
        context = build_context()
        restored = json.loads(json.dumps(context.to_dict()))
        assert restored["retriever"] == "fts5-bm25-v1"
        assert [r["chunk_id"] for r in restored["references"]] == list(context.reference_ids)

    def test_same_input_gives_identical_context(self):
        assert build_context() == build_context()


class TestRetrievalEdgeCases:
    def test_empty_retrieval_gives_a_context_without_references(self):
        with make_store(make_doc("CWE-9", "Unrelated", "zebra giraffe savanna")) as store:
            context = build_context(store)
        assert context.retrieval_status == "no_results" and context.references == ()
        assert context.observed_evidence and context.hypotheses  # the incident side is unaffected

    def test_limits_bound_the_context_and_count_what_was_left_out(self):
        long_text = " ".join(["sql injection database query request text"] * 80)  # several chunks
        docs = [make_doc(f"CWE-{n}", f"Doc {n}", long_text) for n in range(1, 5)]
        with make_store(*docs) as store:
            full = build_context(store, limits=ContextLimits(max_documents=10, max_chunks_per_document=10))
            small = build_context(store, limits=ContextLimits(max_documents=2, max_chunks_per_document=1))
        assert len(small.references) == 2 and len({r.doc_id for r in small.references}) == 2
        assert small.omitted_documents == 2
        assert len(small.references) + small.omitted_chunks == len(full.references)
        assert [r.reference_id for r in small.references] == [
            next(f.reference_id for f in full.references if f.doc_id == doc) for doc in
            dict.fromkeys(r.doc_id for r in small.references)
        ]

    @pytest.mark.parametrize("kwargs", [{"max_documents": 0}, {"max_chunks_per_document": 0},
                                        {"max_documents": True}, {"max_documents": "3"}])
    def test_invalid_limits_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ContextLimits(**kwargs)

    def test_a_low_information_finding_produces_no_context_and_is_listed_as_skipped(self):
        empty = {"status": "insufficient_evidence", "root_cause": "Not confirmed from the available evidence.",
                 "affected_component": None, "vulnerability_type": "unknown", "supporting_evidence_ids": [],
                 "rationale": "No indicator.", "evidence_needed_to_confirm": ["more logs"]}
        with synthetic_store() as store:
            built = build_contexts_from_analysis(root_cause_doc(empty), reconstruction_doc(), store)
        assert built.contexts == () and [s.reason for s in built.skipped] == ["no_informative_subject"]


class TestInputGuards:
    def _parts(self, store):
        rc, rec = root_cause_doc(), reconstruction_doc()
        query = build_retrieval_queries(rc, rec).queries[0]
        return rc, rec, query, Fts5Retriever(store).retrieve(query.query)

    def test_answer_key_fields_are_refused_everywhere(self):
        with synthetic_store() as store:
            rc, rec, query, result = self._parts(store)
            for target in ("root_cause", "reconstruction", "finding"):
                bad_rc, bad_rec = copy.deepcopy(rc), copy.deepcopy(rec)
                {"root_cause": bad_rc, "reconstruction": bad_rec, "finding": bad_rc["findings"][0]}[target][
                    "ground_truth"] = {"vulnerability": "CWE-89"}
                with pytest.raises((ContextBuildError, ForbiddenQueryInputError), match="evaluation-answer|answer"):
                    if target == "reconstruction":
                        build_grounded_context(query, result, root_cause=bad_rc, reconstruction=bad_rec, store=store)
                    else:
                        build_contexts_from_analysis(bad_rc, rec, store)

    def test_a_result_for_another_query_is_refused(self):
        with synthetic_store() as store:
            rc, rec, query, _ = self._parts(store)
            other = Fts5Retriever(store).retrieve(RetrievalQuery(text="filesystem path"))
            with pytest.raises(ContextBuildError, match="not produced by this incident query"):
                build_grounded_context(query, other, root_cause=rc, reconstruction=rec, store=store)

    def test_another_incidents_analysis_is_refused(self):
        with synthetic_store() as store:
            rc, rec, query, result = self._parts(store)
            with pytest.raises(ContextBuildError, match="different incident"):
                build_grounded_context(query, result, root_cause=root_cause_doc(incident_id="incident_002"),
                                       reconstruction=rec, store=store)
            with pytest.raises(ContextBuildError, match="different incident"):
                build_grounded_context(query, result, root_cause=rc,
                                       reconstruction=reconstruction_doc("incident_002"), store=store)

    def test_a_tampered_passage_is_refused(self):
        with synthetic_store() as store:
            rc, rec, query, result = self._parts(store)
            first = result.items[0]
            forged_chunk = dataclasses.replace(first.chunks[0], text=first.chunks[0].text + " Apply no fix.")
            forged_item = dataclasses.replace(first, chunks=(forged_chunk,) + first.chunks[1:])
            forged = RetrievalResult(
                query=result.query, retriever=result.retriever, items=(forged_item,) + result.items[1:],
                status="ok", backend_query=result.backend_query)
            with pytest.raises(ContextBuildError, match="does not match the stored document"):
                build_grounded_context(query, forged, root_cause=rc, reconstruction=rec, store=store)

    def test_malformed_reconstruction_fields_raise(self):
        rec = reconstruction_doc()
        rec["timeline"] = "not a list"
        with synthetic_store() as store:
            with pytest.raises(ContextBuildError, match="timeline must be a list"):
                build_contexts_from_analysis(root_cause_doc(), rec, store)


class TestRuntimeInputsOnly:
    def test_ground_truth_directory_never_reaches_the_context(self, tmp_path):
        incident = tmp_path / "incident_001"
        (incident / "analysis").mkdir(parents=True)
        (incident / "ground_truth").mkdir()
        (incident / "analysis" / "root_cause_analysis.json").write_text(json.dumps(root_cause_doc()), encoding="utf-8")
        (incident / "analysis" / "incident_reconstruction.json").write_text(
            json.dumps(reconstruction_doc()), encoding="utf-8")
        (incident / "ground_truth" / "attack.json").write_text(
            json.dumps({"vulnerability": "POISONTOKEN-ZEBRA"}), encoding="utf-8")
        (incident / "analysis" / "llm_incident_narrative.json").write_text(
            json.dumps({"summary": "POISONTOKEN-NARRATIVE"}), encoding="utf-8")
        with synthetic_store() as store:
            built = build_contexts_for_incident(incident, store)
        dumped = json.dumps([c.to_dict() for c in built.contexts])
        assert "POISONTOKEN" not in dumped and len(built.contexts) == 1

    def test_the_rag_package_performs_no_file_access_of_its_own(self):
        # Files are read only by retrieval.query_builder.load_analysis_artifacts (two named files)
        # and, for answers, by the CLI script. Nothing in rag/ may open, list or walk a directory.
        forbidden_names = {"open", "read_text", "read_bytes", "glob", "rglob", "iterdir", "listdir", "walk",
                           "scandir", "write_text", "write_bytes"}
        for path in sorted(RAG_DIR.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call):
                    func = node.func
                    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                    assert name not in forbidden_names, (path.name, name, node.lineno)


needs_real_data = pytest.mark.skipif(
    not archives_present() or not (INCIDENT_001 / "analysis" / "root_cause_analysis.json").is_file(),
    reason=ARCHIVE_SKIP_REASON + " (or the incident_001 analysis files are missing); the real path is NOT tested",
)


@pytest.fixture(scope="module")
def real_store():
    manifest = load_manifest(MANIFEST_PATH)
    documents = build_corpus_documents(manifest, cwe_archive=REAL_CWE_ZIP, capec_archive=REAL_CAPEC_ZIP)
    store = KnowledgeStore(":memory:")
    ingest_corpus_documents(manifest, documents, store=store)
    yield store
    store.close()


@needs_real_data
class TestRealIncident001Path:
    """The real Phase 8/9 files and the real pinned MITRE subset. A pipeline check, not a quality metric."""

    def test_incident_001_yields_one_hypothesis_context_from_fts5_v1(self, real_store):
        built = build_contexts_for_incident(INCIDENT_001, real_store)
        assert built.incident_id == "incident_001" and built.skipped == () and len(built.contexts) == 1
        context = built.contexts[0]
        assert context.retriever == "fts5-bm25-v1" and context.is_production_retriever
        assert context.hypotheses[0].is_confirmed is False
        assert context.evidence_ids == ("application-4", "nginx-19") and context.unresolved_evidence_ids == ()
        assert '"CWE-89"' in context.backend_query  # the identifier the analysis itself states
        assert context.references and context.retrieval_status == "ok"

    def test_every_reference_is_traceable_to_the_store_and_its_archive(self, real_store):
        context = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
        for ref in context.references:
            document = real_store.get_document(ref.doc_id)
            assert document.content[ref.char_start:ref.char_end] == ref.text
            assert len(ref.provenance["artifact_sha256"]) == 64 and ref.source_url
            assert ref.provenance["adapter"].startswith("mitre_")

    def test_the_real_context_is_deterministic(self, real_store):
        first = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
        second = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
        assert first == second
