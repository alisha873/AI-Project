"""
Tests for `retrieval.query_builder`.

The inline dictionaries mirror the REAL Phase 8/9 JSON structure (verified from
`scripts/run_incident_analysis.py` and `incident_001/analysis/*.json`); the
text is copied from the real `incident_001` artifacts so placeholder handling is
tested against what Phase 9 actually emits. One test reads the real files and
is skipped if they are absent.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from retrieval import query_builder
from retrieval.contract import RESULT_MEANING, RetrievalQuery
from retrieval.fts5 import Fts5Retriever
from retrieval.query_builder import (
    CLAIM_FIELDS_READ,
    FINDING_FIELDS_READ,
    RECONSTRUCTION_FIELDS_READ,
    ROOT_CAUSE_FIELDS_READ,
    TIMELINE_FIELDS_READ,
    ForbiddenQueryInputError,
    QueryBuildError,
    build_queries_for_incident,
    build_retrieval_queries,
    is_placeholder,
    load_analysis_artifacts,
)
from tests.retrieval_helpers import INCIDENT_001, make_doc, make_store

HYPOTHESIS = {
    "status": "hypothesis_not_confirmed",
    "root_cause": "Not confirmed from the available evidence.",
    "affected_component": "Undetermined from request evidence alone.",
    "vulnerability_type": "potential_sql_injection (CWE-89 hypothesis)",
    "supporting_evidence_ids": ["application-4", "nginx-19"],
    "confidence": 0.55,
    "rationale": (
        "One or more normalized events contain SQL-injection-like request syntax. This supports an "
        "injection-attempt hypothesis but does not establish vulnerable server-side query construction, "
        "successful exploitation, or database access."
    ),
    "evidence_needed_to_confirm": [
        "application source or query-construction evidence showing unsafe SQL handling",
        "correlated application/database logs demonstrating the query was executed",
        "request-to-response evidence tied to the same event or request identifier",
    ],
}

INSUFFICIENT = {
    "status": "insufficient_evidence",
    "root_cause": "Unknown; the available evidence does not support a specific root-cause finding.",
    "affected_component": None,
    "vulnerability_type": "unknown",
    "supporting_evidence_ids": [],
    "confidence": 0.0,
    "rationale": "No supported vulnerability-specific indicator was identified by the deterministic baseline.",
    "evidence_needed_to_confirm": [
        "incident-specific application and service logs",
        "relevant source/configuration evidence",
        "correlated evidence linking the suspected behavior to impact",
    ],
}


def root_cause(*findings, incident_id="incident_001"):
    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "agent": "phase_9_root_cause",
        "findings": [copy.deepcopy(f) for f in findings],
        "validation": {"all_supporting_event_ids_exist": True, "ground_truth_used": False},
    }


def reconstruction(incident_id="incident_001", claims=True):
    data = {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "agent": "phase_8_incident_reconstruction",
        "evidence_summary": {"ground_truth_used": False},
        "timeline": [
            {"event_id": "application-4", "action": "GET /search", "source": "application"},
            {"event_id": "nginx-19", "action": "GET /api/search", "source": "nginx"},
            {"event_id": "nginx-10", "action": "GET /api/health", "source": "nginx"},
        ],
        "claims": [
            {
                "claim_id": "claim_001",
                "statement": "Event application-4 contains SQL-injection-like syntax in /search.",
                "status": "observed_indicator",
                "evidence_ids": ["application-4"],
            },
            {
                "claim_id": "claim_002",
                "statement": "Event nginx-19 contains SQL-injection-like syntax in /api/search.",
                "status": "observed_indicator",
                "evidence_ids": ["nginx-19"],
            },
        ],
    }
    if not claims:
        data["claims"] = []
    return data


def only_query(result):
    assert len(result.queries) == 1, result
    return result.queries[0]


class TestMeaningfulQueryFromRealShapedOutput:
    def test_hypothesis_produces_a_meaningful_query(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction()))
        text = built.query.text
        assert "potential sql injection (CWE-89 hypothesis)" in text
        assert "SQL-injection-like syntax in /search" in text  # observed evidence description
        assert "unsafe SQL handling" in text  # evidence gap
        assert built.query.phrases == ("sql injection",)

    def test_placeholders_are_never_turned_into_search_text(self):
        text = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction())).query.text
        assert "Not confirmed" not in text and "Undetermined" not in text

    def test_evidence_ids_are_kept_as_metadata_but_not_in_the_text(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction()))
        assert built.evidence_ids == ("application-4", "nginx-19")
        assert "application-4" not in built.query.text and "nginx-19" not in built.query.text

    def test_a_hypothesis_stays_identified_as_a_hypothesis(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction()))
        assert built.finding_status == "hypothesis_not_confirmed" and built.is_confirmed is False
        by_path = {c.field_path: c.basis for c in built.components}
        assert by_path["findings[0].vulnerability_type"] == "inferred_hypothesis"
        assert by_path["findings[0].rationale"] == "inferred_hypothesis"
        assert by_path["claims[claim_001].statement"] == "observed_evidence"
        assert by_path["findings[0].evidence_needed_to_confirm[0]"] == "evidence_gap"
        assert "confirmed_finding" not in by_path.values()
        assert built.meaning == RESULT_MEANING and "do not confirm" in built.meaning

    def test_only_an_explicit_confirmed_status_is_treated_as_confirmed(self):
        confirmed = {**HYPOTHESIS, "status": "confirmed", "root_cause": "Concatenated SQL in the search handler."}
        built = only_query(build_retrieval_queries(root_cause(confirmed), reconstruction()))
        assert built.is_confirmed is True
        bases = {c.field_path: c.basis for c in built.components}
        assert bases["findings[0].root_cause"] == "confirmed_finding"
        assert bases["claims[claim_001].statement"] == "observed_evidence"  # observation, not finding
        near_miss = {**HYPOTHESIS, "status": "Confirmed "}
        assert only_query(build_retrieval_queries(root_cause(near_miss))).is_confirmed is False

    def test_components_are_in_documented_order_with_claims_in_id_order(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction()))
        assert [c.field_path for c in built.components] == [
            "findings[0].vulnerability_type",
            "claims[claim_001].statement",
            "claims[claim_002].statement",
            "findings[0].rationale",
            "findings[0].evidence_needed_to_confirm[0]",
            "findings[0].evidence_needed_to_confirm[1]",
            "findings[0].evidence_needed_to_confirm[2]",
        ]

    def test_repeated_calls_give_identical_queries(self):
        a = build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction())
        b = build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction())
        assert a == b

    def test_input_is_not_modified(self):
        rc, rec = root_cause(HYPOTHESIS), reconstruction()
        before = copy.deepcopy((rc, rec))
        build_retrieval_queries(rc, rec)
        assert (rc, rec) == before

    def test_limit_is_passed_through(self):
        assert only_query(build_retrieval_queries(root_cause(HYPOTHESIS), limit=3)).query.limit == 3

    def test_a_real_component_and_root_cause_are_used_when_present(self):
        finding = {**HYPOTHESIS, "root_cause": "Search handler concatenates request text into SQL.",
                   "affected_component": "autoshield-api search handler"}
        text = only_query(build_retrieval_queries(root_cause(finding))).query.text
        assert "Search handler concatenates request text into SQL." in text
        assert "autoshield-api search handler" in text


class TestEvidenceDescriptions:
    def test_claim_statements_contribute(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction()))
        assert "Event contains SQL-injection-like syntax in /api/search." in built.query.text

    def test_timeline_action_is_the_fallback_when_no_claim_covers_an_event(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction(claims=False)))
        paths = [c.field_path for c in built.components]
        assert "timeline[application-4].action" in paths and "timeline[nginx-19].action" in paths
        assert "GET /search" in built.query.text and "nginx-10" not in built.query.text

    def test_unrelated_claims_and_events_are_ignored(self):
        rec = reconstruction()
        rec["claims"].append({"claim_id": "claim_009", "statement": "Totally unrelated claim.",
                              "evidence_ids": ["docker-container-1"]})
        text = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), rec)).query.text
        assert "unrelated" not in text

    def test_without_a_reconstruction_no_observed_components_appear(self):
        built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS)))
        assert all(c.basis != "observed_evidence" for c in built.components)


class TestMissingAndOptionalFields:
    def test_only_a_vulnerability_type_gives_exactly_that_text(self):
        built = only_query(build_retrieval_queries(root_cause({"vulnerability_type": "path_traversal"})))
        assert built.query.text == "path traversal" and built.query.phrases == ("path traversal",)
        assert built.finding_status == "unspecified" and built.is_confirmed is False

    def test_null_empty_and_absent_fields_are_treated_alike(self):
        base = {"vulnerability_type": "ssrf"}
        variants = [
            base,
            {**base, "root_cause": None, "affected_component": "", "rationale": "   ",
             "evidence_needed_to_confirm": None, "supporting_evidence_ids": None},
            {**base, "evidence_needed_to_confirm": ["", "  "]},
        ]
        queries = [only_query(build_retrieval_queries(root_cause(v))).query for v in variants]
        assert queries[0] == queries[1] == queries[2]

    def test_nothing_is_invented_for_absent_fields(self):
        built = only_query(build_retrieval_queries(root_cause({"vulnerability_type": "ssrf",
                                                               "rationale": "Fetched an external URL."})))
        assert built.query.text == "ssrf\nFetched an external URL."

    def test_insufficient_evidence_boilerplate_never_becomes_a_query(self):
        result = build_retrieval_queries(root_cause(INSUFFICIENT), reconstruction())
        assert result.queries == ()
        assert [(s.finding_index, s.reason) for s in result.skipped] == [(0, "no_informative_subject")]

    def test_rationale_and_gaps_alone_are_not_a_subject(self):
        finding = {"rationale": "Something happened.", "evidence_needed_to_confirm": ["more logs"]}
        result = build_retrieval_queries(root_cause(finding))
        assert result.queries == () and result.skipped[0].reason == "no_informative_subject"

    def test_no_findings_gives_an_empty_result(self):
        result = build_retrieval_queries(root_cause())
        assert result.queries == () and result.skipped == () and result.incident_id == "incident_001"

    def test_each_finding_gets_its_own_query_or_skip_record(self):
        result = build_retrieval_queries(root_cause(HYPOTHESIS, INSUFFICIENT, {"vulnerability_type": "ssrf"}))
        assert [q.finding_index for q in result.queries] == [0, 2]
        assert [s.finding_index for s in result.skipped] == [1]
        assert result.queries[1].components[0].field_path == "findings[2].vulnerability_type"

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("Not confirmed from the available evidence.", True),
            ("Undetermined from request evidence alone.", True),
            ("unknown", True), ("Unknown; no evidence", True), ("N/A", True),
            ("", True), (None, True), ("   ", True),
            ("autoshield-api search handler", False),
            ("potential_sql_injection (CWE-89 hypothesis)", False),
            ("Unknownhost proxy", False),  # whole-word match: a real name is not a placeholder
            ("Nonexistent-service", False),
            ("None of the available evidence", True),
        ],
    )
    def test_placeholder_detection(self, text, expected):
        assert is_placeholder(text) is expected

    def test_overlong_input_omits_lowest_priority_components_and_says_so(self):
        finding = {**HYPOTHESIS, "evidence_needed_to_confirm": ["gap " + "x" * 1500 for _ in range(4)]}
        built = only_query(build_retrieval_queries(root_cause(finding), reconstruction()))
        assert len(built.query.text) <= 4000
        assert built.omitted_fields and all("evidence_needed_to_confirm" in f for f in built.omitted_fields)
        assert built.components[0].field_path == "findings[0].vulnerability_type"  # priority kept

    def test_a_single_overlong_component_is_truncated_and_reported(self):
        built = only_query(build_retrieval_queries(root_cause({"vulnerability_type": "w " * 4000})))
        assert len(built.query.text) <= 4000
        assert built.omitted_fields == ("findings[0].vulnerability_type (truncated)",)


class TestMalformedInput:
    @pytest.mark.parametrize("bad", [None, [], "text", 5])
    def test_root_cause_must_be_an_object(self, bad):
        with pytest.raises(QueryBuildError):
            build_retrieval_queries(bad)  # type: ignore[arg-type]

    def test_incident_id_is_required(self):
        with pytest.raises(QueryBuildError, match="incident_id"):
            build_retrieval_queries({"findings": []})

    def test_findings_must_be_a_list_of_objects(self):
        with pytest.raises(QueryBuildError, match="findings"):
            build_retrieval_queries({"incident_id": "i", "findings": "none"})
        with pytest.raises(QueryBuildError, match=r"findings\[0\]"):
            build_retrieval_queries({"incident_id": "i", "findings": ["text"]})

    @pytest.mark.parametrize(
        "field, value",
        [("rationale", ["a list"]), ("vulnerability_type", 5), ("evidence_needed_to_confirm", "one string"),
         ("evidence_needed_to_confirm", [1, 2]), ("supporting_evidence_ids", "nginx-19")],
    )
    def test_wrongly_typed_fields_are_errors_not_guesses(self, field, value):
        with pytest.raises(QueryBuildError, match=field):
            build_retrieval_queries(root_cause({"vulnerability_type": "ssrf", field: value}))

    def test_reconstruction_for_another_incident_is_refused(self):
        with pytest.raises(QueryBuildError, match="incident_999"):
            build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction(incident_id="incident_999"))

    def test_malformed_reconstruction_parts_are_errors(self):
        rec = reconstruction()
        rec["claims"] = "oops"
        with pytest.raises(QueryBuildError, match="claims"):
            build_retrieval_queries(root_cause(HYPOTHESIS), rec)
        rec = reconstruction(claims=False)
        rec["timeline"] = {"not": "a list"}
        with pytest.raises(QueryBuildError, match="timeline"):
            build_retrieval_queries(root_cause(HYPOTHESIS), rec)


class TestGroundTruthIsNeverUsed:
    @pytest.mark.parametrize("key", sorted(query_builder.FORBIDDEN_INPUT_KEYS))
    def test_answer_key_fields_are_refused_everywhere_they_could_be_supplied(self, key):
        poisoned_root = {**root_cause(HYPOTHESIS), key: {"anything": 1}}
        with pytest.raises(ForbiddenQueryInputError):
            build_retrieval_queries(poisoned_root)
        poisoned_finding = {**HYPOTHESIS, key: ["cwe:CWE-89"]}
        with pytest.raises(ForbiddenQueryInputError):
            build_retrieval_queries(root_cause(poisoned_finding))
        poisoned_rec = {**reconstruction(), key: {"anything": 1}}
        with pytest.raises(ForbiddenQueryInputError):
            build_retrieval_queries(root_cause(HYPOTHESIS), poisoned_rec)

    def test_the_real_nested_ground_truth_used_flag_is_allowed(self):
        rc = root_cause(HYPOTHESIS)
        assert "ground_truth_used" in rc["validation"]
        build_retrieval_queries(rc, reconstruction())  # does not raise

    def test_only_documented_fields_are_ever_read(self):
        accessed: set[str] = set()

        class Tracked(dict):
            def __init__(self, data):
                super().__init__({k: wrap(v) for k, v in data.items()})

            def get(self, key, default=None):
                accessed.add(key)
                return super().get(key, default)

            def __getitem__(self, key):
                accessed.add(key)
                return super().__getitem__(key)

        def wrap(value):
            if isinstance(value, dict):
                return Tracked(value)
            if isinstance(value, list):
                return [wrap(v) for v in value]
            return value

        rc = root_cause(HYPOTHESIS)
        rc["ground_truth_note"] = "TOKEN-SHOULD-NEVER-APPEAR"  # an unrelated extra field
        rec = reconstruction()
        rec["evidence_summary"]["ground_truth_used"] = False
        built = only_query(build_retrieval_queries(Tracked(rc), Tracked(rec)))
        allowed = (ROOT_CAUSE_FIELDS_READ | FINDING_FIELDS_READ | RECONSTRUCTION_FIELDS_READ
                   | CLAIM_FIELDS_READ | TIMELINE_FIELDS_READ)
        assert accessed <= allowed, sorted(accessed - allowed)
        assert not any("ground_truth" in key for key in accessed)
        assert "validation" not in accessed and "evidence_summary" not in accessed
        assert "TOKEN-SHOULD-NEVER-APPEAR" not in built.query.text

    def test_loader_opens_only_the_two_analysis_files_and_never_ground_truth(self, tmp_path, monkeypatch):
        incident = tmp_path / "incident_001"
        (incident / "analysis").mkdir(parents=True)
        (incident / "ground_truth").mkdir()
        (incident / "ground_truth" / "ground_truth.json").write_text(
            json.dumps({"vulnerability": "POISON-ANSWER-KEY"}), encoding="utf-8")
        (incident / "analysis" / "llm_incident_narrative.json").write_text(
            json.dumps({"narrative": "POISON-LLM-TEXT"}), encoding="utf-8")
        (incident / "analysis" / "root_cause_analysis.json").write_text(
            json.dumps(root_cause(HYPOTHESIS)), encoding="utf-8")
        (incident / "analysis" / "incident_reconstruction.json").write_text(
            json.dumps(reconstruction()), encoding="utf-8")

        opened: list[Path] = []
        original = Path.read_text

        def recording(self, *args, **kwargs):
            opened.append(self)
            return original(self, *args, **kwargs)

        def forbidden(self, *args, **kwargs):
            raise AssertionError("the loader must not list directories")

        monkeypatch.setattr(Path, "read_text", recording)
        for name in ("iterdir", "glob", "rglob"):
            monkeypatch.setattr(Path, name, forbidden)
        built = only_query(build_queries_for_incident(incident))
        assert sorted(p.name for p in opened) == ["incident_reconstruction.json", "root_cause_analysis.json"]
        assert "POISON" not in built.query.text
        assert not any("ground_truth" in str(p) for p in opened)

    def test_query_builder_does_not_depend_on_evaluation_or_the_benchmark(self):
        tree = ast.parse(Path(query_builder.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert "retrieval.evaluation" not in imported
        assert not any("evaluation" in name or "benchmark" in name for name in imported)
        assert "retrieval_benchmark" not in Path(query_builder.__file__).read_text(encoding="utf-8")


class TestLoadingArtifacts:
    def write(self, directory, name, data):
        (directory / "analysis").mkdir(parents=True, exist_ok=True)
        (directory / "analysis" / name).write_text(
            data if isinstance(data, str) else json.dumps(data), encoding="utf-8")

    def test_reconstruction_is_optional(self, tmp_path):
        self.write(tmp_path, "root_cause_analysis.json", root_cause(HYPOTHESIS))
        rc, rec = load_analysis_artifacts(tmp_path)
        assert rc["incident_id"] == "incident_001" and rec is None
        assert only_query(build_queries_for_incident(tmp_path)).evidence_ids == ("application-4", "nginx-19")

    @pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
    def test_unreadable_artifacts_raise_a_clear_error(self, tmp_path, content):
        self.write(tmp_path, "root_cause_analysis.json", content)
        with pytest.raises(QueryBuildError):
            load_analysis_artifacts(tmp_path)

    def test_missing_root_cause_file_raises(self, tmp_path):
        with pytest.raises(QueryBuildError, match="cannot read"):
            load_analysis_artifacts(tmp_path)


@pytest.mark.skipif(
    not (INCIDENT_001 / "analysis" / "root_cause_analysis.json").is_file(),
    reason="incident_001/analysis artifacts are not present; real-artifact query construction is NOT tested",
)
class TestRealIncident001Artifacts:
    def test_real_artifacts_produce_a_hypothesis_query(self):
        result = build_queries_for_incident(INCIDENT_001)
        assert result.incident_id == "incident_001" and result.skipped == ()
        built = only_query(result)
        assert built.finding_status == "hypothesis_not_confirmed" and built.is_confirmed is False
        assert built.evidence_ids == ("application-4", "nginx-19")
        assert built.query.phrases == ("sql injection",)
        assert "CWE-89" in built.query.text  # stated by the analysis itself, not by an answer key
        assert "Not confirmed" not in built.query.text and "Undetermined" not in built.query.text
        assert any(c.basis == "observed_evidence" for c in built.components)

    def test_real_query_runs_through_the_retriever_on_a_synthetic_store(self):
        built = only_query(build_queries_for_incident(INCIDENT_001))
        docs = [
            make_doc("CWE-1", "Query Weakness", "sql injection builds a database query from request text"),
            make_doc("CWE-2", "Other", "filesystem path handling only"),
        ]
        with make_store(*docs) as store:
            result = Fts5Retriever(store).retrieve(built.query)
            assert result.status == "ok" and result.doc_ids[0] == "cwe:CWE-1"


def test_built_query_is_a_valid_retrieval_query():
    built = only_query(build_retrieval_queries(root_cause(HYPOTHESIS), reconstruction()))
    assert isinstance(built.query, RetrievalQuery)
