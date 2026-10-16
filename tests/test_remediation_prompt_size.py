"""
Phase 12 tests: the planner request was reduced because Groq rejected it (HTTP 413, 8456 tokens against a limit of
8000). These tests pin the reduction WITHOUT weakening anything: the model-facing view is smaller, loses no fact,
id or provenance string that the validators or the model need, still excludes withheld passages, and nothing is
truncated. Mocked data only; no model, no network, no tokenizer (sizes use characters and a whitespace-blind
word/punctuation count, which are stated approximations, not model token counts).

`legacy_*` below is a copy of the representation used BEFORE the reduction, kept here as the reference to compare with.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re

import pytest

from rag.prompt import context_for_prompt
from remediation.configuration import (
    UNAVAILABLE,
    ConfigurationConflict,
    ConfigurationContext,
    ConfigurationFact,
    build_configuration_context,
)
from remediation.contract import PLAN_SCHEMA_VERSION
from remediation.grounding import _strings, parse_plan, validate_plan
from remediation.inputs import build_planner_input
from remediation.prompt import (
    OMITTED_PASSAGE_FIELDS,
    SYSTEM_PROMPT,
    build_user_prompt,
    model_view,
    plan_json_schema,
)
from tests.remediation_helpers import (
    OFF_TOPIC,
    RELEVANT_A,
    RELEVANT_B,
    answered,
    good_plan,
    good_proposal,
    make_input,
    reconstruction_with_configuration,
    three_reference_context,
)
from tests.retrieval_helpers import INCIDENT_001
from tests.test_rag_context import needs_real_data, real_store  # noqa: F401  (fixture re-use)

# sha256 of json.dumps(plan_json_schema(), sort_keys=True), recorded BEFORE the reduction. The reduction must not
# touch the strict output schema; a deliberate schema change later must update this value on purpose.
PLAN_SCHEMA_SHA256 = "b6aa12ed69ef9d572f2c87ace52d03a21a28a1ec08dce4e10c7f9de4a219a9a0"


# ------------------------------------------------------------
# the representation used before the reduction (reference copy)
# ------------------------------------------------------------

def legacy_view(planner_input) -> dict:
    view = context_for_prompt(planner_input.context)
    view.pop("reference_ids_to_assess", None)
    usable = {ref.reference_id for ref in planner_input.usable_references}
    view["reference_passages"] = [p for p in view["reference_passages"] if p["reference_id"] in usable]
    view["withheld_reference_count"] = len(planner_input.excluded_references)
    configuration = planner_input.configuration
    view["configuration"] = {
        "status": configuration.status,
        "facts": [
            {
                "fact_id": f.fact_id, "subject": f.subject, "attribute": f.attribute, "value": f.value,
                "detail": f.detail, "source_artifacts": list(f.source_artifacts), "basis": f.basis,
            }
            for f in configuration.facts
        ],
        "conflicts_withheld": [
            {"subject": c.subject, "attribute": c.attribute, "values": list(c.values)} for c in configuration.conflicts
        ],
        "notes": list(configuration.notes),
    }
    return view


def legacy_user_prompt(planner_input) -> str:
    message = {
        "task": (
            f"Produce the remediation plan JSON (schema_version {PLAN_SCHEMA_VERSION!r}) described in the "
            "system message from the context below. Keep observed evidence, hypotheses, reference "
            "passages and configuration facts separate."
        ),
        "context": legacy_view(planner_input),
    }
    return json.dumps(message, ensure_ascii=False, indent=2, sort_keys=False)


def words(text: str) -> int:
    """A whitespace-blind size measure (words and punctuation marks); an approximation, not a model token count."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def decode_configuration(cfg: dict) -> list[tuple]:
    """Rebuild (fact_id, subject, attribute, value, detail, source_artifacts, basis) from the compact configuration."""
    columns = cfg["fact_columns"]
    rows = []
    for group in cfg["facts_by_subject"]:
        for row in group["facts"]:
            fact = dict(zip(columns, row, strict=True))
            basis = fact["basis"] if "basis" in fact else cfg["basis_of_every_fact"]
            rows.append((fact["fact_id"], group["subject"], fact["attribute"], fact["value"], fact["detail"],
                         tuple(cfg["source_sets"][fact["sources"]]), basis))
    return rows


def internal_facts(configuration: ConfigurationContext) -> list[tuple]:
    return [(f.fact_id, f.subject, f.attribute, f.value, f.detail, tuple(f.source_artifacts), f.basis)
            for f in configuration.facts]


def edge(subject: str, relationship: str, target: str, artifact: str, **extra) -> dict:
    return {"source": subject, "source_name": subject, "target": target, "target_name": target,
            "relationship": relationship, "basis": "configured", "evidence_ids": [],
            "source_artifacts": [artifact], "attributes": {"relationship_basis": "configured_startup"}, **extra}


def many_edges(count: int) -> list[dict]:
    return [edge(f"container:svc-{i % 7}", "connects_to", f"ip:10.0.{i // 250}.{i % 250}",
                 f"docker/file_{i % 2}.json") for i in range(count)]


def input_with_configuration(config: ConfigurationContext, labels=None):
    return make_input(config=config, labels=labels)


# ------------------------------------------------------------
# smaller
# ------------------------------------------------------------

def test_the_request_is_smaller_on_a_synthetic_input_with_many_configuration_facts():
    config = build_configuration_context(reconstruction_with_configuration(*many_edges(24)))
    assert len(config.facts) >= 20
    planner_input = input_with_configuration(config)
    new, old = build_user_prompt(planner_input), legacy_user_prompt(planner_input)
    assert len(new) < 0.75 * len(old)
    assert words(new) < 0.9 * words(old)


def test_the_request_is_not_larger_for_a_small_input_either():
    planner_input = make_input()
    assert len(build_user_prompt(planner_input)) < len(legacy_user_prompt(planner_input))


@needs_real_data
def test_the_real_incident_001_request_is_smaller(real_store):
    from rag.pipeline import build_contexts_for_incident
    from retrieval.query_builder import load_analysis_artifacts

    context = build_contexts_for_incident(INCIDENT_001, real_store).contexts[0]
    labels = {ref_id: "not_relevant" for ref_id in context.reference_ids[5:]}   # the first 5 passages stay relevant
    _, reconstruction = load_analysis_artifacts(INCIDENT_001)
    planner_input = build_planner_input(context, answered(context, labels), build_configuration_context(reconstruction))
    assert len(planner_input.usable_references) == 5 and len(planner_input.configuration.facts) >= 10
    new, old = build_user_prompt(planner_input), legacy_user_prompt(planner_input)
    assert len(new) <= 0.70 * len(old)          # measured about 0.62 when this was written
    assert words(new) <= 0.85 * words(old)      # measured about 0.77 (whitespace-blind), so not just indentation
    assert len(new) <= 14_000                   # regression guard for the user message (was 18,610 characters)
    # the strict output schema still goes out unchanged, and the system prompt stays within its recorded size
    assert hashlib.sha256(json.dumps(plan_json_schema(), sort_keys=True).encode()).hexdigest() == PLAN_SCHEMA_SHA256
    assert len(SYSTEM_PROMPT) <= 4_600


# ------------------------------------------------------------
# nothing needed is lost
# ------------------------------------------------------------

def test_every_string_of_the_old_view_is_still_present_except_the_derivable_ones():
    config = build_configuration_context(reconstruction_with_configuration(*many_edges(12)))
    planner_input = input_with_configuration(config)
    old_strings, new_strings = set(_strings(legacy_view(planner_input))), set(_strings(model_view(planner_input)))
    lost = old_strings - new_strings
    derivable = {ref.doc_id for ref in planner_input.usable_references}
    derivable |= {p["document_type"] for p in legacy_view(planner_input)["reference_passages"]}
    derivable |= {p["document_id"] for p in legacy_view(planner_input)["reference_passages"]}
    assert lost <= derivable, sorted(lost - derivable)
    assert set(OMITTED_PASSAGE_FIELDS) == {"document_id", "retrieval_rank", "document_type"}


def test_observed_evidence_hypotheses_retrieval_and_ids_are_byte_identical_to_before():
    planner_input = make_input()
    old, new = legacy_view(planner_input), model_view(planner_input)
    for key in ("incident_id", "observed_evidence", "unresolved_evidence_ids", "hypotheses", "retrieval",
                "withheld_reference_count"):
        assert new[key] == old[key], key
    assert [e["evidence_id"] for e in new["observed_evidence"]] == ["application-4", "nginx-19"]
    assert new["hypotheses"][0]["is_confirmed"] is False
    assert new["hypotheses"][0]["status"] == "hypothesis_not_confirmed"
    assert new["hypotheses"][0]["cited_evidence_ids"] == ["application-4", "nginx-19"]


def test_passage_ids_provenance_and_text_are_unchanged_and_the_dropped_fields_are_derivable():
    planner_input = make_input()
    by_id = {ref.reference_id: ref for ref in planner_input.usable_references}
    old = {p["reference_id"]: p for p in legacy_view(planner_input)["reference_passages"]}
    passages = model_view(planner_input)["reference_passages"]
    assert {p["reference_id"] for p in passages} == {RELEVANT_A, RELEVANT_B}
    for passage in passages:
        ref = by_id[passage["reference_id"]]
        assert passage == {k: v for k, v in old[ref.reference_id].items() if k not in OMITTED_PASSAGE_FIELDS}
        assert passage["text"] == ref.text and passage["source_id"] == ref.source_id
        assert passage["source"] == ref.source and passage["title"] == ref.title
        assert not set(OMITTED_PASSAGE_FIELDS) & set(passage)
        # what was dropped is recoverable from what was kept
        assert ref.reference_id == f"{ref.doc_id}#{ref.reference_id.rsplit('#', 1)[1]}"
        assert passage["reference_id"].rsplit("#", 1)[0] == old[ref.reference_id]["document_id"]


def test_configuration_facts_round_trip_exactly():
    config = build_configuration_context(reconstruction_with_configuration(*many_edges(30)))
    cfg = model_view(input_with_configuration(config))["configuration"]
    assert decode_configuration(cfg) == internal_facts(config) and len(config.facts) >= 30
    assert cfg["status"] == "available" and cfg["basis_of_every_fact"] == "configured"
    assert cfg["fact_columns"] == ["fact_id", "attribute", "value", "detail", "sources"]
    assert [f[0] for f in decode_configuration(cfg)] == [f.fact_id for f in config.facts]


def test_a_basis_that_differs_between_facts_stays_on_each_fact():
    base = build_configuration_context(reconstruction_with_configuration())
    mixed = dataclasses.replace(
        base, facts=(base.facts[0], dataclasses.replace(base.facts[1], basis="other_basis")))
    cfg = model_view(input_with_configuration(mixed))["configuration"]
    assert "basis_of_every_fact" not in cfg and cfg["fact_columns"][-1] == "basis"
    assert decode_configuration(cfg) == internal_facts(mixed)
    assert [row[-1] for row in decode_configuration(cfg)] == ["configured", "other_basis"]


def test_conflicts_notes_status_and_an_empty_configuration_are_preserved():
    base = build_configuration_context(reconstruction_with_configuration())
    conflict = ConfigurationConflict("container:x", "listens_on", ("CFG-8", "CFG-9"), ("80", "8080"))
    cfg = model_view(input_with_configuration(
        dataclasses.replace(base, status="conflicting", conflicts=(conflict,), notes=("two values were reported",)))
    )["configuration"]
    assert cfg["status"] == "conflicting"
    assert cfg["conflicts_withheld"] == [
        {"subject": "container:x", "attribute": "listens_on", "values": ["80", "8080"]}]
    assert cfg["notes"] == ["two values were reported"]
    empty = model_view(input_with_configuration(UNAVAILABLE))["configuration"]
    assert empty["status"] == "unavailable" and empty["facts_by_subject"] == [] and empty["source_sets"] == {}
    assert decode_configuration(empty) == []


def test_the_four_kinds_of_material_stay_distinct_and_unmixed():
    planner_input = make_input()
    view = model_view(planner_input)
    evidence = {e["evidence_id"] for e in view["observed_evidence"]}
    hypotheses = {h["hypothesis_id"] for h in view["hypotheses"]}
    references = {p["reference_id"] for p in view["reference_passages"]}
    facts = {row[0] for row in decode_configuration(view["configuration"])}
    assert evidence == {"application-4", "nginx-19"} and hypotheses == {"H1"} and facts and references
    groups = [evidence, hypotheses, references, facts]
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            assert first.isdisjoint(second)
    assert all(re.fullmatch(r"CFG-\d+", fact_id) for fact_id in facts)
    prompt = json.loads(build_user_prompt(planner_input))
    assert set(prompt) == {"task", "context"}
    assert {"observed_evidence", "hypotheses", "reference_passages", "configuration"} <= set(prompt["context"])


# ------------------------------------------------------------
# excluded passages never leak
# ------------------------------------------------------------

@pytest.mark.parametrize("label", ["not_relevant", "unclear"])
def test_an_excluded_passage_never_reaches_the_prompt(label):
    planner_input = make_input(labels={OFF_TOPIC: label})
    prompt = build_user_prompt(planner_input)
    assert OFF_TOPIC not in prompt and "xml parser entity expansion" not in prompt
    assert "cwe:CWE-3" not in prompt and "reference_ids_to_assess" not in prompt
    assert json.loads(prompt)["context"]["withheld_reference_count"] == 1


def test_an_unassessed_passage_is_not_sent_either():
    planner_input = make_input(with_answer=False)
    view = model_view(planner_input)
    assert view["reference_passages"] == [] and view["withheld_reference_count"] == 3


# ------------------------------------------------------------
# the schema and the grounding checks are unchanged
# ------------------------------------------------------------

def test_the_strict_output_schema_is_unchanged():
    digest = hashlib.sha256(json.dumps(plan_json_schema(), sort_keys=True).encode()).hexdigest()
    assert digest == PLAN_SCHEMA_SHA256


def test_the_new_system_prompt_sentence_describes_the_layout_and_keeps_every_rule():
    for phrase in ("facts_by_subject", "fact_columns", "source_sets", "basis_of_every_fact",
                   "ADVISORY ONLY", "every proposal needs human review", "untrusted data",
                   "configured\n   relationships, not observed activity", "Never call one confirmed",
                   "Do not invent commands", "Never claim a fix eliminates",
                   "exactly the keys of the schema", "has not been run"):
        assert phrase in SYSTEM_PROMPT, phrase


def check(planner_input, **overrides):
    plan = good_plan(planner_input, good_proposal(**overrides))
    return validate_plan(parse_plan(json.dumps(plan)), planner_input)


def test_a_path_that_exists_only_in_the_compact_configuration_is_still_grounded():
    step = {"text": "Compare with the recorded `docker/network_after_attack.json` before changing anything.",
            "origin": "general_practice"}
    assert "ungrounded_literal" not in check(make_input(), implementation_steps=[step]).codes


@pytest.mark.parametrize("text", [
    "Edit `/etc/nginx/nginx.conf` and reload the server.",
    "Run `docker run --privileged app` to test.",
    "Look in `docker/never_supplied.json`.",
])
def test_an_invented_literal_is_still_rejected(text):
    step = {"text": text, "origin": "general_practice"}
    assert "ungrounded_literal" in check(make_input(), implementation_steps=[step]).codes


def test_a_literal_from_a_dropped_field_is_not_grounded_by_it():
    """`document_type` was dropped; a plan may not cite it as if the model had been shown it."""
    step = {"text": "Apply `attack_pattern` handling.", "origin": "general_practice"}
    assert "ungrounded_literal" in check(make_input(), implementation_steps=[step]).codes


# ------------------------------------------------------------
# nothing is silently truncated
# ------------------------------------------------------------

def test_long_passage_text_and_many_facts_are_sent_in_full():
    context = three_reference_context()
    long_text = "sql injection weakness text. " * 2000                      # about 58,000 characters
    stretched = dataclasses.replace(context.references[0], text=long_text)
    context = dataclasses.replace(context, references=(stretched, *context.references[1:]))
    config = build_configuration_context(reconstruction_with_configuration(*many_edges(300)))
    assert len(config.facts) >= 300
    planner_input = build_planner_input(context, answered(context, {OFF_TOPIC: "not_relevant"}), config)
    prompt = build_user_prompt(planner_input)
    data = json.loads(prompt)["context"]
    sent = {p["reference_id"]: p["text"] for p in data["reference_passages"]}
    assert sent[stretched.reference_id] == long_text                          # the full text, no marker, no cut
    assert len(prompt) > len(long_text)
    assert [row[0] for g in data["configuration"]["facts_by_subject"] for row in g["facts"]] == [
        f.fact_id for f in config.facts]                                       # every fact, in order
    assert "truncat" not in prompt.lower() and "\u2026" not in prompt


def test_the_prompt_is_deterministic_valid_json_and_has_no_indentation():
    planner_input = make_input()
    first = build_user_prompt(planner_input)
    assert first == build_user_prompt(make_input())
    assert "\n  " not in first and json.loads(first)["task"].startswith("Produce the remediation plan JSON")
    assert not any(secret in first for secret in ("gsk_", "GROQ_API_KEY", "ground_truth/"))
