"""
Phase 13: the reviewed Incident 001 validation procedure (SQL injection against /api/search). Offline only.

Every test here is deterministic and sends nothing real: request construction is pure data, and the oracle is a
pure function of mocked responses. The scenario stays PENDING joint approval; nothing here approves it, enables live
transport or contacts anything.

The reviewed facts the repository confirms (so they are not invented here):
  * nginx maps host 8080 -> 80 and forwards /api/ to the API, stripping the prefix (infrastructure/docker/...);
  * the API /search runs `... WHERE message LIKE '%{message}%' ORDER BY id` and returns
    {"data": [{"id": int, "message": str}, ...]} (stable ids exist); and
  * records legitimately matching a term are exactly those whose message contains it (the LIKE '%term%' semantics).
The ACTUAL record contents are runtime evidence and are mocked here, never asserted as real.

The oracle's evidence path for this incident is the BASELINE-DERIVED comparison (section C): the controlled baseline
searches establish the known records and confirm the search semantics, and only then is the probe judged by record
identity. The single-control path (section D) is a simpler fallback for scenarios that declare no baseline searches.
"""

from __future__ import annotations

import copy
import json
from urllib.parse import parse_qs

import pytest

from tests.validation_helpers import (
    PROBE_TEXT,
    approved_scenario,
    approved_scenario_with_baselines,
    lab_body,
    run_fake,
    run_fake_baselines,
)
from validation.oracle import (
    SqliSearchClassifier,
    assess_baselines,
    legitimate_match_ids,
    parse_search_records_with_ids,
    records_outside_literal_match,
)
from validation.probe_requests import (
    RequestConstructionError,
    build_baseline_requests,
    build_probe_requests,
    check_constructible,
)
from validation.scenario import (
    SQLI_SCENARIO_ID,
    ReviewerApproval,
    ScenarioDefinition,
    procedure_digest,
    sqli_search_scenario,
)
from validation.transport import TransportResponse

ORIGIN = "http://127.0.0.1:8080"
CLASSIFIER = SqliSearchClassifier()
SCENARIO = approved_scenario()                       # baseline-free: exercises the single-control fallback path
SCENARIO_B = approved_scenario_with_baselines()      # keeps the three reviewed baselines: the Incident 001 path
EXPECTED_PROBE_ENCODING = "%27%20OR%20%271%27%3D%271"  # the reviewed encoded value for ' OR '1'='1
BASELINE_TERMS = ("hello", "security", "autoshield")
RECORD_TEXTS = ("hello world", "security advisory", "autoshield status")  # the fake lab's record messages


def ok(records: list[tuple[int, str]]) -> TransportResponse:
    return TransportResponse.ok(json.dumps({"data": [{"id": i, "message": m} for i, m in records]}))


def good_baselines() -> dict[str, TransportResponse]:
    """Three valid baseline responses: each returns one record whose message contains that baseline's term."""
    return {
        "hello": ok([(1, "hello world")]),
        "security": ok([(2, "security advisory")]),
        "autoshield": ok([(3, "autoshield status")]),
    }


def classify(control, probe):
    return CLASSIFIER.classify(SCENARIO, control, probe)


def classify_baseline(baselines, probe):
    return CLASSIFIER.classify(SCENARIO_B, None, probe, baselines=baselines)


# ======================================================================
# A. the reviewed scenario is represented accurately and stays pending
# ======================================================================

def test_the_reviewed_endpoint_method_parameter_and_entry_point_are_represented():
    scenario = sqli_search_scenario()
    assert scenario.endpoint.method == "GET" and scenario.endpoint.path == "/api/search"
    assert scenario.endpoint.parameter == "message"
    assert scenario.objective_source.extracted_fields["Entry point"] == "http://localhost:8080"
    assert scenario.probe.parameter_value_candidate == PROBE_TEXT


def test_the_reviewed_baseline_searches_are_recorded_as_controlled_inputs():
    terms = tuple(b.term for b in sqli_search_scenario().probe.baseline_searches)
    assert terms == BASELINE_TERMS
    assert all(b.expectation for b in sqli_search_scenario().probe.baseline_searches)


def test_the_scenario_stays_pending_joint_approval_and_is_not_executable():
    review = sqli_search_scenario().review
    assert review.status == "pending_human_review" and review.reviewer is None
    assert review.required_approvers == ["Alisha Ranjan Kumar", "Jihan Chhabra"]
    assert review.approvals == [] and review.missing_approvals() == review.required_approvers
    assert sqli_search_scenario().eligible_for_execution is False


# ======================================================================
# B. request construction and reproducibility (pure data)
# ======================================================================

def test_the_probe_targets_the_search_endpoint_with_the_message_parameter_and_reviewed_encoding():
    pair = build_probe_requests(SCENARIO_B, ORIGIN)
    assert pair.probe.path == "/api/search" and pair.probe.method == "GET"
    assert pair.probe.query == f"message={EXPECTED_PROBE_ENCODING}"
    assert pair.probe.target == f"/api/search?message={EXPECTED_PROBE_ENCODING}"
    # no double-encoding: decoding once returns the original literal, and there is no '%25' (an encoded '%')
    assert parse_qs(pair.probe.query)["message"][0] == PROBE_TEXT
    assert "%25" not in pair.probe.query


def test_the_baseline_requests_are_built_inert_for_each_reviewed_term():
    requests = build_baseline_requests(SCENARIO_B, ORIGIN)
    assert tuple(r.query.split("=", 1)[1] for r in requests) == BASELINE_TERMS  # these terms need no escaping
    for request, term in zip(requests, BASELINE_TERMS):
        assert request.role == "control" and request.attack_shaped is False
        assert request.path == "/api/search" and request.request_id.endswith(f":baseline:{term}")


def test_the_generic_fixture_declares_no_baselines_and_builds_none():
    assert SCENARIO.probe.baseline_searches == []
    assert build_baseline_requests(SCENARIO, ORIGIN) == ()


def test_an_odd_baseline_term_is_percent_encoded_without_double_encoding():
    data = copy.deepcopy(SCENARIO_B.model_dump())
    data["probe"]["baseline_searches"] = [{"term": "a b&c", "expectation": "x"}]
    requests = build_baseline_requests(ScenarioDefinition.model_validate(data), ORIGIN)
    assert requests[0].query == "message=a%20b%26c"


def test_the_data_endpoint_can_never_be_selected_for_this_scenario():
    data = copy.deepcopy(SCENARIO_B.model_dump())
    data["endpoint"]["path"] = "/api/data"
    scenario = ScenarioDefinition.model_validate(data)
    assert check_constructible(scenario) == "endpoint_not_allowed"
    for builder in (build_probe_requests, build_baseline_requests):
        with pytest.raises(RequestConstructionError) as caught:
            builder(scenario, ORIGIN)
        assert caught.value.code == "endpoint_not_allowed"


def test_baseline_construction_fails_closed_for_an_ineligible_scenario():
    with pytest.raises(RequestConstructionError) as caught:
        build_baseline_requests(sqli_search_scenario(), ORIGIN)
    assert caught.value.code == "scenario_not_eligible"


# ======================================================================
# C. the oracle: the baseline-derived legitimate-result-set comparison (the Incident 001 evidence path)
# ======================================================================

def test_assess_baselines_establishes_the_known_records_and_confirms_the_semantics():
    base = assess_baselines(good_baselines())
    assert base.valid and base.reason is None
    assert base.known_records == {1: "hello world", 2: "security advisory", 3: "autoshield status"}
    assert base.terms_validated == BASELINE_TERMS
    assert legitimate_match_ids(base.known_records, PROBE_TEXT) == set()  # no benign record contains the injection


@pytest.mark.parametrize("broken,reason", [
    ({"security": TransportResponse.timeout()}, "control_check_failed"),        # a baseline did not answer
    ({"security": ok([])}, "control_check_failed"),                             # a baseline established no record
    ({"security": ok([(2, "no match here")])}, "control_check_failed"),         # LIKE '%term%' semantics contradicted
    ({"security": TransportResponse.ok(lab_body([]), status_code=500)}, "control_check_failed"),  # not HTTP 200
    ({"security": TransportResponse.ok("<html>down</html>")}, "control_check_failed"),             # unparseable
])
def test_an_unusable_baseline_fails_closed_and_the_probe_is_never_judged(broken, reason):
    baselines = good_baselines()
    baselines.update(broken)
    outcome = classify_baseline(baselines, ok([(1, "hello world")]))  # a probe answer that WOULD look exploited
    assert outcome.status == "inconclusive" and outcome.reason_code == reason
    assert outcome.results["control_request_succeeds"] == "not_met"
    assert outcome.results["record_outside_literal_match"] == "not_evaluated"


def test_a_baseline_that_returns_an_id_with_two_messages_is_contradictory_evidence():
    baselines = good_baselines()
    baselines["autoshield"] = ok([(1, "autoshield status")])  # id 1 was "hello world" in another baseline
    outcome = classify_baseline(baselines, ok([]))
    assert outcome.status == "inconclusive"
    assert outcome.results["control_request_succeeds"] == "not_met"


def test_known_controlled_records_returned_outside_the_match_are_observed_by_identity():
    # the injection returns the whole table; none of those known records contain the literal -> altered predicate
    probe = ok([(1, "hello world"), (2, "security advisory"), (3, "autoshield status")])
    outcome = classify_baseline(good_baselines(), probe)
    assert (outcome.status, outcome.reason_code) == ("objective_achieved", "objective_observed")
    assert outcome.results["record_outside_literal_match"] == "met"
    assert outcome.facts["known_records_leaked_ids"] == "1,2,3"
    assert outcome.facts["legitimate_match_count"] == "0"


def test_only_legitimately_matching_records_are_not_observed():
    # a patched endpoint returns nothing for the injection string (no record contains it) -> a valid, clean result
    outcome = classify_baseline(good_baselines(), ok([]))
    assert outcome.status == "objective_not_observed"
    assert outcome.results["record_outside_literal_match"] == "not_met"
    assert outcome.facts["records_outside_literal_match"] == "0"


def test_a_record_that_genuinely_contains_the_literal_is_not_counted_as_evidence():
    # a record whose message really contains the probe text is a legitimate match, so it is NOT 'outside'
    probe = ok([(5, f"an advisory mentioning {PROBE_TEXT}")])
    outcome = classify_baseline(good_baselines(), probe)
    assert outcome.status == "objective_not_observed"
    assert outcome.facts["records_outside_literal_match"] == "0"  # the literal-containing record is a legit match


def test_outside_records_that_are_not_known_controlled_records_stay_inconclusive():
    # a returned record that is outside the match but was never established by a baseline can not be corroborated
    outcome = classify_baseline(good_baselines(), ok([(99, "mystery note")]))
    assert outcome.status == "inconclusive"
    assert outcome.results["record_outside_literal_match"] == "not_evaluated"
    assert outcome.facts["records_outside_literal_match_ids"] == "99"


def test_a_probe_record_that_reuses_a_known_id_with_a_new_message_fails_closed():
    outcome = classify_baseline(good_baselines(), ok([(1, "tampered message")]))
    assert outcome.status == "inconclusive"
    assert outcome.results["record_outside_literal_match"] == "not_evaluated"


@pytest.mark.parametrize("body", [
    '{"data": [{"id": 1, "message": "a"}, {"id": 1, "message": "b"}]}',   # duplicate id: ambiguous identity
    '{"data": [{"message": "a"}]}',                                       # missing id
    '{"data": [{"id": "x", "message": "a"}]}',                            # non-integer id
    '{"data": [{"id": true, "message": "a"}]}',                           # boolean is not a record id
    '{"data": [{"id": 1}]}',                                             # missing message
    '{"data": [{"id": 1, "message": 5}]}',                               # non-string message
    "<html>not json</html>",                                            # not a record list at all
])
def test_an_unparseable_or_ambiguous_probe_is_inconclusive_never_observed(body):
    outcome = classify_baseline(good_baselines(), TransportResponse.ok(body))
    assert outcome.status == "inconclusive"
    assert outcome.results["record_outside_literal_match"] == "not_evaluated"


@pytest.mark.parametrize("probe,reason", [
    (TransportResponse.timeout(), "timeout"),
    (TransportResponse.unreachable(), "lab_unreachable"),
    (TransportResponse.error(), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=500), "unexpected_response"),
    (TransportResponse.ok(lab_body([]), status_code=302), "unexpected_response"),
])
def test_a_failed_probe_exchange_is_inconclusive(probe, reason):
    outcome = classify_baseline(good_baselines(), probe)
    assert (outcome.status, outcome.reason_code) == ("inconclusive", reason)


def test_the_baseline_oracle_facts_are_auditable_and_expose_no_record_text():
    outcome = classify_baseline(good_baselines(), ok([(1, "hello world"), (2, "security advisory")]))
    assert outcome.status == "objective_achieved"
    assert set(outcome.facts) >= {"baselines_validated", "known_record_count", "known_record_ids",
                                  "probe_record_count", "legitimate_match_count",
                                  "records_outside_literal_match", "known_records_leaked_ids"}
    dumped = json.dumps(outcome.facts)
    for text in RECORD_TEXTS:
        assert text not in dumped  # counts and ids only, never a record message


# ======================================================================
# D. the single-control fallback oracle (for scenarios that declare no baselines)
# ======================================================================

def test_the_fallback_marks_a_valid_control_and_an_unparseable_one_is_unhealthy():
    healthy = classify(ok([(1, "hello world")]), ok([]))
    assert healthy.results["control_request_succeeds"] == "met"
    bad = TransportResponse.ok("<html>down</html>")
    assert CLASSIFIER.control_healthy(SCENARIO, bad) is False
    assert classify(bad, None).results["control_request_succeeds"] == "not_met"


def test_the_fallback_identity_parsing_accepts_only_reliably_identified_records():
    assert parse_search_records_with_ids(json.dumps({"data": []})) == []
    good = parse_search_records_with_ids(json.dumps({"data": [{"id": 2, "message": "m", "extra": 1}]}))
    assert good is not None and good[0].id == 2 and good[0].message == "m"
    assert parse_search_records_with_ids('{"data": [{"id": 1, "message": "a"}, {"id": 1, "message": "a"}]}') is None
    assert records_outside_literal_match(good, "zz") == good and records_outside_literal_match(good, "m") == []


# ======================================================================
# E. the runner: the baseline-gated control flow (offline, deterministic fake)
# ======================================================================

def test_the_runner_sends_three_baselines_then_the_probe_and_observes_the_injection():
    report, transport = run_fake_baselines(vulnerable=True)
    assert len(transport.requests) == 4 and report.execution.requests_sent == 4
    assert report.status == "objective_achieved" and report.execution.completed is True
    control_exchanges = [e for e in report.evidence if e.kind == "control_exchange"]
    probe_exchanges = [e for e in report.evidence if e.kind == "probe_exchange"]
    assert len(control_exchanges) == 3 and len(probe_exchanges) == 1
    assert "known_records_leaked_ids" in probe_exchanges[0].attributes


def test_the_runner_reports_not_observed_against_a_patched_endpoint():
    report, transport = run_fake_baselines(vulnerable=False)
    assert len(transport.requests) == 4 and report.status == "objective_not_observed"


def test_an_unhealthy_baseline_stops_the_probe_from_being_sent():
    # drop record 2, so the 'security' baseline returns nothing: the baselines are insufficient and the probe is held
    report, transport = run_fake_baselines(vulnerable=True, records={1: "hello world", 3: "autoshield status"})
    assert report.status == "inconclusive" and report.reason_code == "control_check_failed"
    assert report.execution.requests_sent == 3 and len(transport.requests) == 3
    assert not [e for e in report.evidence if e.kind == "probe_exchange"]


def test_the_baseline_run_records_each_exact_request_and_exposes_no_record_text():
    report, _ = run_fake_baselines(vulnerable=True)
    for exchange in report.evidence:
        assert exchange.attributes["request_method"] == "GET"
        assert exchange.attributes["request_path"] == "/api/search"
        assert exchange.attributes["request_parameter"] == "message"
    probe = next(e for e in report.evidence if e.kind == "probe_exchange")
    assert probe.attributes["request_encoded_value"] == EXPECTED_PROBE_ENCODING
    assert probe.attributes["request_target"] == f"/api/search?message={EXPECTED_PROBE_ENCODING}"
    dumped = json.dumps([e.model_dump(mode="json") for e in report.evidence])
    for text in RECORD_TEXTS:
        assert text not in dumped  # the only body evidence is a hash and a length


# ======================================================================
# F. gates and E.12 backward compatibility are preserved
# ======================================================================

def test_the_exact_probe_request_is_recorded_in_the_single_control_flow_too():
    report, _ = run_fake(vulnerable=True)
    probe_ev = next(e for e in report.evidence if e.kind == "probe_exchange")
    assert probe_ev.attributes["request_method"] == "GET"
    assert probe_ev.attributes["request_encoded_value"] == EXPECTED_PROBE_ENCODING
    assert probe_ev.attributes["request_target"] == f"/api/search?message={EXPECTED_PROBE_ENCODING}"
    body_keys = [k for k in probe_ev.attributes if "body" in k]
    assert set(body_keys) <= {"body_sha256", "body_length"}


def test_a_full_fake_baseline_run_is_still_refused_as_post_remediation_evidence():
    from validation.contract import ValidationReport
    from validation.handoff import build_phase14_handoff
    report, _ = run_fake_baselines(vulnerable=True)
    assert ValidationReport.model_validate(report.model_dump(mode="json")) == report
    assert report.execution.transport_kind == "fake"
    assert build_phase14_handoff(report).usable_as_post_remediation_evidence is False


# ======================================================================
# G. the dual-approval record is a real gate (model level; the real scenario stays pending)
# ======================================================================

def approve(data: dict, *names: str) -> dict:
    data = copy.deepcopy(data)
    for item in data["ambiguities"]:
        item["resolved"] = True
    data["probe"]["wire_encoding"] = "percent_encoded_utf8_query"
    data["probe"]["control_value"] = "zzqx-no-such-record"
    data["review"]["status"] = "approved"
    data["review"]["reviewer"] = "lead-reviewer"
    digest = procedure_digest(data)
    data["review"]["approvals"] = [
        {
            "name": n, "approved": True, "confirms_procedure": True, "date": "2026-10-16",
            "scenario_id": data["scenario_id"], "procedure_sha256": digest,
        }
        for n in names
    ]
    return data


def test_a_scenario_requiring_two_approvers_can_not_be_approved_until_both_confirm():
    base = sqli_search_scenario().model_dump()
    with pytest.raises(ValueError, match="every required reviewer"):
        ScenarioDefinition.model_validate(approve(base))  # no approvals
    with pytest.raises(ValueError, match="every required reviewer"):
        ScenarioDefinition.model_validate(approve(base, "Alisha Ranjan Kumar"))  # only one
    both = ScenarioDefinition.model_validate(approve(base, "Alisha Ranjan Kumar", "Jihan Chhabra"))
    assert both.eligible_for_execution is True and both.review.missing_approvals() == []


def test_an_approval_must_be_explicit_dated_and_bound_to_the_procedure():
    with pytest.raises(ValueError, match="confirm this procedure"):
        ReviewerApproval(name="X", approved=True, confirms_procedure=True, date=None)
    with pytest.raises(ValueError, match="confirm this procedure"):
        ReviewerApproval(name="X", approved=True, confirms_procedure=False, date="2026-10-16")
    with pytest.raises(ValueError, match="pending approval can not already carry"):
        ReviewerApproval(name="X", approved=False, date="2026-10-16")
    assert ReviewerApproval(name="X").approved is False


def test_a_stale_or_mismatched_approval_cannot_authorize_the_scenario():
    base = sqli_search_scenario().model_dump()
    good = approve(base, "Alisha Ranjan Kumar", "Jihan Chhabra")
    mismatched = copy.deepcopy(good)
    mismatched["review"]["approvals"][0]["scenario_id"] = "other-scenario"
    with pytest.raises(ValueError, match="different scenario"):
        ScenarioDefinition.model_validate(mismatched)
    stale = copy.deepcopy(good)
    stale["review"]["approvals"][1]["procedure_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different procedure digest"):
        ScenarioDefinition.model_validate(stale)
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        ReviewerApproval(
            name="X", approved=True, confirms_procedure=True, date="16 October 2026",
            scenario_id=SQLI_SCENARIO_ID, procedure_sha256="a" * 64,
        )


def test_participation_is_not_approval_the_builtin_record_is_pending():
    assert sqli_search_scenario().review.approvals == []
    assert sqli_search_scenario().eligible_for_execution is False
