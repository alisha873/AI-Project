"""
Phase 13B.1 - the oracle classifier interface, a generic rule engine and the first (SQL-injection) classifier.

An oracle decides what an observation MEANS. It is split in two so neither half can drift:

  1. a CLASSIFIER looks at the transport responses and sets each declared condition to met / not_met /
     not_evaluated (pure functions of the data; no I/O);
  2. a generic RULE ENGINE (`apply_rules`) applies the scenario's own declarative rules to those condition results.
     Nothing about the verdict is hard-coded in a classifier.

The default is always INCONCLUSIVE. A timeout, an unreachable lab, an unexpected status, an unparseable body, a
failed control or missing evidence never become `objective_not_observed`. The first-listed failure decides the
reason code; the result is never silently promoted.

Classifiers are registered per `(oracle_id, version)`. A scenario whose oracle has no registered classifier can not be
run. Registering a classifier does NOT approve a scenario: eligibility is a separate, human decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

from validation.contract import ConditionRecord, ConditionResult, Identifier, ValidationStatus, _Strict
from validation.scenario import OracleRule, OracleSpec, ScenarioDefinition
from validation.transport import TransportResponse


class OracleOutcome(_Strict):
    status: ValidationStatus
    reason_code: Identifier
    conditions: list[ConditionRecord]
    # short, non-secret facts the runner may attach to evidence (counts, never raw bodies)
    facts: dict[str, str] = {}

    @property
    def results(self) -> dict[str, ConditionResult]:
        return {item.condition_id: item.result for item in self.conditions}


@runtime_checkable
class OracleClassifier(Protocol):
    oracle_id: str
    version: str

    def control_healthy(self, scenario: ScenarioDefinition, control: TransportResponse) -> bool:
        """True only if the control answered well enough to justify sending the (attack-shaped) probe."""
        ...

    def baselines_healthy(
        self, scenario: ScenarioDefinition, baseline_responses: Mapping[str, TransportResponse]
    ) -> bool:
        """True only if EVERY controlled baseline search is usable, so the probe may be sent and compared."""
        ...

    def classify(
        self,
        scenario: ScenarioDefinition,
        control: TransportResponse | None,
        probe: TransportResponse | None,
        baselines: Mapping[str, TransportResponse] | None = None,
    ) -> OracleOutcome: ...


def apply_rules(oracle: OracleSpec, results: dict[str, ConditionResult]) -> OracleRule | None:
    """The first rule whose `requires_met` are all met and `requires_not_met` are all not_met. A condition that is
    `not_evaluated` satisfies neither side, so it can never support a conclusive rule."""
    for rule in oracle.rules:
        if all(results.get(c) == "met" for c in rule.requires_met) and all(
            results.get(c) == "not_met" for c in rule.requires_not_met
        ):
            return rule
    return None


# ------------------------------------------------------------------
# the SQL-injection search classifier
# ------------------------------------------------------------------

def parse_search_records(body: object) -> list[str] | None:
    """The record messages of a `/search` body `{"data": [{"message": str, ...}, ...]}`, or None if malformed."""
    if not isinstance(body, str):
        return None
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    messages: list[str] = []
    for item in payload["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("message"), str):
            return None
        messages.append(item["message"])
    return messages


class SearchRecord(_Strict):
    """One returned record with its STABLE identifier, so record identity drives the comparison (not a count)."""

    id: int
    message: str


def parse_search_records_with_ids(body: object) -> list[SearchRecord] | None:
    """The records of a `/search` body as `(id, message)` pairs, or None if the record IDENTITY is not reliable.

    Returns None (so the caller must treat the evidence as insufficient, never as success) when the body is not a
    parseable `{"data": [...]}` object, a record is not an object, an `id` is missing or is not a plain integer
    (a bool is rejected), a `message` is missing or is not a string, or two records share an id. This is the
    gate that stops 'more records than expected' from meaning anything unless the records are reliably identified."""
    if not isinstance(body, str):
        return None
    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return None
    records: list[SearchRecord] = []
    seen: set[int] = set()
    for item in payload["data"]:
        if not isinstance(item, dict):
            return None
        identifier, message = item.get("id"), item.get("message")
        if not isinstance(identifier, int) or isinstance(identifier, bool) or not isinstance(message, str):
            return None
        if identifier in seen:
            return None  # ambiguous record identity: the same id twice can not be compared reliably
        seen.add(identifier)
        records.append(SearchRecord(id=identifier, message=message))
    return records


def records_outside_literal_match(records: list[SearchRecord], literal: str) -> list[SearchRecord]:
    """Records whose message does NOT contain the literal search text. A correct substring search for that text
    can only return records that contain it, so any such record shows the predicate was altered."""
    return [record for record in records if literal not in record.message]


# ------------------------------------------------------------------
# the baseline-derived legitimate-result-set comparison (the reviewed Incident 001 procedure)
#
# A bare "the message does not contain the probe text" check is NOT sufficient evidence of SQL injection: it never
# confirms what the endpoint's legitimate behaviour is, and a record could legitimately contain the probe text. The
# reviewed procedure instead runs the controlled baseline searches first. Each baseline both (a) establishes a known,
# reliably-identified controlled record and (b) CONFIRMS the documented `LIKE '%term%'` semantics (every record a
# baseline returns must contain that baseline's term). Only then is the probe judged: the objective is Observed only
# when the probe returns a KNOWN controlled record that a correct search for the literal would not return, so the
# conclusion rests on record identity and confirmed semantics, not on a message-substring heuristic.
# ------------------------------------------------------------------

@dataclass(frozen=True)
class BaselineEvidence:
    """What the controlled baseline searches established, or a non-secret reason they are not usable.

    `known_records` maps a stable record id to its message for every record the baselines corroborated; it is empty
    unless `valid` is True. `terms_validated` is the baseline search terms that passed (reviewer-chosen, not secret)."""

    valid: bool
    reason: str | None
    known_records: dict[int, str]
    terms_validated: tuple[str, ...]


def assess_baselines(baseline_responses: Mapping[str, TransportResponse]) -> BaselineEvidence:
    """Validate the controlled baseline searches and establish the corroborated known-record set.

    A baseline is usable only if it answered HTTP 200 with a parseable record list (reliable ids), returned at least
    one record, and EVERY returned record contains that baseline's term (the documented substring semantics). The
    known set is the union by stable id across baselines; the same id returned with two different messages is a
    contradiction. Any failure fails closed with a non-secret reason code (never success)."""
    if not baseline_responses:
        return BaselineEvidence(False, "evidence_insufficient", {}, ())
    known: dict[int, str] = {}
    validated: list[str] = []
    for term, response in baseline_responses.items():
        records = (parse_search_records_with_ids(response.body)
                   if isinstance(response, TransportResponse) and response.outcome == "response" else None)
        if (not isinstance(response, TransportResponse) or response.outcome != "response"
                or response.status_code != 200 or records is None):
            return BaselineEvidence(False, "control_check_failed", {}, ())
        if not records:
            return BaselineEvidence(False, "control_check_failed", {}, ())  # a baseline must establish a known record
        for record in records:
            if term not in record.message:  # the documented LIKE '%term%' semantics are contradicted: do not trust it
                return BaselineEvidence(False, "control_check_failed", {}, ())
            if record.id in known and known[record.id] != record.message:
                return BaselineEvidence(False, "evidence_insufficient", {}, ())  # contradictory record identity
            known[record.id] = record.message
        validated.append(term)
    return BaselineEvidence(True, None, known, tuple(validated))


def legitimate_match_ids(known_records: Mapping[int, str], literal: str) -> set[int]:
    """The ids of the known controlled records that a correct substring search for `literal` would legitimately
    return (its message contains the literal). Normally empty, because the benign controlled records do not contain
    an injection string; a record that genuinely contains it is thus never miscounted as evidence."""
    return {rid for rid, message in known_records.items() if literal in message}


def _failure_reason(response: TransportResponse | None, *, is_control: bool, parseable: bool) -> str | None:
    """The inconclusive reason for a failed exchange, or None when the exchange was healthy."""
    if response is None:
        return "evidence_insufficient"
    if response.outcome == "timeout":
        return "timeout"
    if response.outcome == "unreachable":
        return "lab_unreachable"
    if response.outcome == "error":
        return "unexpected_response"
    if response.status_code != 200 or not parseable:
        return "control_check_failed" if is_control else "unexpected_response"
    return None


class SqliSearchClassifier:
    """Classifier for oracle `sqli-search-record-set-oracle` version `1.0`."""

    oracle_id = "sqli-search-record-set-oracle"
    version = "1.0"

    def control_healthy(self, scenario: ScenarioDefinition, control: TransportResponse) -> bool:
        return (control.outcome == "response" and control.status_code == 200
                and parse_search_records_with_ids(control.body) is not None)

    def baselines_healthy(
        self, scenario: ScenarioDefinition, baseline_responses: Mapping[str, TransportResponse]
    ) -> bool:
        """Every controlled baseline must be usable before the probe is sent (see `assess_baselines`)."""
        return assess_baselines(baseline_responses).valid

    def classify(
        self,
        scenario: ScenarioDefinition,
        control: TransportResponse | None,
        probe: TransportResponse | None,
        baselines: Mapping[str, TransportResponse] | None = None,
    ) -> OracleOutcome:
        if baselines is not None:
            return self._classify_with_baselines(scenario, probe, baselines)
        oracle = scenario.oracle
        literal = scenario.probe.parameter_value_candidate
        results: dict[str, ConditionResult] = {c.condition_id: "not_evaluated" for c in oracle.conditions}
        facts: dict[str, str] = {}

        control_records = (parse_search_records_with_ids(control.body)
                           if control and control.outcome == "response" else None)
        control_failure = _failure_reason(control, is_control=True, parseable=control_records is not None)
        if control is not None:
            results["control_request_succeeds"] = "met" if control_failure is None else "not_met"
        if control_records is not None:
            facts["control_record_count"] = str(len(control_records))

        probe_records = (parse_search_records_with_ids(probe.body)
                         if probe and probe.outcome == "response" else None)
        probe_failure = _failure_reason(probe, is_control=False, parseable=probe_records is not None)
        if probe is not None:
            results["probe_response_parseable"] = "met" if probe_failure is None else "not_met"
        if probe_records is not None and probe_failure is None:
            outside = records_outside_literal_match(probe_records, literal)
            # The comparison is by stable id, not by count: a record counts as 'outside' only because it is reliably
            # identified and its message does not contain the literal search text.
            results["record_outside_literal_match"] = "met" if outside else "not_met"
            facts["probe_record_count"] = str(len(probe_records))
            facts["records_outside_literal_match"] = str(len(outside))
            facts["legitimate_match_count"] = str(len(probe_records) - len(outside))
            if outside:
                facts["records_outside_literal_match_ids"] = ",".join(str(r.id) for r in sorted(outside, key=lambda r: r.id))

        rule = apply_rules(oracle, results)
        if rule is not None:
            status, reason = rule.status, rule.reason_code
        else:
            status = "inconclusive"
            reason = control_failure or probe_failure or "evidence_insufficient"
        return OracleOutcome(
            status=status, reason_code=reason, facts=facts,
            conditions=[ConditionRecord(condition_id=c.condition_id, result=results[c.condition_id])
                        for c in oracle.conditions],
        )

    def _classify_with_baselines(
        self,
        scenario: ScenarioDefinition,
        probe: TransportResponse | None,
        baseline_responses: Mapping[str, TransportResponse],
    ) -> OracleOutcome:
        """The reviewed Incident 001 comparison: judge the probe against the legitimate result set that the
        controlled baseline searches established, by stable record identity and confirmed search semantics."""
        oracle = scenario.oracle
        literal = scenario.probe.parameter_value_candidate
        results: dict[str, ConditionResult] = {c.condition_id: "not_evaluated" for c in oracle.conditions}
        facts: dict[str, str] = {}

        base = assess_baselines(baseline_responses)
        results["control_request_succeeds"] = "met" if base.valid else "not_met"
        if base.valid:
            facts["baselines_validated"] = str(len(base.terms_validated))
            facts["known_record_count"] = str(len(base.known_records))
            facts["known_record_ids"] = ",".join(str(i) for i in sorted(base.known_records))

        probe_records = (parse_search_records_with_ids(probe.body)
                         if probe and probe.outcome == "response" else None)
        probe_failure = _failure_reason(probe, is_control=False, parseable=probe_records is not None)
        if probe is not None:
            results["probe_response_parseable"] = "met" if probe_failure is None else "not_met"

        if base.valid and probe_records is not None and probe_failure is None:
            legitimate = legitimate_match_ids(base.known_records, literal)
            # records the probe returned that a correct substring search for the literal would NOT return
            outside = [r for r in probe_records if literal not in r.message]
            # a KNOWN controlled record whose message differs from the baseline's: contradictory evidence, fail closed
            contradiction = any(r.id in base.known_records and base.known_records[r.id] != r.message
                                for r in probe_records)
            # KNOWN controlled records the probe returned but that are outside the legitimate match: the predicate was
            # altered to return records it should not. This is the identity-corroborated signal (not a bare substring).
            leaked = [r for r in probe_records if r.id in base.known_records and r.id not in legitimate]
            facts["probe_record_count"] = str(len(probe_records))
            facts["legitimate_match_count"] = str(len(legitimate))
            facts["records_outside_literal_match"] = str(len(outside))
            if outside:
                facts["records_outside_literal_match_ids"] = ",".join(
                    str(r.id) for r in sorted(outside, key=lambda r: r.id))
            if leaked:
                facts["known_records_leaked"] = str(len(leaked))
                facts["known_records_leaked_ids"] = ",".join(str(r.id) for r in sorted(leaked, key=lambda r: r.id))
            if contradiction:
                results["record_outside_literal_match"] = "not_evaluated"  # fail closed on contradictory identity
            elif leaked:
                results["record_outside_literal_match"] = "met"  # corroborated by known controlled record identity
            elif outside:
                results["record_outside_literal_match"] = "not_evaluated"  # outside but uncorroborated -> inconclusive
            else:
                results["record_outside_literal_match"] = "not_met"  # a valid comparison: only legitimate matches

        rule = apply_rules(oracle, results)
        if rule is not None:
            status, reason = rule.status, rule.reason_code
        else:
            status = "inconclusive"
            reason = (base.reason if not base.valid else None) or probe_failure or "evidence_insufficient"
        return OracleOutcome(
            status=status, reason_code=reason, facts=facts,
            conditions=[ConditionRecord(condition_id=c.condition_id, result=results[c.condition_id])
                        for c in oracle.conditions],
        )


class ClassifierRegistry:
    def __init__(self, *classifiers: OracleClassifier) -> None:
        self._items: dict[tuple[str, str], OracleClassifier] = {}
        for item in classifiers:
            self.register(item)

    def register(self, classifier: OracleClassifier) -> None:
        key = (classifier.oracle_id, classifier.version)
        if key in self._items:
            raise ValueError(f"a classifier for {key} is already registered")
        self._items[key] = classifier

    def get(self, oracle: OracleSpec) -> OracleClassifier | None:
        return self._items.get((oracle.oracle_id, oracle.version))


def default_classifiers() -> ClassifierRegistry:
    return ClassifierRegistry(SqliSearchClassifier())
