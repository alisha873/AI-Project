"""
Phase 13 - approved-scenario definitions and the registry (INERT DATA, no networking).

A `ScenarioDefinition` separates, on purpose:

- the human-readable objective and where it came from (`objective_source`, with a pinned hash);
- the endpoint the scenario would be allowed to touch and the lab services it may traverse (`endpoint`);
- the conditions needed to evaluate the objective and the oracle that classifies an observation
  (`oracle`: conditions and declarative classification rules, NOT code);
- the evidence an executed check would have to collect, and the ways the result could be wrong;
- the review state. A scenario is ELIGIBLE for execution only if a named human reviewer approved it and no
  blocking ambiguity remains open.

Nothing here sends a request, builds a URL, resolves a name or runs a command. The probe value kept in the
definition is an inert, quoted candidate string (`inert_not_executable=True`): it is data for a future,
separately approved runner to review, never an instruction. There is no generic exploit runner.

The registry lets later approved scenarios be added WITHOUT changing the report contract, which only ever
refers to a scenario by id, version and hash.

Where the first scenario's objective comes from
-----------------------------------------------
`incident_001/metadata/attack_request.txt` (tracked in git since commit f8f078f, "Complete Phase 2
controlled incident and ground truth") is the operator-side record of the attack that the Phase 2 lab run
generated. It is NOT in `ground_truth/` and is not an answer key, and it is not read at run time: this module
pins a reviewed copy of its fields and its hash, and `check_source_drift` reports any difference from the
file's current text. The file is free text and is never interpreted as instructions. Its ambiguities are
recorded in `AMBIGUITIES` and keep the scenario ineligible until a human resolves and approves them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import model_validator

from validation.contract import (
    CONCLUSIVE_STATUSES,
    REASONS_BY_STATUS,
    Identifier,
    ReviewStatus,
    Sha256,
    Text,
    _Strict,
)

SCENARIO_SCHEMA_VERSION = "1.0"

SQLI_SCENARIO_ID = "sqli-search-incident-001"
SOURCE_PATH = "incident_001/metadata/attack_request.txt"
REQUIRED_SOURCE_KEYS = (
    "Incident", "Attack", "CWE", "Method", "Endpoint", "Parameter", "Payload", "Entry point", "Target",
    "Database", "Expected effect",
)


def normalized_text_sha256(text: str) -> str:
    """SHA-256 of the text with line endings normalized to LF and one trailing newline.

    (The file is LF in git and CRLF in a Windows working tree; the pin must not depend on that.)
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ============================================================
# MODELS
# ============================================================

class ObjectiveSource(_Strict):
    kind: Identifier
    path: Identifier
    incident_id: Identifier
    sha256: Sha256
    extracted_fields: dict[str, str]
    obtained_by: Identifier
    provenance_note: Text
    ground_truth_consulted: Literal[False] = False


class EndpointSpec(_Strict):
    """What the scenario may touch. A scenario names an endpoint and services, never a host or an address."""

    method: Literal["GET"]
    path: Identifier
    parameter: Identifier
    entry_service: Identifier
    target_service: Identifier
    data_service: Identifier

    @model_validator(mode="after")
    def _path(self) -> "EndpointSpec":
        if not self.path.startswith("/") or any(ch in self.path for ch in "?#\\ \t\r\n") or "//" in self.path:
            raise ValueError("path must be an absolute path without query, fragment, backslash or whitespace")
        return self


class BaselineSearch(_Strict):
    """A legitimate, benign search the oracle uses as a controlled comparison input (e.g. `hello`).

    `term` is the search text; `expectation` is the reviewer-stated, human-readable expectation (e.g. that it
    returns its one known record). It is INERT data: nothing here sends it, and the actual records it returns are
    runtime evidence, never invented in this file."""

    term: Text
    expectation: Text


class ProbeSpec(_Strict):
    """The original probe value as an INERT quoted string. Not executable, not reviewed for execution here."""

    parameter_value_candidate: Text
    inert_not_executable: Literal[True] = True
    # Reviewer-provided legitimate searches used as controlled comparison inputs. Empty by default (backward
    # compatible); the built-in scenario carries the reviewed `hello` / `security` / `autoshield` searches.
    baseline_searches: list[BaselineSearch] = []
    # "unspecified" until a reviewer decides. "percent_encoded_utf8_query" means: the parameter name and value are
    # UTF-8 percent-encoded with only RFC 3986 unreserved characters left as they are, sent as a query string of
    # exactly one parameter, with no headers beyond what a transport adds on its own.
    wire_encoding: Literal["unspecified", "percent_encoded_utf8_query"] = "unspecified"
    control_description: Text
    # The concrete benign control value. None until a reviewer supplies one; request construction needs it.
    control_value: Text | None = None


class OracleCondition(_Strict):
    condition_id: Identifier
    description: Text


class OracleRule(_Strict):
    """When the named conditions hold, the observation is classified as `status` for `reason_code`."""

    status: Literal["objective_achieved", "objective_not_observed", "inconclusive"]
    reason_code: Identifier
    requires_met: list[Identifier]
    requires_not_met: list[Identifier] = []
    rationale: Text

    @model_validator(mode="after")
    def _reason(self) -> "OracleRule":
        if self.reason_code not in REASONS_BY_STATUS[self.status]:
            raise ValueError(f"reason {self.reason_code!r} is not allowed for status {self.status!r}")
        return self


class OracleSpec(_Strict):
    oracle_id: Identifier
    version: Identifier
    conditions: list[OracleCondition]
    rules: list[OracleRule]
    required_evidence: list[Text]
    default_when_no_rule_applies: Literal["inconclusive"] = "inconclusive"
    possible_false_conclusions: list[Text]

    @model_validator(mode="after")
    def _consistent(self) -> "OracleSpec":
        ids = [c.condition_id for c in self.conditions]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("an oracle needs unique condition ids")
        for rule in self.rules:
            if not (set(rule.requires_met) | set(rule.requires_not_met)) <= set(ids):
                raise ValueError("an oracle rule refers to an unknown condition")
            if set(rule.requires_met) & set(rule.requires_not_met):
                raise ValueError("an oracle rule can not require a condition both met and not met")
            if rule.status in CONCLUSIVE_STATUSES and not rule.requires_met:
                raise ValueError("a conclusive rule must require at least one condition to be met")
        if not self.required_evidence or not self.possible_false_conclusions:
            raise ValueError("an oracle must state its required evidence and its possible false conclusions")
        return self


class Ambiguity(_Strict):
    code: Identifier
    description: Text
    blocking: bool
    resolved: bool = False


class ReviewerApproval(_Strict):
    """One named reviewer's EXPLICIT approval of the final procedure. Pending until `approved` is True.

    An approval only counts when the reviewer confirms THIS procedure (`confirms_procedure`) and records the date.
    Participation in the review, a technical comment, or a stated role is not an approval."""

    name: Text
    approved: bool = False
    confirms_procedure: bool = False
    date: Text | None = None

    @model_validator(mode="after")
    def _explicit(self) -> "ReviewerApproval":
        if self.approved and not (self.confirms_procedure and self.date):
            raise ValueError("an approval must confirm this procedure and record its date")
        if not self.approved and (self.confirms_procedure or self.date):
            raise ValueError("a pending approval can not already carry a confirmation or a date")
        return self


class ScenarioReview(_Strict):
    status: ReviewStatus
    reviewer: Identifier | None = None
    note: Text
    # Named reviewers who must EACH explicitly approve before the scenario can be approved. Empty keeps the legacy
    # single-reviewer behaviour (used by the test fixtures); the built-in scenario names both required reviewers.
    required_approvers: list[Text] = []
    approvals: list[ReviewerApproval] = []

    @model_validator(mode="after")
    def _approvals(self) -> "ScenarioReview":
        names = [a.name for a in self.approvals]
        if len(set(names)) != len(names):
            raise ValueError("an approval is recorded at most once per reviewer")
        return self

    def missing_approvals(self) -> list[str]:
        """Required reviewers who have not yet explicitly approved this procedure."""
        granted = {a.name for a in self.approvals if a.approved}
        return [name for name in self.required_approvers if name not in granted]


class ScenarioDefinition(_Strict):
    schema_version: Literal["1.0"] = SCENARIO_SCHEMA_VERSION
    scenario_id: Identifier
    definition_version: Identifier
    family: Identifier
    title: Text
    objective_summary: Text
    weakness_reference: Identifier | None = None
    incident_id: Identifier
    objective_source: ObjectiveSource
    endpoint: EndpointSpec
    probe: ProbeSpec
    oracle: OracleSpec
    limitations: list[Text]
    ambiguities: list[Ambiguity]
    review: ScenarioReview

    @model_validator(mode="after")
    def _consistent(self) -> "ScenarioDefinition":
        if self.objective_source.incident_id != self.incident_id:
            raise ValueError("the objective source belongs to a different incident")
        if not self.limitations:
            raise ValueError("a scenario must state its limitations")
        codes = [a.code for a in self.ambiguities]
        if len(set(codes)) != len(codes):
            raise ValueError("ambiguity codes must be unique")
        if self.review.status == "approved":
            if not self.review.reviewer:
                raise ValueError("an approved scenario needs a named reviewer")
            if self.open_blocking_ambiguities():
                raise ValueError("a scenario with open blocking ambiguities can not be approved")
            if self.probe.wire_encoding == "unspecified" or not self.probe.control_value:
                raise ValueError("an approved scenario must fix its wire encoding and its control value")
            if self.review.missing_approvals():
                raise ValueError("every required reviewer must explicitly approve before the scenario is approved")
        return self

    def open_blocking_ambiguities(self) -> list[str]:
        return [a.code for a in self.ambiguities if a.blocking and not a.resolved]

    @property
    def eligible_for_execution(self) -> bool:
        return self.review.status == "approved" and not self.open_blocking_ambiguities()

    def definition_sha256(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================
# CANDIDATE-SOURCE PARSING (pure text; the caller reads the file)
# ============================================================

def parse_candidate_source(text: str) -> tuple[dict[str, str], list[str]]:
    """Split `Key: value` lines into fields and list every problem (missing, duplicate, unknown, malformed).

    The text is treated strictly as data. Nothing in it is executed or turned into a request.
    """
    fields: dict[str, str] = {}
    problems: list[str] = []
    for number, line in enumerate(text.replace("\r\n", "\n").split("\n"), 1):
        if not line.strip():
            continue
        key, separator, value = line.partition(": ")
        if not separator or not key.strip():
            problems.append(f"line {number} is not a 'Key: value' line")
        elif key in fields:
            problems.append(f"duplicate key {key!r}")
        elif key not in REQUIRED_SOURCE_KEYS:
            problems.append(f"unknown key {key!r}")
        else:
            fields[key] = value.strip()
    problems.extend(f"missing key {key!r}" for key in REQUIRED_SOURCE_KEYS if key not in fields)
    return fields, problems


def check_source_drift(definition: ScenarioDefinition, text: str) -> list[str]:
    """Differences between a scenario's pinned objective source and the CURRENT text of that source."""
    source = definition.objective_source
    fields, problems = parse_candidate_source(text)
    drift = list(problems)
    if normalized_text_sha256(text) != source.sha256:
        drift.append("the source text no longer matches the pinned hash; re-review the scenario")
    drift.extend(
        f"field {key!r} differs from the reviewed value"
        for key, value in source.extracted_fields.items()
        if fields.get(key) != value
    )
    return drift


# ============================================================
# THE FIRST SCENARIO: SQL injection against the search endpoint (incident_001)
# ============================================================

_SOURCE_SHA256 = "5a6331396a7ea3843e825e14270fda43c50b03f0ad8986970d88cf93f3347a70"

_SOURCE_FIELDS = {
    "Incident": "incident_001",
    "Attack": "SQL Injection",
    "CWE": "CWE-89",
    "Method": "GET",
    "Endpoint": "/api/search",
    "Parameter": "message",
    "Payload": "' OR '1'='1",
    "Entry point": "http://localhost:8080",
    "Target": "autoshield-api",
    "Database": "autoshield-postgres",
    "Expected effect": "alter SQL predicate and return records beyond the legitimate search match",
}

AMBIGUITIES = (
    Ambiguity(
        code="oracle_not_specified_by_source",
        description="The source states only the expected effect in prose ('return records beyond the "
                    "legitimate search match'). The measurable conditions in this definition are AutoShield's "
                    "own interpretation and need human review.",
        blocking=True,
    ),
    Ambiguity(
        code="wire_encoding_unspecified",
        description="The source gives the payload as text, not the exact bytes or URL encoding sent, and says "
                    "nothing about headers.",
        blocking=True,
    ),
    Ambiguity(
        code="control_value_unspecified",
        description="The source names no control request. The oracle needs a concrete benign control value "
                    "(found missing during the Phase 13B.1 review); none is invented here.",
        blocking=True,
    ),
    Ambiguity(
        code="entry_point_is_a_candidate_only",
        description="'Entry point: http://localhost:8080' names where the original attack was sent. It "
                    "authorizes nothing; the lab target must come from an approved lab configuration with an "
                    "independently established lab identity.",
        blocking=True,
    ),
    Ambiguity(
        code="source_records_the_generated_attack",
        description="The file records the attack AutoShield generated in the lab (Phase 2), not an approved "
                    "validation procedure or what a real attacker would try.",
        blocking=True,
    ),
    Ambiguity(
        code="table_contents_unknown_at_validation_time",
        description="Whether a record in the lab database happens to contain the probe text is unknown until "
                    "the check runs; the oracle's third condition can then be wrong in either direction.",
        blocking=False,
    ),
)

_ORACLE = OracleSpec(
    oracle_id="sqli-search-record-set-oracle",
    version="1.0",
    conditions=[
        OracleCondition(
            condition_id="control_request_succeeds",
            description="A control request to the same endpoint with a benign value that no legitimate record "
                        "is expected to match completes with HTTP 200 and a parseable record list.",
        ),
        OracleCondition(
            condition_id="probe_response_parseable",
            description="The probe request completes within the timeout with HTTP 200 and a parseable record "
                        "list.",
        ),
        OracleCondition(
            condition_id="record_outside_literal_match",
            description="The probe response contains at least one record whose message does not contain the "
                        "probe text itself. A correct search for that literal text can return only records "
                        "that contain it, so such a record shows the search predicate was altered.",
        ),
    ],
    rules=[
        OracleRule(
            status="objective_achieved",
            reason_code="objective_observed",
            requires_met=["probe_response_parseable", "record_outside_literal_match"],
            rationale="Records beyond the legitimate search match were returned for the probe.",
        ),
        OracleRule(
            status="objective_not_observed",
            reason_code="objective_not_observed_under_tested_conditions",
            requires_met=["control_request_succeeds", "probe_response_parseable"],
            requires_not_met=["record_outside_literal_match"],
            rationale="The lab answered a healthy control and a parseable probe, and no record outside the "
                      "literal match was returned. This covers this one probe only.",
        ),
    ],
    required_evidence=[
        "The control and probe observations (status, record count, a hash of the response body) with times.",
        "Each returned record's stable identifier, so records outside the legitimate search match can be "
        "identified reliably rather than inferred from a count or a string heuristic.",
        "The legitimate baseline searches and their observed records, as a controlled comparison.",
        "The lab identity and lab-state identifier the check ran against.",
        "For a post-remediation result, the attestation that the remediation was applied.",
    ],
    possible_false_conclusions=[
        "objective_achieved can be wrong if a stored record legitimately contains the probe text pattern or "
        "the application changed for an unrelated reason.",
        "objective_not_observed does not show the weakness is gone: other payloads, encodings, endpoints, "
        "second-order paths or error-based techniques were not tested.",
        "objective_not_observed can also come from an unrelated fault that makes the probe return nothing "
        "while a healthy control passes.",
        "Any response that is not a parseable HTTP 200 (timeout, 5xx, redirect, block page) is inconclusive, "
        "never a success.",
    ],
)


def sqli_search_scenario() -> ScenarioDefinition:
    """The first scenario, in its current state: reviewed for extraction, NOT approved, so NOT eligible."""
    return ScenarioDefinition(
        scenario_id=SQLI_SCENARIO_ID,
        definition_version="1.0",
        family="sql_injection",
        title="SQL injection against the search endpoint (incident_001)",
        objective_summary="Check whether a search request carrying the original SQL-injection-shaped text can "
                          "still make the search return records beyond the legitimate search match.",
        weakness_reference="CWE-89",
        incident_id="incident_001",
        objective_source=ObjectiveSource(
            kind="operator_metadata_file",
            path=SOURCE_PATH,
            incident_id="incident_001",
            sha256=_SOURCE_SHA256,
            extracted_fields=dict(_SOURCE_FIELDS),
            obtained_by="manual_reading_of_candidate_source_by_ai_assistant",
            provenance_note="Tracked file added in commit f8f078f (2026-10-09, Phase 2). It records the attack "
                            "the lab run generated. It is not in the incident's answer-key directory, no "
                            "answer-key file was consulted, and the text is treated as data, never as "
                            "instructions.",
        ),
        endpoint=EndpointSpec(
            method="GET",
            path="/api/search",
            parameter="message",
            entry_service="autoshield-nginx",
            target_service="autoshield-api",
            data_service="autoshield-postgres",
        ),
        probe=ProbeSpec(
            parameter_value_candidate=_SOURCE_FIELDS["Payload"],
            control_description="A benign value that no legitimate record is expected to contain, sent to the "
                                "same endpoint, used only to show the endpoint answers normally.",
            baseline_searches=[
                BaselineSearch(term="hello", expectation="A reviewer-stated legitimate search; it is expected to "
                               "return its one known record whose message contains 'hello'."),
                BaselineSearch(term="security", expectation="A reviewer-stated legitimate search; it is expected "
                               "to return its one known record whose message contains 'security'."),
                BaselineSearch(term="autoshield", expectation="A reviewer-stated legitimate search; it is "
                               "expected to return its one known record whose message contains 'autoshield'."),
            ],
        ),
        oracle=_ORACLE,
        limitations=[
            "One probe value and one endpoint: not a survey of SQL-injection techniques.",
            "The oracle looks at returned records only; it does not detect error-based, blind or "
            "second-order injection.",
            "The definition was extracted by an AI assistant and has had no human review.",
        ],
        ambiguities=list(AMBIGUITIES),
        review=ScenarioReview(
            status="pending_human_review",
            reviewer=None,
            note="Extraction checked against the source text; the oracle and the open ambiguities have not "
                 "been reviewed or approved by a human. The final procedure needs the explicit approval of both "
                 "named reviewers; both are pending.",
            required_approvers=["Alisha Ranjan Kumar", "Jihan Chhabra"],
            approvals=[],
        ),
    )


# ============================================================
# REGISTRY
# ============================================================

class ScenarioDefinitionError(ValueError):
    """A scenario definition is invalid or duplicates an existing id."""


class ScenarioRegistry:
    """Holds validated, approved-for-registration definitions. Registering does not make one eligible."""

    def __init__(self, definitions: tuple[ScenarioDefinition, ...] | list[ScenarioDefinition] = ()) -> None:
        self._definitions: dict[str, ScenarioDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: object) -> ScenarioDefinition:
        if not isinstance(definition, ScenarioDefinition):
            raise ScenarioDefinitionError("only a ScenarioDefinition can be registered")
        try:  # re-validate: an instance built with model_construct() would skip every check
            checked = ScenarioDefinition.model_validate(definition.model_dump())
        except ValueError as exc:
            raise ScenarioDefinitionError(f"invalid scenario definition: {exc}") from exc
        if checked.scenario_id in self._definitions:
            raise ScenarioDefinitionError(f"duplicate scenario id {checked.scenario_id!r}")
        self._definitions[checked.scenario_id] = checked
        return checked

    def get(self, scenario_id: object) -> ScenarioDefinition | None:
        return self._definitions.get(scenario_id) if isinstance(scenario_id, str) else None

    def ids(self) -> list[str]:
        return sorted(self._definitions)


def default_registry() -> ScenarioRegistry:
    """A fresh registry holding only the one initial scenario (ineligible until a human approves it)."""
    return ScenarioRegistry([sqli_search_scenario()])
