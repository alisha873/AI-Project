from __future__ import annotations

import json

from evidence.formatter import EvidenceContextFormatter
from evidence.schema import (
    EvidenceExtraction,
    IncidentEvidenceContext,
)
from llm.base import LLMClient


class EvidenceExtractionAgent:
    """
    Identifies evidence relevant to an incident.

    This agent does not diagnose the attack, determine root cause,
    propose remediation, or perform security validation.
    """

    SYSTEM_PROMPT = """
    You are the Evidence Extraction Agent in AutoShield.

    Your job is ONLY to select evidence relevant to the incident.

    You are NOT an attack-analysis agent.

    You must NOT:
    - identify or name the attack
    - determine the root cause
    - identify a vulnerability
    - propose a remediation
    - recommend a security fix
    - perform penetration testing
    - infer facts that are not explicitly supported by the provided evidence
    - invent event IDs or artifact paths

    INPUT SECTION HEADINGS ARE NOT OUTPUT FIELD NAMES.

    The input may contain these section headings:
    - NORMALIZED_EVENTS
    - CONFIGURATION_ARTIFACTS
    - NETWORK_ARTIFACTS
    - SUPPORTING_EVIDENCE_ARTIFACTS

    These are categories of INPUT evidence.

    NEVER copy those headings into your output JSON.

    Your output JSON MUST contain exactly these seven keys:

    1. incident_id
    2. relevant_events
    3. relevant_configuration
    4. relevant_network_events
    5. relevant_network_artifacts
    6. evidence_summary
    7. confidence

    The output structure is:

    {
    "incident_id": "incident_001",
    "relevant_events": [],
    "relevant_configuration": [],
    "relevant_network_events": [],
    "relevant_network_artifacts": [],
    "evidence_summary": "",
    "confidence": 0.0
    }

    FIELD RULES:

    incident_id:
    - Must equal the INCIDENT_ID provided in the input.

    relevant_events:
    - Select normalized event IDs that materially help reconstruct the
    incident timeline or establish what happened.
    - Prioritize direct observations of suspicious or incident-related
    activity over general environment inventory.
    - Prefer events with a clear temporal, endpoint, request, or entity
    relationship to the incident.
    - Include contextual events only when they help establish the sequence
    before, during, or after relevant activity.
    - Container-state and network-membership events are contextual evidence.
    Do not select them merely because they exist.
    - Every returned event ID MUST exist in NORMALIZED_EVENTS.
    - Return [] if no event is relevant.

    relevant_configuration:
    - A list of artifact paths from CONFIGURATION_ARTIFACTS.
    - Every returned path MUST exist in CONFIGURATION_ARTIFACTS.
    - Return [] if no configuration artifact is relevant.

    relevant_network_events:
    - Select only normalized event IDs that document relevant
    network-facing activity.
    - HTTP request events may qualify when their request path, query,
    timestamp, source, destination, or response is relevant to the
    incident.
    - Prefer the most directly relevant requests; do not include every
    HTTP request simply because it exists.
    - Container-state events and network-membership events are not
    network activity events for this field.
    - A network event may also appear in relevant_events when it serves
    both purposes.
    - Every returned event ID MUST exist in NORMALIZED_EVENTS.
    - Return [] if no relevant network event exists.

    EVIDENCE SELECTION POLICY:
    1. Identify direct observations relevant to the incident.
    2. Identify a small amount of contextual evidence that helps explain
    the sequence of events.
    3. Include configuration artifacts only when they describe the
    affected environment or help assess the incident.
    4. Select network artifacts when they may contain relevant traffic.
    5. Do not maximize the number of selected items.
    6. Prefer a precise, defensible subset over broad environment inventory.
    7. Never infer that an event is malicious solely because it occurred
    during the incident window.
    8. Do not diagnose the attack or propose remediation.

    relevant_network_artifacts:
    - A list of artifact paths from NETWORK_ARTIFACTS.
    - Every returned path MUST exist in NETWORK_ARTIFACTS.
    - Return [] if no network artifact is relevant.

    evidence_summary:
    - Briefly describe what the selected evidence contains.
    - Describe observable evidence only.
    - Do NOT explain what caused the incident.
    - Do NOT name the attack.
    - Do NOT recommend a fix.

    confidence:
    - A number between 0.0 and 1.0.
    - Represents confidence that the selected evidence is relevant.
    - Do not use confidence to express attack severity.

    IMPORTANT:
    - You are selecting evidence, not diagnosing the incident.
    - Prefer precise evidence over selecting everything.
    - Do not create new evidence.
    - Do not create new event IDs.
    - Do not create new artifact paths.
    - Do not copy input section headings as output keys.
    - Return ONLY one JSON object.
    - Do not return Markdown.
    - Do not return explanatory prose before or after the JSON.

    Before returning the answer, verify that:
    1. all seven required keys are present;
    2. no other keys are present;
    3. every selected event ID exists in NORMALIZED_EVENTS;
    4. every selected configuration path exists in CONFIGURATION_ARTIFACTS;
    5. every selected network artifact exists in NETWORK_ARTIFACTS;
    6. the evidence_summary contains no diagnosis or remediation;
    7. confidence is between 0.0 and 1.0.
    """.strip()

    def __init__(
        self,
        llm_client: LLMClient,
        formatter: EvidenceContextFormatter | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.formatter = formatter or EvidenceContextFormatter()

    def extract(
        self,
        context: IncidentEvidenceContext,
    ) -> EvidenceExtraction:
        formatted_context = self.formatter.format(context)

        response = self.llm_client.generate(
        system_prompt=self.SYSTEM_PROMPT,
        user_prompt=formatted_context,
        response_format=EvidenceExtraction.model_json_schema()
    )

        print("\n===== RAW LLM RESPONSE =====")
        print(response)
        print("===== END RAW LLM RESPONSE =====\n")

        extraction = self._parse_response(response)

        self._validate_references(
            extraction,
            context,
        )

        return extraction

    @staticmethod
    def _parse_response(response: str) -> EvidenceExtraction:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Evidence Extraction Agent returned invalid JSON"
            ) from exc

        try:
            return EvidenceExtraction.model_validate(payload)
        except ValueError as exc:
            raise ValueError(
                "Evidence Extraction Agent returned invalid "
                "EvidenceExtraction schema"
            ) from exc

    @staticmethod
    def _validate_references(
        extraction: EvidenceExtraction,
        context: IncidentEvidenceContext,
    ) -> None:

        if extraction.incident_id != context.incident_id:
            raise ValueError(
                f"Evidence Extraction Agent returned incident ID "
                f"'{extraction.incident_id}', expected "
                f"'{context.incident_id}'"
            )

        valid_event_ids = {
            event.event_id
            for event in context.events
        }

        event_types_by_id = {
            event.event_id: event.event_type
            for event in context.events
        }

        valid_configuration = set(
            context.configuration_artifacts
        )

        valid_network_artifacts = set(
            context.network_artifacts
        )

        invalid_events = (
            set(extraction.relevant_events)
            - valid_event_ids
        )

        invalid_network_events = (
            set(extraction.relevant_network_events)
            - valid_event_ids
        )

        invalid_configuration = (
            set(extraction.relevant_configuration)
            - valid_configuration
        )

        invalid_network_artifacts = (
            set(extraction.relevant_network_artifacts)
            - valid_network_artifacts
        )

        if invalid_events:
            raise ValueError(
                "Evidence Extraction Agent referenced unknown "
                f"event IDs: {sorted(invalid_events)}"
            )

        if invalid_network_events:
            raise ValueError(
                "Evidence Extraction Agent referenced unknown "
                f"network event IDs: "
                f"{sorted(invalid_network_events)}"
            )


        non_network_event_ids = [
            event_id
            for event_id in extraction.relevant_network_events
            if event_types_by_id[event_id] != "http_request"
        ]

        if non_network_event_ids:
            raise ValueError(
                "Evidence Extraction Agent referenced non-network event "
                f"IDs in relevant_network_events: "
                f"{sorted(non_network_event_ids)}"
            )


        if invalid_configuration:
            raise ValueError(
                "Evidence Extraction Agent referenced unknown "
                f"configuration artifacts: "
                f"{sorted(invalid_configuration)}"
            )

        if invalid_network_artifacts:
            raise ValueError(
                "Evidence Extraction Agent referenced unknown "
                f"network artifacts: "
                f"{sorted(invalid_network_artifacts)}"
            )