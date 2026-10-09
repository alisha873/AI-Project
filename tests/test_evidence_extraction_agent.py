import json

import pytest

from agents.evidence_extraction import EvidenceExtractionAgent

from llm.base import LLMClient
from evidence.schema import (
    EvidenceEvent,
    EvidenceExtraction,
    IncidentEvidenceContext,
)

class FakeLLMClient(LLMClient):
    def __init__(self, response: str) -> None:
        self.response = response
        self.response_format = None
        self.user_prompt = None

    def generate(
    self,
    system_prompt: str,
    user_prompt: str,
    response_format: dict | None = None,
    ) -> str:
        self.response_format = response_format
        self.user_prompt = user_prompt
        return self.response


def build_context() -> IncidentEvidenceContext:
    event = EvidenceEvent(
        event_id="event-001",
        timestamp="2026-10-09T10:37:49Z",
        source="nginx",
        event_type="http_request",
        source_entity="client",
        destination_entity="autoshield-api",
        action="GET",
        attributes={
            "path": "/api/search",
            "status": 200,
        },
    )

    return IncidentEvidenceContext(
        incident_id="incident_001",
        events=[event],
        configuration_artifacts=[
            "docker/containers_after_attack.json",
        ],
        network_artifacts=[
            "network/attack.pcap",
        ],
        evidence_artifacts=[
            "logs/nginx.log",
        ],
    )


def test_extract_valid_evidence():
    response = json.dumps(
        {
            "incident_id": "incident_001",
            "relevant_events": ["event-001"],
            "relevant_configuration": [
                "docker/containers_after_attack.json",
            ],
            "relevant_network_events": ["event-001"],
            "relevant_network_artifacts": [
                "network/attack.pcap",
            ],
            "evidence_summary": (
                "The selected evidence contains an HTTP request "
                "to the application search endpoint."
            ),
            "confidence": 0.9,
        }
    )

    fake_client = FakeLLMClient(response)

    agent = EvidenceExtractionAgent(
        llm_client=fake_client
    )

    result = agent.extract(build_context())

    assert fake_client.response_format == (
        EvidenceExtraction.model_json_schema()
    )

    assert result.incident_id == "incident_001"
    assert result.relevant_events == ["event-001"]
    assert result.relevant_network_events == ["event-001"]
    assert result.relevant_configuration == [
        "docker/containers_after_attack.json"
    ]
    assert result.relevant_network_artifacts == [
        "network/attack.pcap"
    ]
    assert result.confidence == 0.9


def test_rejects_invalid_json():
    agent = EvidenceExtractionAgent(
        llm_client=FakeLLMClient(
            "this is not json"
        )
    )

    with pytest.raises(
        ValueError,
        match="invalid JSON",
    ):
        agent.extract(build_context())


def test_rejects_unknown_event_reference():
    response = json.dumps(
        {
            "incident_id": "incident_001",
            "relevant_events": ["event-999"],
            "relevant_configuration": [],
            "relevant_network_events": [],
            "relevant_network_artifacts": [],
            "evidence_summary": "Selected evidence.",
            "confidence": 0.8,
        }
    )

    agent = EvidenceExtractionAgent(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError,
        match="unknown event IDs",
    ):
        agent.extract(build_context())


def test_rejects_unknown_artifact_reference():
    response = json.dumps(
        {
            "incident_id": "incident_001",
            "relevant_events": [],
            "relevant_configuration": [
                "docker/fake_config.json"
            ],
            "relevant_network_events": [],
            "relevant_network_artifacts": [],
            "evidence_summary": "Selected evidence.",
            "confidence": 0.8,
        }
    )

    agent = EvidenceExtractionAgent(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError,
        match="unknown configuration artifacts",
    ):
        agent.extract(build_context())


def test_rejects_unknown_network_artifact():
    response = json.dumps(
        {
            "incident_id": "incident_001",
            "relevant_events": [],
            "relevant_configuration": [],
            "relevant_network_events": [],
            "relevant_network_artifacts": [
                "network/fake.pcap"
            ],
            "evidence_summary": "Selected evidence.",
            "confidence": 0.8,
        }
    )

    agent = EvidenceExtractionAgent(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError,
        match="unknown network artifacts",
    ):
        agent.extract(build_context())


def test_rejects_non_network_event_in_network_events():
    context = build_context()

    # Reuse the existing event ID, but make it a container-state event.
    context.events[0].event_type = "container_state"

    response = json.dumps(
        {
            "incident_id": "incident_001",
            "relevant_events": [],
            "relevant_configuration": [],
            "relevant_network_events": ["event-001"],
            "relevant_network_artifacts": [],
            "evidence_summary": "Selected evidence.",
            "confidence": 0.8,
        }
    )

    agent = EvidenceExtractionAgent(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError,
        match="network event",
    ):
        agent.extract(context)


def test_rejects_hidden_packet_event_in_network_events():
    context = build_context()
    context.events[0].event_id = "packet-hidden-001"
    context.events[0].event_type = "network_packet"

    response = json.dumps(
        {
            "incident_id": "incident_001",
            "relevant_events": [],
            "relevant_configuration": [],
            "relevant_network_events": ["packet-hidden-001"],
            "relevant_network_artifacts": [],
            "evidence_summary": "Selected evidence.",
            "confidence": 0.8,
        }
    )
    fake_client = FakeLLMClient(response)
    agent = EvidenceExtractionAgent(llm_client=fake_client)

    with pytest.raises(ValueError, match="network event"):
        agent.extract(context)

    assert "packet-hidden-001" not in fake_client.user_prompt


def test_rejects_mismatched_incident_id():
    response = json.dumps(
        {
            "incident_id": "incident_999",
            "relevant_events": [],
            "relevant_configuration": [],
            "relevant_network_events": [],
            "relevant_network_artifacts": [],
            "evidence_summary": "Selected evidence.",
            "confidence": 0.8,
        }
    )

    agent = EvidenceExtractionAgent(
        llm_client=FakeLLMClient(response)
    )

    with pytest.raises(
        ValueError,
        match="incident ID",
    ):
        agent.extract(build_context())
