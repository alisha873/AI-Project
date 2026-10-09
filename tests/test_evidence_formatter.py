from pathlib import Path

from evidence.context import EvidenceContextBuilder

from evidence.formatter import EvidenceContextFormatter
from evidence.schema import (
    EvidenceEvent,
    IncidentEvidenceContext,
)
INCIDENT_DIR = Path("incident_001")


def test_formatter_contains_incident_id():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    formatted = EvidenceContextFormatter().format(
        context
    )

    assert "INCIDENT_ID: incident_001" in formatted


def test_formatter_contains_all_non_packet_event_ids():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    formatted = EvidenceContextFormatter().format(
        context
    )

    for event in context.events:
        if event.event_type == "network_packet":
            assert event.event_id not in formatted
        else:
            assert event.event_id in formatted


def test_formatter_contains_artifacts():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    formatted = EvidenceContextFormatter().format(
        context
    )

    expected_artifacts = (
        context.configuration_artifacts
        + context.network_artifacts
        + context.evidence_artifacts
    )

    for artifact in expected_artifacts:
        assert artifact in formatted


def test_formatter_excludes_ground_truth():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    formatted = EvidenceContextFormatter().format(
        context
    )

    assert "ground_truth" not in formatted.lower()
    assert "expected_remediation" not in formatted.lower()
    assert "root_cause" not in formatted.lower()


def test_formatter_is_deterministic():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    formatter = EvidenceContextFormatter()

    first = formatter.format(context)
    second = formatter.format(context)

    assert first == second

def test_network_packets_are_compact():
    event = EvidenceEvent(
        event_id="packet-001",
        timestamp="2026-10-09T10:37:49Z",
        source="pcap",
        event_type="network_packet",
        source_entity="127.0.0.1:50000",
        destination_entity="127.0.0.1:8080",
        action="TCP",
        attributes={
            "protocol": "TCP",
            "source_port": 50000,
            "destination_port": 8080,
            "packet_length": 74,
            "large_payload": "x" * 5000,
        },
    )

    context = IncidentEvidenceContext(
        incident_id="incident_001",
        events=[event],
    )

    original_event = event.model_dump(mode="json")
    formatted = EvidenceContextFormatter().format(context)

    assert "packet-001" not in formatted
    assert "network_packet" not in formatted
    assert "large_payload" not in formatted
    assert "xxxxx" not in formatted
    assert event.model_dump(mode="json") == original_event


def test_network_packet_remains_canonical_but_is_not_presented_individually():
    event = EvidenceEvent(
        event_id="packet-123",
        timestamp="2026-10-09T10:37:49Z",
        source="pcap",
        event_type="network_packet",
        source_entity="127.0.0.1:50000",
        destination_entity="127.0.0.1:8080",
        action="TCP",
        attributes={
            "protocol": "TCP",
        },
    )

    context = IncidentEvidenceContext(
        incident_id="incident_001",
        events=[event],
    )

    original_event = event.model_dump(mode="json")
    formatted = EvidenceContextFormatter().format(context)

    assert len(context.events) == 1
    assert context.events[0].event_id == "packet-123"
    assert context.events[0].event_type == "network_packet"
    assert "packet-123" not in formatted
    assert event.model_dump(mode="json") == original_event


def test_packet_omission_keeps_network_artifact_and_other_event_types():
    events = [
        EvidenceEvent(
            event_id="http-preserved",
            timestamp="2026-10-09T10:37:49Z",
            source="nginx",
            event_type="http_request",
            source_entity="client",
            destination_entity="autoshield-api",
            action="GET /api/data",
            attributes={"method": "GET", "path": "/api/data", "status": 200},
        ),
        EvidenceEvent(
            event_id="container-preserved",
            timestamp="2026-10-09T10:37:50Z",
            source="docker",
            event_type="container_state",
            source_entity="autoshield-api",
            action="running",
            attributes={"image": "autoshield-api:latest", "running": True},
        ),
        EvidenceEvent(
            event_id="membership-preserved",
            timestamp="2026-10-09T10:37:51Z",
            source="docker",
            event_type="network_membership",
            source_entity="autoshield-api",
            destination_entity="autoshield_net",
            action="attached_to_network",
            attributes={"ip_address": "172.19.0.6"},
        ),
        EvidenceEvent(
            event_id="packet-omitted-from-presentation",
            timestamp="2026-10-09T10:37:52Z",
            source="pcap",
            event_type="network_packet",
            source_entity="172.19.0.3:45000",
            destination_entity="172.19.0.6:8000",
            action="TCP request bytes",
            attributes={"protocol": "TCP", "source_port": 45000},
        ),
    ]
    context = IncidentEvidenceContext(
        incident_id="incident_001",
        events=events,
        network_artifacts=["network/attack.pcap"],
    )
    canonical_before = [event.model_dump(mode="json") for event in context.events]

    formatted = EvidenceContextFormatter().format(context)

    assert "event_id=http-preserved" in formatted
    assert "event_type=http_request" in formatted
    assert "event_id=container-preserved" in formatted
    assert "event_type=container_state" in formatted
    assert "event_id=membership-preserved" in formatted
    assert "event_type=network_membership" in formatted
    assert "network/attack.pcap" in formatted
    assert "packet-omitted-from-presentation" not in formatted
    assert "event_type=network_packet" not in formatted
    assert [event.model_dump(mode="json") for event in context.events] == canonical_before


def test_packet_omission_is_deterministic():
    event = EvidenceEvent(
        event_id="packet-deterministic",
        timestamp="2026-10-09T10:37:49Z",
        source="pcap",
        event_type="network_packet",
        source_entity="10.0.0.1",
        destination_entity="10.0.0.2",
        action="TCP",
    )
    context = IncidentEvidenceContext(
        incident_id="incident_001",
        events=[event],
        network_artifacts=["network/attack.pcap"],
    )
    formatter = EvidenceContextFormatter()

    assert formatter.format(context) == formatter.format(context)


def test_non_network_events_keep_attributes():
    event = EvidenceEvent(
        event_id="http-001",
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

    context = IncidentEvidenceContext(
        incident_id="incident_001",
        events=[event],
    )

    formatted = EvidenceContextFormatter().format(context)

    assert "event_id=http-001" in formatted
    assert "path='/api/search'" in formatted
    assert "status=200" in formatted


def test_compact_event_preserves_identity_type_entities_and_action():
    event = EvidenceEvent(
        event_id="request-compact-001",
        timestamp="2026-10-09T10:37:49Z",
        source="nginx",
        event_type="http_request",
        source_entity="client:51000",
        destination_entity="autoshield-api",
        action="GET",
        attributes={
            "method": "GET",
            "path": "/api/data",
            "query": "id=7",
            "status": 200,
            "user_agent": "x" * 3000,
            "evidence_sources": ["nginx", "api"],
        },
    )
    context = IncidentEvidenceContext(incident_id="incident_001", events=[event])

    formatted = EvidenceContextFormatter().format(context)

    assert "event_id=request-compact-001" in formatted
    assert "event_type=http_request" in formatted
    assert "source_entity=client:51000" in formatted
    assert "destination_entity=autoshield-api" in formatted
    assert "action=GET" in formatted
    assert "path='/api/data'" in formatted
    assert "query='id=7'" in formatted
    assert "status=200" in formatted
    assert "user_agent" not in formatted
    assert "evidence_sources" not in formatted
    assert len(formatted) < 500


def test_artifact_paths_are_preserved_in_compact_format():
    context = IncidentEvidenceContext(
        incident_id="incident_001",
        configuration_artifacts=["docker/containers_after_attack.json"],
        network_artifacts=["network/attack.pcap"],
        evidence_artifacts=["logs/nginx.log", "metadata/attack_request.txt"],
    )

    formatted = EvidenceContextFormatter().format(context)

    for artifact in (
        "docker/containers_after_attack.json",
        "network/attack.pcap",
        "logs/nginx.log",
        "metadata/attack_request.txt",
    ):
        assert artifact in formatted


def test_network_packet_is_omitted_without_mutating_canonical_event():
    event = EvidenceEvent(
        event_id="packet-compact-001",
        timestamp="2026-10-09T10:37:49Z",
        source="pcap",
        event_type="network_packet",
        source_entity="10.0.0.4:51000",
        destination_entity="10.0.0.9:5432",
        action="TCP 10.0.0.4:51000 -> 10.0.0.9:5432",
        attributes={
            "protocol": "TCP",
            "source_port": 51000,
            "destination_port": 5432,
            "captured_length": 74,
            "original_length": 74,
            "source_ip": "10.0.0.4",
            "destination_ip": "10.0.0.9",
            "packet_payload": "sensitive" * 1000,
        },
    )
    before = event.model_dump(mode="json")
    context = IncidentEvidenceContext(
        incident_id="incident_001",
        events=[event],
        network_artifacts=["network/attack.pcap"],
    )
    formatted = EvidenceContextFormatter().format(context)

    assert "packet-compact-001" not in formatted
    assert "network_packet" not in formatted
    assert "10.0.0.4:51000" not in formatted
    assert "10.0.0.9:5432" not in formatted
    assert "protocol='TCP'" not in formatted
    assert "source_port=51000" not in formatted
    assert "destination_port=5432" not in formatted
    assert "captured_length=74" not in formatted
    assert "original_length=74" not in formatted
    assert "network/attack.pcap" in formatted
    assert "packet_payload" not in formatted
    assert "sensitive" not in formatted
    assert event.model_dump(mode="json") == before


def test_formatter_does_not_expose_ground_truth_attributes():
    event = EvidenceEvent(
        event_id="event-ground-truth-check",
        timestamp="2026-10-09T10:37:49Z",
        source="docker",
        event_type="container_state",
        source_entity="autoshield-api",
        action="running",
        attributes={
            "image": "autoshield-api:latest",
            "ground_truth": "secret-answer",
            "expected_remediation": "secret-remediation",
            "attack_type": "secret-classification",
        },
    )

    formatted = EvidenceContextFormatter().format(
        IncidentEvidenceContext(incident_id="incident_001", events=[event])
    )

    assert "ground_truth" not in formatted.lower()
    assert "secret-answer" not in formatted
    assert "expected_remediation" not in formatted
    assert "secret-remediation" not in formatted
    assert "attack_type" not in formatted
    assert "secret-classification" not in formatted