from pathlib import Path

from evidence.context import EvidenceContextBuilder


INCIDENT_DIR = Path("incident_001")


def test_builder_loads_normalized_events():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    assert context.incident_id == "incident_001"
    assert len(context.events) == 50


def test_builder_collects_expected_artifacts():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    assert context.configuration_artifacts == [
        "docker/containers_after_attack.json",
        "docker/network_after_attack.json",
    ]

    assert context.network_artifacts == [
        "network/attack.pcap",
    ]

    assert context.evidence_artifacts == [
        "database/application_data_after_attack.txt",
        "logs/api.log",
        "logs/nginx.log",
        "logs/postgres.log",
        "metadata/attack_request.txt",
    ]


def test_builder_excludes_ground_truth():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    all_artifacts = (
        context.configuration_artifacts
        + context.network_artifacts
        + context.evidence_artifacts
    )

    assert not any(
        "ground_truth" in artifact
        for artifact in all_artifacts
    )


def test_builder_preserves_event_ids():
    context = EvidenceContextBuilder().build(
        INCIDENT_DIR
    )

    event_ids = [
        event.event_id
        for event in context.events
    ]

    assert len(event_ids) == 50
    assert len(set(event_ids)) == 50