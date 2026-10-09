from pathlib import Path
from collections import Counter

from evidence.pipeline import EvidencePipeline


INCIDENT_DIR = Path("incident_001")


def test_pipeline_processes_incident():
    pipeline = EvidencePipeline()

    events = pipeline.process(INCIDENT_DIR)

    assert events


def test_pipeline_produces_expected_event_types():
    pipeline = EvidencePipeline()

    events = pipeline.process(INCIDENT_DIR)

    counts = Counter(
        event.event_type
        for event in events
    )

    assert counts["container_state"] == 6
    assert counts["network_membership"] == 6
    assert counts["http_request"] == 14
    assert counts["network_packet"] == 24


def test_pipeline_deduplicates_docker_membership():
    pipeline = EvidencePipeline()

    events = pipeline.process(INCIDENT_DIR)

    network_events = [
        event
        for event in events
        if event.event_type == "network_membership"
    ]

    assert len(network_events) == 6

    api_event = next(
        event
        for event in network_events
        if event.source_entity == "autoshield-api"
    )

    assert len(
        api_event.attributes["source_event_ids"]
    ) == 2


def test_pipeline_resolves_docker_entities():
    pipeline = EvidencePipeline()

    events = pipeline.process(INCIDENT_DIR)

    nginx_events = [
        event
        for event in events
        if (
            event.source == "nginx"
            and event.event_type == "http_request"
        )
    ]

    assert nginx_events

    api_events = [
        event
        for event in nginx_events
        if (
            event.attributes.get(
                "destination_canonical_entity"
            )
            == "container:autoshield-api"
        )
    ]

    assert api_events


def test_pipeline_keeps_unknown_attacker_identity():
    pipeline = EvidencePipeline()

    events = pipeline.process(INCIDENT_DIR)

    nginx_events = [
        event
        for event in events
        if (
            event.source == "nginx"
            and event.source_entity == "192.168.65.1"
        )
    ]

    assert nginx_events

    for event in nginx_events:
        assert event.attributes["source_entity_type"] == "ip"
        assert (
            "source_canonical_entity"
            not in event.attributes
        )