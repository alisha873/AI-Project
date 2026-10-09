from pathlib import Path

from evidence.parsers.docker import DockerParser


INCIDENT_DOCKER_DIR = Path("incident_001/docker")


def test_parse_container_snapshot():
    parser = DockerParser()

    events = parser.parse(
        INCIDENT_DOCKER_DIR / "containers_after_attack.json"
    )

    assert events

    container_events = [
        event for event in events
        if event.event_type == "container_state"
    ]

    network_events = [
        event for event in events
        if event.event_type == "network_membership"
    ]

    assert len(container_events) == 6
    assert len(network_events) == 6

    api_event = next(
        event
        for event in container_events
        if event.source_entity == "autoshield-api"
    )

    assert api_event.action == "running"
    assert api_event.attributes["running"] is True
    assert api_event.attributes["privileged"] is False


def test_parse_network_snapshot():
    parser = DockerParser()

    events = parser.parse(
        INCIDENT_DOCKER_DIR / "network_after_attack.json"
    )

    assert len(events) == 6

    api_event = next(
        event
        for event in events
        if event.source_entity == "autoshield-api"
    )

    assert api_event.destination_entity == "autoshield_net"
    assert api_event.action == "attached_to_network"
    assert api_event.attributes["ip_address"] == "172.19.0.6"
    assert api_event.attributes["driver"] == "bridge"


def test_container_and_network_parsers_use_same_schema():
    parser = DockerParser()

    container_events = parser.parse(
        INCIDENT_DOCKER_DIR / "containers_after_attack.json"
    )
    network_events = parser.parse(
        INCIDENT_DOCKER_DIR / "network_after_attack.json"
    )

    for event in container_events + network_events:
        assert event.source == "docker"
        assert event.event_id
        assert event.timestamp
        assert event.event_type
        assert event.action
        assert isinstance(event.attributes, dict)