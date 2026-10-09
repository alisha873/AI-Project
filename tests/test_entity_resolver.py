from pathlib import Path

from evidence.entity_resolver import EntityResolver
from evidence.parsers.docker import DockerParser
from evidence.schema import EvidenceEvent


def test_entity_resolver_builds_docker_identity_index():
    parser = DockerParser()

    events = parser.parse(
        Path("incident_001/docker/containers_after_attack.json")
    )

    resolver = EntityResolver()
    resolver.build(events)

    assert resolver.resolve_ip("172.19.0.6") == "container:autoshield-api"

    assert (
        resolver.resolve_container_id(
            "b71b5355ee7ea2a33320bc37cb03265c4cb95c1cd43259f4e9187d5894547f47"
        )
        == "container:autoshield-api"
    )

    assert resolver.resolve_name("autoshield-api") == "container:autoshield-api"


def test_entity_resolver_tracks_all_containers():
    parser = DockerParser()

    events = parser.parse(
        Path("incident_001/docker/containers_after_attack.json")
    )

    resolver = EntityResolver()
    resolver.build(events)

    entities = resolver.entities()

    assert len(entities) == 6

    names = {entity.name for entity in entities}

    assert names == {
        "autoshield-nginx",
        "autoshield-api",
        "autoshield-postgres",
        "autoshield-redis",
        "autoshield-juice-shop",
        "autoshield-linux-admin",
    }


def test_unknown_entity_returns_none():
    resolver = EntityResolver()

    assert resolver.resolve_ip("192.0.2.123") is None
    assert resolver.resolve_container_id("unknown") is None
    assert resolver.resolve_name("unknown") is None

def test_resolve_event_preserves_original_entities():
    parser = DockerParser()

    events = parser.parse(
        Path("incident_001/docker/containers_after_attack.json")
    )

    resolver = EntityResolver()
    resolver.build(events)

    event = EvidenceEvent(
        event_id="test-http-1",
        timestamp="2026-10-09T11:07:46Z",
        source="nginx",
        event_type="http_request",
        source_entity="192.168.65.1",
        destination_entity="172.19.0.6",
        action="GET /api/search",
        attributes={
            "status": 200,
        },
    )

    resolved = resolver.resolve_event(event)

    assert resolved.source_entity == "192.168.65.1"
    assert resolved.destination_entity == "172.19.0.6"

    assert resolved.attributes["source_entity_type"] == "ip"
    assert resolved.attributes["destination_entity_type"] == "ip"

    assert "source_canonical_entity" not in resolved.attributes

    assert (
        resolved.attributes["destination_canonical_entity"]
        == "container:autoshield-api"
    )


def test_resolve_event_does_not_mutate_original():
    parser = DockerParser()

    events = parser.parse(
        Path("incident_001/docker/containers_after_attack.json")
    )

    resolver = EntityResolver()
    resolver.build(events)

    event = EvidenceEvent(
        event_id="test-http-2",
        timestamp="2026-10-09T11:07:46Z",
        source="nginx",
        event_type="http_request",
        source_entity="192.168.65.1",
        destination_entity="172.19.0.6",
        action="GET /api/search",
        attributes={
            "status": 200,
        },
    )

    original_attributes = dict(event.attributes)

    resolved = resolver.resolve_event(event)

    assert event.attributes == original_attributes
    assert "destination_canonical_entity" not in event.attributes

    assert (
        resolved.attributes["destination_canonical_entity"]
        == "container:autoshield-api"
    )