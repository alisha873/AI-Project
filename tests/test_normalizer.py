from pathlib import Path

from evidence.entity_resolver import EntityResolver
from evidence.normalizer import EvidenceNormalizer
from evidence.parsers.docker import DockerParser


DOCKER_DIR = Path("incident_001/docker")


def build_resolver():
    parser = DockerParser()

    events = parser.parse(
        DOCKER_DIR / "containers_after_attack.json"
    )

    resolver = EntityResolver()
    resolver.build(events)

    return resolver


def test_normalizer_deduplicates_docker_network_membership():
    parser = DockerParser()

    container_events = parser.parse(
        DOCKER_DIR / "containers_after_attack.json"
    )

    network_events = parser.parse(
        DOCKER_DIR / "network_after_attack.json"
    )

    resolver = build_resolver()
    normalizer = EvidenceNormalizer(resolver)

    normalized = normalizer.normalize(
        container_events + network_events
    )

    network_events = [
        event
        for event in normalized
        if event.event_type == "network_membership"
    ]

    assert len(network_events) == 6


def test_normalizer_preserves_duplicate_provenance():
    parser = DockerParser()

    container_events = parser.parse(
        DOCKER_DIR / "containers_after_attack.json"
    )

    network_events = parser.parse(
        DOCKER_DIR / "network_after_attack.json"
    )

    resolver = build_resolver()
    normalizer = EvidenceNormalizer(resolver)

    normalized = normalizer.normalize(
        container_events + network_events
    )

    api_network = next(
        event
        for event in normalized
        if (
            event.event_type == "network_membership"
            and event.source_entity == "autoshield-api"
        )
    )

    assert len(api_network.attributes["source_event_ids"]) == 2
    assert (
        "docker-container-network-1-autoshield_net"
        in api_network.attributes["source_event_ids"]
    )
    assert (
        "docker-network-0-b71b5355ee7e"
        in api_network.attributes["source_event_ids"]
    )


def test_normalizer_preserves_canonical_entity():
    parser = DockerParser()

    container_events = parser.parse(
        DOCKER_DIR / "containers_after_attack.json"
    )

    resolver = build_resolver()
    normalizer = EvidenceNormalizer(resolver)

    normalized = normalizer.normalize(container_events)

    api_network = next(
        event
        for event in normalized
        if (
            event.event_type == "network_membership"
            and event.source_entity == "autoshield-api"
        )
    )

    assert (
        api_network.attributes["source_canonical_entity"]
        == "container:autoshield-api"
    )


def test_normalizer_keeps_container_state_events():
    parser = DockerParser()

    events = parser.parse(
        DOCKER_DIR / "containers_after_attack.json"
    )

    resolver = build_resolver()
    normalizer = EvidenceNormalizer(resolver)

    normalized = normalizer.normalize(events)

    container_events = [
        event
        for event in normalized
        if event.event_type == "container_state"
    ]

    assert len(container_events) == 6