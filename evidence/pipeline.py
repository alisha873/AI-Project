from __future__ import annotations

from pathlib import Path

from evidence.entity_resolver import EntityResolver
from evidence.normalizer import EvidenceNormalizer
from evidence.parsers import (
    ApplicationParser,
    AuthParser,
    DockerParser,
    NginxParser,
    PcapngParser,
)
from evidence.schema import EvidenceEvent


class EvidencePipeline:
    """
    End-to-end forensic evidence ingestion pipeline.

    Raw evidence is parsed into EvidenceEvents, canonical entities
    are resolved from Docker evidence, and observations are
    normalized and deduplicated.

    This layer performs no attack classification or AI inference.
    """

    def process(self, incident_dir: Path) -> list[EvidenceEvent]:
        """
        Process all supported evidence available in an incident
        directory.
        """

        incident_dir = Path(incident_dir)

        docker_events = self._parse_docker(incident_dir)

        resolver = EntityResolver()
        resolver.build(docker_events)

        events: list[EvidenceEvent] = []
        events.extend(docker_events)

        nginx_paths = [
            incident_dir / "logs" / "nginx.log",
        ]
        for nginx_path in nginx_paths:
            self._parse_optional(events, NginxParser(), nginx_path)

        self._parse_optional(
            events,
            ApplicationParser(),
            incident_dir / "logs" / "api.log",
        )

        self._parse_optional(
            events,
            AuthParser(),
            incident_dir / "logs" / "linux_admin" / "auth.log",
        )

        self._parse_optional(
            events,
            PcapngParser(),
            incident_dir / "network" / "attack.pcap",
        )

        normalizer = EvidenceNormalizer(resolver)

        return normalizer.normalize(events)

    @staticmethod
    def _parse_docker(
        incident_dir: Path,
    ) -> list[EvidenceEvent]:
        import json

        parser = DockerParser()
        events: list[EvidenceEvent] = []

        # Legacy incident layout.
        legacy_containers = (
            incident_dir / "docker" / "containers_after_attack.json"
        )
        legacy_networks = (
            incident_dir / "docker" / "network_after_attack.json"
        )

        if legacy_containers.exists():
            events.extend(parser.parse(legacy_containers))

        if legacy_networks.exists():
            events.extend(parser.parse(legacy_networks))

        # Current capture layout: individual Docker inspection snapshots.
        container_dir = incident_dir / "configs" / "containers"

        if container_dir.exists():
            snapshots = []

            for path in sorted(container_dir.glob("*.json")):
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)

                if isinstance(data, list):
                    snapshots.extend(data)
                elif isinstance(data, dict):
                    snapshots.append(data)

            if snapshots:
                events.extend(
                    parser._parse_containers_from_data(snapshots)
                )

        # Current capture's network inspection snapshot.
        current_networks = incident_dir / "network" / "autoshield_net.json"

        if current_networks.exists():
            events.extend(parser.parse(current_networks))

        return events


    @staticmethod
    def _parse_optional(
        events: list[EvidenceEvent],
        parser,
        path: Path,
    ) -> None:
        if path.exists():
            events.extend(parser.parse(path))