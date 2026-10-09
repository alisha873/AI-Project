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

        self._parse_optional(
            events,
            NginxParser(),
            incident_dir / "logs" / "nginx.log",
        )

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
        parser = DockerParser()
        events: list[EvidenceEvent] = []

        for filename in (
            "containers_after_attack.json",
            "network_after_attack.json",
        ):
            path = incident_dir / "docker" / filename

            if path.exists():
                events.extend(parser.parse(path))

        return events

    @staticmethod
    def _parse_optional(
        events: list[EvidenceEvent],
        parser,
        path: Path,
    ) -> None:
        if path.exists():
            events.extend(parser.parse(path))