from __future__ import annotations

import json
from pathlib import Path

from evidence.schema import (
    EvidenceEvent,
    IncidentEvidenceContext,
)


class EvidenceContextBuilder:
    """
    Builds the controlled evidence context presented to the
    Evidence Extraction Agent.

    Ground-truth evaluation artifacts are never included.
    """

    def build(self, incident_dir: Path) -> IncidentEvidenceContext:
        incident_dir = Path(incident_dir)

        events = self._load_events(
            incident_dir / "normalized" / "events.jsonl"
        )

        configuration_artifacts = self._collect_existing(
            incident_dir,
            [
                "docker/containers_after_attack.json",
                "docker/network_after_attack.json",
            ],
        )

        network_artifacts = self._collect_existing(
            incident_dir,
            [
                "network/attack.pcap",
            ],
        )

        evidence_artifacts = self._collect_existing(
            incident_dir,
            [
                "database/application_data_after_attack.txt",
                "logs/api.log",
                "logs/nginx.log",
                "logs/postgres.log",
                "metadata/attack_request.txt",
            ],
        )

        return IncidentEvidenceContext(
            incident_id=incident_dir.name,
            events=events,
            configuration_artifacts=configuration_artifacts,
            network_artifacts=network_artifacts,
            evidence_artifacts=evidence_artifacts,
        )

    @staticmethod
    def _load_events(path: Path) -> list[EvidenceEvent]:
        if not path.exists():
            raise FileNotFoundError(
                f"Normalized evidence not found: {path}"
            )

        events: list[EvidenceEvent] = []

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue

                try:
                    payload = json.loads(line)
                    events.append(
                        EvidenceEvent.model_validate(payload)
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid normalized event at "
                        f"{path}:{line_number}"
                    ) from exc

        return events

    @staticmethod
    def _collect_existing(
        incident_dir: Path,
        relative_paths: list[str],
    ) -> list[str]:
        artifacts: list[str] = []

        for relative_path in relative_paths:
            path = incident_dir / relative_path

            if path.exists():
                artifacts.append(relative_path)

        return artifacts