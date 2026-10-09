from __future__ import annotations

from evidence.schema import EvidenceEvent, IncidentEvidenceContext


class EvidenceContextFormatter:
    """
    Converts the controlled incident context into a deterministic,
    model-ready evidence representation.

    This formatter does not interpret or diagnose evidence.
    """

    _ATTRIBUTE_FIELDS = {
        "container_state": (
            "image",
            "network_mode",
            "running",
            "restart_count",
            "exit_code",
            "oom_killed",
            "exposed_ports",
            "published_ports",
        ),
        "network_membership": (
            "network_id",
            "ip_address",
            "gateway",
            "aliases",
            "mac_address",
        ),
        "http_request": (
            "method",
            "path",
            "query",
            "status",
            "status_text",
            "upstream",
            "upstream_status",
            "request_time",
            "body_bytes",
            "client_port",
        ),
        "network_packet": (
            "protocol",
            "source_port",
            "destination_port",
            "src_port",
            "dst_port",
            "captured_length",
            "original_length",
            "packet_length",
            "length",
        ),
    }

    _OMIT_ATTRIBUTE_FIELDS = frozenset(
        {
            "evidence_sources",
            "source_event_ids",
            "source_canonical_entity",
            "source_entity_type",
            "destination_canonical_entity",
            "destination_entity_type",
            "large_payload",
            "payload",
            "raw_payload",
            "ground_truth",
            "attack_type",
            "root_cause",
            "expected_attack_path",
            "expected_remediation",
            "attacker_objective",
        }
    )

    def format(self, context: IncidentEvidenceContext) -> str:
        sections: list[str] = []

        sections.append(
            f"INCIDENT_ID: {context.incident_id}"
        )

        sections.append(
            self._format_events(context.events)
        )

        sections.append(
            self._format_artifacts(
                "CONFIGURATION_ARTIFACTS",
                context.configuration_artifacts,
            )
        )

        sections.append(
            self._format_artifacts(
                "NETWORK_ARTIFACTS",
                context.network_artifacts,
            )
        )

        sections.append(
            self._format_artifacts(
                "SUPPORTING_EVIDENCE_ARTIFACTS",
                context.evidence_artifacts,
            )
        )

        return "\n\n".join(sections)

    @classmethod
    def _format_events(
        cls,
        events: list[EvidenceEvent],
    ) -> str:
        lines = ["NORMALIZED_EVENTS"]

        for event in events:
            if event.event_type == "network_packet":
                continue

            record = (
                f"- event_id={event.event_id} "
                f"event_type={event.event_type} "
                f"timestamp={event.timestamp.isoformat()} "
                f"source={event.source} "
                f"source_entity={event.source_entity or '-'} "
                f"destination_entity={event.destination_entity or '-'} "
                f"action={event.action}"
            )
            attributes = cls._format_attributes(event)
            if attributes:
                record += f" attributes={{{attributes}}}"
            lines.append(record)

        return "\n".join(lines)

    @classmethod
    def _format_attributes(cls, event: EvidenceEvent) -> str:
        attributes = event.attributes
        fields = cls._ATTRIBUTE_FIELDS.get(event.event_type)
        if fields is None:
            fields = tuple(sorted(attributes))

        formatted = []
        for key in fields:
            if key not in attributes or cls._is_excluded_attribute(key):
                continue
            value = attributes[key]
            if value is None:
                continue
            formatted.append(f"{key}={value!r}")
        return ",".join(formatted)

    @classmethod
    def _is_excluded_attribute(cls, key: str) -> bool:
        normalized = key.lower()
        return (
            normalized in cls._OMIT_ATTRIBUTE_FIELDS
            or "ground_truth" in normalized
            or normalized.startswith("expected_")
        )

    @staticmethod
    def _format_artifacts(
        heading: str,
        artifacts: list[str],
    ) -> str:
        lines = [heading]

        if not artifacts:
            lines.append("- none")
            return "\n".join(lines)

        for artifact in artifacts:
            lines.append(f"- {artifact}")

        return "\n".join(lines)