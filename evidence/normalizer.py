from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from evidence.entity_resolver import EntityResolver
from evidence.schema import EvidenceEvent


class EvidenceNormalizer:
    """
    Normalizes parsed forensic events into a unified event stream.

    Responsibilities:
    - resolve canonical entities
    - deduplicate semantically identical observations
    - preserve evidence provenance

    This layer does not perform attack classification or inference.
    """

    def __init__(self, resolver: EntityResolver) -> None:
        self.resolver = resolver

    def normalize(
        self,
        events: Iterable[EvidenceEvent],
    ) -> list[EvidenceEvent]:
        """
        Resolve and deduplicate a collection of evidence events.
        """

        normalized: OrderedDict[
            tuple,
            EvidenceEvent,
        ] = OrderedDict()

        for event in events:
            resolved = self.resolver.resolve_event(event)
            key = self._deduplication_key(resolved)

            if key not in normalized:
                attributes = dict(resolved.attributes)

                attributes["evidence_sources"] = [
                    event.source
                ]

                attributes["source_event_ids"] = [
                    event.event_id
                ]

                normalized[key] = resolved.model_copy(
                    update={"attributes": attributes}
                )

                continue

            existing = normalized[key]

            attributes = dict(existing.attributes)

            evidence_sources = list(
                attributes.get("evidence_sources", [])
            )

            if event.source not in evidence_sources:
                evidence_sources.append(event.source)

            source_event_ids = list(
                attributes.get("source_event_ids", [])
            )

            if event.event_id not in source_event_ids:
                source_event_ids.append(event.event_id)

            attributes["evidence_sources"] = evidence_sources
            attributes["source_event_ids"] = source_event_ids

            normalized[key] = existing.model_copy(
                update={"attributes": attributes}
            )

        return list(normalized.values())

    @staticmethod
    def _deduplication_key(event: EvidenceEvent) -> tuple:
        """
        Build a semantic identity for an observation.

        Docker network membership is deduplicated across independent
        Docker snapshot files. Other event types remain distinct
        unless their semantic identity is identical.
        """

        if event.event_type == "network_membership":
            return (
                event.event_type,
                event.source_entity,
                event.destination_entity,
                event.action,
                event.attributes.get("ip_address"),
                event.attributes.get("container_id"),
                event.attributes.get("network_id"),
            )

        return (
            event.event_type,
            event.source,
            event.source_entity,
            event.destination_entity,
            event.action,
            event.timestamp,
            event.event_id,
        )