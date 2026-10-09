from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from evidence.schema import EvidenceEvent
import ipaddress

@dataclass
class EntityRecord:
    """
    Canonical identity reconstructed from Docker evidence.
    """

    canonical_id: str
    entity_type: str
    name: str

    container_id: str | None = None
    ip_addresses: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    dns_names: set[str] = field(default_factory=set)


class EntityResolver:
    """
    Builds canonical entity mappings from normalized Docker evidence.

    The resolver uses observed evidence only. It does not infer
    attack semantics or classify events.
    """

    def __init__(self) -> None:
        self._by_ip: dict[str, str] = {}
        self._by_container_id: dict[str, str] = {}
        self._by_name: dict[str, str] = {}
        self._entities: dict[str, EntityRecord] = {}

    def build(self, events: Iterable[EvidenceEvent]) -> None:
        """
        Build the resolver index from Docker evidence events.
        """

        for event in events:
            if event.source != "docker":
                continue

            if event.event_type != "network_membership":
                continue

            container_name = event.source_entity
            if not container_name:
                continue

            attributes = event.attributes

            container_id = attributes.get("container_id")
            ip_address = attributes.get("ip_address")

            canonical_id = f"container:{container_name}"

            entity = self._entities.get(canonical_id)

            if entity is None:
                entity = EntityRecord(
                    canonical_id=canonical_id,
                    entity_type="container",
                    name=container_name,
                    container_id=container_id,
                )
                self._entities[canonical_id] = entity

            if container_id:
                entity.container_id = container_id
                self._by_container_id[container_id] = canonical_id

            if ip_address:
                entity.ip_addresses.add(ip_address)
                self._by_ip[ip_address] = canonical_id

            for alias in attributes.get("aliases", []):
                entity.aliases.add(alias)
                self._by_name[alias] = canonical_id

            for dns_name in attributes.get("dns_names", []):
                entity.dns_names.add(dns_name)
                self._by_name[dns_name] = canonical_id

            self._by_name[container_name] = canonical_id

    def resolve_ip(self, ip_address: str | None) -> str | None:
        """
        Resolve an observed IP address to a canonical entity.
        """

        if not ip_address:
            return None

        return self._by_ip.get(ip_address)

    def resolve_container_id(self, container_id: str | None) -> str | None:
        """
        Resolve a Docker container ID to a canonical entity.
        """

        if not container_id:
            return None

        return self._by_container_id.get(container_id)

    def resolve_name(self, name: str | None) -> str | None:
        """
        Resolve a container name, alias, or DNS name.
        """

        if not name:
            return None

        return self._by_name.get(name)

    def get_entity(self, canonical_id: str) -> EntityRecord | None:
        return self._entities.get(canonical_id)

    def entities(self) -> list[EntityRecord]:
        return list(self._entities.values())

    def resolve_event(self, event: EvidenceEvent) -> EvidenceEvent:
        """
        Enrich an EvidenceEvent with canonical entity resolution.

        Original source_entity and destination_entity values are
        preserved exactly as observed. Canonical identities are
        added to attributes when they can be established from
        observed Docker evidence.
        """

        attributes = dict(event.attributes)

        source_entity = event.source_entity
        destination_entity = event.destination_entity

        source_canonical = (
            self.resolve_ip(source_entity)
            or self.resolve_name(source_entity)
        )

        destination_canonical = (
            self.resolve_ip(destination_entity)
            or self.resolve_name(destination_entity)
        )

        if source_entity:
            try:
                ipaddress.ip_address(source_entity)
                attributes["source_entity_type"] = "ip"
            except ValueError:
                if self.resolve_name(source_entity):
                    attributes["source_entity_type"] = "name"

        if destination_entity:
            try:
                ipaddress.ip_address(destination_entity)
                attributes["destination_entity_type"] = "ip"
            except ValueError:
                if self.resolve_name(destination_entity):
                    attributes["destination_entity_type"] = "name"

        if source_canonical is not None:
            attributes["source_canonical_entity"] = source_canonical

        if destination_canonical is not None:
            attributes["destination_canonical_entity"] = destination_canonical

        return event.model_copy(
            update={
                "attributes": attributes,
            }
        )