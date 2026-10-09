import json
from pathlib import Path

from evidence.parsers.base import EvidenceParser
from evidence.schema import EvidenceEvent


class DockerParser(EvidenceParser):
    source = "docker"

    def parse(self, path: Path) -> list[EvidenceEvent]:
        if path.name == "containers_after_attack.json":
            return self._parse_containers(path)

        if path.name == "network_after_attack.json":
            return self._parse_networks(path)

        raise ValueError(f"Unsupported Docker evidence file: {path}")

    def _load_json(self, path: Path):
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _parse_containers(self, path: Path) -> list[EvidenceEvent]:
        containers = self._load_json(path)
        events: list[EvidenceEvent] = []

        for index, container in enumerate(containers):
            container_id = container["Id"]
            name = container["Name"].lstrip("/")
            state = container.get("State", {})
            host_config = container.get("HostConfig", {})
            config = container.get("Config", {})
            network_settings = container.get("NetworkSettings", {})

            events.append(
                EvidenceEvent(
                    event_id=f"docker-container-{index}",
                    timestamp=state.get("StartedAt", container["Created"]),
                    source=self.source,
                    event_type="container_state",
                    source_entity=name,
                    action=state.get("Status", "unknown"),
                    attributes={
                        "container_id": container_id,
                        "image": config.get("Image"),
                        "network_mode": host_config.get("NetworkMode"),
                        "running": state.get("Running"),
                        "paused": state.get("Paused"),
                        "restarting": state.get("Restarting"),
                        "oom_killed": state.get("OOMKilled"),
                        "dead": state.get("Dead"),
                        "restart_count": container.get("RestartCount"),
                        "exit_code": state.get("ExitCode"),
                        "privileged": host_config.get("Privileged"),
                        "readonly_rootfs": host_config.get("ReadonlyRootfs"),
                        "exposed_ports": sorted(
                            (config.get("ExposedPorts") or {}).keys()
                        ),
                        "published_ports": self._published_ports(
                            network_settings.get("Ports") or {}
                        ),
                    },
                )
            )

            for network_name, network in (
                network_settings.get("Networks") or {}
            ).items():
                events.append(
                    EvidenceEvent(
                        event_id=f"docker-container-network-{index}-{network_name}",
                        timestamp=state.get("StartedAt", container["Created"]),
                        source=self.source,
                        event_type="network_membership",
                        source_entity=name,
                        destination_entity=network_name,
                        action="attached_to_network",
                        attributes={
                            "container_id": container_id,
                            "network_id": network.get("NetworkID"),
                            "endpoint_id": network.get("EndpointID"),
                            "ip_address": network.get("IPAddress"),
                            "gateway": network.get("Gateway"),
                            "mac_address": network.get("MacAddress"),
                            "aliases": network.get("Aliases", []),
                            "dns_names": network.get("DNSNames", []),
                        },
                    )
                )

        return events

    def _parse_networks(self, path: Path) -> list[EvidenceEvent]:
        networks = self._load_json(path)
        events: list[EvidenceEvent] = []

        for network_index, network in enumerate(networks):
            network_name = network["Name"]

            for container_id, container in (
                network.get("Containers") or {}
            ).items():
                events.append(
                    EvidenceEvent(
                        event_id=(
                            f"docker-network-{network_index}-"
                            f"{container_id[:12]}"
                        ),
                        timestamp=network["Created"],
                        source=self.source,
                        event_type="network_membership",
                        source_entity=container["Name"],
                        destination_entity=network_name,
                        action="attached_to_network",
                        attributes={
                            "container_id": container_id,
                            "endpoint_id": container.get("EndpointID"),
                            "ip_address": self._strip_prefix(
                                container.get("IPv4Address")
                            ),
                            "ipv6_address": container.get("IPv6Address"),
                            "mac_address": container.get("MacAddress"),
                            "network_id": network.get("Id"),
                            "driver": network.get("Driver"),
                            "scope": network.get("Scope"),
                        },
                    )
                )

        return events

    @staticmethod
    def _published_ports(ports: dict) -> list[dict]:
        published = []

        for container_port, bindings in ports.items():
            for binding in bindings or []:
                published.append(
                    {
                        "container_port": container_port,
                        "host_ip": binding.get("HostIp"),
                        "host_port": binding.get("HostPort"),
                    }
                )

        return published

    @staticmethod
    def _strip_prefix(address: str | None) -> str | None:
        if not address:
            return address

        return address.split("/", 1)[0]