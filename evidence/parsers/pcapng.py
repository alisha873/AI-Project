import ipaddress
import struct
from datetime import datetime, timezone
from pathlib import Path

from evidence.parsers.base import EvidenceParser
from evidence.schema import EvidenceEvent


SECTION_HEADER_BLOCK = 0x0A0D0D0A
INTERFACE_DESCRIPTION_BLOCK = 0x00000001
ENHANCED_PACKET_BLOCK = 0x00000006

LINKTYPE_NULL = 0
AF_INET = 2
AF_INET6 = 24

IPV4 = 4
IPV6 = 6
TCP = 6
UDP = 17


class PcapngParser(EvidenceParser):
    source = "pcapng"

    def parse(self, path: Path) -> list[EvidenceEvent]:
        data = path.read_bytes()

        if len(data) < 12:
            raise ValueError("PCAPNG file is too short")

        events: list[EvidenceEvent] = []
        interfaces: dict[int, dict] = {}

        offset = 0
        endian = None

        while offset < len(data):
            if offset + 12 > len(data):
                raise ValueError("Truncated PCAPNG block header")

            block_type = struct.unpack_from(
                "<I",
                data,
                offset,
            )[0]

            if block_type == SECTION_HEADER_BLOCK:
                endian, offset = self._parse_section_header(
                    data,
                    offset,
                )
                continue

            if endian is None:
                raise ValueError(
                    "PCAPNG block encountered before Section Header"
                )

            block_length = struct.unpack_from(
                endian + "I",
                data,
                offset + 4,
            )[0]

            if block_length < 12:
                raise ValueError(
                    f"Invalid PCAPNG block length: {block_length}"
                )

            if offset + block_length > len(data):
                raise ValueError("Truncated PCAPNG block")

            if block_type == INTERFACE_DESCRIPTION_BLOCK:
                interface_id = len(interfaces)
                interfaces[interface_id] = self._parse_interface(
                    data,
                    offset,
                    endian,
                )

            elif block_type == ENHANCED_PACKET_BLOCK:
                event = self._parse_enhanced_packet(
                    data,
                    offset,
                    block_length,
                    endian,
                    interfaces,
                )

                if event is not None:
                    events.append(event)

            offset += block_length

        return events

    def _parse_section_header(
        self,
        data: bytes,
        offset: int,
    ) -> tuple[str, int]:
        raw_magic = data[offset + 8 : offset + 12]

        if raw_magic == bytes.fromhex("4d3c2b1a"):
            endian = "<"
        elif raw_magic == bytes.fromhex("1a2b3c4d"):
            endian = ">"
        else:
            raise ValueError(
                f"Unsupported PCAPNG byte-order magic: "
                f"{raw_magic.hex()}"
            )

        block_length = struct.unpack_from(
            endian + "I",
            data,
            offset + 4,
        )[0]

        if block_length < 28:
            raise ValueError(
                "Invalid PCAPNG Section Header Block"
            )

        return endian, offset + block_length

    def _parse_interface(
        self,
        data: bytes,
        offset: int,
        endian: str,
    ) -> dict:
        link_type, _, snaplen = struct.unpack_from(
            endian + "HHI",
            data,
            offset + 8,
        )

        return {
            "link_type": link_type,
            "snaplen": snaplen,
            "timestamp_resolution": 1_000_000,
        }

    def _parse_enhanced_packet(
        self,
        data: bytes,
        offset: int,
        block_length: int,
        endian: str,
        interfaces: dict[int, dict],
    ) -> EvidenceEvent | None:
        (
            interface_id,
            timestamp_high,
            timestamp_low,
            captured_length,
            original_length,
        ) = struct.unpack_from(
            endian + "IIIII",
            data,
            offset + 8,
        )

        interface = interfaces.get(interface_id)

        if interface is None:
            return None

        packet_start = offset + 28
        packet_end = packet_start + captured_length

        if packet_end > offset + block_length - 4:
            raise ValueError("Truncated Enhanced Packet Block")

        packet = data[packet_start:packet_end]

        timestamp_units = (
            (timestamp_high << 32) | timestamp_low
        )

        timestamp = datetime.fromtimestamp(
            timestamp_units / interface["timestamp_resolution"],
            tz=timezone.utc,
        )

        decoded = self._decode_packet(
            packet,
            interface["link_type"],
        )

        if decoded is None:
            return None

        source_ip = decoded["source_ip"]
        destination_ip = decoded["destination_ip"]

        source_entity = None
        destination_entity = None

        return EvidenceEvent(
            event_id=f"pcapng-packet-{offset}",
            timestamp=timestamp,
            source=self.source,
            event_type="network_packet",
            source_entity=source_entity or source_ip,
            destination_entity=destination_entity or destination_ip,
            action=(
                f"{decoded['protocol']} "
                f"{source_ip}:{decoded.get('source_port', '')}"
                f" -> "
                f"{destination_ip}:{decoded.get('destination_port', '')}"
            ),
            attributes={
                "source_ip": source_ip,
                "destination_ip": destination_ip,
                "protocol": decoded["protocol"],
                "source_port": decoded.get("source_port"),
                "destination_port": decoded.get("destination_port"),
                "captured_length": captured_length,
                "original_length": original_length,
                "interface_id": interface_id,
            },
        )

    def _decode_packet(
        self,
        packet: bytes,
        link_type: int,
    ) -> dict | None:
        if link_type == LINKTYPE_NULL:
            return self._decode_null_link(packet)

        return None

    def _decode_null_link(self, packet: bytes) -> dict | None:
        if len(packet) < 4:
            return None

        family = struct.unpack_from("<I", packet, 0)[0]

        if family == AF_INET:
            return self._decode_ipv4(packet[4:])

        if family == AF_INET6:
            return self._decode_ipv6(packet[4:])

        return None

    def _decode_ipv4(self, packet: bytes) -> dict | None:
        if len(packet) < 20:
            return None

        version = packet[0] >> 4
        header_length = (packet[0] & 0x0F) * 4

        if version != IPV4 or len(packet) < header_length:
            return None

        protocol_number = packet[9]

        source_ip = str(
            ipaddress.IPv4Address(packet[12:16])
        )
        destination_ip = str(
            ipaddress.IPv4Address(packet[16:20])
        )

        result = {
            "protocol": self._protocol_name(protocol_number),
            "source_ip": source_ip,
            "destination_ip": destination_ip,
        }

        if protocol_number in (TCP, UDP):
            if len(packet) < header_length + 4:
                return result

            source_port, destination_port = struct.unpack_from(
                "!HH",
                packet,
                header_length,
            )

            result["source_port"] = source_port
            result["destination_port"] = destination_port

        return result

    def _decode_ipv6(self, packet: bytes) -> dict | None:
        if len(packet) < 40:
            return None

        if packet[0] >> 4 != IPV6:
            return None

        next_header = packet[6]

        source_ip = str(
            ipaddress.IPv6Address(packet[8:24])
        )
        destination_ip = str(
            ipaddress.IPv6Address(packet[24:40])
        )

        result = {
            "protocol": self._protocol_name(next_header),
            "source_ip": source_ip,
            "destination_ip": destination_ip,
        }

        if next_header in (TCP, UDP):
            if len(packet) < 44:
                return result

            source_port, destination_port = struct.unpack_from(
                "!HH",
                packet,
                40,
            )

            result["source_port"] = source_port
            result["destination_port"] = destination_port

        return result

    @staticmethod
    def _protocol_name(protocol_number: int) -> str:
        return {
            1: "ICMPv4",
            6: "TCP",
            17: "UDP",
            58: "ICMPv6",
        }.get(
            protocol_number,
            f"IP protocol {protocol_number}",
        )