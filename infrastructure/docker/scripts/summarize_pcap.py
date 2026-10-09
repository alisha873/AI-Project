#!/usr/bin/env python3
"""Summarize observed classic-PCAP traffic without inferring unobserved events."""

import json
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PCAP_MAGICS = {
    bytes.fromhex("d4c3b2a1"): ("<", 1_000_000),
    bytes.fromhex("a1b2c3d4"): (">", 1_000_000),
    bytes.fromhex("4d3cb2a1"): ("<", 1_000_000_000),
    bytes.fromhex("a1b23c4d"): (">", 1_000_000_000),
}
PROTOCOL_NAMES = {1: "ICMPv4", 6: "TCP", 17: "UDP", 58: "ICMPv6"}
APPLICATION_PORTS = {80, 3000, 8000, 5432, 6379}


def read_network_labels(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as source:
        network_data = json.load(source)
    labels = {}
    for network in network_data:
        for container in network.get("Containers", {}).values():
            name = container.get("Name")
            for key in ("IPv4Address", "IPv6Address"):
                address = container.get(key, "").split("/", 1)[0]
                if name and address:
                    labels[address] = name
    return labels


def parse_packet(frame: bytes, link_type: int) -> dict | None:
    offset = 0
    if link_type == 1:  # Ethernet
        if len(frame) < 14:
            return None
        ether_type = struct.unpack_from("!H", frame, 12)[0]
        offset = 14
        while ether_type in (0x8100, 0x88A8, 0x9100):  # VLAN tags
            if len(frame) < offset + 4:
                return None
            ether_type = struct.unpack_from("!H", frame, offset + 2)[0]
            offset += 4
    elif link_type == 113:  # Linux cooked capture
        if len(frame) < 16:
            return None
        ether_type = struct.unpack_from("!H", frame, 14)[0]
        offset = 16
    elif link_type == 276:  # Linux cooked capture v2
        if len(frame) < 20:
            return None
        ether_type = struct.unpack_from("!H", frame, 0)[0]
        offset = 20
    else:
        return {"protocol": f"link_type_{link_type}"}

    if ether_type == 0x0806:  # ARP
        if len(frame) < offset + 8:
            return {"protocol": "ARP"}
        hardware_len, protocol_len = frame[offset + 4], frame[offset + 5]
        source_offset = offset + 8 + hardware_len
        target_offset = source_offset + protocol_len + hardware_len
        if protocol_len == 4 and len(frame) >= target_offset + protocol_len:
            return {
                "protocol": "ARP",
                "source_ip": ".".join(str(value) for value in frame[source_offset : source_offset + 4]),
                "destination_ip": ".".join(str(value) for value in frame[target_offset : target_offset + 4]),
            }
        return {"protocol": "ARP"}

    if ether_type == 0x0800:  # IPv4
        if len(frame) < offset + 20:
            return {"protocol": "IPv4"}
        header_length = (frame[offset] & 0x0F) * 4
        protocol_number = frame[offset + 9]
        source_ip = ".".join(str(value) for value in frame[offset + 12 : offset + 16])
        destination_ip = ".".join(str(value) for value in frame[offset + 16 : offset + 20])
        transport_offset = offset + header_length
        protocol = PROTOCOL_NAMES.get(protocol_number, f"IPv4 protocol {protocol_number}")
    elif ether_type == 0x86DD:  # IPv6
        if len(frame) < offset + 40:
            return {"protocol": "IPv6"}
        protocol_number = frame[offset + 6]
        source_bytes = frame[offset + 8 : offset + 24]
        destination_bytes = frame[offset + 24 : offset + 40]
        source_ip = ":".join(f"{value:02x}" for value in struct.unpack("!8H", source_bytes))
        destination_ip = ":".join(f"{value:02x}" for value in struct.unpack("!8H", destination_bytes))
        transport_offset = offset + 40
        protocol = PROTOCOL_NAMES.get(protocol_number, f"IPv6 next header {protocol_number}")
    else:
        return {"protocol": f"EtherType 0x{ether_type:04x}"}

    result = {
        "protocol": protocol,
        "source_ip": source_ip,
        "destination_ip": destination_ip,
    }
    if protocol_number in (6, 17) and len(frame) >= transport_offset + 4:
        result["source_port"], result["destination_port"] = struct.unpack_from("!HH", frame, transport_offset)
    return result


def load_request_results(path: Path) -> list[dict[str, str]]:
    results = []
    if not path.is_file():
        return results
    for line in path.read_text(encoding="utf-8").splitlines():
        request_path, separator, status = line.partition("\t")
        if separator:
            results.append({"method": "GET", "path": request_path, "http_status": status})
    return results


def main() -> int:
    if len(sys.argv) not in (5, 6):
        print("Usage: summarize_pcap.py PCAP SUMMARY_JSON NETWORK_JSON REQUEST_RESULTS [CAPTURE_MODE]", file=sys.stderr)
        return 2

    pcap_path, summary_path, network_path, request_path = map(Path, sys.argv[1:5])
    capture_mode = sys.argv[5] if len(sys.argv) == 6 else "unspecified"
    data = pcap_path.read_bytes()
    if len(data) < 24:
        raise ValueError(f"PCAP is shorter than its global header: {pcap_path}")
    if data[:4] not in PCAP_MAGICS:
        raise ValueError(f"Unsupported PCAP magic bytes: {data[:4].hex()}")

    endian, timestamp_resolution = PCAP_MAGICS[data[:4]]
    _, _, _, _, _, link_type = struct.unpack_from(endian + "HHIIII", data, 4)
    offset = 24
    packet_count = 0
    protocol_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    destination_counts: Counter[str] = Counter()
    flow_counts: Counter[tuple] = Counter()
    flow_times: dict[tuple, list[float]] = {}

    while offset < len(data):
        if offset + 16 > len(data):
            raise ValueError("Truncated PCAP packet record header")
        seconds, fraction, captured_length, _ = struct.unpack_from(endian + "IIII", data, offset)
        offset += 16
        if offset + captured_length > len(data):
            raise ValueError("Truncated PCAP packet data")
        frame = data[offset : offset + captured_length]
        offset += captured_length
        timestamp = seconds + fraction / timestamp_resolution
        packet_count += 1

        decoded = parse_packet(frame, link_type) or {"protocol": "undecoded"}
        protocol = decoded["protocol"]
        protocol_counts[protocol] += 1
        source_ip = decoded.get("source_ip")
        destination_ip = decoded.get("destination_ip")
        if source_ip:
            source_counts[source_ip] += 1
        if destination_ip:
            destination_counts[destination_ip] += 1

        if source_ip and destination_ip:
            key = (
                source_ip,
                destination_ip,
                protocol,
                decoded.get("source_port"),
                decoded.get("destination_port"),
            )
            flow_counts[key] += 1
            flow_times.setdefault(key, []).append(timestamp)

    labels = read_network_labels(network_path)
    flows = []
    application_flows = []
    timestamps = []
    for (source_ip, destination_ip, protocol, source_port, destination_port), count in flow_counts.items():
        packet_times = flow_times[(source_ip, destination_ip, protocol, source_port, destination_port)]
        timestamps.extend(packet_times)
        flow = {
            "source_ip": source_ip,
            "source_component": labels.get(source_ip),
            "destination_ip": destination_ip,
            "destination_component": labels.get(destination_ip),
            "protocol": protocol,
            "source_port": source_port,
            "destination_port": destination_port,
            "packet_count": count,
            "first_packet_timestamp_utc": datetime.fromtimestamp(min(packet_times), timezone.utc).isoformat(),
            "last_packet_timestamp_utc": datetime.fromtimestamp(max(packet_times), timezone.utc).isoformat(),
        }
        flows.append(flow)
        if source_port in APPLICATION_PORTS or destination_port in APPLICATION_PORTS:
            application_flows.append(flow)

    summary = {
        "schema_version": 1,
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pcap_file": pcap_path.name,
        "capture_mode": capture_mode,
        "packet_count": packet_count,
        "protocol_counts": dict(sorted(protocol_counts.items())),
        "source_addresses": [
            {"address": address, "container": labels.get(address), "packet_count": count}
            for address, count in sorted(source_counts.items())
        ],
        "destination_addresses": [
            {"address": address, "container": labels.get(address), "packet_count": count}
            for address, count in sorted(destination_counts.items())
        ],
        "duration_seconds": round(max(timestamps) - min(timestamps), 6) if timestamps else None,
        "flows": flows,
    }
    if capture_mode == "baseline_collector_generated_normal_requests":
        summary["normal_application_flows"] = application_flows
        summary["normal_requests"] = load_request_results(request_path)
    else:
        summary["observed_application_flows"] = application_flows
        summary["collector_generated_requests"] = []
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summarized {packet_count} observed packet(s) into {summary_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())