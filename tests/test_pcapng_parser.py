from pathlib import Path

from evidence.parsers.pcapng import PcapngParser


INCIDENT_PCAP = Path("incident_001/network/attack.pcap")


def test_parse_pcapng():
    parser = PcapngParser()

    events = parser.parse(INCIDENT_PCAP)

    assert events
    assert len(events) == 24


def test_pcapng_contains_tcp_observations():
    parser = PcapngParser()

    events = parser.parse(INCIDENT_PCAP)

    tcp_events = [
        event
        for event in events
        if event.attributes["protocol"] == "TCP"
    ]

    assert tcp_events

    for event in tcp_events:
        assert event.attributes["source_ip"]
        assert event.attributes["destination_ip"]
        assert event.attributes["source_port"] is not None
        assert event.attributes["destination_port"] is not None


def test_pcapng_first_packet():
    parser = PcapngParser()

    events = parser.parse(INCIDENT_PCAP)

    first = events[0]

    assert first.source == "pcapng"
    assert first.event_type == "network_packet"
    assert first.attributes["protocol"] == "TCP"
    assert first.attributes["source_ip"] == "127.0.0.1"
    assert first.attributes["destination_ip"] == "127.0.0.1"
    assert first.attributes["source_port"] == 57828
    assert first.attributes["destination_port"] == 8080


def test_pcapng_common_schema():
    parser = PcapngParser()

    events = parser.parse(INCIDENT_PCAP)

    for event in events:
        assert event.event_id
        assert event.timestamp
        assert event.source
        assert event.event_type
        assert event.source_entity
        assert event.destination_entity
        assert event.action
        assert isinstance(event.attributes, dict)