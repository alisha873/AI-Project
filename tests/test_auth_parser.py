from pathlib import Path

from evidence.parsers.auth import AuthParser


BASELINE_AUTH_LOG = Path("baseline/logs/linux_admin/auth.log")


def test_parse_auth_log():
    parser = AuthParser()

    events = parser.parse(BASELINE_AUTH_LOG)

    assert len(events) == 2
    assert all(event.source == "linux_auth" for event in events)
    assert all(
        event.event_type == "authentication_failure"
        for event in events
    )


def test_parse_authentication_failure():
    parser = AuthParser()

    events = parser.parse(BASELINE_AUTH_LOG)
    event = events[0]

    assert event.source_entity == "172.19.0.6"
    assert event.destination_entity is None
    assert event.action == "authentication_failed"

    assert event.attributes["user"] == "analyst"
    assert event.attributes["rhost"] == "172.19.0.6"
    assert event.attributes["tty"] == "ssh"


def test_auth_parser_preserves_duplicate_observations():
    parser = AuthParser()

    events = parser.parse(BASELINE_AUTH_LOG)

    assert events[0].event_id != events[1].event_id
    assert events[0].timestamp == events[1].timestamp
    assert events[0].source_entity == events[1].source_entity


def test_auth_parser_uses_common_schema():
    parser = AuthParser()

    events = parser.parse(BASELINE_AUTH_LOG)

    for event in events:
        assert event.event_id
        assert event.timestamp
        assert event.source
        assert event.event_type
        assert event.source_entity
        assert event.action
        assert isinstance(event.attributes, dict)