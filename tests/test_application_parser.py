from pathlib import Path

from evidence.parsers.application import ApplicationParser


INCIDENT_API_LOG = Path("incident_001/logs/api.log")


def test_parse_application_log():
    parser = ApplicationParser()

    events = parser.parse(INCIDENT_API_LOG)

    assert len(events) == 4

    assert all(event.source == "application" for event in events)
    assert all(event.event_type == "http_request" for event in events)


def test_parse_sqli_request_as_http_observation():
    parser = ApplicationParser()

    events = parser.parse(INCIDENT_API_LOG)

    event = next(
        event
        for event in events
        if "OR" in event.attributes["query"]
    )

    assert event.source_entity == "172.19.0.7"
    assert event.action == "GET /search"
    assert event.attributes["method"] == "GET"
    assert event.attributes["path"] == "/search"
    assert event.attributes["status"] == 200
    assert event.attributes["status_text"] == "OK"


def test_application_parser_keeps_query_separate():
    parser = ApplicationParser()

    events = parser.parse(INCIDENT_API_LOG)

    event = next(
        event
        for event in events
        if event.attributes["path"] == "/search"
        and "security" in event.attributes["query"]
    )

    assert event.action == "GET /search"
    assert event.attributes["query"] == "message=security"


def test_application_parser_uses_common_schema():
    parser = ApplicationParser()

    events = parser.parse(INCIDENT_API_LOG)

    for event in events:
        assert event.event_id
        assert event.timestamp
        assert event.source
        assert event.event_type
        assert event.source_entity
        assert event.action
        assert isinstance(event.attributes, dict)