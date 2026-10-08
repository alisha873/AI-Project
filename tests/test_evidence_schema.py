from datetime import datetime, timezone

from evidence.schema import EvidenceEvent


def test_evidence_event_creation():
    event = EvidenceEvent(
        event_id="EVT-001",
        timestamp=datetime.now(timezone.utc),
        source="nginx",
        event_type="http_request",
        source_entity="external",
        destination_entity="webapp",
        action="GET",
        attributes={
            "path": "/",
            "status": 200,
        },
    )

    assert event.event_id == "EVT-001"
    assert event.source == "nginx"
    assert event.attributes["status"] == 200