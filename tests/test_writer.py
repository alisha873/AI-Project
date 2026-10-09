import json
from pathlib import Path

from evidence.pipeline import EvidencePipeline
from evidence.writer import EvidenceEventWriter


INCIDENT_DIR = Path("incident_001")


def test_writer_creates_jsonl(tmp_path):
    pipeline = EvidencePipeline()
    events = pipeline.process(INCIDENT_DIR)

    output_path = tmp_path / "normalized" / "events.jsonl"

    writer = EvidenceEventWriter()
    writer.write(events, output_path)

    assert output_path.exists()

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == len(events)


def test_writer_produces_valid_json_per_line(tmp_path):
    pipeline = EvidencePipeline()
    events = pipeline.process(INCIDENT_DIR)

    output_path = tmp_path / "events.jsonl"

    EvidenceEventWriter().write(
        events,
        output_path,
    )

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert lines

    for line in lines:
        payload = json.loads(line)

        assert payload["event_id"]
        assert payload["timestamp"]
        assert payload["source"]
        assert payload["event_type"]
        assert payload["action"]
        assert isinstance(payload["attributes"], dict)


def test_writer_preserves_event_order(tmp_path):
    pipeline = EvidencePipeline()
    events = pipeline.process(INCIDENT_DIR)

    output_path = tmp_path / "events.jsonl"

    EvidenceEventWriter().write(
        events,
        output_path,
    )

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    payloads = [
        json.loads(line)
        for line in lines
    ]

    assert [
        payload["event_id"]
        for payload in payloads
    ] == [
        event.event_id
        for event in events
    ]