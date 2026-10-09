from pathlib import Path

import pytest

from evidence.parsers.base import EvidenceParser
from evidence.schema import EvidenceEvent


class DummyParser(EvidenceParser):
    source = "test"

    def parse(self, path: Path) -> list[EvidenceEvent]:
        return [
            EvidenceEvent(
                event_id="TEST-001",
                timestamp="2026-10-09T00:00:00Z",
                source=self.source,
                event_type="test_event",
                source_entity="source",
                destination_entity="destination",
                action="test",
            )
        ]


def test_parser_contract_returns_evidence_events():
    parser = DummyParser()

    events = parser.parse(Path("dummy.log"))

    assert isinstance(events, list)
    assert len(events) == 1
    assert isinstance(events[0], EvidenceEvent)
    assert events[0].source == "test"


def test_parser_contract_requires_parse_implementation():
    with pytest.raises(TypeError):
        EvidenceParser()