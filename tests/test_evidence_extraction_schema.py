import pytest
from pydantic import ValidationError

from evidence.schema import EvidenceExtraction


def test_evidence_extraction_defaults():
    result = EvidenceExtraction(
        incident_id="incident_001"
    )

    assert result.relevant_events == []
    assert result.relevant_configuration == []
    assert result.evidence_summary == ""
    assert result.confidence == 0.0
    assert result.relevant_network_events == []
    assert result.relevant_network_artifacts == []


def test_evidence_extraction_accepts_structured_output():
    result = EvidenceExtraction(
        incident_id="incident_001",
        relevant_events=[
            "nginx-event-1",
            "application-event-2",
        ],
        relevant_configuration=[
            "docker/containers_after_attack.json",
        ],
        relevant_network_events=[
        "pcapng-packet-100",
        ],
        relevant_network_artifacts=[
            "network/attack.pcap",
        ],
        evidence_summary="Relevant HTTP and network activity was identified.",
        confidence=0.92,
    )

    assert result.incident_id == "incident_001"
    assert len(result.relevant_events) == 2
    assert len(result.relevant_configuration) == 1
    assert len(result.relevant_network_events) == 1
    assert len(result.relevant_network_artifacts) == 1
    assert result.confidence == 0.92


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_evidence_extraction_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValidationError):
        EvidenceExtraction(
            incident_id="incident_001",
            confidence=confidence,
        )


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_evidence_extraction_accepts_confidence_boundaries(confidence):
    result = EvidenceExtraction(
        incident_id="incident_001",
        confidence=confidence,
    )

    assert result.confidence == confidence


def test_evidence_extraction_rejects_unexpected_fields():
    with pytest.raises(ValidationError, match="root_cause"):
        EvidenceExtraction(
            incident_id="incident_001",
            root_cause="unexpected diagnostic field",
        )