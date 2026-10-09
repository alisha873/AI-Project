from pathlib import Path
import json
from evidence.schema import IncidentEvidenceContext


from evidence.context import EvidenceContextBuilder


def test_context_builder_excludes_ground_truth(tmp_path: Path):
    incident_dir = tmp_path / "incident_001"
    (incident_dir / "normalized").mkdir(parents=True)
    (incident_dir / "ground_truth").mkdir()

    event = {
        "event_id": "event-001",
        "timestamp": "2026-10-09T10:37:49Z",
        "source": "nginx",
        "event_type": "http_request",
        "source_entity": "client",
        "destination_entity": "autoshield-api",
        "action": "GET",
        "attributes": {"path": "/api/search", "status": 200},
    }

    (incident_dir / "normalized" / "events.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )

    # Create a ground-truth file containing a distinctive secret marker.
    (incident_dir / "ground_truth" / "ground_truth.json").write_text(
        json.dumps({"secret_marker": "DO_NOT_INCLUDE"}),
        encoding="utf-8",
    )

    context = EvidenceContextBuilder().build(incident_dir)

    assert context.incident_id == "incident_001"
    assert "ground_truth/ground_truth.json" not in (
        context.configuration_artifacts
    )
    assert "ground_truth/ground_truth.json" not in (
        context.network_artifacts
    )
    assert "ground_truth/ground_truth.json" not in (
        context.evidence_artifacts
    )

    # Check all string content exposed by the context.
    context_text = context.model_dump_json()
    assert "DO_NOT_INCLUDE" not in context_text
    assert "ground_truth.json" not in context_text

def test_context_defaults():
    context = IncidentEvidenceContext(
        incident_id="incident_001"
    )

    assert context.events == []
    assert context.configuration_artifacts == []
    assert context.network_artifacts == []
    assert context.evidence_artifacts == []


def test_context_accepts_structured_evidence():
    context = IncidentEvidenceContext(
        incident_id="incident_001",
        configuration_artifacts=[
            "docker/containers_after_attack.json",
            "docker/network_after_attack.json",
        ],
        network_artifacts=[
            "network/attack.pcap",
        ],
        evidence_artifacts=[
            "logs/nginx.log",
            "logs/api.log",
        ],
    )

    assert len(context.configuration_artifacts) == 2
    assert len(context.network_artifacts) == 1
    assert len(context.evidence_artifacts) == 2