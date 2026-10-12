
import json
from pathlib import Path

from graph.builder import DeterministicGraphBuilder
from evaluation.gnn_path_metrics import map_ground_truth_path


INCIDENT_DIR = Path("incident_001")


def test_ground_truth_path_mapping_reports_missing_nodes():
    graph = DeterministicGraphBuilder().build(INCIDENT_DIR)
    ground_truth = json.loads(
        (INCIDENT_DIR / "ground_truth" / "ground_truth.json").read_text()
    )

    result = map_ground_truth_path(
        graph,
        ground_truth["expected_attack_path"],
    )

    assert result.mapped_path == [
        "container:autoshield-nginx",
        "container:autoshield-api",
        "container:autoshield-postgres",
    ]
    assert result.unmapped_labels == ["attacker"]

