import pytest
from pydantic import ValidationError

from evaluation.gnn_dataset import (
    CandidateEdgeLabel,
    ScenarioGraphRecord,
    ScenarioGroundTruth,
    ScenarioSplit,
    validate_scenario_splits,
    EdgeLabelAssessment,
    positive_labels_from_ground_truth,
    PreparedEdgeExample,
    prepare_edge_examples
)


def test_graph_record_rejects_ground_truth_field():
    with pytest.raises(ValidationError):
        ScenarioGraphRecord.model_validate(
            {
                "scenario_id": "scenario_001",
                "graph": {"nodes": [], "edges": [], "attack_path": []},
                "ground_truth": {"expected_attack_path": ["a", "b"]},
            }
        )


def test_ground_truth_is_stored_separately_from_graph():
    record = ScenarioGraphRecord.model_validate(
        {
            "scenario_id": "scenario_001",
            "graph": {"nodes": [], "edges": [], "attack_path": []},
        }
    )
    truth = ScenarioGroundTruth.model_validate(
        {
            "scenario_id": "scenario_001",
            "expected_attack_path": ["a", "b"],
            "positive_edges": [
                {"source": "a", "target": "b", "edge_type": "accesses", "label": 1}
            ],
        }
    )

    assert record.scenario_id == truth.scenario_id
    assert not hasattr(record, "expected_attack_path")
    assert truth.expected_attack_path == ["a", "b"]


def test_candidate_edge_rejects_non_binary_label():
    with pytest.raises(ValidationError):
        CandidateEdgeLabel(
            source="a",
            target="b",
            edge_type="accesses",
            label=2,
        )


def test_scenario_split_accepts_known_split_names():
    for split_name in ("train", "validation", "test"):
        split = ScenarioSplit(
            scenario_id="scenario_001",
            split=split_name,
        )
        assert split.split == split_name


def test_scenario_split_rejects_unknown_split_name():
    with pytest.raises(ValidationError):
        ScenarioSplit(
            scenario_id="scenario_001",
            split="training",
        )

def test_scenario_splits_return_mapping():
    assignments = [
        ScenarioSplit(scenario_id="scenario_001", split="train"),
        ScenarioSplit(scenario_id="scenario_002", split="validation"),
        ScenarioSplit(scenario_id="scenario_003", split="test"),
    ]

    assert validate_scenario_splits(assignments) == {
        "scenario_001": "train",
        "scenario_002": "validation",
        "scenario_003": "test",
    }


def test_scenario_cannot_be_assigned_to_multiple_splits():
    assignments = [
        ScenarioSplit(scenario_id="scenario_001", split="train"),
        ScenarioSplit(scenario_id="scenario_001", split="test"),
    ]

    with pytest.raises(ValueError, match="assigned more than once"):
        validate_scenario_splits(assignments)

def test_positive_assessment_converts_to_binary_label():
    assessment = EdgeLabelAssessment(
        source="service:nginx",
        target="service:api",
        edge_type="forwards",
        status="positive",
        justification="Controlled scenario ground truth confirms this transition.",
    )

    label = assessment.to_binary_label()

    assert label is not None
    assert label.label == 1


def test_confirmed_negative_assessment_converts_to_zero():
    assessment = EdgeLabelAssessment(
        source="service:nginx",
        target="service:postgres",
        edge_type="connects_to",
        status="negative",
        justification="Controlled scenario confirms this transition did not occur.",
    )

    label = assessment.to_binary_label()

    assert label is not None
    assert label.label == 0


def test_unknown_assessment_is_excluded_from_binary_labels():
    assessment = EdgeLabelAssessment(
        source="service:api",
        target="service:postgres",
        edge_type="communicates_with",
        status="unknown",
        justification="No conclusive observation is available.",
    )

    assert assessment.to_binary_label() is None


def test_label_assessment_rejects_missing_justification():
    with pytest.raises(ValidationError):
        EdgeLabelAssessment(
            source="service:api",
            target="service:postgres",
            edge_type="connects_to",
            status="negative",
            justification="",
        )

def test_expected_path_does_not_imply_unrelated_edges_are_negative():
    # Knowing the expected path alone cannot establish that another
    # candidate transition definitely did not occur.
    assessment = EdgeLabelAssessment(
        source="service:nginx",
        target="service:postgres",
        edge_type="connects_to",
        status="unknown",
        justification=(
            "The expected attack path does not establish whether this "
            "alternative transition occurred."
        ),
    )

    assert assessment.to_binary_label() is None

def test_positive_labels_use_existing_graph_edges_only():
    graph_record = ScenarioGraphRecord.model_validate(
        {
            "scenario_id": "scenario_001",
            "graph": {
                "nodes": [
                    {
                        "node_id": "service:nginx",
                        "node_type": "service",
                        "name": "nginx",
                    },
                    {
                        "node_id": "service:api",
                        "node_type": "service",
                        "name": "api",
                    },
                    {
                        "node_id": "service:postgres",
                        "node_type": "service",
                        "name": "postgres",
                    },
                ],
                "edges": [
                    {
                        "source": "service:nginx",
                        "target": "service:api",
                        "edge_type": "forwards",
                        "basis": "observed",
                    },
                    {
                        "source": "service:api",
                        "target": "service:postgres",
                        "edge_type": "depends_on",
                        "basis": "configured",
                    },
                ],
                "attack_path": [],
            },
        }
    )
    truth = ScenarioGroundTruth(
        scenario_id="scenario_001",
        expected_attack_path=["nginx", "api", "postgres"],
        positive_edges=[
            {
                "source": "service:nginx",
                "target": "service:api",
                "edge_type": "forwards",
                "label": 1,
            }
        ],
    )

    labels = positive_labels_from_ground_truth(graph_record, truth)

    assert [
        (label.source, label.target, label.edge_type)
        for label in labels
    ] == [
        ("service:nginx", "service:api", "forwards"),
    ]


def test_positive_label_extraction_rejects_mismatched_scenario_ids():
    graph_record = ScenarioGraphRecord(
        scenario_id="scenario_001",
        graph={"nodes": [], "edges": [], "attack_path": []},
    )
    truth = ScenarioGroundTruth(
        scenario_id="scenario_002",
        expected_attack_path=["a", "b"],
        positive_edges=[
            {
                "source": "a",
                "target": "b",
                "edge_type": "accesses",
                "label": 1,
            }
        ],
    )

    with pytest.raises(ValueError, match="scenario IDs differ"):
        positive_labels_from_ground_truth(graph_record, truth)

def test_prepare_edge_examples_keeps_scenarios_in_their_assigned_splits():
    graph_record = ScenarioGraphRecord(
        scenario_id="scenario_001",
        graph={
            "nodes": [],
            "edges": [
                {
                    "source": "service:nginx",
                    "target": "service:api",
                    "edge_type": "forwards",
                    "basis": "observed",
                }
            ],
            "attack_path": [],
        },
    )
    truth = ScenarioGroundTruth(
        scenario_id="scenario_001",
        expected_attack_path=["nginx", "api"],
        positive_edges=[
            {
                "source": "service:nginx",
                "target": "service:api",
                "edge_type": "forwards",
                "label": 1,
            }
        ],
    )
    assignments = [
        ScenarioSplit(scenario_id="scenario_001", split="train")
    ]

    examples = prepare_edge_examples([graph_record], [truth], assignments)

    assert len(examples) == 1
    assert examples[0].scenario_id == "scenario_001"
    assert examples[0].split == "train"
    assert examples[0].edge.label == 1


def test_prepare_edge_examples_rejects_labels_absent_from_graph():
    graph_record = ScenarioGraphRecord(
        scenario_id="scenario_001",
        graph={"nodes": [], "edges": [], "attack_path": []},
    )
    truth = ScenarioGroundTruth(
        scenario_id="scenario_001",
        expected_attack_path=["nginx", "api"],
        positive_edges=[
            {
                "source": "service:nginx",
                "target": "service:api",
                "edge_type": "forwards",
                "label": 1,
            }
        ],
    )
    assignments = [
        ScenarioSplit(scenario_id="scenario_001", split="train")
    ]

    with pytest.raises(ValueError, match="absent from graph"):
        prepare_edge_examples([graph_record], [truth], assignments)

def test_prepare_edge_examples_rejects_duplicate_labels():
    graph_record = ScenarioGraphRecord(
        scenario_id="scenario_001",
        graph={
            "nodes": [],
            "edges": [
                {
                    "source": "service:nginx",
                    "target": "service:api",
                    "edge_type": "forwards",
                    "basis": "observed",
                }
            ],
            "attack_path": [],
        },
    )
    label = {
        "source": "service:nginx",
        "target": "service:api",
        "edge_type": "forwards",
        "label": 1,
    }
    truth = ScenarioGroundTruth(
        scenario_id="scenario_001",
        expected_attack_path=["nginx", "api"],
        positive_edges=[label, label],
    )
    assignments = [
        ScenarioSplit(scenario_id="scenario_001", split="train")
    ]

    with pytest.raises(ValueError, match="Duplicate labeled edge"):
        prepare_edge_examples([graph_record], [truth], assignments)
