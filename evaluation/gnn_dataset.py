"""Dataset records for leakage-aware attack-graph evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidence.schema import AttackGraph

from evaluation.gnn_path_metrics import map_ground_truth_path

class CandidateEdgeLabel(BaseModel):
    """Evaluation label for one candidate typed transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    edge_type: str = Field(min_length=1)
    label: Literal[0, 1]

class EdgeLabelAssessment(BaseModel):
    """Evaluation-only assessment of a candidate transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    edge_type: str = Field(min_length=1)
    status: Literal["positive", "negative", "unknown"]
    justification: str = Field(min_length=1)

    def to_binary_label(self) -> CandidateEdgeLabel | None:
        """Return a training label only when the transition is known."""

        if self.status == "unknown":
            return None

        return CandidateEdgeLabel(
            source=self.source,
            target=self.target,
            edge_type=self.edge_type,
            label=1 if self.status == "positive" else 0,
        )


class ScenarioGraphRecord(BaseModel):
    """Observed graph input, identified independently of its labels."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    graph: AttackGraph


class ScenarioGroundTruth(BaseModel):
    """Evaluation-only labels; never pass this object to encode_graph."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    expected_attack_path: list[str] = Field(min_length=2)
    positive_edges: list[CandidateEdgeLabel] = Field(min_length=1)


class ScenarioSplit(BaseModel):
    """Assign each complete scenario to exactly one dataset split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    split: Literal["train", "validation", "test"]

class PreparedEdgeExample(BaseModel):
    """A labeled edge assigned to a scenario-level dataset split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    split: Literal["train", "validation", "test"]
    edge: CandidateEdgeLabel

def validate_scenario_splits(
    assignments: list[ScenarioSplit],
) -> dict[str, str]:
    """Return scenario-to-split mapping, rejecting duplicates."""

    mapping: dict[str, str] = {}

    for assignment in assignments:
        if assignment.scenario_id in mapping:
            raise ValueError(
                f"Scenario assigned more than once: {assignment.scenario_id}"
            )

        mapping[assignment.scenario_id] = assignment.split

    return mapping

def positive_labels_from_ground_truth(
    record: ScenarioGraphRecord,
    truth: ScenarioGroundTruth,
) -> list[CandidateEdgeLabel]:
    """Extract supported positive transitions without inferring negatives.

    A transition is emitted only when both adjacent ground-truth labels
    map to graph nodes and the graph contains a direct edge between them.
    The graph's edge type is retained; no edge type is invented.
    """

    if record.scenario_id != truth.scenario_id:
        raise ValueError("Graph record and ground truth scenario IDs differ")

    mapping = map_ground_truth_path(
        record.graph,
        truth.expected_attack_path,
    )
    mapped_positions = mapping.mapped_positions

    labels: list[CandidateEdgeLabel] = []

    for index in range(len(mapped_positions) - 1):
        source = mapped_positions[index]
        target = mapped_positions[index + 1]

        if source is None or target is None:
            continue

        matching_edges = [
            edge
            for edge in record.graph.edges
            if edge.source == source and edge.target == target
        ]

        for edge in matching_edges:
            # Configured relationships describe architecture, not proof
            # that the attack traversed that relationship.
            if edge.basis != "observed":
                continue

            labels.append(
                CandidateEdgeLabel(
                    source=edge.source,
                    target=edge.target,
                    edge_type=edge.edge_type,
                    label=1,
                )
            )

    return labels

def prepare_edge_examples(
    graph_records: list[ScenarioGraphRecord],
    ground_truth_records: list[ScenarioGroundTruth],
    assignments: list[ScenarioSplit],
) -> list[PreparedEdgeExample]:
    """Prepare explicitly labeled edges without mixing labels into graph input."""

    split_by_scenario = validate_scenario_splits(assignments)
    graphs_by_scenario = {record.scenario_id: record for record in graph_records}
    truth_by_scenario = {
        truth.scenario_id: truth for truth in ground_truth_records
    }

    if len(graphs_by_scenario) != len(graph_records):
        raise ValueError("Duplicate graph record scenario IDs")

    if len(truth_by_scenario) != len(ground_truth_records):
        raise ValueError("Duplicate ground-truth scenario IDs")

    if set(graphs_by_scenario) != set(truth_by_scenario):
        raise ValueError("Graph and ground-truth scenario sets differ")

    if set(split_by_scenario) != set(graphs_by_scenario):
        raise ValueError("Every graph scenario must have exactly one split")

    examples: list[PreparedEdgeExample] = []
    seen_labels: set[tuple[str, str, str, int]] = set()

    for scenario_id, graph_record in graphs_by_scenario.items():
        truth = truth_by_scenario[scenario_id]
        graph_edges = {
            (edge.source, edge.target, edge.edge_type)
            for edge in graph_record.graph.edges
        }

        for label in truth.positive_edges:
            edge_key = (label.source, label.target, label.edge_type)

            if edge_key not in graph_edges:
                raise ValueError(
                    f"Labeled edge is absent from graph for {scenario_id}: "
                    f"{edge_key}"
                )

            label_key = (
                label.source,
                label.target,
                label.edge_type,
                label.label,
            )
            if label_key in seen_labels:
                raise ValueError(
                    f"Duplicate labeled edge for {scenario_id}: {label_key}"
                )
            seen_labels.add(label_key)

            examples.append(
                PreparedEdgeExample(
                    scenario_id=scenario_id,
                    split=split_by_scenario[scenario_id],
                    edge=label,
                )
            )

    return examples