
"""Dataset records for leakage-aware attack-graph evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidence.schema import AttackGraph
from evaluation.gnn_path_metrics import map_ground_truth_path


class CandidateEdgeLabel(BaseModel):
    """Binary evaluation label for one candidate typed transition."""

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
        """Convert known assessments to binary labels; omit unknowns."""
        if self.status == "unknown":
            return None

        return CandidateEdgeLabel(
            source=self.source,
            target=self.target,
            edge_type=self.edge_type,
            label=1 if self.status == "positive" else 0,
        )


class ScenarioGraphRecord(BaseModel):
    """Graph input, kept separate from its evaluation labels."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    graph: AttackGraph


class ScenarioGroundTruth(BaseModel):
    """Evaluation-only ground truth; never pass this object to the GNN."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    expected_attack_path: list[str] = Field(default_factory=list)
    attack_path_complete: bool
    edge_assessments: list[EdgeLabelAssessment] = Field(min_length=1)


class ScenarioSplit(BaseModel):
    """Assign each complete scenario to exactly one dataset split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    split: Literal["train", "validation", "test"]


class PreparedEdgeExample(BaseModel):
    """A known edge label assigned to a scenario-level dataset split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(pattern=r"^scenario_[0-9]{3}$")
    split: Literal["train", "validation", "test"]
    edge: CandidateEdgeLabel


def validate_scenario_splits(
    assignments: list[ScenarioSplit],
) -> dict[str, str]:
    """Return scenario-to-split mapping, rejecting duplicate assignments."""
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
    """Extract observed graph edges matching adjacent mapped path nodes.

    This helper does not infer negative labels. Even a complete ground-truth
    path does not make configured-only edges observed attack transitions.
    """
    if record.scenario_id != truth.scenario_id:
        raise ValueError("Graph record and ground truth scenario IDs differ")

    mapping = map_ground_truth_path(record.graph, truth.expected_attack_path)
    mapped_positions = mapping.mapped_positions
    labels: list[CandidateEdgeLabel] = []

    for index in range(len(mapped_positions) - 1):
        source = mapped_positions[index]
        target = mapped_positions[index + 1]

        if source is None or target is None:
            continue

        for edge in record.graph.edges:
            if edge.source != source or edge.target != target:
                continue
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
    """Prepare explicit known labels without leaking labels into graph input."""
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

    for scenario_id, graph_record in graphs_by_scenario.items():
        truth = truth_by_scenario[scenario_id]
        graph_edges = {
            (edge.source, edge.target, edge.edge_type)
            for edge in graph_record.graph.edges
        }
        seen_statuses: dict[tuple[str, str, str], str] = {}

        for assessment in truth.edge_assessments:
            edge_key = (
                assessment.source,
                assessment.target,
                assessment.edge_type,
            )

            if edge_key not in graph_edges:
                raise ValueError(
                    f"Labeled edge is absent from graph for {scenario_id}: "
                    f"{edge_key}"
                )

            previous_status = seen_statuses.get(edge_key)
            if previous_status is not None:
                if previous_status == assessment.status:
                    raise ValueError(
                        f"Duplicate labeled edge for {scenario_id}: {edge_key}"
                    )
                raise ValueError(
                    f"Conflicting labels for {scenario_id}: {edge_key}"
                )
            seen_statuses[edge_key] = assessment.status

            binary_label = assessment.to_binary_label()
            if binary_label is None:
                continue

            examples.append(
                PreparedEdgeExample(
                    scenario_id=scenario_id,
                    split=split_by_scenario[scenario_id],
                    edge=binary_label,
                )
            )

    return examples
