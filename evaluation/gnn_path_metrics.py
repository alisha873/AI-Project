"""Evaluation-only mapping and coverage metrics for expected attack paths."""

from __future__ import annotations

from dataclasses import dataclass

from evidence.schema import AttackGraph


@dataclass
class PathMapping:
    mapped_path: list[str]
    unmapped_labels: list[str]
    # One entry per expected-path position; None means unmapped.
    mapped_positions: list[str | None]


@dataclass
class PathCoverage:
    node_coverage: float
    transition_coverage: float
    mapped_nodes: int
    expected_nodes: int
    represented_transitions: int
    expected_transitions: int


def map_ground_truth_path(
    graph: AttackGraph,
    expected_path: list[str],
) -> PathMapping:
    """Map ground-truth labels to graph IDs without altering the graph."""

    node_ids = {node.node_id for node in graph.nodes}
    nodes_by_name: dict[str, list[str]] = {}

    for node in graph.nodes:
        nodes_by_name.setdefault(node.name, []).append(node.node_id)

    mapped_path: list[str] = []
    unmapped_labels: list[str] = []
    mapped_positions: list[str | None] = []

    for label in expected_path:
        node_id: str | None = None

        # The external attacker is not currently represented as a graph node.
        if label != "attacker":
            container_id = f"container:{label}"

            if container_id in node_ids:
                node_id = container_id
            else:
                candidates = nodes_by_name.get(label, [])
                if len(candidates) == 1:
                    node_id = candidates[0]

        mapped_positions.append(node_id)

        if node_id is None:
            unmapped_labels.append(label)
        else:
            mapped_path.append(node_id)

    return PathMapping(
        mapped_path=mapped_path,
        unmapped_labels=unmapped_labels,
        mapped_positions=mapped_positions,
    )


def measure_path_coverage(
    graph: AttackGraph,
    expected_path: list[str],
    mapping: PathMapping,
) -> PathCoverage:
    """Measure direct-edge coverage; never modifies the graph."""

    if len(mapping.mapped_positions) != len(expected_path):
        raise ValueError(
            "mapped_positions must contain one entry per expected-path label"
        )

    expected_nodes = len(expected_path)
    mapped_nodes = sum(
        node_id is not None for node_id in mapping.mapped_positions
    )

    edge_pairs = {
        (edge.source, edge.target)
        for edge in graph.edges
    }

    expected_transitions = max(expected_nodes - 1, 0)
    represented_transitions = 0

    for index in range(len(expected_path) - 1):
        source = mapping.mapped_positions[index]
        target = mapping.mapped_positions[index + 1]

        # Count only transitions between adjacent, successfully mapped labels.
        if source is not None and target is not None:
            if (source, target) in edge_pairs:
                represented_transitions += 1

    return PathCoverage(
        node_coverage=(
            mapped_nodes / expected_nodes if expected_nodes else 1.0
        ),
        transition_coverage=(
            represented_transitions / expected_transitions
            if expected_transitions
            else 1.0
        ),
        mapped_nodes=mapped_nodes,
        expected_nodes=expected_nodes,
        represented_transitions=represented_transitions,
        expected_transitions=expected_transitions,
    )
