
"""Evidence-aware candidate path search for AutoShield attack graphs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from evidence.schema import AttackGraph
import torch

from graph.gnn import AttackGraphGCN, encode_graph

@dataclass(frozen=True)
class PathHop:
    source: str
    target: str
    edge_type: str
    basis: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePath:
    nodes: tuple[str, ...]
    hops: tuple[PathHop, ...]
    score: float


# These weights rank candidate paths; they are not calibrated probabilities.
EDGE_WEIGHTS = {
    "accesses": 1.0,
    "forwards": 1.0,
    "communicates_with": 0.9,
    "authenticates_to": 0.9,
    "connects_to": 0.5,
    "depends_on": 0.6,
    "hosts": 0.3,
    "executes": 0.2,
}

OBSERVED_EDGE_FACTOR = 1.0
CONFIGURED_EDGE_FACTOR = 0.25


def find_candidate_paths(
    graph: AttackGraph,
    source: str,
    target: str,
    *,
    max_hops: int = 6,
    max_paths: int = 10,
    observed_only: bool = False,
) -> list[CandidatePath]:
    """Find simple directed paths using graph edges only."""

    if max_hops < 1 or max_paths < 1:
        raise ValueError("max_hops and max_paths must be positive")

    node_ids = {node.node_id for node in graph.nodes}
    if source not in node_ids or target not in node_ids:
        return []
    if source == target:
        return [CandidatePath(nodes=(source,), hops=(), score=1.0)]

    adjacency: dict[str, list] = {}
    for edge in graph.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            continue
        if observed_only and edge.basis != "observed":
            continue
        adjacency.setdefault(edge.source, []).append(edge)

    for edges in adjacency.values():
        edges.sort(key=lambda edge: (edge.target, edge.edge_type))

    queue = deque([(source, (source,), ())])
    results: list[CandidatePath] = []

    while queue and len(results) < max_paths:
        current, path_nodes, path_hops = queue.popleft()

        if len(path_hops) >= max_hops:
            continue

        for edge in adjacency.get(current, []):
            if edge.target in path_nodes:
                continue

            hop = PathHop(
                source=edge.source,
                target=edge.target,
                edge_type=edge.edge_type,
                basis=edge.basis,
                evidence_ids=tuple(sorted(set(edge.evidence_ids))),
            )
            next_nodes = path_nodes + (edge.target,)
            next_hops = path_hops + (hop,)

            if edge.target == target:
                # Mean edge weight rewards paths with stronger relationships
                # without making longer paths automatically score higher.
                score = sum(
                    EDGE_WEIGHTS.get(item.edge_type, 0.1)
                    * (
                        OBSERVED_EDGE_FACTOR
                        if item.basis == "observed"
                        else CONFIGURED_EDGE_FACTOR
                    )
                    for item in next_hops
                ) / len(next_hops)
                results.append(
                    CandidatePath(
                        nodes=next_nodes,
                        hops=next_hops,
                        score=score,
                    )
                )
            else:
                queue.append((edge.target, next_nodes, next_hops))

    return sorted(
        results,
        key=lambda path: (-path.score, len(path.hops), path.nodes),
    )


def score_candidate_path_with_gcn(
    graph: AttackGraph,
    path: CandidatePath,
    *,
    model: AttackGraphGCN | None = None,
) -> float:
    """Return an experimental GCN embedding-coherence score.

    This is not a probability and is not a trained attack-path prediction.
    """

    if not path.nodes:
        return 0.0

    tensors = encode_graph(graph)
    node_index = {
        node_id: index
        for index, node_id in enumerate(tensors.node_ids)
    }

    if any(node_id not in node_index for node_id in path.nodes):
        raise ValueError("Candidate path contains nodes absent from the graph")

    if len(path.nodes) == 1:
        return 1.0

    model = model or AttackGraphGCN()
    model.eval()

    with torch.no_grad():
        embeddings = model(
            tensors.node_features,
            tensors.edge_index,
            tensors.edge_type_ids,
            tensors.edge_basis_ids,
        )

    path_embeddings = torch.stack(
        [embeddings[node_index[node_id]] for node_id in path.nodes]
    )

    source_embeddings = path_embeddings[:-1]
    target_embeddings = path_embeddings[1:]

    similarities = torch.nn.functional.cosine_similarity(
        source_embeddings,
        target_embeddings,
        dim=1,
        eps=1e-8,
    )

    return float(similarities.mean().clamp(-1.0, 1.0).item())
