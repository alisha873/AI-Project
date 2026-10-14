
"""Compare deterministic and GNN-guided attack-path ranking."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import torch

from evidence.schema import AttackGraph
from graph.gnn import AttackGraphGCN, EdgeScorer, EDGE_TYPES, encode_graph
from graph.path_reasoning import CandidatePath, find_candidate_paths
from evaluation.gnn_path_metrics import map_ground_truth_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "scenarios"
MODEL_DIR = ROOT / "evaluation" / "results" / "models"
OUTPUT = ROOT / "evaluation" / "results" / "gnn_path_evaluation.json"


def resolve_label(graph: AttackGraph, label: str) -> str | None:
    """Resolve ground-truth labels to service nodes where possible."""
    ids = {node.node_id for node in graph.nodes}

    if label == "attacker":
        return None

    name = label.removeprefix("autoshield-")
    service_id = f"service:{name}"

    # Prefer service nodes because attack paths describe service transitions.
    if service_id in ids:
        return service_id

    container_id = f"container:{label}"
    if container_id in ids:
        return container_id

    matches = [node.node_id for node in graph.nodes if node.name == label]
    return matches[0] if len(matches) == 1 else None


def load_models() -> tuple[AttackGraphGCN, EdgeScorer]:
    gcn = AttackGraphGCN()
    scorer = EdgeScorer()

    gcn.load_state_dict(torch.load(
        MODEL_DIR / "attack_graph_gcn.pt",
        map_location="cpu",
        weights_only=True,
    ))
    scorer.load_state_dict(torch.load(
        MODEL_DIR / "edge_scorer.pt",
        map_location="cpu",
        weights_only=True,
    ))

    gcn.eval()
    scorer.eval()
    return gcn, scorer


def score_path_with_edge_model(
    graph: AttackGraph,
    path: CandidatePath,
    gcn: AttackGraphGCN,
    scorer: EdgeScorer,
) -> float:
    """Mean predicted edge score; not a calibrated path probability."""
    if not path.hops:
        return 0.0

    tensors = encode_graph(graph)
    node_index = {node_id: i for i, node_id in enumerate(tensors.node_ids)}

    with torch.no_grad():
        embeddings = gcn(
            tensors.node_features,
            tensors.edge_index,
            tensors.edge_type_ids,
            tensors.edge_basis_ids,
        )

        source_ids = []
        target_ids = []
        relation_ids = []

        for hop in path.hops:
            if hop.edge_type not in EDGE_TYPES:
                raise ValueError(f"Unknown edge type: {hop.edge_type}")

            source_ids.append(node_index[hop.source])
            target_ids.append(node_index[hop.target])
            relation_ids.append(EDGE_TYPES.index(hop.edge_type))

        logits = scorer(
            embeddings[torch.tensor(source_ids, dtype=torch.long)],
            embeddings[torch.tensor(target_ids, dtype=torch.long)],
            torch.tensor(relation_ids, dtype=torch.long),
        )
        return float(torch.sigmoid(logits).mean().item())


def path_metrics(
    predicted: list[str],
    expected: list[str],
    graph: AttackGraph,
) -> dict:
    predicted_nodes = set(predicted)
    expected_nodes = set(expected)

    common_nodes = predicted_nodes & expected_nodes
    precision = len(common_nodes) / len(predicted_nodes) if predicted_nodes else 0.0
    recall = len(common_nodes) / len(expected_nodes) if expected_nodes else 0.0

    predicted_transitions = set(zip(predicted, predicted[1:]))
    expected_transitions = set(zip(expected, expected[1:]))
    common_transitions = predicted_transitions & expected_transitions

    transition_recall = (
        len(common_transitions) / len(expected_transitions)
        if expected_transitions else 1.0
    )
    transition_precision = (
        len(common_transitions) / len(predicted_transitions)
        if predicted_transitions else 0.0
    )

    graph_edges = {(edge.source, edge.target) for edge in graph.edges}
    graph_valid = all(
        (source, target) in graph_edges
        for source, target in zip(predicted, predicted[1:])
    )

    return {
        "node_precision": precision,
        "node_recall": recall,
        "transition_precision": transition_precision,
        "transition_recall": transition_recall,
        "exact_path_match": predicted == expected,
        "graph_valid": graph_valid,
    }


def main() -> None:
    assignments = json.loads((DATA / "gnn_splits.json").read_text())
    gcn, scorer = load_models()

    results = []

    for assignment in assignments:
        sid = assignment["scenario_id"]
        split = assignment["split"]

        graph = AttackGraph.model_validate(json.loads(
            (DATA / f"{sid}_graph.json").read_text()
        ))
        truth = json.loads(
            (DATA / f"{sid}_ground_truth.json").read_text()
        )

        expected_labels = truth.get("expected_attack_path", [])
        if not expected_labels:
            results.append({
                "scenario_id": sid,
                "split": split,
                "status": "no_expected_attack_path",
                "note": "Benign or path-unspecified scenario; excluded from path metrics.",
            })
            continue

        # Ground truth is used here only after prediction, for evaluation.
        mapped = map_ground_truth_path(graph, expected_labels)
        expected = [
            resolve_label(graph, label)
            for label in expected_labels
            if label != "attacker"
        ]
        expected = [node_id for node_id in expected if node_id is not None]

        if len(expected) < 2:
            results.append({
                "scenario_id": sid,
                "split": split,
                "status": "insufficient_mappable_ground_truth",
                "unmapped_labels": mapped.unmapped_labels,
            })
            continue

        # Use the known external-ingress service as the search source.
        # Do not use the ground-truth final node to choose a prediction target.
        source = resolve_label(graph, "autoshield-nginx")
        if source is None:
            results.append({
                "scenario_id": sid,
                "split": split,
                "status": "ingress_service_unmapped",
            })
            continue

        service_targets = sorted({
            node.node_id for node in graph.nodes
            if node.node_type == "service" and node.node_id != source
        })

        candidates = []
        for target in service_targets:
            candidates.extend(find_candidate_paths(
                graph, source, target, max_hops=6, max_paths=100
            ))

        if not candidates:
            results.append({
                "scenario_id": sid,
                "split": split,
                "status": "no_candidate_paths",
                "unmapped_labels": mapped.unmapped_labels,
            })
            continue

        # Deterministic baseline uses the existing evidence-aware path score.
        deterministic = max(
            candidates,
            key=lambda p: (p.score, -len(p.hops), p.nodes),
        )

        # GNN ranking uses the trained edge scorer across each path's hops.
        gnn_ranked = [
            (score_path_with_edge_model(graph, path, gcn, scorer), path)
            for path in candidates
        ]
        gnn_score, gnn_path = max(
            gnn_ranked,
            key=lambda item: (item[0], -len(item[1].hops), item[1].nodes),
        )

        results.append({
            "scenario_id": sid,
            "split": split,
            "status": "evaluated",
            "expected_path": expected,
            "unmapped_ground_truth_labels": mapped.unmapped_labels,
            "mapping_coverage": (
                len(expected) / max(
                    1, sum(label != "attacker" for label in expected_labels)
                )
            ),
            "deterministic": {
                "predicted_path": list(deterministic.nodes),
                "ranking_score": deterministic.score,
                **path_metrics(list(deterministic.nodes), expected, graph),
            },
            "gnn_guided": {
                "predicted_path": list(gnn_path.nodes),
                "mean_edge_score": gnn_score,
                **path_metrics(list(gnn_path.nodes), expected, graph),
            },
        })

    # Aggregate only evaluated attack-path scenarios.
    evaluated = [item for item in results if item["status"] == "evaluated"]
    summary = {}

    for method in ("deterministic", "gnn_guided"):
        summary[method] = {}
        for metric in (
            "node_precision",
            "node_recall",
            "transition_precision",
            "transition_recall",
            "exact_path_match",
            "graph_valid",
        ):
            values = [item[method][metric] for item in evaluated]
            summary[method][metric] = mean(values) if values else None

    report = {
        "experiment": "gnn_guided_attack_path_ranking",
        "model_weights": {
            "gcn": str(MODEL_DIR / "attack_graph_gcn.pt"),
            "edge_scorer": str(MODEL_DIR / "edge_scorer.pt"),
        },
        "evaluated_attack_scenarios": len(evaluated),
        "total_scenarios": len(assignments),
        "aggregate_metrics": summary,
        "results": results,
        "limitations": [
            "This is a preliminary path-ranking experiment, not proof of attack execution.",
            "The ingress service is assumed to be service:nginx when resolvable.",
            "The mean edge score is not a calibrated path probability.",
            "The current edge scorer may not generalize across scenarios with near-identical graph structure.",
            "Ground-truth mapping and missing entities affect measured coverage.",
            "Test results should be interpreted cautiously with a small dataset.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(OUTPUT),
        "evaluated_attack_scenarios": len(evaluated),
        "aggregate_metrics": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
