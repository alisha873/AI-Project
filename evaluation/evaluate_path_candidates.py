#!/usr/bin/env python3
"""Evidence-aware path ranking for scenario_001.

Run from repository root:
    PYTHONPATH=. python evaluation/evaluate_path_candidates.py

Important: configured dependencies are impact hypotheses, not observed attack transitions.
The report therefore separates fully observed paths from paths containing configured edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation import complete_path_reconstruction as base
from evidence.schema import AttackGraph

ROOT = Path(__file__).resolve().parents[1]
GRAPH_FILE = ROOT / "evaluation/scenarios/scenario_001_graph.json"
GT_FILE = ROOT / "evaluation/scenarios/scenario_001_ground_truth.json"
MODEL_FILE = ROOT / "evaluation/results/models/attack_graph_gcn.pt"
OUT_FILE = ROOT / "evaluation/results/path_candidate_ranking.json"
CANDIDATE_TARGETS = ("service:juice-shop", "service:postgres", "service:redis")


def load_graph(path: Path) -> AttackGraph:
    doc = json.loads(path.read_text())
    return AttackGraph.model_validate(doc.get("graph", doc))


def is_observed(edge) -> bool:
    basis = str(base.field(edge, "basis", "unknown")).lower()
    return basis == "observed" or basis.endswith(".observed")


def deterministic_score(edges) -> float:
    if not edges:
        return 0.0
    values = []
    for edge in edges:
        relation = str(base.field(edge, "edge_type", ""))
        provenance_factor = 1.0 if is_observed(edge) else 0.25
        values.append(base.EDGE_WEIGHTS.get(relation, 0.1) * provenance_factor)
    return sum(values) / len(values)


def main() -> None:
    for path in (GRAPH_FILE, GT_FILE, MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.relative_to(ROOT)}")

    graph = load_graph(GRAPH_FILE)
    nodes = list(graph.nodes.values()) if isinstance(graph.nodes, dict) else list(graph.nodes)
    node_ids = {base.node_id(node) for node in nodes}
    node_ids.discard(None)

    gt_doc = json.loads(GT_FILE.read_text())
    gt = gt_doc.get("ground_truth", gt_doc)
    expected = base.normalize_gt_path(gt.get("expected_attack_path", []), nodes)
    if len(expected) < 2:
        raise ValueError(f"Could not map ground-truth path to graph nodes: {expected}")
    source, true_target = expected[0], expected[-1]

    gcn = base.build_gcn(base.checkpoint_state(MODEL_FILE))
    x, edge_index, edge_types, edge_bases = base.encoded_tensors(graph)
    with torch.no_grad():
        embeddings = gcn(x, edge_index, edge_types, edge_bases)

    ordered_node_ids = sorted(
        base.node_id(node) for node in nodes if base.node_id(node) is not None
    )
    if len(ordered_node_ids) != embeddings.shape[0]:
        raise ValueError("Node ID ordering/count mismatch; refusing to score paths.")
    node_index = {node_id: index for index, node_id in enumerate(ordered_node_ids)}

    rows = []
    unavailable = []
    for target in CANDIDATE_TARGETS:
        if target not in node_ids:
            unavailable.append({"target": target, "status": "target_not_in_graph"})
            continue
        paths = base.enumerate_paths(graph, source, target, max_hops=8, limit=500)
        if not paths:
            unavailable.append({"target": target, "status": "no_directed_path"})
            continue
        for path, edges in paths:
            raw_gnn_score, coherence, evidence_score = base.score_path(
                path, edges, embeddings, node_index
            )
            observed_count = sum(is_observed(edge) for edge in edges)
            observed_fraction = observed_count / len(edges) if edges else 0.0
            configured_edges = [
                {
                    "source": str(base.field(edge, "source", "")),
                    "target": str(base.field(edge, "target", "")),
                    "edge_type": str(base.field(edge, "edge_type", "")),
                }
                for edge in edges if not is_observed(edge)
            ]
            # Keep the raw GNN score visible. This adjusted score discounts paths
            # whose transitions are not directly supported by observed evidence.
            adjusted_score = raw_gnn_score * (0.25 + 0.75 * observed_fraction)
            row = {
                "target": target,
                "path": path,
                "edge_types": [str(base.field(edge, "edge_type", "")) for edge in edges],
                "edge_bases": [str(base.field(edge, "basis", "unknown")) for edge in edges],
                "raw_gnn_guided_score": raw_gnn_score,
                "evidence_adjusted_gnn_score": adjusted_score,
                "gnn_embedding_coherence": coherence,
                "evidence_score": evidence_score,
                "deterministic_score": deterministic_score(edges),
                "observed_edge_count": observed_count,
                "total_edge_count": len(edges),
                "observed_edge_fraction": observed_fraction,
                "configured_edges": configured_edges,
                "path_evidence_status": (
                    "observed_transitions_only" if not configured_edges
                    else "impact_hypothesis_contains_configured_transition"
                ),
                "is_ground_truth_path": path == expected,
                "is_ground_truth_target": target == true_target,
            }
            rows.append(row)

    raw_ranked = sorted(rows, key=lambda row: row["raw_gnn_guided_score"], reverse=True)
    adjusted_ranked = sorted(rows, key=lambda row: row["evidence_adjusted_gnn_score"], reverse=True)
    baseline_ranked = sorted(rows, key=lambda row: row["deterministic_score"], reverse=True)
    observed_only = [row for row in rows if row["path_evidence_status"] == "observed_transitions_only"]
    impact_hypotheses = [row for row in rows if row["path_evidence_status"] != "observed_transitions_only"]

    def rank_of(ranked, predicate):
        return next((index for index, row in enumerate(ranked, start=1) if predicate(row)), None)

    result = {
        "experiment": "scenario_001_evidence_aware_candidate_path_ranking",
        "status": "exploratory_evidence_aware_ranking",
        "ground_truth_path": expected,
        "ground_truth_target": true_target,
        "candidate_targets": list(CANDIDATE_TARGETS),
        "candidate_path_count": len(rows),
        "raw_gnn_ground_truth_path_rank": rank_of(raw_ranked, lambda row: row["is_ground_truth_path"]),
        "evidence_adjusted_gnn_ground_truth_path_rank": rank_of(adjusted_ranked, lambda row: row["is_ground_truth_path"]),
        "deterministic_ground_truth_path_rank": rank_of(baseline_ranked, lambda row: row["is_ground_truth_path"]),
        "raw_gnn_top_path": raw_ranked[0]["path"] if raw_ranked else None,
        "evidence_adjusted_gnn_top_path": adjusted_ranked[0]["path"] if adjusted_ranked else None,
        "deterministic_top_path": baseline_ranked[0]["path"] if baseline_ranked else None,
        "observed_transition_paths": sorted(observed_only, key=lambda row: row["evidence_adjusted_gnn_score"], reverse=True),
        "impact_hypotheses": sorted(impact_hypotheses, key=lambda row: row["evidence_adjusted_gnn_score"], reverse=True),
        "raw_gnn_ranked_candidates": raw_ranked,
        "evidence_adjusted_gnn_ranked_candidates": adjusted_ranked,
        "deterministic_ranked_candidates": baseline_ranked,
        "unavailable_candidates": unavailable,
        "limitations": [
            "This compares different downstream targets, not alternate routes to the same target.",
            "A configured dependency is not evidence that the attack traversed that dependency.",
            "The evidence adjustment is an explicit heuristic, not a learned probability.",
            "The GCN was not trained directly on complete attack paths.",
            "One scenario cannot establish generalization or GNN superiority.",
        ],
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "candidate_path_count": len(rows),
        "raw_gnn_ground_truth_path_rank": result["raw_gnn_ground_truth_path_rank"],
        "evidence_adjusted_gnn_ground_truth_path_rank": result["evidence_adjusted_gnn_ground_truth_path_rank"],
        "deterministic_ground_truth_path_rank": result["deterministic_ground_truth_path_rank"],
        "raw_gnn_top_path": result["raw_gnn_top_path"],
        "evidence_adjusted_gnn_top_path": result["evidence_adjusted_gnn_top_path"],
        "observed_transition_path_count": len(observed_only),
        "impact_hypothesis_count": len(impact_hypotheses),
        "result_file": str(OUT_FILE.relative_to(ROOT)),
        "interpretation": "Observed transitions and configured-impact hypotheses are reported separately.",
    }, indent=2))


if __name__ == "__main__":
    main()
