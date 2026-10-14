#!/usr/bin/env python3
"""Evaluate independently documented paths without mixing path semantics.

Run from the repository root:
    PYTHONPATH=. python evaluation/evaluate_verified_paths.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation import complete_path_reconstruction as base
from evidence.schema import AttackGraph

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "evaluation/results/models/attack_graph_gcn.pt"
OUT = ROOT / "evaluation/results/verified_path_comparison.json"


def load_graph(path: Path):
    doc = json.loads(path.read_text())
    return AttackGraph.model_validate(doc.get("graph", doc))


def nodes_of(graph):
    nodes = graph.nodes
    return list(nodes.values()) if isinstance(nodes, dict) else list(nodes)


def deterministic_score(edges):
    if not edges:
        return 0.0
    values = []
    for edge in edges:
        rel = str(base.field(edge, "edge_type", ""))
        basis = str(base.field(edge, "basis", "observed")).lower()
        # Handle both plain string and enum-style serializations.
        observed = basis == "observed" or basis.endswith(".observed")
        values.append(base.EDGE_WEIGHTS.get(rel, 0.1) * (1.0 if observed else 0.25))
    return sum(values) / len(values)


def metrics(expected, predicted):
    expected_nodes, predicted_nodes = set(expected), set(predicted)
    expected_edges = set(zip(expected, expected[1:]))
    predicted_edges = set(zip(predicted, predicted[1:]))
    return {
        "exact_path_match": predicted == expected,
        "node_coverage": len(expected_nodes & predicted_nodes) / max(1, len(expected_nodes)),
        "transition_coverage": len(expected_edges & predicted_edges) / max(1, len(expected_edges)),
    }


def evaluate_case(case_id, path_type, graph_path, expected):
    graph = load_graph(graph_path)
    graph_nodes = nodes_of(graph)
    graph_node_ids = {base.node_id(node) for node in graph_nodes}
    graph_node_ids.discard(None)
    missing = [node for node in expected if node not in graph_node_ids]
    if missing:
        return {
            "case_id": case_id,
            "path_type": path_type,
            "status": "invalid_ground_truth_nodes",
            "missing_nodes": missing,
            "graph_path": str(graph_path.relative_to(ROOT)),
        }

    source, target = expected[0], expected[-1]
    candidates = base.enumerate_paths(graph, source, target, max_hops=8, limit=500)
    if not candidates:
        return {
            "case_id": case_id,
            "path_type": path_type,
            "status": "no_candidate_path",
            "ground_truth_path": expected,
            "candidate_path_count": 0,
        }

    gcn = evaluate_case.gcn
    x, edge_index, edge_types, edge_bases = base.encoded_tensors(graph)
    with torch.no_grad():
        embeddings = gcn(x, edge_index, edge_types, edge_bases)
    ordered_ids = sorted(base.node_id(node) for node in graph_nodes if base.node_id(node) is not None)
    if embeddings.shape[0] != len(ordered_ids):
        raise ValueError(
            f"{case_id}: embedding count {embeddings.shape[0]} != graph node count {len(ordered_ids)}"
        )
    node_index = {node_id: index for index, node_id in enumerate(ordered_ids)}

    ranked = []
    for path, edges in candidates:
        gnn_score, coherence, evidence_score = base.score_path(path, edges, embeddings, node_index)
        ranked.append({
            "path": path,
            "edge_types": [str(base.field(edge, "edge_type", "")) for edge in edges],
            "edge_bases": [str(base.field(edge, "basis", "unknown")) for edge in edges],
            "gnn_guided_score": gnn_score,
            "gnn_embedding_coherence": coherence,
            "deterministic_score": deterministic_score(edges),
            "hybrid_evidence_score": evidence_score,
        })

    by_gnn = sorted(ranked, key=lambda row: row["gnn_guided_score"], reverse=True)
    by_baseline = sorted(ranked, key=lambda row: row["deterministic_score"], reverse=True)
    gnn_path, baseline_path = by_gnn[0]["path"], by_baseline[0]["path"]
    return {
        "case_id": case_id,
        "path_type": path_type,
        "status": "evaluated",
        "ground_truth_path": expected,
        "candidate_path_count": len(ranked),
        "gnn_guided_prediction": gnn_path,
        "deterministic_prediction": baseline_path,
        "gnn_guided_metrics": metrics(expected, gnn_path),
        "deterministic_metrics": metrics(expected, baseline_path),
        "gnn_guided_top_candidates": by_gnn[:10],
        "deterministic_top_candidates": by_baseline[:10],
    }


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL.relative_to(ROOT)}")

    evaluate_case.gcn = base.build_gcn(base.checkpoint_state(MODEL))

    # Use the previously recorded, independently documented incident_001 path.
    incident001_report = json.loads((ROOT / "evaluation/results/attack_path_reconstruction.json").read_text())
    expected001 = incident001_report["ground_truth_path"]

    # This second label is specifically an observed HTTP request-routing path,
    # not a lateral-movement or compromise-impact path.
    gt045 = json.loads((ROOT / "evaluation/results/incident_045_path_ground_truth.json").read_text())
    expected045 = gt045["ground_truth_path"]

    cases = [
        evaluate_case(
            "incident_001",
            "attack_impact_path",
            ROOT / "evaluation/scenarios/scenario_001_graph.json",
            expected001,
        ),
        evaluate_case(
            "incident_045",
            "observed_http_request_path",
            ROOT / "evaluation/results/incident_045_graph.json",
            expected045,
        ),
    ]

    evaluated = [case for case in cases if case.get("status") == "evaluated"]
    output = {
        "experiment": "verified_path_reconstruction_comparison",
        "status": "prototype_two_cases_no_multi_candidate_evidence",
        "cases_evaluated": len(evaluated),
        "cases_with_multiple_candidate_paths": sum(
            case.get("candidate_path_count", 0) > 1 for case in evaluated
        ),
        "cases": cases,
        "interpretation": (
            "Path types are intentionally reported separately: incident_001 is an "
            "attack-impact path, while incident_045 is an observed HTTP request path. "
            "Do not average them into a single performance claim."
        ),
        "limitations": [
            "The cases have different path semantics and are not a homogeneous benchmark.",
            "Only candidate ranking with multiple plausible routes can test ranking quality.",
            "The GNN-guided score uses GCN embedding coherence plus evidence weights; it is not a path-supervised probability.",
            "incident_045's 200 response and admin-token response indicate apparent authentication bypass, but the token itself is not stored in this report.",
        ],
        "next_requirement": (
            "Collect an independently verified case with at least two plausible directed "
            "paths between the same ingress and target, then compare GNN-guided and "
            "deterministic rankings on that same path task."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "status": output["status"],
        "cases_evaluated": output["cases_evaluated"],
        "cases_with_multiple_candidate_paths": output["cases_with_multiple_candidate_paths"],
        "cases": [
            {
                "case_id": case.get("case_id"),
                "path_type": case.get("path_type"),
                "status": case.get("status"),
                "candidate_path_count": case.get("candidate_path_count"),
                "gnn_exact_match": case.get("gnn_guided_metrics", {}).get("exact_path_match"),
                "baseline_exact_match": case.get("deterministic_metrics", {}).get("exact_path_match"),
            }
            for case in cases
        ],
        "result_file": str(OUT.relative_to(ROOT)),
        "next_requirement": output["next_requirement"],
    }, indent=2))


if __name__ == "__main__":
    main()
