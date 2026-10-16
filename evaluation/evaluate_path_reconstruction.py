#!/usr/bin/env python3
"""Audit and compare GNN-guided vs deterministic attack-path ranking.

Run from repository root:
    PYTHONPATH=. python evaluation/evaluate_path_reconstruction.py

Only complete, explicitly supplied expected_attack_path labels are treated as
path ground truth. Attack-intent labels are never promoted into path labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation import complete_path_reconstruction as base
from evidence.schema import AttackGraph

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "evaluation/scenarios"
MODEL_PATH = ROOT / "evaluation/results/models/attack_graph_gcn.pt"
OUTPUT_PATH = ROOT / "evaluation/results/path_reconstruction_comparison.json"


def load_graph(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    payload = doc.get("graph", doc)
    return AttackGraph.model_validate(payload)


def get_nodes(graph):
    return list(graph.nodes.values()) if isinstance(graph.nodes, dict) else list(graph.nodes)


def edge_provenance_score(path_edges):
    if not path_edges:
        return 0.0
    scores = []
    for edge in path_edges:
        relation = str(base.field(edge, "edge_type", ""))
        basis = str(base.field(edge, "basis", "observed"))
        scores.append(base.EDGE_WEIGHTS.get(relation, 0.1) * (1.0 if basis == "observed" else 0.25))
    return sum(scores) / len(scores)


def metrics(expected, predicted):
    expected_nodes = set(expected)
    predicted_nodes = set(predicted)
    expected_edges = set(zip(expected, expected[1:]))
    predicted_edges = set(zip(predicted, predicted[1:]))
    return {
        "exact_path_match": predicted == expected,
        "node_coverage": len(expected_nodes & predicted_nodes) / max(1, len(expected_nodes)),
        "transition_coverage": len(expected_edges & predicted_edges) / max(1, len(expected_edges)),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH.relative_to(ROOT)}")

    gcn = base.build_gcn(base.checkpoint_state(MODEL_PATH))
    cases = []
    excluded = []
    for gt_path in sorted(SCENARIO_DIR.glob("scenario_*_ground_truth.json")):
        scenario_id = gt_path.name.replace("_ground_truth.json", "")
        graph_path = SCENARIO_DIR / f"{scenario_id}_graph.json"
        if not graph_path.exists():
            excluded.append({"scenario_id": scenario_id, "reason": "missing graph artifact"})
            continue

        gt_doc = json.loads(gt_path.read_text(encoding="utf-8"))
        gt = gt_doc.get("ground_truth", gt_doc)
        raw_expected = gt.get("expected_attack_path", [])
        if not raw_expected or not gt.get("attack_path_complete", False):
            excluded.append({
                "scenario_id": scenario_id,
                "reason": "no complete, independently specified attack path",
            })
            continue

        graph = load_graph(graph_path)
        nodes = get_nodes(graph)
        expected = base.normalize_gt_path(raw_expected, nodes)
        if len(expected) < 2:
            excluded.append({
                "scenario_id": scenario_id,
                "reason": "ground-truth path could not be mapped to graph service nodes",
            })
            continue

        source, target = expected[0], expected[-1]
        candidates = base.enumerate_paths(graph, source, target, max_hops=8, limit=500)
        if not candidates:
            cases.append({
                "scenario_id": scenario_id,
                "ground_truth_path": expected,
                "candidate_path_count": 0,
                "status": "no_candidate_path",
            })
            continue

        x, edge_index, edge_types, edge_bases = base.encoded_tensors(graph)
        with torch.no_grad():
            embeddings = gcn(x, edge_index, edge_types, edge_bases)
        node_ids = sorted(base.node_id(node) for node in nodes if base.node_id(node) is not None)
        if embeddings.shape[0] != len(node_ids):
            raise ValueError(
                f"{scenario_id}: GCN embedding count {embeddings.shape[0]} "
                f"does not match node count {len(node_ids)}"
            )
        node_index = {node_id: i for i, node_id in enumerate(node_ids)}

        ranked = []
        for path, path_edges in candidates:
            gnn_score, coherence, evidence_score = base.score_path(
                path, path_edges, embeddings, node_index
            )
            deterministic_score = edge_provenance_score(path_edges)
            ranked.append({
                "path": path,
                "edge_types": [str(base.field(edge, "edge_type", "")) for edge in path_edges],
                "edge_bases": [str(base.field(edge, "basis", "unknown")) for edge in path_edges],
                "gnn_guided_score": gnn_score,
                "gnn_embedding_coherence": coherence,
                "deterministic_evidence_score": deterministic_score,
                "evidence_score_in_hybrid": evidence_score,
            })

        gnn_ranked = sorted(ranked, key=lambda row: row["gnn_guided_score"], reverse=True)
        baseline_ranked = sorted(ranked, key=lambda row: row["deterministic_evidence_score"], reverse=True)
        gnn_pred = gnn_ranked[0]["path"]
        baseline_pred = baseline_ranked[0]["path"]
        cases.append({
            "scenario_id": scenario_id,
            "status": "evaluated",
            "ground_truth_path": expected,
            "candidate_path_count": len(ranked),
            "gnn_guided_prediction": gnn_pred,
            "deterministic_prediction": baseline_pred,
            "gnn_guided_metrics": metrics(expected, gnn_pred),
            "deterministic_metrics": metrics(expected, baseline_pred),
            "gnn_guided_top_candidates": gnn_ranked[:10],
            "deterministic_top_candidates": baseline_ranked[:10],
        })

    evaluated = [case for case in cases if case.get("status") == "evaluated"]
    multi_candidate = [case for case in evaluated if case["candidate_path_count"] > 1]
    summary = {
        "experiment": "gnn_guided_vs_deterministic_attack_path_reconstruction",
        "status": (
            "multi_path_evaluation_available"
            if len(evaluated) >= 2 and len(multi_candidate) >= 1
            else "insufficient_independent_path_ground_truth"
        ),
        "complete_ground_truth_cases_evaluated": len(evaluated),
        "cases_with_multiple_candidate_paths": len(multi_candidate),
        "mean_gnn_exact_match": (
            sum(case["gnn_guided_metrics"]["exact_path_match"] for case in evaluated) / len(evaluated)
            if evaluated else None
        ),
        "mean_deterministic_exact_match": (
            sum(case["deterministic_metrics"]["exact_path_match"] for case in evaluated) / len(evaluated)
            if evaluated else None
        ),
        "cases": cases,
        "excluded_ground_truth_artifacts": excluded,
        "limitations": [
            "Only complete, explicitly supplied expected_attack_path labels are evaluated.",
            "Attack-intent labels are not treated as ground-truth attack paths.",
            "One candidate path cannot test whether a ranking method distinguishes alternatives.",
            "The GNN-guided score uses GCN embedding coherence and evidence weights; it is not a path-supervised probability.",
        ],
        "next_requirement": (
            "Add independently verified complete path labels for at least two more incidents, "
            "including at least one graph with multiple plausible candidate paths, before making "
            "a multi-case performance claim."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "complete_ground_truth_cases_evaluated": summary["complete_ground_truth_cases_evaluated"],
        "cases_with_multiple_candidate_paths": summary["cases_with_multiple_candidate_paths"],
        "mean_gnn_exact_match": summary["mean_gnn_exact_match"],
        "mean_deterministic_exact_match": summary["mean_deterministic_exact_match"],
        "excluded_artifacts": len(excluded),
        "result_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "next_requirement": summary["next_requirement"],
    }, indent=2))


if __name__ == "__main__":
    main()
