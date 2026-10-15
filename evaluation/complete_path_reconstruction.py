#!/usr/bin/env python3
"""Run a first GNN-guided attack-path reconstruction experiment.

Run from the repository root:
    PYTHONPATH=. python evaluation/complete_path_reconstruction.py

This evaluates the independently documented incident_001 path only. The current
scenario dataset's attack-intent labels are not treated as ground-truth attack paths.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch

from evidence.schema import AttackGraph
from graph.gnn import AttackGraphGCN, encode_graph

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / "evaluation/scenarios/scenario_001_graph.json"
GROUND_TRUTH_PATH = ROOT / "evaluation/scenarios/scenario_001_ground_truth.json"
MODEL_PATH = ROOT / "evaluation/results/models/attack_graph_gcn.pt"
OUTPUT_PATH = ROOT / "evaluation/results/attack_path_reconstruction.json"

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


def field(obj: Any, name: str, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "gcn_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Could not read a model state_dict from {path}")
    return payload


def build_gcn(state: dict[str, torch.Tensor]) -> AttackGraphGCN:
    layers = [
        (key, value)
        for key, value in state.items()
        if key.endswith("self_linear.weight") and getattr(value, "ndim", 0) == 2
    ]
    if len(layers) < 2:
        raise ValueError("Could not infer GCN dimensions from saved checkpoint.")
    layers.sort(key=lambda item: item[0])
    input_dim = int(layers[0][1].shape[1])
    hidden_dim = int(layers[0][1].shape[0])
    output_dim = int(layers[-1][1].shape[0])
    values = {
        "input_dim": input_dim, "in_channels": input_dim,
        "hidden_dim": hidden_dim, "hidden_channels": hidden_dim,
        "output_dim": output_dim, "out_channels": output_dim,
        "embedding_dim": output_dim,
    }
    kwargs = {}
    for name, param in inspect.signature(AttackGraphGCN).parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif param.default is inspect.Parameter.empty:
            raise ValueError(f"Cannot infer required AttackGraphGCN parameter: {name}")
    model = AttackGraphGCN(**kwargs)
    model.load_state_dict(state)
    model.eval()
    return model


def encoded_tensors(graph):
    encoded = encode_graph(graph)
    if isinstance(encoded, dict):
        x = encoded.get("node_features", encoded.get("x"))
        edge_index = encoded.get("edge_index")
        edge_types = encoded.get("edge_type_ids", encoded.get("edge_types"))
        edge_bases = encoded.get("edge_basis_ids", encoded.get("edge_bases"))
    elif hasattr(encoded, "node_features"):
        x, edge_index = encoded.node_features, encoded.edge_index
        edge_types = getattr(encoded, "edge_type_ids", None)
        edge_bases = getattr(encoded, "edge_basis_ids", None)
    else:
        if len(encoded) < 4:
            raise ValueError("encode_graph returned fewer than four values")
        x, edge_index, edge_types, edge_bases = encoded[:4]
    if x is None or edge_index is None:
        raise ValueError("Could not identify node features and edge index.")
    if edge_types is None:
        edge_types = torch.zeros(edge_index.shape[1], dtype=torch.long)
    if edge_bases is None:
        edge_bases = torch.zeros(edge_index.shape[1], dtype=torch.long)
    return x, edge_index, edge_types, edge_bases


def enumerate_paths(graph, source: str, target: str, max_hops: int = 8, limit: int = 100):
    adjacency: dict[str, list[Any]] = {}
    for edge in graph.edges:
        adjacency.setdefault(str(field(edge, "source")), []).append(edge)
    found = []

    def walk(node, path, path_edges):
        if len(found) >= limit:
            return
        if node == target:
            found.append((path[:], path_edges[:]))
            return
        if len(path_edges) >= max_hops:
            return
        for edge in adjacency.get(node, []):
            nxt = str(field(edge, "target"))
            if nxt not in path:
                walk(nxt, path + [nxt], path_edges + [edge])

    walk(source, [source], [])
    return found


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return 0.0 if denom <= 1e-12 else float(torch.dot(a, b) / denom)


def score_path(path, path_edges, embeddings, node_index):
    if not path_edges:
        return 0.0, 0.0, 0.0
    coherence = sum(
        cosine(embeddings[node_index[a]], embeddings[node_index[b]])
        for a, b in zip(path, path[1:])
    ) / len(path_edges)
    evidence = []
    for edge in path_edges:
        relation = str(field(edge, "edge_type", ""))
        basis = str(field(edge, "basis", "observed"))
        evidence.append(EDGE_WEIGHTS.get(relation, 0.1) * (1.0 if basis == "observed" else 0.25))
    evidence_score = sum(evidence) / len(evidence)
    return 0.5 * coherence + 0.5 * evidence_score, coherence, evidence_score


def node_id(node):
    for key in ("node_id", "id", "entity_id", "key"):
        value = field(node, key)
        if value is not None:
            return str(value)
    return None


def node_label(node):
    parts = []
    for key in ("name", "label", "service_name", "container_name", "value"):
        value = field(node, key)
        if value is not None:
            parts.append(str(value))
    attrs = field(node, "attributes", {}) or {}
    if isinstance(attrs, dict):
        parts.extend(str(v) for v in attrs.values() if isinstance(v, (str, int, float)))
    return " ".join(parts).lower()


def node_kind(node):
    return str(field(node, "node_type", field(node, "type", ""))).lower()


def find_service_node(nodes, service):
    aliases = {
        "nginx": ("nginx",),
        "api": ("autoshield-api", "service api", " api"),
        "postgres": ("postgres", "postgresql"),
    }[service]
    matches = []
    for node in nodes:
        label = node_label(node)
        kind = node_kind(node)
        if any(alias in label for alias in aliases) and (
            kind == "service" or str(node_id(node)).lower().startswith(("service:", "svc:"))
        ):
            matches.append(node_id(node))
    matches = [item for item in matches if item is not None]
    if len(matches) == 1:
        return matches[0]
    # Fallback for graph IDs whose names encode service identity.
    for node in nodes:
        identifier = (node_id(node) or "").lower()
        label = node_label(node)
        if service in identifier and any(alias in label or alias in identifier for alias in aliases):
            return node_id(node)
    return None


def normalize_gt_path(labels, nodes):
    service_ids = {
        "nginx": find_service_node(nodes, "nginx"),
        "api": find_service_node(nodes, "api"),
        "postgres": find_service_node(nodes, "postgres"),
    }
    result = []
    for label in labels:
        text = str(label).lower()
        if text == "attacker":
            continue
        if "nginx" in text:
            candidate = service_ids["nginx"]
        elif "postgres" in text or "database" in text:
            candidate = service_ids["postgres"]
        elif "autoshield-api" in text or text == "api":
            candidate = service_ids["api"]
        else:
            candidate = None
        if candidate and (not result or result[-1] != candidate):
            result.append(candidate)
    return result


def main():
    for path in (GRAPH_PATH, GROUND_TRUTH_PATH, MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path.relative_to(ROOT)}")

    graph_document = json.loads(GRAPH_PATH.read_text(encoding=" utf-8\))
    # Scenario graph artifacts wrap the AttackGraph under a top-level "graph" key.
    graph_payload = graph_document.get("graph", graph_document)
    graph = AttackGraph.model_validate(graph_payload)
    gt_document = json.loads(GROUND_TRUTH_PATH.read_text(encoding=" utf-8\))
    # Ground-truth artifacts may also be wrapped; accept either representation.
    gt = gt_document.get("ground_truth", gt_document)
    graph_nodes = list(graph.nodes.values()) if isinstance(graph.nodes, dict) else list(graph.nodes)
    graph_node_ids = {node_id(node) for node in graph_nodes if node_id(node) is not None}
    expected = normalize_gt_path(gt.get("expected_attack_path", []), graph_nodes)
    if len(expected) < 2:
        details = [{"id": node_id(n), "type": node_kind(n), "label": node_label(n)} for n in graph_nodes]
        raise ValueError(f"Could not map ground-truth path. Mapped={expected}; graph nodes={json.dumps(details[:40])}")

    # The ingress and impacted service are fixed from incident evidence; the
    # intermediate route is generated from the graph and not supplied to the decoder.
    source, target = expected[0], expected[-1]
    candidates = enumerate_paths(graph, source, target)
    if not candidates:
        raise ValueError(f"No directed graph path found from {source} to {target}")

    gcn = build_gcn(checkpoint_state(MODEL_PATH))
    x, edge_index, edge_types, edge_bases = encoded_tensors(graph)
    with torch.no_grad():
        embeddings = gcn(x, edge_index, edge_types, edge_bases)
    # encode_graph orders nodes by node_id, as used by the graph encoder.
    node_ids = sorted(node_id(node) for node in graph_nodes if node_id(node) is not None)
    if embeddings.shape[0] != len(node_ids):
        raise ValueError("Embedding/node count mismatch; refusing to report misleading metrics.")
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    scored = []
    for path, edges in candidates:
        score, coherence, evidence = score_path(path, edges, embeddings, node_index)
        scored.append({
            "path": path,
            "edge_types": [str(field(edge, "edge_type", "")) for edge in edges],
            "edge_bases": [str(field(edge, "basis", "unknown")) for edge in edges],
            "combined_score": score,
            "gnn_embedding_coherence": coherence,
            "evidence_score": evidence,
        })
    scored.sort(key=lambda row: row["combined_score"], reverse=True)
    predicted = scored[0]["path"]

    gt_transitions = set(zip(expected, expected[1:]))
    pred_transitions = set(zip(predicted, predicted[1:]))
    result = {
        "experiment": "incident_001_gnn_guided_attack_path_reconstruction",
        "status": "prototype",
        "ground_truth_path": expected,
        "predicted_path": predicted,
        "exact_path_match": predicted == expected,
        "node_coverage": len(set(expected) & set(predicted)) / max(1, len(set(expected))),
        "transition_coverage": len(gt_transitions & pred_transitions) / max(1, len(gt_transitions)),
        "candidate_path_count": len(scored),
        "candidate_paths": scored,
        "scoring_note": (
            "Experimental ranking combines GCN embedding cosine coherence with edge "
            "type and provenance weights. It is not a calibrated probability or a "
            "separately path-supervised GNN."
        ),
        "evaluation_limitations": [
            "Only one independently documented incident path is evaluated.",
            "A high score here does not establish generalization to other attack families.",
            "Configured dependencies are weaker evidence than observed runtime communication.",
            "The 36 scenario attack-intent labels are not treated as path ground truth."
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "ground_truth_path": expected,
        "predicted_path": predicted,
        "exact_path_match": result["exact_path_match"],
        "node_coverage": round(result["node_coverage"], 4),
        "transition_coverage": round(result["transition_coverage"], 4),
        "candidate_path_count": len(scored),
        "result_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "warning": "Prototype result; see evaluation_limitations in the JSON report."
    }, indent=2))


if __name__ == "__main__":
    main()
