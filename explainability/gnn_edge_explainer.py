"""SHAP explanations for a candidate edge in AutoShield's GCN.

The explanation concerns the model's raw edge-scoring logit. It does not
establish exploitation or a confirmed attack transition.
"""
from __future__ import annotations

import numpy as np
import shap
import torch

from graph.gnn import EDGE_TYPES, encode_graph
from graph.http_features import HTTP_FEATURE_DIM


def explain_candidate_edge(
    graph,
    gcn,
    scorer,
    source_id: str,
    target_id: str,
    edge_type: str,
    *,
    background_samples: int = 100,
    seed: int = 7,
) -> dict:
    """Explain an edge logit by perturbing HTTP features of its endpoints.

    The graph topology, node types, node provenance and queried edge type
    remain fixed. Only the HTTP feature dimensions at the two endpoints
    are varied. Background values are drawn from the graph's own encoded
    endpoint features; this is a local, scenario-specific explanation.
    """
    if edge_type not in EDGE_TYPES:
        raise ValueError(f"Unsupported edge type: {edge_type}")
    if background_samples < 1:
        raise ValueError("background_samples must be positive")

    tensors = encode_graph(graph)
    index = {node_id: i for i, node_id in enumerate(tensors.node_ids)}
    if source_id not in index or target_id not in index:
        raise ValueError("Source or target node is absent from the graph")

    source_idx, target_idx = index[source_id], index[target_id]
    base_features = tensors.node_features.detach().cpu().clone()
    edge_index = tensors.edge_index
    edge_type_ids = tensors.edge_type_ids
    edge_basis_ids = tensors.edge_basis_ids
    relation_id = EDGE_TYPES.index(edge_type)

    # Node features are [node type one-hot, basis one-hot, HTTP features].
    http_start = base_features.shape[1] - HTTP_FEATURE_DIM
    if http_start < 0:
        raise ValueError("Unexpected encoded node feature dimension")

    endpoint_values = torch.cat([
        base_features[source_idx, http_start:],
        base_features[target_idx, http_start:],
    ]).numpy().astype(float)

    names = (
        [f"source_http_feature_{i}" for i in range(HTTP_FEATURE_DIM)]
        + [f"target_http_feature_{i}" for i in range(HTTP_FEATURE_DIM)]
    )

    # Use observed endpoint HTTP vectors as a small, graph-local background.
    # Repetition is intentional if the graph has few nodes.
    node_http = base_features[:, http_start:].numpy().astype(float)
    if len(node_http) == 0:
        raise ValueError("Cannot explain an empty graph")
    rng = np.random.default_rng(seed)
    source_rows = rng.integers(0, len(node_http), size=background_samples)
    target_rows = rng.integers(0, len(node_http), size=background_samples)
    background = np.concatenate(
        [node_http[source_rows], node_http[target_rows]], axis=1
    )

    gcn.eval()
    scorer.eval()

    def predict(batch):
        batch = np.asarray(batch, dtype=float)
        if batch.ndim != 2 or batch.shape[1] != 2 * HTTP_FEATURE_DIM:
            raise ValueError("Unexpected endpoint feature matrix shape")
        if not np.isfinite(batch).all():
            raise ValueError("Features must be finite")

        outputs = []
        with torch.no_grad():
            for row in batch:
                features = base_features.clone()
                features[source_idx, http_start:] = torch.as_tensor(
                    row[:HTTP_FEATURE_DIM], dtype=features.dtype
                )
                features[target_idx, http_start:] = torch.as_tensor(
                    row[HTTP_FEATURE_DIM:], dtype=features.dtype
                )

                embeddings = gcn(
                    features,
                    edge_index,
                    edge_type_ids,
                    edge_basis_ids,
                )
                logits = scorer(
                    embeddings[source_idx].reshape(1, -1),
                    embeddings[target_idx].reshape(1, -1),
                    torch.tensor([relation_id], dtype=torch.long),
                )
                if logits.numel() != 1:
                    raise ValueError("EdgeScorer must return one logit")
                outputs.append(float(logits.reshape(-1)[0].item()))

        result = np.asarray(outputs, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("EdgeScorer returned a non-finite output")
        return result

    explainer = shap.Explainer(predict, background, feature_names=names)
    explanation = explainer(endpoint_values.reshape(1, -1))
    score = float(predict(endpoint_values.reshape(1, -1))[0])
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    values = np.asarray(explanation.values).reshape(-1)

    contributions = []
    for name, value in zip(names, values):
        value = float(value)
        contributions.append({
            "feature": name,
            "contribution": value,
            "direction": (
                "increases_edge_logit" if value > 1e-10
                else "decreases_edge_logit" if value < -1e-10
                else "no_change"
            ),
        })
    contributions.sort(
        key=lambda item: abs(item["contribution"]), reverse=True
    )

    return {
        "method": "SHAP",
        "edge": {
            "source": source_id,
            "target": target_id,
            "type": edge_type,
        },
        "output_type": "raw_edge_logit",
        "score": score,
        "baseline": base,
        "contributions": contributions,
        "reconstructed_score": base + sum(
            item["contribution"] for item in contributions
        ),
        "interpretation": (
            "Local explanation of endpoint HTTP features for this candidate "
            "edge, with graph topology, node types, provenance and edge type "
            "held fixed. This is not proof of exploitation."
        ),
    }
