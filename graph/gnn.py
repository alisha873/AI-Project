"""PyTorch GCN baseline for AutoShield attack graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from evidence.schema import AttackGraph


NODE_TYPES = (
    "container",
    "service",
    "ip",
    "port",
    "process",
    "user",
)

EDGE_TYPES = (
    "communicates_with",
    "connects_to",
    "accesses",
    "executes",
    "authenticates_to",
    "hosts",
    "forwards",
    "depends_on",
)

EDGE_BASES = ("configured", "observed")

@dataclass
class GraphTensors:
    """Tensor representation of an AutoShield graph."""

    node_ids: list[str]
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_type_ids: torch.Tensor
    edge_basis_ids: torch.Tensor


def encode_graph(graph: AttackGraph) -> GraphTensors:
    """Convert an AttackGraph into deterministic tensors."""

    nodes = sorted(graph.nodes, key=lambda node: node.node_id)
    node_ids = [node.node_id for node in nodes]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    # One-hot node type plus configured/observed basis.
    features = []
    for node in nodes:
        type_features = [
            float(node.node_type == node_type)
            for node_type in NODE_TYPES
        ]
        basis_features = [
            float(node.basis == "configured"),
            float(node.basis == "observed"),
        ]
        features.append(type_features + basis_features)

    node_features = torch.tensor(features, dtype=torch.float32)

    edge_records = sorted(
        graph.edges,
        key=lambda edge: (
            edge.source,
            edge.target,
            edge.edge_type,
        ),
    )

    edge_pairs = []
    edge_type_ids = []
    edge_basis_ids = []

    for edge in edge_records:
        if edge.source not in node_index or edge.target not in node_index:
            continue

        edge_pairs.append(
            [node_index[edge.source], node_index[edge.target]]
        )
        edge_type_ids.append(EDGE_TYPES.index(edge.edge_type))

        if edge.basis not in EDGE_BASES:
            raise ValueError(f"Unsupported edge basis: {edge.basis}")

        edge_basis_ids.append(EDGE_BASES.index(edge.basis))

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long,
        ).t().contiguous()
        edge_type_tensor = torch.tensor(
            edge_type_ids,
            dtype=torch.long,
        )
        edge_basis_tensor = torch.tensor(
            edge_basis_ids,
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type_tensor = torch.empty((0,), dtype=torch.long)
        edge_basis_tensor = torch.empty((0,), dtype=torch.long)

    return GraphTensors(
        node_ids=node_ids,
        node_features=node_features,
        edge_index=edge_index,
        edge_type_ids=edge_type_tensor,
        edge_basis_ids=edge_basis_tensor,
    )


class GraphConvolution(nn.Module):
    """Graph convolution with separate transformations per edge type."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_edge_types: int = len(EDGE_TYPES),
    ) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.num_edge_types = num_edge_types
        self.num_edge_bases = len(EDGE_BASES)

        self.edge_linears = nn.ModuleList(
            nn.Linear(in_features, out_features, bias=False)
            for _ in range(num_edge_types * self.num_edge_bases)
        )

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor | None = None,
        edge_basis_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_count = features.shape[0]
        output = self.self_linear(features)

        if edge_index.numel() == 0:
            return output

        source, target = edge_index.to(features.device)
        if edge_type_ids is None:
            edge_type_ids = torch.zeros(
                source.shape[0],
                dtype=torch.long,
                device=features.device,
            )
        else:
            edge_type_ids = edge_type_ids.to(features.device)

        if edge_basis_ids is None:
            edge_basis_ids = torch.zeros(
                source.shape[0],
                dtype=torch.long,
                device=features.device,
            )
        else:
            edge_basis_ids = edge_basis_ids.to(features.device)

        if edge_basis_ids.shape[0] != source.shape[0]:
            raise ValueError("Edge basis IDs must match the number of edges")

        if torch.any(edge_basis_ids < 0) or torch.any(
            edge_basis_ids >= self.num_edge_bases
        ):
            raise ValueError("Unsupported edge basis ID")

        for relation_id, transform in enumerate(self.edge_linears):
            relation_type_id = relation_id // self.num_edge_bases
            basis_id = relation_id % self.num_edge_bases

            mask = (
                (edge_type_ids == relation_type_id)
                & (edge_basis_ids == basis_id)
            )
            if not torch.any(mask):
                continue

            relation_sources = source[mask]
            relation_targets = target[mask]
            messages = transform(features[relation_sources])

            aggregated = torch.zeros(
                (node_count, messages.shape[1]),
                dtype=messages.dtype,
                device=messages.device,
            )
            aggregated.index_add_(0, relation_targets, messages)

            counts = torch.zeros(
                node_count,
                dtype=messages.dtype,
                device=messages.device,
            )
            counts.index_add_(
                0,
                relation_targets,
                torch.ones_like(relation_targets, dtype=messages.dtype),
            )

            output = output + aggregated / counts.clamp(min=1).unsqueeze(1)

        return output



class AttackGraphGCN(nn.Module):
    """Two-layer relation-aware GCN producing node embeddings."""

    def __init__(
        self,
        input_dim: int = len(NODE_TYPES) + 2,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, embedding_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor | None = None,
        edge_basis_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.conv1(
            node_features, edge_index, edge_type_ids, edge_basis_ids
        )
        hidden = F.relu(hidden)
        hidden = self.conv2(
            hidden, edge_index, edge_type_ids, edge_basis_ids
        )
        return hidden


class EdgeScorer(nn.Module):
    """Learn to score candidate typed edges between graph nodes."""

    def __init__(
        self,
        embedding_dim: int = 16,
        num_edge_types: int = len(EDGE_TYPES),
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()

        self.edge_type_embedding = nn.Embedding(
            num_edge_types,
            embedding_dim,
        )

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        edge_type_features = self.edge_type_embedding(edge_type_ids)

        features = torch.cat(
            [
                source_embeddings,
                target_embeddings,
                torch.abs(source_embeddings - target_embeddings),
                edge_type_features,
            ],
            dim=1,
        )

        return self.network(features).squeeze(-1)
