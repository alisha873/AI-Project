"""Training utilities for the AutoShield GNN edge scorer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from evaluation.gnn_dataset import PreparedEdgeExample
from evidence.schema import AttackGraph
from graph.gnn import EDGE_TYPES, AttackGraphGCN, EdgeScorer, encode_graph


@dataclass
class TrainingResult:
    gcn: AttackGraphGCN
    scorer: EdgeScorer
    train_losses: list[float]
    validation_losses: list[float]
    validation_metrics: dict[str, float]


def _require_both_classes(
    examples: list[PreparedEdgeExample],
    split_name: str,
) -> None:
    classes = {example.edge.label for example in examples}
    if classes != {0, 1}:
        raise ValueError(
            f"{split_name} split must contain both positive and negative labels"
        )


def _calculate_metrics(
    predictions: list[int],
    labels: list[int],
) -> dict[str, float]:
    true_positive = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    true_negative = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    false_positive = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    false_negative = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    total = len(labels)
    accuracy = (true_positive + true_negative) / total if total else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def train_gnn_edge_scorer(
    graph_by_scenario: dict[str, AttackGraph],
    examples: list[PreparedEdgeExample],
    *,
    epochs: int = 50,
    learning_rate: float = 0.01,
    seed: int = 7,
) -> TrainingResult:
    """Train on explicit labels with scenario-level validation."""

    if epochs < 1:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    train_examples = [e for e in examples if e.split == "train"]
    validation_examples = [e for e in examples if e.split == "validation"]

    if not train_examples or not validation_examples:
        raise ValueError("Training and validation examples are both required")

    if any(e.scenario_id not in graph_by_scenario for e in examples):
        raise ValueError("An example references a missing scenario graph")

    scenario_splits: dict[str, str] = {}
    for example in examples:
        previous = scenario_splits.setdefault(example.scenario_id, example.split)
        if previous != example.split:
            raise ValueError(
                f"Scenario appears in multiple splits: {example.scenario_id}"
            )

    _require_both_classes(train_examples, "Training")
    _require_both_classes(validation_examples, "Validation")

    torch.manual_seed(seed)

    gcn = AttackGraphGCN()
    scorer = EdgeScorer()
    optimizer = torch.optim.Adam(
        list(gcn.parameters()) + list(scorer.parameters()),
        lr=learning_rate,
    )
    loss_fn = nn.BCEWithLogitsLoss()

    def score_examples(
        batch: list[PreparedEdgeExample],
    ) -> tuple[torch.Tensor, list[int]]:
        logits_by_scenario = []
        labels_by_scenario = []

        by_scenario: dict[str, list[PreparedEdgeExample]] = {}
        for example in batch:
            by_scenario.setdefault(example.scenario_id, []).append(example)

        for scenario_id, scenario_examples in by_scenario.items():
            tensors = encode_graph(graph_by_scenario[scenario_id])
            embeddings = gcn(
                tensors.node_features,
                tensors.edge_index,
                tensors.edge_type_ids,
                tensors.edge_basis_ids,
            )
            node_index = {
                node_id: index
                for index, node_id in enumerate(tensors.node_ids)
            }

            source_indices = []
            target_indices = []
            relation_ids = []
            labels = []

            for example in scenario_examples:
                edge = example.edge
                if edge.source not in node_index or edge.target not in node_index:
                    raise ValueError(
                        f"Labeled edge references missing node in {scenario_id}"
                    )
                if edge.edge_type not in EDGE_TYPES:
                    raise ValueError(f"Unsupported edge type: {edge.edge_type}")

                source_indices.append(node_index[edge.source])
                target_indices.append(node_index[edge.target])
                relation_ids.append(EDGE_TYPES.index(edge.edge_type))
                labels.append(edge.label)

            logits = scorer(
                embeddings[torch.tensor(source_indices, dtype=torch.long)],
                embeddings[torch.tensor(target_indices, dtype=torch.long)],
                torch.tensor(relation_ids, dtype=torch.long),
            )
            logits_by_scenario.append(logits)
            labels_by_scenario.extend(labels)

        return torch.cat(logits_by_scenario), labels_by_scenario

    train_losses: list[float] = []
    validation_losses: list[float] = []

    for _ in range(epochs):
        gcn.train()
        scorer.train()
        optimizer.zero_grad()

        train_logits, train_labels = score_examples(train_examples)
        train_targets = torch.tensor(train_labels, dtype=torch.float32)
        train_loss = loss_fn(train_logits, train_targets)
        train_loss.backward()
        optimizer.step()
        train_losses.append(float(train_loss.detach().item()))

        gcn.eval()
        scorer.eval()
        with torch.no_grad():
            validation_logits, validation_labels = score_examples(
                validation_examples
            )
            validation_targets = torch.tensor(
                validation_labels,
                dtype=torch.float32,
            )
            validation_loss = loss_fn(validation_logits, validation_targets)

        validation_losses.append(float(validation_loss.item()))

    probabilities = torch.sigmoid(validation_logits)
    predictions = (probabilities >= 0.5).to(torch.int64).tolist()

    return TrainingResult(
        gcn=gcn,
        scorer=scorer,
        train_losses=train_losses,
        validation_losses=validation_losses,
        validation_metrics=_calculate_metrics(predictions, validation_labels),
    )
