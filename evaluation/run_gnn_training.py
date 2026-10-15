"""Train, evaluate, and save the AutoShield GNN baseline."""

import json
from pathlib import Path

import torch

from evaluation.gnn_dataset import (
    ScenarioGraphRecord,
    ScenarioGroundTruth,
    ScenarioSplit,
    prepare_edge_examples,
)
from evaluation.train_gnn import train_gnn_edge_scorer, _calculate_metrics
from graph.gnn import EDGE_TYPES, encode_graph

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "scenarios"
OUTPUT = ROOT / "evaluation" / "results"
MODEL_DIR = OUTPUT / "models"


def main():
    assignments = [
        ScenarioSplit.model_validate(item)
        for item in json.loads((DATA / "gnn_splits.json").read_text(encoding=" utf-8\))
    ]

    graphs, truths = [], []
    for assignment in assignments:
        sid = assignment.scenario_id
        graphs.append(ScenarioGraphRecord.model_validate({
            "scenario_id": sid,
            "graph": json.loads((DATA / f"{sid}_graph.json").read_text(encoding=" utf-8\)),
        }))
        truths.append(ScenarioGroundTruth.model_validate(
            json.loads((DATA / f"{sid}_ground_truth.json").read_text(encoding=" utf-8\))
        ))

    graph_by_scenario = {record.scenario_id: record.graph for record in graphs}
    examples = prepare_edge_examples(graphs, truths, assignments)

    result = train_gnn_edge_scorer(
        graph_by_scenario, examples, epochs=50, learning_rate=0.01, seed=7
    )

    test_examples = [e for e in examples if e.split == "test"]
    probabilities, labels = [], []
    result.gcn.eval()
    result.scorer.eval()

    with torch.no_grad():
        by_scenario = {}
        for example in test_examples:
            by_scenario.setdefault(example.scenario_id, []).append(example)

        for sid, scenario_examples in by_scenario.items():
            tensors = encode_graph(graph_by_scenario[sid])
            embeddings = result.gcn(
                tensors.node_features,
                tensors.edge_index,
                tensors.edge_type_ids,
                tensors.edge_basis_ids,
            )
            node_index = {node_id: i for i, node_id in enumerate(tensors.node_ids)}

            source_indices, target_indices, relation_ids = [], [], []
            for example in scenario_examples:
                source_indices.append(node_index[example.edge.source])
                target_indices.append(node_index[example.edge.target])
                relation_ids.append(EDGE_TYPES.index(example.edge.edge_type))

            logits = result.scorer(
                embeddings[torch.tensor(source_indices, dtype=torch.long)],
                embeddings[torch.tensor(target_indices, dtype=torch.long)],
                torch.tensor(relation_ids, dtype=torch.long),
            )
            probabilities.extend(torch.sigmoid(logits).tolist())
            labels.extend(e.edge.label for e in scenario_examples)

    predictions = [int(p >= 0.5) for p in probabilities]
    test_metrics = _calculate_metrics(predictions, labels)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    torch.save(result.gcn.state_dict(), MODEL_DIR / "attack_graph_gcn.pt")
    torch.save(result.scorer.state_dict(), MODEL_DIR / "edge_scorer.pt")

    report = {
        "experiment": "initial_gnn_edge_scorer_baseline",
        "seed": 7,
        "epochs": 50,
        "learning_rate": 0.01,
        "scenario_count": len(graphs),
        "labeled_example_count": len(examples),
        "train_examples": sum(e.split == "train" for e in examples),
        "validation": {
            "metrics": result.validation_metrics,
            "final_loss": result.validation_losses[-1],
            "examples": sum(e.split == "validation" for e in examples),
        },
        "test": {
            "metrics": test_metrics,
            "examples": len(labels),
            "predictions": [
                {
                    "probability": round(p, 4),
                    "predicted": pred,
                    "actual": label,
                }
                for p, pred, label in zip(probabilities, predictions, labels)
            ],
        },
        "final_train_loss": result.train_losses[-1],
        "note": (
            "Preliminary results from a very small dataset; not evidence of "
            "generalization to unseen environments."
        ),
    }
    (OUTPUT / "gnn_baseline_results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print(json.dumps(report, indent=2))
    print(f"\nSaved GNN weights: {MODEL_DIR / 'attack_graph_gcn.pt'}")
    print(f"Saved edge scorer: {MODEL_DIR / 'edge_scorer.pt'}")
    print(f"Saved metrics: {OUTPUT / 'gnn_baseline_results.json'}")


if __name__ == "__main__":
    main()
