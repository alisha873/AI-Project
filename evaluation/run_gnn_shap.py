"""Explain one test-split candidate edge with the saved AutoShield GCN."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from evaluation.gnn_dataset import (
    ScenarioGraphRecord,
    ScenarioGroundTruth,
    ScenarioSplit,
    prepare_edge_examples,
)
from explainability.gnn_edge_explainer import explain_candidate_edge
from graph.gnn import AttackGraphGCN, EdgeScorer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evaluation" / "scenarios"
MODEL_DIR = ROOT / "evaluation" / "results" / "models"
OUTPUT = ROOT / "evaluation" / "results" / "gnn_shap_explanation.json"


def main() -> None:
    splits = [
        ScenarioSplit.model_validate(item)
        for item in json.loads(
            (DATA / "gnn_splits.json").read_text(encoding="utf-8")
        )
    ]

    graphs = []
    truths = []
    for assignment in splits:
        sid = assignment.scenario_id
        graphs.append(ScenarioGraphRecord.model_validate({
            "scenario_id": sid,
            "graph": json.loads(
                (DATA / f"{sid}_graph.json").read_text(encoding="utf-8")
            ),
        }))
        truths.append(ScenarioGroundTruth.model_validate(
            json.loads(
                (DATA / f"{sid}_ground_truth.json").read_text(encoding="utf-8")
            )
        ))

    examples = prepare_edge_examples(graphs, truths, splits)
    test_examples = [example for example in examples if example.split == "test"]
    if not test_examples:
        raise RuntimeError("No labeled test-split candidate edges were found")

    graph_by_scenario = {record.scenario_id: record.graph for record in graphs}
    selected = test_examples[0]
    graph = graph_by_scenario[selected.scenario_id]

    gcn = AttackGraphGCN()
    scorer = EdgeScorer()
    gcn_path = MODEL_DIR / "attack_graph_gcn.pt"
    scorer_path = MODEL_DIR / "edge_scorer.pt"

    if not gcn_path.is_file() or not scorer_path.is_file():
        raise FileNotFoundError(
            "Trained checkpoints not found. Run "
            "'python -m evaluation.run_gnn_training' first."
        )

    gcn.load_state_dict(torch.load(
        gcn_path, map_location="cpu", weights_only=True
    ))
    scorer.load_state_dict(torch.load(
        scorer_path, map_location="cpu", weights_only=True
    ))
    gcn.eval()
    scorer.eval()

    edge = selected.edge
    result = explain_candidate_edge(
        graph,
        gcn,
        scorer,
        source_id=edge.source,
        target_id=edge.target,
        edge_type=edge.edge_type,
    )
    result["scenario_id"] = selected.scenario_id
    result["dataset_split"] = "test"
    result["evaluation_label"] = edge.label
    result["evaluation_label_note"] = (
        "Scenario assessment label for evaluation only; not model input."
    )
    result["checkpoint_files"] = [
        str(gcn_path.relative_to(ROOT)),
        str(scorer_path.relative_to(ROOT)),
    ]
    result["dataset_note"] = (
        "Preliminary scenario-based evaluation. The raw edge logit is not "
        "a calibrated probability and does not prove exploitation."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SHAP explanation written to: {OUTPUT}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
