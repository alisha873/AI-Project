from __future__ import annotations

import json
from pathlib import Path

from evaluation.evidence_metrics import precision_recall_f1
import argparse

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_category(
    name: str,
    predicted: list[str],
    expected: list[str],
) -> None:
    metrics = precision_recall_f1(predicted, expected)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Expected items:    {len(set(expected))}")
    print(f"Predicted items:   {len(set(predicted))}")
    print(f"True positives:    {metrics['true_positives']}")
    print(f"False positives:   {metrics['false_positives']}")
    print(f"False negatives:   {metrics['false_negatives']}")
    print(f"Precision:         {metrics['precision']:.3f}")
    print(f"Recall:            {metrics['recall']:.3f}")
    print(f"F1:                {metrics['f1']:.3f}")

    unexpected = sorted(set(predicted) - set(expected))
    missed = sorted(set(expected) - set(predicted))

    print(f"Over-selected:     {unexpected}")
    print(f"Missed:            {missed}")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extraction",
        required=True,
        help="Path to the extraction JSON to evaluate",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    labels = load_json(
        project_root
        / "evaluation"
        / "fixtures"
        / "incident_001_evidence_labels.json"
    )


    extraction_path = Path(args.extraction)

    if not extraction_path.is_absolute():
        extraction_path = project_root / extraction_path

    if not extraction_path.is_file():
        parser.error(f"extraction file does not exist: {extraction_path}")

    extraction = load_json(extraction_path)


    if labels["incident_id"] != extraction["incident_id"]:
        raise ValueError("Incident IDs do not match.")

    print(f"Evidence evaluation: {extraction['incident_id']}")

    categories = [
        (
            "Relevant events",
            extraction["relevant_events"],
            labels["relevant_event_ids"],
        ),
        (
            "Relevant network events",
            extraction["relevant_network_events"],
            labels["relevant_network_event_ids"],
        ),
        (
            "Relevant configuration",
            extraction["relevant_configuration"],
            labels["relevant_configuration_paths"],
        ),
        (
            "Relevant network artifacts",
            extraction["relevant_network_artifacts"],
            labels["relevant_network_artifact_paths"],
        ),
    ]

    for name, predicted, expected in categories:
        evaluate_category(name, predicted, expected)


if __name__ == "__main__":
    main()
