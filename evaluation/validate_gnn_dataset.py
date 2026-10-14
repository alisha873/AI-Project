#!/usr/bin/env python3
"""Read-only structural validator for AutoShield Phase 7 scenario artifacts.

Run from the repository root:
    PYTHONPATH=. python evaluation/validate_gnn_dataset.py

Optional:
    PYTHONPATH=. python evaluation/validate_gnn_dataset.py --start 3 --count 12
    PYTHONPATH=. python evaluation/validate_gnn_dataset.py --strict-labels

This script never edits artifacts. It checks structural consistency and flags
labeling patterns that deserve human review; it does not prove an exploit worked.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "evaluation" / "scenarios"
VALID_SPLITS = ("train", "validation", "test")


def load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        errors.append(f"Missing file: {path.relative_to(ROOT)}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Cannot read JSON {path.relative_to(ROOT)}: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"Expected a JSON object in {path.relative_to(ROOT)}")
        return None
    return value


def scenario_id(number: int) -> str:
    return f"scenario_{number:03d}"


def validate_one(number: int, errors: list[str], warnings: list[str]) -> dict[str, Any] | None:
    sid = scenario_id(number)
    graph_path = SCENARIO_DIR / f"{sid}_graph.json"
    truth_path = SCENARIO_DIR / f"{sid}_ground_truth.json"
    manifest_path = SCENARIO_DIR / f"{sid}_manifest.json"

    graph = load_json(graph_path, errors)
    truth = load_json(truth_path, errors)
    manifest = load_json(manifest_path, errors)
    if graph is None or truth is None or manifest is None:
        return None

    if truth.get("scenario_id") != sid:
        errors.append(f"{sid}: ground-truth scenario_id is {truth.get('scenario_id')!r}")
    if manifest.get("scenario_id") != sid:
        errors.append(f"{sid}: manifest scenario_id is {manifest.get('scenario_id')!r}")

    graph_obj = graph.get("graph", graph)
    nodes = graph_obj.get("nodes")
    edges = graph_obj.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        errors.append(f"{sid}: graph must contain list fields 'nodes' and 'edges'")
        nodes, edges = [], []

    node_ids: set[str] = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict) or not isinstance(node.get("node_id"), str):
            errors.append(f"{sid}: graph node #{idx} has no string node_id")
            continue
        if node["node_id"] in node_ids:
            errors.append(f"{sid}: duplicate graph node_id {node['node_id']!r}")
        node_ids.add(node["node_id"])

    edge_keys: set[tuple[str, str, str]] = set()
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            errors.append(f"{sid}: graph edge #{idx} is not an object")
            continue
        source = edge.get("source")
        target = edge.get("target")
        edge_type = edge.get("edge_type")
        # Support serialized models that use source_id/target_id as well.
        source = source if isinstance(source, str) else edge.get("source_id")
        target = target if isinstance(target, str) else edge.get("target_id")
        if not isinstance(source, str) or not isinstance(target, str) or not isinstance(edge_type, str):
            errors.append(f"{sid}: graph edge #{idx} lacks source, target, or edge_type")
            continue
        if source not in node_ids:
            errors.append(f"{sid}: edge #{idx} source {source!r} is not a graph node")
        if target not in node_ids:
            errors.append(f"{sid}: edge #{idx} target {target!r} is not a graph node")
        edge_keys.add((source, target, edge_type))

    expected_path = truth.get("expected_attack_path", [])
    if not isinstance(expected_path, list) or any(not isinstance(item, str) for item in expected_path):
        errors.append(f"{sid}: expected_attack_path must be a list of strings")
        expected_path = []
    if expected_path and len(expected_path) < 2:
        errors.append(f"{sid}: non-empty expected_attack_path must contain at least 2 entries")
    if not isinstance(truth.get("attack_path_complete"), bool):
        errors.append(f"{sid}: attack_path_complete must be boolean")

    assessments = truth.get("edge_assessments")
    if not isinstance(assessments, list) or not assessments:
        errors.append(f"{sid}: edge_assessments must be a non-empty list")
        assessments = []
    seen_assessments: dict[tuple[str, str, str], str] = {}
    for idx, assessment in enumerate(assessments):
        if not isinstance(assessment, dict):
            errors.append(f"{sid}: edge assessment #{idx} is not an object")
            continue
        source, target, edge_type = (
            assessment.get("source"), assessment.get("target"), assessment.get("edge_type")
        )
        status = assessment.get("status")
        if not all(isinstance(x, str) for x in (source, target, edge_type)):
            errors.append(f"{sid}: edge assessment #{idx} lacks source/target/edge_type strings")
            continue
        if status not in {"positive", "negative", "unknown"}:
            errors.append(f"{sid}: edge assessment #{idx} has invalid status {status!r}")
            continue
        key = (source, target, edge_type)
        if key in seen_assessments:
            errors.append(f"{sid}: duplicate assessment for edge {key}")
        seen_assessments[key] = status
        if key not in edge_keys:
            errors.append(f"{sid}: assessed edge {key} is not present in the graph")
        if not isinstance(assessment.get("justification"), str) or not assessment["justification"].strip():
            errors.append(f"{sid}: edge assessment #{idx} needs a non-empty justification")

    reference = manifest.get("ground_truth_reference")
    if not isinstance(reference, str) or not reference:
        errors.append(f"{sid}: manifest ground_truth_reference is missing")
    else:
        resolved = (SCENARIO_DIR / reference).resolve()
        if resolved != truth_path.resolve():
            # References may be just the filename or a relative path from repo root.
            candidates = {(SCENARIO_DIR / truth_path.name).resolve(), (ROOT / reference).resolve()}
            if resolved not in candidates and (ROOT / reference).resolve() != truth_path.resolve():
                errors.append(f"{sid}: manifest ground_truth_reference does not resolve to {truth_path.name}")

    family = manifest.get("scenario_family")
    if not isinstance(family, str) or not family:
        errors.append(f"{sid}: manifest scenario_family is missing")
    if manifest.get("environment") != "isolated_docker_lab":
        errors.append(f"{sid}: manifest environment must be 'isolated_docker_lab'")

    return {
        "scenario_id": sid,
        "nodes": len(nodes),
        "edges": len(edges),
        "family": family if isinstance(family, str) else "unknown",
        "assessments": assessments,
        "edge_keys": edge_keys,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=3, help="first scenario number (default: 3)")
    parser.add_argument("--count", type=int, default=12, help="number of consecutive scenarios (default: 12)")
    parser.add_argument(
        "--strict-labels", action="store_true",
        help="exit non-zero when label-quality warnings are found (structural checks still run)",
    )
    args = parser.parse_args()
    if args.start < 1 or args.count < 1:
        parser.error("--start and --count must be positive")

    errors: list[str] = []
    warnings: list[str] = []
    records = [
        record
        for number in range(args.start, args.start + args.count)
        if (record := validate_one(number, errors, warnings)) is not None
    ]

    print("AutoShield Phase 7 dataset validation (read-only)")
    print(f"Repository: {ROOT}")
    print(f"Expected scenarios: {args.count} ({scenario_id(args.start)}–{scenario_id(args.start + args.count - 1)})")
    print(f"Complete artifact triples: {len(records)}/{args.count}")

    if records:
        print("\nScenario summary:")
        print(f"{'Scenario':<15} {'Family':<24} {'Nodes':>5} {'Edges':>5} {'Labels':>6}")
        for record in records:
            print(
                f"{record['scenario_id']:<15} {record['family']:<24} "
                f"{record['nodes']:>5} {record['edges']:>5} {len(record['assessments']):>6}"
            )

    # The current generator assigns four consecutive IDs to each split, but it
    # does not persist split membership in each manifest. Make that limitation explicit.
    if args.start == 3 and args.count == 12:
        inferred = {
            "train": list(range(3, 7)),
            "validation": list(range(7, 11)),
            "test": list(range(11, 15)),
        }
        print("\nSplit note:")
        print("  Current generator's split assignment is inferred from scenario ID ranges:")
        for split, numbers in inferred.items():
            print(f"  {split}: {', '.join(scenario_id(n) for n in numbers)}")
        warnings.append(
            "Split membership is not stored as an explicit field in the scenario artifacts; "
            "the displayed train/validation/test assignment is inferred from the generator's ID order."
        )

    # Detect a suspicious pattern: the same edge is called positive in some
    # scenarios and negative in others. That may be valid only if scenario-specific
    # evidence supports the distinction; it should never be assumed automatically.
    edge_statuses: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    edge_type_statuses: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for assessment in record["assessments"]:
            if isinstance(assessment, dict) and assessment.get("status") in {"positive", "negative"}:
                source, target, edge_type = (
                    assessment.get("source"), assessment.get("target"), assessment.get("edge_type")
                )
                if all(isinstance(x, str) for x in (source, target, edge_type)):
                    edge_statuses[(source, target, edge_type)].add(assessment["status"])
                    edge_type_statuses[edge_type].add(assessment["status"])

    for key, statuses in edge_statuses.items():
        if {"positive", "negative"} <= statuses:
            warnings.append(
                f"Same graph edge has both positive and negative labels across scenarios: {key}. "
                "Verify that labels are scenario-specific and evidence-backed."
            )

    if edge_type_statuses.get("forwards") == {"positive", "negative"}:
        warnings.append(
            "The 'forwards' relation is labeled both positive and negative. A shared service-forwarding "
            "edge alone does not establish attack versus benign behavior; review those labels manually."
        )

    all_families = Counter(record["family"] for record in records)
    print("\nScenario families:")
    for family, count in sorted(all_families.items()):
        print(f"  {family}: {count}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for message in errors:
            print(f"  [ERROR] {message}")
    else:
        print("\nStructural checks: PASS")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for message in warnings:
            print(f"  [WARN] {message}")
    else:
        print("Label-quality warnings: none detected by these heuristics")

    print("\nImportant: a structural pass is not proof that attack-path labels are semantically correct.")
    print("This validator does not train a model and does not modify any files.")

    if errors or (args.strict_labels and warnings):
        print("\nRESULT: FAIL")
        return 1
    if warnings:
        print("\nRESULT: STRUCTURE CHECKED — REVIEW WARNINGS BEFORE TRAINING")
    else:
        print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
