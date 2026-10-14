#!/usr/bin/env python3
"""Run AutoShield Phase 8 (incident reconstruction) and Phase 9 (root cause).

Usage from repository root:
    PYTHONPATH=. python scripts/run_incident_analysis.py incident_001

The first version is deterministic and evidence-first: it creates auditable JSON
outputs before adding LLM synthesis. It never reads ground_truth.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def normalized_request_query(event: dict[str, Any]) -> str:
    """Normalize encoding only; do not reorder or otherwise rewrite query semantics."""
    attrs = event.get("attributes") or {}
    raw = attrs.get("query")
    if raw is None:
        return ""
    return unquote(unquote(str(raw))).strip()


def timestamp_delta_seconds(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    """Return absolute timestamp distance when both timestamps parse."""
    def parse(value: Any):
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    a = parse(left.get("timestamp"))
    b = parse(right.get("timestamp"))
    if a is None or b is None:
        return None
    # Treat naive timestamps as UTC only to permit comparison; record that the
    # correlation remains a candidate rather than a confirmed identity.
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    return abs((a - b).total_seconds())


def jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def resolve_incident_dir(incident_id: str) -> Path:
    if not re.fullmatch(r"incident_[0-9]{3}", incident_id):
        raise ValueError("Incident ID must match incident_NNN, e.g. incident_001.")

    candidates = [
        ROOT / incident_id,
        ROOT / "incidents" / incident_id,
        ROOT / "data" / "incidents" / incident_id,
        ROOT / "infrastructure" / "incidents" / incident_id,
    ]
    matches = [p.resolve() for p in candidates if (p / "docker" / "containers_after_attack.json").is_file()]

    if not matches:
        for path in ROOT.rglob("containers_after_attack.json"):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            candidate = path.parent.parent.resolve()
            if candidate.name == incident_id and (candidate / "docker" / "network_after_attack.json").is_file():
                matches.append(candidate)

    unique = sorted(set(matches))
    if not unique:
        raise FileNotFoundError(
            f"Could not locate {incident_id}/docker/containers_after_attack.json under {ROOT}."
        )
    if len(unique) > 1:
        raise RuntimeError(
            f"Found multiple directories for {incident_id}: "
            + ", ".join(str(p) for p in unique)
            + ". Pass --incident-dir to choose one."
        )
    return unique[0]


def load_or_normalize_events(incident_dir: Path) -> list[dict[str, Any]]:
    """Reuse normalized events when present; otherwise call the existing pipeline."""
    events_path = incident_dir / "normalized" / "events.jsonl"
    if not events_path.is_file():
        from evidence.pipeline import EvidencePipeline

        events = EvidencePipeline().process(incident_dir)
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(jsonable(event), sort_keys=True) + "\n")

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {events_path}:{line_number}") from exc
        if not isinstance(item, dict) or not item.get("event_id"):
            raise ValueError(f"Normalized event at line {line_number} has no event_id.")
        events.append(item)
    return events


def load_graph(incident_dir: Path) -> dict[str, Any]:
    from graph.builder import DeterministicGraphBuilder

    graph = DeterministicGraphBuilder().build(incident_dir)
    return jsonable(graph)


def event_text(event: dict[str, Any]) -> str:
    attrs = event.get("attributes") or {}
    values = [
        event.get("source", ""),
        event.get("event_type", ""),
        event.get("action", ""),
        event.get("source_entity", ""),
        event.get("destination_entity", ""),
        attrs.get("method", ""),
        attrs.get("path", ""),
        attrs.get("query", ""),
        attrs.get("url", ""),
        attrs.get("request", ""),
        attrs.get("message", ""),
        attrs.get("user_agent", ""),
    ]
    return unquote(unquote(" ".join(str(v) for v in values if v is not None))).lower()


def is_sqli_like(event: dict[str, Any]) -> bool:
    attrs = event.get("attributes") or {}
    text = event_text(event)
    request_text = " ".join(
        str(attrs.get(k, "")) for k in ("query", "path", "url", "request", "message", "body")
    ).lower()
    request_text = unquote(unquote(request_text))
    if not request_text.strip():
        request_text = text
    patterns = (
        r"\bunion\s+(?:all\s+)?select\b",
        r"\bor\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+",
        r"'\s*or\s*'[^']*'\s*=\s*'[^']*",
        r"\bwaitfor\s+delay\b",
        r"\bsleep\s*\(",
        r";\s*drop\s+table\b",
        r"--\s*(?:$|\w)",
    )
    return any(re.search(pattern, request_text, re.IGNORECASE) for pattern in patterns)


def stage_for_event(event: dict[str, Any], suspicious: bool) -> tuple[str, str]:
    """Return a conservative activity label and whether it is a confirmed attack stage."""
    attrs = event.get("attributes") or {}
    kind = str(event.get("event_type", "")).lower()
    source = str(event.get("source", "")).lower()
    action = str(event.get("action", "")).lower()

    if suspicious:
        return "exploitation_attempt_indicator", "attempt_indicator_observed_success_unconfirmed"
    if any(word in kind + " " + source + " " + action for word in ("auth", "login", "ssh")):
        return "authentication_activity_observed", "activity_observed_outcome_requires_evidence"
    if attrs.get("method") or attrs.get("path") or "http" in kind:
        return "web_request_observed", "activity_observed"
    if any(word in kind + " " + source for word in ("packet", "network", "pcap", "tcp", "udp")):
        return "network_activity_observed", "activity_observed"
    if any(word in kind + " " + source for word in ("docker", "container", "process")):
        return "system_state_or_process_event", "activity_observed"
    return "unclassified_event", "observed_event_stage_unknown"


def node_index(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes = graph.get("nodes", [])
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    return {str(node.get("node_id")): node for node in nodes if node.get("node_id")}


def reconstruct_incident(
    incident_id: str,
    events: list[dict[str, Any]],
    graph: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda e: (str(e.get("timestamp", "")), str(e.get("event_id", ""))))
    event_ids = {str(e["event_id"]) for e in ordered}
    suspicious_ids = {str(e["event_id"]) for e in ordered if is_sqli_like(e)}

    timeline = []
    claims = []
    for event in ordered:
        event_id = str(event["event_id"])
        suspicious = event_id in suspicious_ids
        stage, interpretation = stage_for_event(event, suspicious)
        attrs = event.get("attributes") or {}
        is_http = str(event.get("event_type", "")).lower() == "http_request" or bool(
            attrs.get("method") or attrs.get("path")
        )
        request_context = None
        if is_http:
            request_context = {
                "method": attrs.get("method"),
                "path": attrs.get("path"),
                "query": attrs.get("query"),
                "status": attrs.get("status"),
                "upstream": attrs.get("upstream"),
                "upstream_status": attrs.get("upstream_status"),
                "request_id": attrs.get("request_id"),
            }
        timeline.append({
            "timestamp": event.get("timestamp"),
            "event_id": event_id,
            "source": event.get("source"),
            "event_type": event.get("event_type"),
            "action": event.get("action"),
            "source_entity": event.get("source_entity"),
            "destination_entity": event.get("destination_entity"),
            "stage_label": stage,
            "interpretation_status": interpretation,
            "request_context": request_context,
            "attributes": attrs,
        })
        if suspicious:
            location = attrs.get("path") or attrs.get("url") or "request data"
            claims.append({
                "claim_id": f"claim_{len(claims) + 1:03d}",
                "statement": f"Event {event_id} contains SQL-injection-like syntax in {location}.",
                "status": "observed_indicator",
                "evidence_ids": [event_id],
                "observed_at": event.get("timestamp"),
                "source": event.get("source"),
                "request_context": request_context,
                "confidence": 0.65,
                "limitation": "A request pattern alone does not prove successful exploitation or unsafe query construction.",
            })

    # Cross-source request correlation is deliberately conservative. Matching
    # payloads and near-identical timestamps are candidate evidence, not proof that
    # two log records describe the same request. A shared request ID is stronger.
    http_events = []
    for event in ordered:
        attrs = event.get("attributes") or {}
        if str(event.get("event_type", "")).lower() == "http_request" or attrs.get("method") or attrs.get("path"):
            http_events.append(event)

    event_correlations = []
    correlation_ids_by_event: dict[str, list[str]] = {str(e["event_id"]): [] for e in ordered}
    for i, left in enumerate(http_events):
        for right in http_events[i + 1:]:
            left_id, right_id = str(left["event_id"]), str(right["event_id"])
            if left_id not in suspicious_ids and right_id not in suspicious_ids:
                continue
            if str(left.get("source", "")) == str(right.get("source", "")):
                continue

            left_attrs = left.get("attributes") or {}
            right_attrs = right.get("attributes") or {}
            left_method = str(left_attrs.get("method") or "").upper()
            right_method = str(right_attrs.get("method") or "").upper()
            left_query = normalized_request_query(left)
            right_query = normalized_request_query(right)
            delta = timestamp_delta_seconds(left, right)
            same_request_id = bool(
                left_attrs.get("request_id")
                and right_attrs.get("request_id")
                and str(left_attrs.get("request_id")) == str(right_attrs.get("request_id"))
            )
            same_query = bool(left_query) and left_query == right_query
            same_method = bool(left_method) and left_method == right_method
            near_in_time = delta is not None and delta <= 5.0

            # Emit only explainable candidates; don't correlate merely because the
            # events are nearby in time.
            if not same_request_id and not (same_query and same_method and near_in_time):
                continue

            correlation_id = f"correlation_{len(event_correlations) + 1:03d}"
            if same_request_id:
                status = "shared_request_id_candidate"
                strength = "stronger_candidate"
            else:
                status = "candidate_not_confirmed"
                strength = "strong_candidate" if same_query and same_method and near_in_time else "weak_candidate"

            left_context = {
                "method": left_attrs.get("method"),
                "path": left_attrs.get("path"),
                "query": left_attrs.get("query"),
                "status": left_attrs.get("status"),
                "upstream": left_attrs.get("upstream"),
                "upstream_status": left_attrs.get("upstream_status"),
                "request_id": left_attrs.get("request_id"),
            }
            right_context = {
                "method": right_attrs.get("method"),
                "path": right_attrs.get("path"),
                "query": right_attrs.get("query"),
                "status": right_attrs.get("status"),
                "upstream": right_attrs.get("upstream"),
                "upstream_status": right_attrs.get("upstream_status"),
                "request_id": right_attrs.get("request_id"),
            }
            signals = []
            if same_request_id:
                signals.append("request_id_matches")
            if same_query:
                signals.append("decoded_query_matches_exactly")
            if same_method:
                signals.append("http_method_matches")
            if near_in_time:
                signals.append("timestamps_within_5_seconds")
            if left_context["path"] != right_context["path"]:
                signals.append("paths_differ_possible_proxy_rewrite_not_assumed")

            event_correlations.append({
                "correlation_id": correlation_id,
                "status": status,
                "strength": strength,
                "event_ids": [left_id, right_id],
                "events": [
                    {
                        "event_id": left_id,
                        "timestamp": left.get("timestamp"),
                        "source": left.get("source"),
                        "request_context": left_context,
                    },
                    {
                        "event_id": right_id,
                        "timestamp": right.get("timestamp"),
                        "source": right.get("source"),
                        "request_context": right_context,
                    },
                ],
                "signals": signals,
                "timestamp_delta_seconds": round(delta, 6) if delta is not None else None,
                "decision_rule": (
                    "Candidate correlation requires either a shared non-empty request_id, "
                    "or an exact decoded query match plus same HTTP method and timestamps "
                    "within five seconds across different sources."
                ),
                "limitation": (
                    "This is a candidate correlation, not proof of request identity. "
                    "Different paths may reflect proxy rewriting, but that transformation "
                    "must be verified from proxy configuration or a request ID before being asserted."
                ),
            })
            correlation_ids_by_event[left_id].append(correlation_id)
            correlation_ids_by_event[right_id].append(correlation_id)

    for entry in timeline:
        entry["correlation_ids"] = correlation_ids_by_event.get(str(entry["event_id"]), [])

    nodes = node_index(graph)
    edges = graph.get("edges", [])
    if isinstance(edges, dict):
        edges = list(edges.values())
    observed_relationships = []
    configured_relationships = []
    for edge in edges:
        basis = str(edge.get("basis", "unknown")).lower()
        evidence_ids = sorted(set(str(x) for x in edge.get("evidence_ids", []) if x))
        row = {
            "source": edge.get("source"),
            "source_name": nodes.get(str(edge.get("source")), {}).get("name", edge.get("source")),
            "target": edge.get("target"),
            "target_name": nodes.get(str(edge.get("target")), {}).get("name", edge.get("target")),
            "relationship": edge.get("edge_type"),
            "basis": basis,
            "evidence_ids": evidence_ids,
            "source_artifacts": sorted(set(str(x) for x in edge.get("source_artifacts", []) if x)),
            "attributes": edge.get("attributes") or {},
            "aggregation_note": (
                "This edge aggregates multiple evidence events. Its attributes may describe only one "
                "event and must not be interpreted as request-specific details."
            ) if len(evidence_ids) > 1 else None,
        }
        if basis == "observed":
            observed_relationships.append(row)
        elif basis == "configured":
            configured_relationships.append(row)

    # Correlate a suspicious request with a graph relationship only by exact event ID.
    # Use the normalized event's own fields for request details; never copy details from
    # an aggregated edge's attributes into a per-request claim.
    request_level_relationships = []
    event_by_id = {str(event["event_id"]): event for event in ordered}
    for event_id in sorted(suspicious_ids):
        event = event_by_id[event_id]
        attrs = event.get("attributes") or {}
        matching_edges = [
            edge for edge in edges
            if str(edge.get("basis", "")).lower() == "observed"
            and event_id in {str(x) for x in edge.get("evidence_ids", [])}
        ]
        for edge in matching_edges:
            request_level_relationships.append({
                "event_id": event_id,
                "timestamp": event.get("timestamp"),
                "relationship": edge.get("edge_type"),
                "source": edge.get("source"),
                "source_name": nodes.get(str(edge.get("source")), {}).get("name", edge.get("source")),
                "target": edge.get("target"),
                "target_name": nodes.get(str(edge.get("target")), {}).get("name", edge.get("target")),
                "basis": "observed",
                "request_context": {
                    "method": attrs.get("method"),
                    "path": attrs.get("path"),
                    "query": attrs.get("query"),
                    "status": attrs.get("status"),
                    "upstream": attrs.get("upstream"),
                    "upstream_status": attrs.get("upstream_status"),
                    "request_id": attrs.get("request_id"),
                },
                "evidence_ids": [event_id],
                "relationship_evidence_note": (
                    "The graph edge cites this event ID, establishing event-to-edge provenance; "
                    "the relationship does not by itself prove successful exploitation or impact."
                ),
            })

    event_type_counts = Counter(str(e.get("event_type", "unknown")) for e in ordered)
    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "agent": "phase_8_incident_reconstruction",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evidence_summary": {
            "normalized_event_count": len(ordered),
            "graph_node_count": len(nodes),
            "graph_edge_count": len(edges),
            "event_type_counts": dict(sorted(event_type_counts.items())),
            "sqli_like_indicator_event_ids": sorted(suspicious_ids),
            "ground_truth_used": False,
        },
        "timeline": timeline,
        "event_correlations": event_correlations,
        "correlation_policy": {
            "status_vocabulary": {
                "candidate_not_confirmed": "Multiple independent fields match, but event identity is not proven.",
                "shared_request_id_candidate": "A shared request ID is present; validate its uniqueness and logging semantics before treating it as definitive.",
            },
            "no_silent_merges": True,
            "path_rewrites_are_not_assumed": True,
        },
        "claims": claims,
        "graph_relationships": {
            "observed": observed_relationships,
            "configured_only": configured_relationships,
            "request_level_relationships_for_suspicious_indicators": request_level_relationships,
        },
        "attack_chain_assessment": {
            "confirmed_attack_transitions": [],
            "observed_relationships": observed_relationships,
            "configured_impact_hypotheses": configured_relationships,
            "request_level_relationships_for_suspicious_indicators": request_level_relationships,
            "event_correlations": event_correlations,
            "note": (
                "Graph relationships are not automatically attack transitions. "
                "Cross-source event correlations are explicitly marked as candidates unless "
                "stronger request-identity evidence is available. No confirmed multi-stage "
                "attack chain is asserted unless the evidence directly establishes it."
            ),
        },
        "unconfirmed_stages": [
            "initial access attribution",
            "successful exploitation",
            "privilege change",
            "lateral movement",
            "data access or exfiltration",
        ],
    }


def root_cause_analysis(
    incident_id: str,
    reconstruction: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    event_by_id = {str(e["event_id"]): e for e in events}
    indicator_ids = reconstruction["evidence_summary"]["sqli_like_indicator_event_ids"]

    if indicator_ids:
        supporting = [event_id for event_id in indicator_ids if event_id in event_by_id]
        finding = {
            "status": "hypothesis_not_confirmed",
            "root_cause": "Not confirmed from the available evidence.",
            "affected_component": "Undetermined from request evidence alone.",
            "vulnerability_type": "potential_sql_injection (CWE-89 hypothesis)",
            "supporting_evidence_ids": supporting,
            "confidence": 0.55,
            "rationale": (
                "One or more normalized events contain SQL-injection-like request syntax. "
                "This supports an injection-attempt hypothesis but does not establish vulnerable "
                "server-side query construction, successful exploitation, or database access."
            ),
            "evidence_needed_to_confirm": [
                "application source or query-construction evidence showing unsafe SQL handling",
                "correlated application/database logs demonstrating the query was executed",
                "request-to-response evidence tied to the same event or request identifier",
            ],
        }
    else:
        finding = {
            "status": "insufficient_evidence",
            "root_cause": "Unknown; the available evidence does not support a specific root-cause finding.",
            "affected_component": None,
            "vulnerability_type": "unknown",
            "supporting_evidence_ids": [],
            "confidence": 0.0,
            "rationale": "No supported vulnerability-specific indicator was identified by the deterministic baseline.",
            "evidence_needed_to_confirm": [
                "incident-specific application and service logs",
                "relevant source/configuration evidence",
                "correlated evidence linking the suspected behavior to impact",
            ],
        }

    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "agent": "phase_9_root_cause",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_reconstruction": "incident_reconstruction.json",
        "findings": [finding],
        "validation": {
            "all_supporting_event_ids_exist": all(
                event_id in event_by_id for event_id in finding["supporting_evidence_ids"]
            ),
            "ground_truth_used": False,
            "principle": "No evidence, no claim.",
        },
    }


def validate_outputs(
    reconstruction: dict[str, Any],
    root_cause: dict[str, Any],
    events: list[dict[str, Any]],
) -> None:
    valid_ids = {str(e["event_id"]) for e in events}
    for claim in reconstruction.get("claims", []):
        invalid = set(claim.get("evidence_ids", [])) - valid_ids
        if invalid:
            raise ValueError(f"Reconstruction claim contains unknown event IDs: {sorted(invalid)}")
    for finding in root_cause.get("findings", []):
        invalid = set(finding.get("supporting_evidence_ids", [])) - valid_ids
        if invalid:
            raise ValueError(f"Root-cause finding contains unknown event IDs: {sorted(invalid)}")
        if finding.get("status") == "confirmed" and not finding.get("supporting_evidence_ids"):
            raise ValueError("A confirmed root-cause finding must cite supporting evidence.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("incident_id", help="Incident ID, e.g. incident_001")
    parser.add_argument(
        "--incident-dir",
        type=Path,
        help="Optional explicit incident directory if it is outside the standard locations.",
    )
    args = parser.parse_args()

    try:
        incident_dir = args.incident_dir.resolve() if args.incident_dir else resolve_incident_dir(args.incident_id)
        if incident_dir.name != args.incident_id:
            raise ValueError(f"--incident-dir basename must be {args.incident_id!r}.")
        if not (incident_dir / "docker" / "containers_after_attack.json").is_file():
            raise FileNotFoundError(f"Missing Docker snapshot in {incident_dir}")

        events = load_or_normalize_events(incident_dir)
        graph = load_graph(incident_dir)
        reconstruction = reconstruct_incident(args.incident_id, events, graph)
        root_cause = root_cause_analysis(args.incident_id, reconstruction, events)
        validate_outputs(reconstruction, root_cause, events)

        output_dir = incident_dir / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        reconstruction_path = output_dir / "incident_reconstruction.json"
        root_cause_path = output_dir / "root_cause_analysis.json"
        reconstruction_path.write_text(json.dumps(reconstruction, indent=2) + "\n", encoding="utf-8")
        root_cause_path.write_text(json.dumps(root_cause, indent=2) + "\n", encoding="utf-8")

        print(json.dumps({
            "status": "success",
            "incident_id": args.incident_id,
            "normalized_events": len(events),
            "graph_nodes": reconstruction["evidence_summary"]["graph_node_count"],
            "graph_edges": reconstruction["evidence_summary"]["graph_edge_count"],
            "timeline_entries": len(reconstruction["timeline"]),
            "evidence_linked_claims": len(reconstruction["claims"]),
            "root_cause_status": root_cause["findings"][0]["status"],
            "outputs": [
                str(reconstruction_path.relative_to(incident_dir)),
                str(root_cause_path.relative_to(incident_dir)),
            ],
            "mode": "deterministic_evidence_first_baseline",
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"Incident analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
