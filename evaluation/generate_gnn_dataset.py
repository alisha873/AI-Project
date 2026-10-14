#!/usr/bin/env python3
"""Generate 12 isolated AutoShield GNN scenarios from the local Docker lab.

From repository root:
  PYTHONPATH=. python evaluation/generate_gnn_dataset.py --dry-run
  PYTHONPATH=. python evaluation/generate_gnn_dataset.py

Refuses to overwrite incident/scenario artifacts. Uses only localhost HTTP requests.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8080"
CAPTURE = ROOT / "infrastructure/docker/scripts/create_incident.sh"
SCENARIO_DIR = ROOT / "evaluation/scenarios"

@dataclass(frozen=True)
class Case:
    family: str
    slug: str
    description: str
    method: str
    path: str
    target: str
    attack: bool
    split: str
    body: dict[str, Any] | None = None

# The three groups are attack-attempt scenarios, not claims of successful exploitation.
# Each family contributes one scenario to train, validation, and test.

CASES = (
    # TRAIN: 8 scenarios
    Case("sql_injection", "sqli_boolean", "Boolean SQL injection probe", "GET", "/api/search?message=%27%20OR%20%271%27%3D%271", "api", True, "train"),
    Case("authentication_abuse", "auth_invalid_1", "Invalid login attempt", "POST", "/rest/user/login", "juice-shop", True, "train", {"email":"dataset-invalid-1@example.invalid","password":"not-a-real-password"}),
    Case("web_exploitation", "xss_probe", "XSS-shaped query", "GET", "/?q=%3Cscript%3Ealert(1)%3C%2Fscript%3E", "juice-shop", True, "train"),
    Case("benign", "benign_home", "Ordinary homepage request", "GET", "/", "juice-shop", False, "train"),
    Case("sql_injection", "sqli_numeric", "Numeric SQL injection-shaped input", "GET", "/api/search?message=1%20OR%201%3D1", "api", True, "train"),
    Case("authentication_abuse", "auth_invalid_3", "Different invalid account login", "POST", "/rest/user/login", "juice-shop", True, "train", {"email":"dataset-invalid-3@example.invalid","password":"invalid-password"}),
    Case("web_exploitation", "traversal_encoded", "Encoded traversal-shaped request", "GET", "/ftp/..%252fpackage.json", "juice-shop", True, "train"),
    Case("benign", "benign_search_train", "Ordinary product search", "GET", "/rest/products/search?q=banana", "juice-shop", False, "train"),

    # VALIDATION: 8 scenarios
    Case("sql_injection", "sqli_comment", "SQL comment-style probe", "GET", "/api/search?message=%27%20OR%201%3D1%20--", "api", True, "validation"),
    Case("authentication_abuse", "auth_invalid_2", "Second invalid login attempt", "POST", "/rest/user/login", "juice-shop", True, "validation", {"email":"dataset-invalid-2@example.invalid","password":"not-a-real-password"}),
    Case("web_exploitation", "traversal_probe", "Path traversal-shaped request", "GET", "/ftp/..%2fpackage.json", "juice-shop", True, "validation"),
    Case("benign", "benign_api_search", "Ordinary API search", "GET", "/api/search?message=autoshield%20benign%20control", "api", False, "validation"),
    Case("sql_injection", "sqli_quote", "Quote-based SQL injection-shaped input", "GET", "/api/search?message=%27%20OR%20%27x%27%3D%27x", "api", True, "validation"),
    Case("authentication_abuse", "auth_empty", "Empty credential login attempt", "POST", "/rest/user/login", "juice-shop", True, "validation", {"email":"","password":""}),
    Case("web_exploitation", "encoded_script", "Encoded script-shaped query", "GET", "/?q=%3Csvg%20onload%3Dalert(1)%3E", "juice-shop", True, "validation"),
    Case("benign", "benign_search_validation", "Ordinary product search", "GET", "/rest/products/search?q=orange", "juice-shop", False, "validation"),

    # TEST: 8 scenarios
    Case("sql_injection", "sqli_union_text", "UNION-shaped SQL input", "GET", "/api/search?message=%27%20UNION%20SELECT%20NULL--", "api", True, "test"),
    Case("authentication_abuse", "auth_sql_payload", "SQL-shaped login input", "POST", "/rest/user/login", "juice-shop", True, "test", {"email":"' OR 1=1--","password":"invalid"}),
    Case("web_exploitation", "product_search_probe", "Special-character product search", "GET", "/rest/products/search?q=%27%3C%3E%22", "juice-shop", True, "test"),
    Case("benign", "benign_product_search", "Ordinary product search", "GET", "/rest/products/search?q=apple", "juice-shop", False, "test"),
    Case("sql_injection", "sqli_parenthesis", "Parenthesized SQL-shaped input", "GET", "/api/search?message=%27%29%20OR%20%281%3D1", "api", True, "test"),
    Case("authentication_abuse", "auth_unknown_user", "Unknown-user login attempt", "POST", "/rest/user/login", "juice-shop", True, "test", {"email":"unknown-dataset-user@example.invalid","password":"invalid"}),
    Case("web_exploitation", "encoded_traversal_test", "Alternative encoded traversal-shaped request", "GET", "/ftp/..%252f..%252fetc%252fhosts", "juice-shop", True, "test"),
    Case("benign", "benign_home_test", "Second ordinary homepage request", "GET", "/", "juice-shop", False, "test"),
)


def run(cmd: list[str], cwd: Path = ROOT) -> str:
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout.strip()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def choose_nginx_service() -> str:
    services = [
        x.strip()
        for x in run([
            "docker", "compose",
            "-f", "infrastructure/docker/docker-compose.yml",
            "config", "--services"
        ]).splitlines()
        if x.strip()
    ]
    matches = [x for x in services if x.lower() in {"nginx", "autoshield-nginx"}]
    if len(matches) != 1:
        matches = [x for x in services if "nginx" in x.lower()]
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely identify Nginx Compose service; found services: {services}")
    return matches[0]

def send(case: Case) -> dict[str, Any]:
    url = BASE_URL + case.path
    headers = {"User-Agent": "AutoShield-Dataset-Runner/1.0"}
    data = None
    if case.body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(case.body).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method=case.method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status = response.status
            response.read(256)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.read(256)
    # Server errors are recorded as scenario evidence rather than discarded.
    return {"method": case.method, "path": case.path, "status": status}

def assign_ids() -> list[tuple[str, str]]:
    pairs = []
    # scenario_002 is preserved because it was built from mixed historical logs.
    sid, iid = 3, 9
    for _ in CASES:
        while any((SCENARIO_DIR / f"scenario_{sid:03d}_{suffix}.json").exists()
                  for suffix in ("graph", "ground_truth", "manifest")):
            sid += 1
        while (ROOT / f"incident_{iid:03d}").exists():
            iid += 1
        pairs.append((f"scenario_{sid:03d}", f"incident_{iid:03d}"))
        sid += 1; iid += 1
    return pairs

def graph_to_dict(graph: Any) -> dict[str, Any]:
    if hasattr(graph, "model_dump"):
        return graph.model_dump(mode="json")
    return graph.dict()

def main() -> int:
    global BASE_URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="list actions without changing Docker or writing artifacts")
    parser.add_argument("--base-url", default=BASE_URL, help="must be the local HTTP lab endpoint")
    args = parser.parse_args()
    BASE_URL = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        parser.error("--base-url must use http://localhost or http://127.0.0.1")
    if not CAPTURE.is_file():
        raise FileNotFoundError(f"Capture script not found: {CAPTURE}")
    nginx = choose_nginx_service()
    ids = assign_ids()
    print(f"Root: {ROOT}\nTarget: {BASE_URL}\nNginx service: {nginx}\nScenarios: {len(CASES)} (8 per split)")
    if not args.dry_run:
        try:
            urllib.request.urlopen(BASE_URL + "/", timeout=5).close()
        except Exception as exc:
            raise RuntimeError(f"Lab is not reachable at {BASE_URL}: {exc}") from exc
    summary = []
    for n, (case, (scenario_id, incident_id)) in enumerate(zip(CASES, ids), 1):
        incident_dir = ROOT / incident_id
        paths = [SCENARIO_DIR / f"{scenario_id}_{s}.json" for s in ("graph", "ground_truth", "manifest")]
        if incident_dir.exists() or any(p.exists() for p in paths):
            raise FileExistsError(f"Refusing to overwrite {scenario_id} or {incident_id}")
        print(f"\n[{n}/24] {scenario_id} ({case.family}, {case.split}) — {case.description}")
        print(f"  {case.method} {BASE_URL}{case.path}")
        if args.dry_run:
            summary.append({"scenario_id": scenario_id, "incident_id": incident_id, "split": case.split, "dry_run": True})
            continue
        # Reset Nginx's container log so each capture has a bounded HTTP window.
        print(run([
            "docker", "compose",
            "-f", "infrastructure/docker/docker-compose.yml",
            "up", "-d", "--force-recreate", nginx
        ]))
        time.sleep(2)
        request_result = send(case)
        time.sleep(1)
        print(run(["bash", str(CAPTURE), incident_id]))

        sys.path.insert(0, str(ROOT))
        from evidence.pipeline import EvidencePipeline
        try:
            from evidence.writer import EvidenceEventWriter
        except ImportError:
            from evidence.formatter import EvidenceEventWriter
        from graph.builder import DeterministicGraphBuilder

        events = EvidencePipeline().process(incident_dir)
        normalized = incident_dir / "normalized" / "events.jsonl"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        EvidenceEventWriter().write(events, normalized)
        graph = DeterministicGraphBuilder().build(incident_dir)
        write_json(SCENARIO_DIR / f"{scenario_id}_graph.json", graph_to_dict(graph))

        target_id = f"service:{case.target}"
        def is_target(edge: Any) -> bool:
            target = edge.target.lower()
            wanted = case.target.lower()
            if wanted == "juice-shop":
                target_match = "juice" in target or "shop" in target
            else:
                target_match = target.endswith(":" + wanted)
            return edge.edge_type == "forwards" and edge.source.endswith(":nginx") and target_match
        edge = next((e for e in graph.edges if is_target(e)), None)
        if edge is None:
            raise RuntimeError(f"{scenario_id}: expected observed forwards edge service:nginx -> {target_id} absent; raw evidence kept, labels not written")
        target_container = "autoshield-api" if case.target == "api" else "autoshield-juice-shop"
        truth = {
            "scenario_id": scenario_id,
            "expected_attack_path": ["attacker", "autoshield-nginx", target_container] if case.attack else [],
            "attack_path_complete": True,
            "edge_assessments": [{
                "source": edge.source,
                "target": edge.target,
                "edge_type": edge.edge_type,
                "status": "unknown",
                "justification": (
                    "The observed forwarding edge establishes routing to the target service, "
                    "but does not establish whether the request exploited a vulnerability "
                    "or represents a malicious attack-path transition."
                ),
            }],
        }
        write_json(SCENARIO_DIR / f"{scenario_id}_ground_truth.json", truth)
        manifest = {
            "schema_version": "1.0",
            "scenario_id": scenario_id,
            "scenario_family": case.family,
            "split": case.split,
            "description": case.description,
            "environment": "isolated_docker_lab",
            "execution": {"method": "scripted"},
            "target_allowlist": [BASE_URL],
            "ground_truth_reference": f"{scenario_id}_ground_truth.json",
        }
        write_json(SCENARIO_DIR / f"{scenario_id}_manifest.json", manifest)
        summary.append({"scenario_id": scenario_id, "incident_id": incident_id, "family": case.family,
                        "split": case.split, "request_status": request_result["status"],
                        "nodes": len(graph.nodes), "edges": len(graph.edges),
                        "label": "positive" if case.attack else "negative"})
    
    split_counts = {
        split: sum(1 for item in summary if item["split"] == split)
        for split in ("train", "validation", "test")
    }
    report = {
        "dataset": "AutoShield GNN controlled scenarios",
        "scenario_count": len(summary),
        "dry_run": args.dry_run,
        "split_counts": split_counts,
        "results": summary,
        "caveats": [
            "Attack cases are controlled attempts, not confirmed exploitation.",
            "Forwarding-edge labels remain unknown unless evidence supports a stronger claim.",
            "Dataset size alone does not establish model generalization.",
        ],
    }

    print("\n" + json.dumps(report, indent=2))
    if not args.dry_run:
        write_json(SCENARIO_DIR / "dataset_generation_summary.json", report)
        write_json(SCENARIO_DIR / "gnn_splits.json", [
            {"scenario_id": item["scenario_id"], "split": item["split"]} for item in summary
        ])
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
