#!/usr/bin/env python
"""
Read-only verifier for benchmark_11c.json (AutoShield Phase 11, Milestone 11C preflight).

What it does
------------
* Rebuilds the 23-record MITRE corpus IN MEMORY from the pinned archives (the
  same builder the repository already uses) and recomputes the corpus
  fingerprint exactly as `retrieval.evaluation.corpus_fingerprint` does.
* Checks the benchmark file structure, unique/valid IDs, categories, labels,
  evidence quotes (verbatim, in the stated section), exclusivity checks,
  declared cross-lane links, absence checks for behaviour cases, and
  duplicates against the v1/v2 benchmarks.
* Prints the SHA-256 of the benchmark file.

What it never does
------------------
* It never runs a retriever. It does not import `retrieval`, FTS5, any
  embedding model or the knowledge store; it asserts that none are loaded.
  Phrase searches are plain substring checks over the source text, used only
  to verify labelling claims against the sources.
* It writes nothing to disk.

Usage (repository root):
    .\\.venv\\Scripts\\python.exe -B data/retrieval_benchmark/preflight_11c/verify_benchmark_11c.py
Exit code 0 = all checks passed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark_11c.json"
V1 = ROOT / "data/retrieval_benchmark/retrieval_benchmark_v1.json"
V2 = ROOT / "data/retrieval_benchmark/retrieval_benchmark_v2.json"
sys.path.insert(0, str(ROOT))

CATEGORIES = {
    "forensic_observation", "terminology", "weakness_consequence", "attack_technique",
    "mitigation_detection", "disambiguation", "identifier_lookup", "low_information", "unanswerable",
}
QUERY_STYLES = {"descriptive", "terminology", "short", "identifier"}
EXPECTATIONS = {"relevant_documents", "low_information", "no_relevant_documents"}
IDENT_RE = re.compile(r"\b(CWE|CAPEC|CVE)[-_ \u2010-\u2015\u2212]([0-9]+)", re.IGNORECASE)
FP_RE = re.compile(r"[0-9a-f]{64}")

errors: list[str] = []


def fail(msg: str) -> None:
    errors.append(msg)


def norm(text: str) -> str:
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", text).strip()


def split_sections(content: str) -> dict[str, str]:
    parts = re.split(r"(?m)^## ", content)
    out = {"__head__": parts[0]}
    for part in parts[1:]:
        name, _, body = part.partition("\n")
        out[name.strip()] = body
    return out


def load_corpus():
    from ingestion.mitre_corpus import build_corpus_documents, load_manifest

    manifest = load_manifest(ROOT / "data/knowledge_sources/manifest.json")
    return build_corpus_documents(
        manifest,
        cwe_archive=ROOT / "data/knowledge_sources/raw/cwec_v4.20.xml.zip",
        capec_archive=ROOT / "data/knowledge_sources/raw/capec_v3.9.zip",
    )


def ident_of(doc_id: str) -> str:
    return doc_id.split(":", 1)[1]  # "CWE-78"


def main() -> int:
    raw = BENCH.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    bench = json.loads(raw.decode("utf-8"))

    docs = {d.doc_id: d for d in load_corpus()}
    doc_ids = set(docs)
    lines = [f"{i}\t{docs[i].document_sha256}" for i in sorted(doc_ids)]
    fingerprint = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    sections = {i: split_sections(docs[i].content) for i in doc_ids}
    norm_full = {i: norm(docs[i].content) for i in doc_ids}
    norm_sec = {i: {k: norm(v) for k, v in s.items()} for i, s in sections.items()}

    # -- top level -------------------------------------------------------
    if bench.get("format") != "autoshield.retrieval_benchmark.v3":
        fail("format must be autoshield.retrieval_benchmark.v3")
    corpus = bench.get("corpus", {})
    if corpus.get("document_count") != len(doc_ids):
        fail(f"corpus.document_count {corpus.get('document_count')} != {len(doc_ids)}")
    if not FP_RE.fullmatch(str(corpus.get("corpus_fingerprint", ""))):
        fail("corpus.corpus_fingerprint is not a 64-hex SHA-256")
    elif corpus["corpus_fingerprint"] != fingerprint:
        fail(f"corpus fingerprint mismatch: file {corpus['corpus_fingerprint']} vs recomputed {fingerprint}")
    if bench.get("review_summary", {}).get("human_review") != "not_performed":
        fail("review_summary.human_review must be 'not_performed'")
    if bench.get("label_provenance", {}).get("retrieval_run_on_these_cases_before_freeze") is not False:
        fail("label_provenance.retrieval_run_on_these_cases_before_freeze must be false")

    cases = bench.get("cases", [])
    behaviour = bench.get("behaviour_cases", [])
    all_cases = cases + behaviour

    # -- IDs ------------------------------------------------------------
    ids = [c.get("case_id") for c in all_cases]
    if len(ids) != len(set(ids)):
        fail("duplicate case_id values")
    for cid in ids:
        if not isinstance(cid, str) or not re.fullmatch(r"RB3-[FB][0-9]{2}", cid):
            fail(f"bad case_id {cid!r}")

    # -- duplicates vs v1/v2 ---------------------------------------------
    previous: set[str] = set()
    for path in (V1, V2):
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data.get("cases") or data.get("questions") or []
        for item in items:
            q = item.get("question") or item.get("query")
            if q:
                previous.add(norm(q).casefold())
    for c in all_cases:
        if norm(c.get("question", "")).casefold() in previous:
            fail(f"{c['case_id']}: question duplicates a v1/v2 question")
    new_q = [norm(c["question"]).casefold() for c in all_cases]
    if len(new_q) != len(set(new_q)):
        fail("duplicate questions within v3")

    # -- per case ---------------------------------------------------------
    quote_count = 0
    exclusivity_count = 0
    lanes = {"cwe": 0, "capec": 0}
    for c in all_cases:
        cid = c.get("case_id", "?")
        for key, allowed in (("category", CATEGORIES), ("query_style", QUERY_STYLES), ("expectation", EXPECTATIONS)):
            if c.get(key) not in allowed:
                fail(f"{cid}: {key}={c.get(key)!r} not allowed")
        if c.get("split") != "final_heldout":
            fail(f"{cid}: split must be final_heldout")
        if not isinstance(c.get("question"), str) or not c["question"].strip():
            fail(f"{cid}: empty question")
        # identifiers in questions only for identifier_lookup
        has_ident = bool(IDENT_RE.search(c.get("question", "")))
        if has_ident and c.get("category") != "identifier_lookup":
            fail(f"{cid}: identifier in question but category is {c.get('category')}")
        if c.get("category") == "identifier_lookup" and not has_ident:
            fail(f"{cid}: identifier_lookup question has no identifier")

        rel = c.get("relevant_doc_ids", [])
        if len(rel) != len(set(rel)):
            fail(f"{cid}: duplicate relevant_doc_ids")
        for d in rel + c.get("lenient_extra_doc_ids", []):
            if d not in doc_ids:
                fail(f"{cid}: unknown doc id {d}")

        if c in behaviour:
            if rel:
                fail(f"{cid}: behaviour case must have no relevant docs")
            for phrase in c.get("absence_checks", []):
                hits = [i for i in doc_ids if phrase.casefold() in norm_full[i].casefold()]
                if hits:
                    fail(f"{cid}: absence check {phrase!r} found in {sorted(hits)}")
            continue

        if c.get("expectation") != "relevant_documents" or not rel:
            fail(f"{cid}: scored case needs expectation relevant_documents and labels")
        for d in rel:
            lanes[d.split(":")[0]] += 1
        # lenient extras
        for extra in c.get("lenient_extra_doc_ids", []):
            if extra in rel:
                fail(f"{cid}: lenient extra {extra} is also strict relevant")
            declared = False
            for strict in rel:
                if extra.split(":")[0] == strict.split(":")[0]:
                    continue
                for a, b in ((strict, extra), (extra, strict)):
                    rel_text = "\n".join(
                        v for k, v in sections[a].items() if k in ("Related Weaknesses", "Related Attack Patterns")
                    )
                    if re.search(rf"\b{re.escape(ident_of(b))}\b", rel_text):
                        declared = True
            if not declared:
                fail(f"{cid}: lenient extra {extra} is not a declared cross-lane link of {rel}")
        # evidence
        evid_docs = set()
        for ev in c.get("evidence", []):
            d, sec, quote = ev.get("doc_id"), ev.get("section"), ev.get("quote", "")
            evid_docs.add(d)
            quote_count += 1
            if d not in rel:
                fail(f"{cid}: evidence for {d}, which is not a relevant doc")
                continue
            if sec not in norm_sec[d]:
                fail(f"{cid}: section {sec!r} not found in {d}; have {sorted(norm_sec[d])}")
                continue
            if norm(quote) not in norm_sec[d][sec]:
                fail(f"{cid}: quote not found verbatim in {d} / {sec}: {quote!r}")
        for d in rel:
            if d not in evid_docs:
                fail(f"{cid}: relevant doc {d} has no evidence quote")
        # considered_not_relevant
        for item in c.get("considered_not_relevant", []):
            if item.get("doc_id") not in doc_ids:
                fail(f"{cid}: considered_not_relevant unknown doc {item.get('doc_id')}")
            if item.get("doc_id") in rel:
                fail(f"{cid}: {item.get('doc_id')} is both relevant and not relevant")
            if not item.get("reason"):
                fail(f"{cid}: considered_not_relevant entry without reason")
        # exclusivity
        for chk in c.get("exclusivity_checks", []):
            exclusivity_count += 1
            phrase = norm(chk["phrase"]).casefold()
            scope = chk.get("section")  # optional: search only inside this section
            haystack = (lambda i: norm_sec[i].get(scope, "")) if scope else (lambda i: norm_full[i])
            hits = sorted(i for i in doc_ids if phrase in haystack(i).casefold())
            if hits != sorted(chk["expected_docs"]):
                fail(f"{cid}: exclusivity {chk['phrase']!r}: expected {sorted(chk['expected_docs'])}, found in {hits}")

    # -- no retriever/model imported -------------------------------------
    for forbidden in ("retrieval", "retrieval.fts5", "retrieval.fts5_v2", "sentence_transformers", "onnxruntime", "transformers", "tokenizers"):
        if forbidden in sys.modules:
            fail(f"forbidden module imported: {forbidden}")

    # -- summary ----------------------------------------------------------
    by_cat: dict[str, int] = {}
    for c in cases:
        by_cat[c["category"]] = by_cat.get(c["category"], 0) + 1
    print("benchmark file      :", BENCH.relative_to(ROOT))
    print("benchmark SHA-256   :", sha)
    print("recomputed corpus fp:", fingerprint)
    print("scored cases        :", len(cases))
    print("behaviour cases     :", len(behaviour))
    print("total case ids      :", len(ids), "(unique:", len(set(ids)), ")")
    print("relevant labels     : CWE", lanes["cwe"], "| CAPEC", lanes["capec"])
    print("evidence quotes     :", quote_count, "| exclusivity checks:", exclusivity_count)
    print("scored by category  :", dict(sorted(by_cat.items())))
    if errors:
        print(f"\nFAILED with {len(errors)} problem(s):")
        for e in errors:
            print(" -", e)
        return 1
    print("\nALL CHECKS PASSED (no retriever was run; nothing was written).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
