"""
Phase 11 (Milestone 11B) - the two knowledge-type retrieval lanes, and the
written specification of the dual-lane architecture.

This module holds DEFINITIONS and a SPECIFICATION. It contains no search code,
no embedding code and no fusion code. Nothing here is a vector retriever; the
dual-lane design is documented, NOT implemented.

Architecture decisions for the initial prototype (agreed in Milestone 11B)
---------------------------------------------------------------------------
A. TWO KNOWLEDGE-TYPE LANES over ONE corpus and ONE store.
   - Weakness lane: CWE records (`doc_type == "weakness"`): software
     weaknesses, their consequences, mitigations and detection guidance.
   - Attack-pattern lane: CAPEC records (`doc_type == "attack_pattern"`):
     attack methods, prerequisites, execution flow, consequences and
     mitigations.
   A lane is only a `doc_types` filter on a `RetrievalQuery`; no second store
   or index is created. The existing FTS5 retriever is the baseline for both.

B. ONE general-purpose embedding model shared by both lanes (future work).
   No model is chosen, installed or fine-tuned here. Requirements for the
   eventual choice (`EMBEDDING_MODEL_REQUIREMENTS` below) must all be recorded
   before a model is adopted.

C. SOURCE-LEVEL RECIPROCAL RANK FUSION (RRF) of the two lanes' ranked lists
   (future work; specified below, NOT implemented).

D. CVE/NVD is DEFERRED. The prototype uses CWE and CAPEC only. CVE/NVD (and
   similar vulnerability-instance data) may be introduced later only if an
   evaluation shows a specific need for instance-level knowledge.

E. NO FINE-TUNING of BERT or any embedding model on the current corpus. A
   23-record subset cannot support a defensible fine-tuning claim.

An honest caveat about C (found while formalizing it)
-----------------------------------------------------
With the lanes defined purely by `doc_type`, they are DISJOINT: a document is
in exactly one lane. RRF rewards a document that appears near the top of
SEVERAL lists, but a document can appear in only one lane's list, so no
agreement signal exists and RRF reduces to a rank-balanced interleaving of
the two lanes. That may still be what is wanted (guaranteeing that both
weakness knowledge and attack-pattern knowledge are shown), but it must be
measured against the simplest alternative: ONE unfiltered query with the same
retriever. The next milestone should include that comparison instead of
assuming fusion helps.

RRF specification (to be implemented in the next milestone)
-----------------------------------------------------------
Inputs: one `RetrievalResult` per lane. Each is already SOURCE-LEVEL (one
item per distinct `doc_id`, chunks folded into the item), so fusion never
treats overlapping chunks of one source as independent documents.

Score: for each distinct `doc_id` d,

    rrf(d) = sum over lanes L in which d was returned of  1 / (k + rank_L(d))

with `rank_L(d)` the 1-based rank in lane L's list (`RetrievedItem.rank`) and
the constant `k` configurable (default `RRF_DEFAULT_K` = 60, the value
commonly used in the RRF literature; it must be recorded with every run, and
only positive integers >= `RRF_MIN_K` = 1 are valid). Raw lane scores (BM25,
cosine similarity) are NEVER mixed: only ranks are used, which is why RRF
needs no score normalization.

Ordering and deterministic tie-breaking, in this order:
  1. higher `rrf(d)` first;
  2. smaller best (minimum) lane rank first;
  3. lane order as listed in `LANES` (weakness lane before attack-pattern lane)
     for the lane in which the document was found;
  4. `doc_id` in ascending string order.
No step depends on dictionary order, hashing or the clock, so identical lane
results always fuse identically.

What a fused item must keep (a future `FusedItem` would carry):
  - identity and provenance exactly as in `RetrievedItem`: `doc_id`, `source`,
    `source_id`, `doc_type`, `title`, `source_url`;
  - the matched chunks (`ChunkMatch`: `chunk_id`, offsets, text) from the lane
    that returned them, each tagged with its lane;
  - per lane: the lane name, the lane rank and the lane's own raw score (kept
    for display, never fused);
  - the fused RRF score, the constant `k` used, and the fused rank (1..n).
The full document and its provenance stay in `KnowledgeStore.get_document`.
Fused results remain REFERENCE KNOWLEDGE (`RESULT_MEANING`): they do not
confirm or refute an incident hypothesis.

Requirements for the future embedding model (record before adopting one)
-----------------------------------------------------------------------
See `EMBEDDING_MODEL_REQUIREMENTS`. Nothing is installed in this milestone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from retrieval.contract import VALID_DOC_TYPES, InvalidRetrievalQueryError, RetrievalQuery

RRF_DEFAULT_K = 60
RRF_MIN_K = 1

# Facts to record, and thresholds to decide, BEFORE a model is adopted.
EMBEDDING_MODEL_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("local_execution", "Runs fully offline at inference time on the developer's Windows machine; weights are "
                        "downloaded once, pinned and stored locally. No call to a hosted API."),
    ("licence", "Model weights AND any bundled tokenizer/data must have a licence that permits the project's use "
                "(permissive such as Apache-2.0 or MIT is preferred); record the licence text and its source."),
    ("dimensions", "Record the output dimension and the maximum input length in tokens; the maximum input length "
                   "must cover a chunk (up to 1200 characters, roughly 300 tokens) without truncation."),
    ("reproducibility", "Pin the exact model revision (commit hash), tokenizer and library versions; record the "
                        "normalization (for example L2) and similarity measure; embeddings are stored with the "
                        "model id, revision and dimension so a mismatch is detected, and the same input must give "
                        "the same vector within a stated numerical tolerance."),
    ("resources", "CPU-only operation must be feasible; record download size, peak memory and time to embed the "
                  "corpus. A GPU may not be assumed."),
    ("scope", "One general-purpose model shared by both lanes; no fine-tuning, training pipeline or second model "
              "until an evaluation on a larger, independently labelled benchmark justifies it."),
    ("evaluation", "Judged against the frozen FTS5 baseline on the same corpus fingerprint and benchmark version, "
                   "tuned on the development split only, with the held-out split run once."),
)


@dataclass(frozen=True)
class Lane:
    """A named retrieval lane: a filter over `SourceDocument.doc_type` values."""

    name: str
    doc_types: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if not self.name or not self.doc_types:
            raise ValueError("a lane needs a name and at least one doc_type")
        unknown = sorted(set(self.doc_types) - VALID_DOC_TYPES)
        if unknown:
            raise ValueError(f"lane {self.name!r} has unknown doc_type(s) {unknown}")


WEAKNESS_LANE = Lane(
    name="weakness",
    doc_types=("weakness",),
    description="CWE weaknesses: what is wrong in software, with consequences, detection and mitigations.",
)
ATTACK_PATTERN_LANE = Lane(
    name="attack_pattern",
    doc_types=("attack_pattern",),
    description="CAPEC attack patterns: how an attack is carried out, with prerequisites, flow and mitigations.",
)

# The order is part of the deterministic tie-break rule above.
LANES: tuple[Lane, ...] = (WEAKNESS_LANE, ATTACK_PATTERN_LANE)


def lane_by_name(name: str) -> Lane:
    for lane in LANES:
        if lane.name == name:
            return lane
    raise InvalidRetrievalQueryError(f"unknown lane {name!r}; expected one of {[lane.name for lane in LANES]}")


def lane_for_doc_type(doc_type: str) -> Lane | None:
    """The lane that owns `doc_type`, or None if no lane covers it (for example a guideline)."""
    for lane in LANES:
        if doc_type in lane.doc_types:
            return lane
    return None


def lane_query(query: RetrievalQuery, lane: Lane) -> RetrievalQuery:
    """The same query restricted to one lane (the query's own doc_types are replaced, not intersected)."""
    return dataclasses.replace(query, doc_types=lane.doc_types)
