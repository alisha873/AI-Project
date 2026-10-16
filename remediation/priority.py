"""
Phase 12 - deterministic, documented proposal priority.

The model never supplies a rank. After a plan passes validation, `rank_proposals` orders the proposals
by the fixed rule in `remediation.contract.PRIORITY_RULE`:

    (1) evidence linkage      incident_linked before general_hardening
    (2) severity              high, medium, low, unknown
    (3) disruption            low, medium, high, unknown
    (4) reversibility         reversible, partially_reversible, unknown, irreversible
    (5) model's own order     (ties only; keeps the result deterministic)

The evidence linkage is computed from the citations. Severity, disruption and reversibility are the
model's uncalibrated ratings. The resulting `priority_rank` is an ORDINAL position, not a risk score and
not a probability; the output states this (`PRIORITY_CRITERIA`).
"""

from __future__ import annotations

from remediation.contract import PriorityBasis, ProposalDraft

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
_DISRUPTION_ORDER = {"low": 0, "medium": 1, "high": 2, "unknown": 3}
_REVERSIBILITY_ORDER = {"reversible": 0, "partially_reversible": 1, "unknown": 2, "irreversible": 3}


def priority_basis(draft: ProposalDraft) -> PriorityBasis:
    return PriorityBasis(
        evidence_linkage="incident_linked" if draft.observed_evidence_ids else "general_hardening",
        severity=draft.severity.level,
        disruption=draft.disruption,
        reversibility=draft.reversibility,
    )


def _sort_key(index: int, basis: PriorityBasis) -> tuple[int, int, int, int, int]:
    return (
        0 if basis.evidence_linkage == "incident_linked" else 1,
        _SEVERITY_ORDER[basis.severity],
        _DISRUPTION_ORDER[basis.disruption],
        _REVERSIBILITY_ORDER[basis.reversibility],
        index,
    )


def rank_proposals(drafts: list[ProposalDraft]) -> list[tuple[int, int, PriorityBasis]]:
    """[(original_index, priority_rank starting at 1, basis)] in priority order."""
    bases = [priority_basis(draft) for draft in drafts]
    order = sorted(range(len(drafts)), key=lambda i: _sort_key(i, bases[i]))
    return [(index, rank, bases[index]) for rank, index in enumerate(order, start=1)]
