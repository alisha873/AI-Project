"""
Phase 14 — evidence-based Judge Agent.

`JudgeAgent.evaluate` reads a Phase 13 `ValidationReport`, rebuilds the Phase 14 handoff, and returns a
versioned `JudgeAssessment`. The decision is a deterministic check of the recorded evidence. It does not call
a model, send a request, or treat a proposal, a confidence number, or an upstream status string as proof.

Nothing in this package performs live validation. A mocked or dry-run result stays labelled as such.
"""

from judge.agent import JudgeAgent, JudgeInputError, evaluate_payload
from judge.contract import (
    DECISIONS,
    JUDGE_SCHEMA_VERSION,
    JudgeAssessment,
    serialize,
)

__all__ = [
    "DECISIONS",
    "JUDGE_SCHEMA_VERSION",
    "JudgeAgent",
    "JudgeAssessment",
    "JudgeInputError",
    "evaluate_payload",
    "serialize",
]
