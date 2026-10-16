"""
Phase 12 - the context-aware remediation planner.

`plan_remediation(planner_input, llm_client)` turns one prepared `PlannerInput` into a
`RemediationPlanResult`. It reasons through the existing provider-neutral `LLMClient.generate(...)`
(Groq in production; nothing here is Groq-specific and no second client exists) and accepts the model's
output ONLY if it passes `remediation.grounding`.

The planner is ADVISORY. It reads no file, writes nothing, changes no container and launches no test.

Outcome statuses (`RemediationPlanResult.status`)
-------------------------------------------------
planned                            validated plan with at least one proposal; `plan` holds it.
not_planned_insufficient_context   the prepared input is insufficient (no observed evidence, no reference
                                   material, no validated relevance assessment or no relevant passage). The
                                   model is NOT called. `plan` is an explicit insufficient-evidence plan.
model_declared_insufficient        the model validly said the material supports no proposal; `plan` is an
                                   insufficient-evidence plan carrying the model's reason.
llm_error                          the client raised (API error, timeout, network); no plan.
rejected_invalid_response          the response is not valid JSON or does not match the schema; no plan.
rejected_grounding                 structurally valid but validation found issues; no plan; the issues and
                                   the raw response are kept for audit.

No retry, no repair, no fallback text, no partial plan: a failure is reported as a failure. ("No retry" is
about this module: it calls `generate()` once. The Groq SDK underneath may itself re-send a request after a
timeout, 408, 409, 429 or 5xx; see `llm.groq.GroqLLMClient`.)

Derived by code, never by the model: the proposal ID (content hash of the validated draft), the priority
rank and basis, the linked hypotheses with their Phase 9 status, the reference provenance and the
"proposed, not executed, human review required" markers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from llm.base import LLMClient
from rag.answer import describe_llm
from remediation.contract import (
    CONFIDENCE_MEANING,
    PLAN_CAVEATS,
    PRIORITY_CRITERIA,
    InsufficiencyReason,
    PlanDraft,
    ProposalDraft,
    ReferenceCitation,
    RemediationPlan,
    RemediationProposal,
)
from remediation.grounding import InvalidPlanError, PlanValidation, parse_plan, validate_plan
from remediation.inputs import PlannerInput, linked_hypotheses
from remediation.priority import rank_proposals
from remediation.prompt import SYSTEM_PROMPT, build_user_prompt, plan_json_schema

PlanStatus = Literal[
    "planned",
    "not_planned_insufficient_context",
    "model_declared_insufficient",
    "llm_error",
    "rejected_invalid_response",
    "rejected_grounding",
]

MODEL_DECLARED_INSUFFICIENT = "model_declared_insufficient"
_MAX_ERROR_CHARS = 300
_SECRET_PATTERNS = (
    re.compile(r"gsk_[A-Za-z0-9_\-]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    # header-like pairs: Authorization: Bearer <token>, 'Authorization': 'Basic ...', x-api-key=...
    re.compile(
        r"(?:authorization|proxy-authorization|x-api-key)[\"']?\s*[:=]\s*[\"']?(?:(?:bearer|basic|token)\s+)?[^\s,}\"']+",
        re.IGNORECASE,
    ),
    re.compile(r"bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE),
    re.compile(r"api[_\-]?key[\"']?\s*[:=]\s*\S+", re.IGNORECASE),
)
# A provider error body can carry the model's partial output (`failed_generation`), which may contain
# incident text. Everything from that key to the end of the message is dropped.
_FAILED_GENERATION = re.compile(r"failed_generation[\"']?\s*:\s*.*", re.IGNORECASE | re.DOTALL)


def redact_credentials(text: object) -> str:
    """
    Remove credentials from stored model text WITHOUT shortening or rewording it.

    Only API-key-like strings, the configured key values, authorization headers and bearer tokens are
    replaced by `[redacted]`. Ordinary wording, reference ids and evidence ids are left exactly as they
    were, and nothing is truncated (unlike `scrub_message`, which is for short error messages). The
    `failed_generation` rule is deliberately NOT applied: that key belongs to a provider error body, and
    this function is for a model answer that the provider actually returned. It is a pattern-based
    redaction, not a proof that no sensitive text remains.
    """
    message = text if isinstance(text, str) else str(text)
    for name in ("GROQ_API_KEY", "LLM_API_KEY"):
        secret = os.environ.get(name, "")
        if len(secret) >= 8:
            message = message.replace(secret, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub("[redacted]", message)
    return message


def scrub_message(text: object, limit: int = _MAX_ERROR_CHARS) -> str:
    """
    A short diagnostic string that is safe to store and print.

    Removes API-key-like strings, the configured key values, authorization headers and the provider's
    `failed_generation` body, then truncates to `limit` characters. It is a pattern-based scrub, not a proof
    that no sensitive text remains, which is why the result is also truncated.
    """
    message = text if isinstance(text, str) else str(text)
    for name in ("GROQ_API_KEY", "LLM_API_KEY"):
        secret = os.environ.get(name, "")
        if len(secret) >= 8:
            message = message.replace(secret, "[redacted]")
    for pattern in _SECRET_PATTERNS:
        message = pattern.sub("[redacted]", message)
    message = _FAILED_GENERATION.sub("failed_generation: [omitted]", message)
    return message[:limit]


def safe_error_message(exc: BaseException) -> str:
    """The exception type and a short message with anything key-like removed (never the full text)."""
    return f"{type(exc).__name__}: {scrub_message(str(exc))}"


def _error_kind(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower():
        return "timeout"
    return "provider_error"


REJECTED_PLAN_STATUSES = ("rejected_invalid_response", "rejected_grounding")
RAW_RESPONSE_WARNING = (
    "raw_response is the model's UNVALIDATED output that was NOT accepted (the plan was rejected). It may be "
    "wrong, may contain fabricated or excluded ids and may echo sensitive incident details. Credential-like "
    "strings were redacted, nothing else was altered. It is kept only for diagnosis and is not evidence."
)


@dataclass(frozen=True)
class RemediationPlanResult:
    incident_id: str
    finding_index: int
    status: PlanStatus
    message: str
    plan: RemediationPlan | None = None
    validation: PlanValidation | None = None
    raw_response: str | None = None
    llm: str | None = None
    error_kind: str | None = None

    @property
    def accepted(self) -> bool:
        """True only for a validated plan that contains proposals."""
        return self.status == "planned"

    @property
    def model_called(self) -> bool:
        """True if the planner attempted a model call; `llm` stays None when the input was insufficient."""
        return self.llm is not None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "incident_id": self.incident_id,
            "finding_index": self.finding_index,
            "status": self.status,
            "message": self.message,
            "error_kind": self.error_kind,
            "llm": self.llm,
            "planner_model_called": self.model_called,
            "plan": self.plan.model_dump(mode="json") if self.plan is not None else None,
            "validation": None if self.validation is None else {
                "passed": self.validation.passed,
                "note": self.validation.note,
                "issues": [
                    {"code": i.code, "location": i.location, "message": i.message} for i in self.validation.issues
                ],
            },
            "raw_response": self.raw_response,
        }
        if self.status in REJECTED_PLAN_STATUSES and self.raw_response is not None:
            data["raw_response_warning"] = RAW_RESPONSE_WARNING
        return data


# ============================================================
# PLAN ASSEMBLY (everything trustworthy is derived here, not by the model)
# ============================================================

def _proposal_id(incident_id: str, finding_index: int, draft: ProposalDraft) -> str:
    canonical = json.dumps(draft.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"REM-{incident_id}-F{finding_index}-{digest}"


def _build_proposals(planner_input: PlannerInput, drafts: list[ProposalDraft]) -> list[RemediationProposal]:
    context = planner_input.context
    passages = {ref.reference_id: ref for ref in planner_input.usable_references}
    hypotheses = {item.hypothesis_id: item for item in linked_hypotheses(context)}
    built: list[RemediationProposal] = []
    for index, rank, basis in rank_proposals(drafts):
        draft = drafts[index]
        references = [
            ReferenceCitation(
                reference_id=ref.reference_id, doc_id=ref.doc_id, chunk_id=ref.chunk_id, ordinal=ref.ordinal,
                char_start=ref.char_start, char_end=ref.char_end, retrieval_rank=ref.document_rank,
                source=ref.source, source_id=ref.source_id, source_version=ref.source_version,
                doc_type=ref.doc_type, title=ref.title, source_url=ref.source_url, license=ref.license,
                chunk_sha256=ref.chunk_sha256, document_content_sha256=ref.document_content_sha256,
                provenance=dict(ref.provenance), relevance_label="relevant",
            )
            for ref in (passages[ref_id] for ref_id in draft.reference_ids)
        ]
        built.append(
            RemediationProposal(
                **draft.model_dump(),
                proposal_id=_proposal_id(context.incident_id, context.finding_index, draft),
                priority_rank=rank,
                priority_basis=basis,
                linked_hypotheses=[hypotheses[h] for h in draft.hypothesis_ids],
                references=references,
            )
        )
    built.sort(key=lambda proposal: proposal.priority_rank)
    return built


def _assemble(
    planner_input: PlannerInput,
    reasons: list[InsufficiencyReason],
    proposals: list[RemediationProposal],
    unresolved_questions: list[str],
    limitations: list[str],
) -> RemediationPlan:
    context = planner_input.context
    return RemediationPlan(
        incident_id=context.incident_id,
        finding_index=context.finding_index,
        outcome="proposals" if proposals else "insufficient_evidence",
        insufficient_reasons=reasons,
        proposals=proposals,
        unresolved_questions=unresolved_questions,
        limitations=limitations,
        input_summary=planner_input.summary(),
        priority_criteria=list(PRIORITY_CRITERIA),
        confidence_meaning=CONFIDENCE_MEANING,
        caveats=list(PLAN_CAVEATS),
    )


# ============================================================
# PUBLIC API
# ============================================================

def plan_remediation(planner_input: PlannerInput, llm_client: LLMClient) -> RemediationPlanResult:
    """Generate and validate one remediation plan; see the module docstring for the statuses."""
    if not isinstance(planner_input, PlannerInput):
        raise TypeError("planner_input must be a PlannerInput")
    if not isinstance(llm_client, LLMClient):
        raise TypeError("llm_client must be an LLMClient")

    context = planner_input.context
    base = {"incident_id": context.incident_id, "finding_index": context.finding_index}

    if not planner_input.ready:
        plan = _assemble(planner_input, list(planner_input.insufficiency), [], [], [])
        return RemediationPlanResult(
            status="not_planned_insufficient_context",
            message="The prepared input is insufficient, so the model was not called and no remediation "
                    "was proposed: " + ", ".join(reason.code for reason in planner_input.insufficiency) + ".",
            plan=plan,
            **base,
        )

    llm = describe_llm(llm_client)
    try:
        response = llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(planner_input),
            response_format=plan_json_schema(),
        )
    except Exception as exc:  # any provider failure; only the type, a kind and a scrubbed message are kept
        return RemediationPlanResult(
            status="llm_error",
            message=f"The language model call failed ({safe_error_message(exc)}); no plan was produced.",
            llm=llm,
            error_kind=_error_kind(exc),
            **base,
        )

    raw = response if isinstance(response, str) else None
    # Rejected output is kept for diagnosis only, so it is credential-redacted (see RAW_RESPONSE_WARNING);
    # an accepted plan keeps `raw` exactly as the model returned it.
    raw_rejected = redact_credentials(raw) if raw is not None else None
    try:
        draft: PlanDraft = parse_plan(response)
    except InvalidPlanError as exc:
        return RemediationPlanResult(
            status="rejected_invalid_response", message=str(exc), raw_response=raw_rejected, llm=llm, **base
        )

    validation = validate_plan(draft, planner_input)
    if not validation.passed:
        return RemediationPlanResult(
            status="rejected_grounding",
            message=f"The plan was rejected: {len(validation.issues)} grounding issue(s).",
            validation=validation,
            raw_response=raw_rejected,
            llm=llm,
            **base,
        )

    if draft.outcome == "insufficient_evidence":
        plan = _assemble(
            planner_input,
            [InsufficiencyReason(code=MODEL_DECLARED_INSUFFICIENT, message=draft.insufficient_reason.strip())],
            [],
            list(draft.unresolved_questions),
            list(draft.limitations),
        )
        return RemediationPlanResult(
            status="model_declared_insufficient",
            message="The model reported that the supplied material does not support a remediation proposal.",
            plan=plan, validation=validation, raw_response=raw, llm=llm, **base,
        )

    plan = _assemble(
        planner_input, [], _build_proposals(planner_input, list(draft.proposals)),
        list(draft.unresolved_questions), list(draft.limitations),
    )
    return RemediationPlanResult(
        status="planned",
        message="The plan passed grounding validation (this does not prove that any proposal is correct or safe).",
        plan=plan, validation=validation, raw_response=raw, llm=llm, **base,
    )
