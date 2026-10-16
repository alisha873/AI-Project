"""
Phase 11 (Milestone 11D) - grounded answer generation through the existing `LLMClient`.

`generate_grounded_answer(context, llm_client)` turns one `GroundedContext` into a
`GroundedAnswerResult`. The language model is called ONLY when there is something
to ground an answer on, and its output is accepted ONLY if `rag.citations` passes it.

Outcome statuses (`GroundedAnswerResult.status`)
------------------------------------------------
answered                          validated answer; `answer` holds it.
not_generated_no_incident_evidence  the context has no citable observed evidence; the model is NOT
                                  called ("no evidence, no claim").
not_generated_no_reference_material retrieval returned nothing; the model is NOT called, so it cannot
                                  invent supporting knowledge. The incident side of the context is
                                  unaffected and still stored.
llm_error                         the client raised; only the exception type and a short message are kept.
rejected_invalid_response         the response was not a JSON object.
rejected_citations                parsed, but validation found issues; `answer` is None and the issues
                                  are in `validation`. The raw response is kept for audit only.

No retry, no repair and no fallback text: a rejected answer is reported as rejected. ("No retry" is about
this module: it calls `generate()` once. The Groq SDK underneath may itself re-send a request after a
timeout, 408, 409, 429 or 5xx; see `llm.groq.GroqLLMClient`.)

Irrelevant retrieval
--------------------
FTS5 v1 has no relevance threshold: any document sharing query words is returned, and
earlier measurements showed it returns documents even for out-of-corpus questions
(PROJECT_STATUS section 3.6 and 3.8). This module therefore does NOT claim to detect
irrelevant retrieval. It asks the model to assess each passage (`reference_relevance`), refuses
to let a passage assessed "not_relevant" be cited, and allows empty guidance. That assessment is
the model's own judgement and has not been evaluated.

`LLMClient.generate(system_prompt, user_prompt, response_format)` is used unchanged. The schema is
passed as `response_format`; providers differ in how they honour it, so nothing relies on it.
No live provider was called in Milestone 11D.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from llm.base import LLMClient
from llm.token_budget import RequestNotSentError
from rag.citations import (
    CitationValidation,
    InvalidAnswerError,
    parse_answer_json,
    validate_answer,
)
from rag.context import GroundedContext
from rag.prompt import SYSTEM_PROMPT, answer_json_schema, build_user_prompt

AnswerStatus = Literal[
    "answered",
    "not_generated_no_incident_evidence",
    "not_generated_no_reference_material",
    "llm_error",
    "rejected_invalid_response",
    "rejected_citations",
]

_MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class GroundedAnswerResult:
    context: GroundedContext
    status: AnswerStatus
    message: str
    answer: dict[str, Any] | None = None
    validation: CitationValidation | None = None
    raw_response: str | None = None
    llm: str | None = None
    # Whether a request actually reached the provider. False only when the pre-send budget refused the
    # prompt (no network call happened); True otherwise, including a provider error after a call was made.
    request_sent: bool = True

    @property
    def accepted(self) -> bool:
        return self.status == "answered"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_llm(client: LLMClient) -> str:
    """A short, non-secret description such as 'GroqLLMClient(openai/gpt-oss-20b)'."""
    model = getattr(client, "model", None)
    name = type(client).__name__
    return f"{name}({model})" if isinstance(model, str) and model else name


def generate_grounded_answer(
    context: GroundedContext,
    llm_client: LLMClient,
    *,
    max_characters_per_passage: int | None = None,
) -> GroundedAnswerResult:
    """Generate and validate one answer; see the module docstring for the statuses.

    `max_characters_per_passage` (optional) shortens only reference-passage text sent to the model, keeping
    all identifiers and provenance; it does not change validation, which runs on the stored context.
    """
    if not isinstance(context, GroundedContext):
        raise TypeError("context must be a GroundedContext")
    if not isinstance(llm_client, LLMClient):
        raise TypeError("llm_client must be an LLMClient")

    if not context.observed_evidence:
        return GroundedAnswerResult(
            context,
            "not_generated_no_incident_evidence",
            "No citable incident evidence is available for this finding, so no answer was generated.",
        )
    if not context.references:
        return GroundedAnswerResult(
            context,
            "not_generated_no_reference_material",
            "Retrieval returned no reference passage, so no reference guidance was generated "
            "and no support was invented. The hypothesis stays unsupported by reference knowledge.",
        )

    llm = describe_llm(llm_client)
    try:
        response = llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_user_prompt(context, max_characters_per_passage=max_characters_per_passage),
            response_format=answer_json_schema(),
        )
    except RequestNotSentError as exc:  # oversized prompt OR no per-minute budget left; nothing was sent
        return GroundedAnswerResult(
            context, "llm_error",
            f"No request was sent: {str(exc)[:_MAX_ERROR_CHARS]}.",
            llm=llm, request_sent=False,
        )
    except Exception as exc:  # any provider failure; the type and a short message are kept, nothing else
        message = f"{type(exc).__name__}: {str(exc)[:_MAX_ERROR_CHARS]}"
        return GroundedAnswerResult(context, "llm_error", f"The language model call failed ({message}).", llm=llm)

    try:
        parsed = parse_answer_json(response)
    except InvalidAnswerError as exc:
        return GroundedAnswerResult(
            context, "rejected_invalid_response", str(exc), raw_response=_raw(response), llm=llm
        )

    validation = validate_answer(parsed, context)
    if not validation.passed:
        return GroundedAnswerResult(
            context,
            "rejected_citations",
            f"The answer was rejected: {len(validation.issues)} citation or grounding issue(s).",
            validation=validation,
            raw_response=_raw(response),
            llm=llm,
        )
    return GroundedAnswerResult(
        context,
        "answered",
        "The answer passed citation validation (this does not prove every statement is entailed).",
        answer=parsed,
        validation=validation,
        raw_response=_raw(response),
        llm=llm,
    )


def _raw(response: object) -> str | None:
    return response if isinstance(response, str) else None
