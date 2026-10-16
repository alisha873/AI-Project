"""
Phase 12 - deterministic preparation of the planner input.

`build_planner_input(context, answer_result, configuration)` takes the verified Phase 11D
`GroundedContext` (observed evidence, Phase 9 hypotheses, retrieved reference passages with provenance),
the Phase 11D answer for that same context, and a `ConfigurationContext`, and decides two things
WITHOUT calling any model:

1. Which retrieved passages the planner may use.
   The relevance labels come from the Phase 11D answer's `reference_relevance`, and only from an answer
   that was accepted (`status == "answered"`) and that is re-validated here. Only passages labelled
   "relevant" are offered. "not_relevant" passages are excluded, "unclear" passages are excluded (a
   passage nobody could call relevant must not justify a recommendation) and passages without a label are
   excluded as "unassessed". Relevant means "possibly useful reference material", never proof.

2. Whether there is enough to plan at all (`insufficiency`).
   No observed evidence, no retrieved reference material, no validated relevance assessment or no relevant
   passage each give an explicit reason and the planner will not call the model. Missing or conflicting
   configuration is NOT a reason to stop: the plan can still be useful, but every proposal must then say
   that it depends on configuration details nobody supplied (`remediation.grounding`).

The relevance labels are the answering model's own judgement (PROJECT_STATUS section 3.9); they are
a filter, not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.answer import GroundedAnswerResult
from rag.citations import validate_answer
from rag.context import GroundedContext, ReferencePassage
from remediation.configuration import UNAVAILABLE, ConfigurationContext
from remediation.contract import (
    ConfigurationConflictSummary,
    ExcludedReference,
    InputSummary,
    InsufficiencyReason,
    LinkedHypothesis,
)

NO_OBSERVED_EVIDENCE = "no_observed_evidence"
NO_REFERENCE_MATERIAL = "no_reference_material"
RELEVANCE_NOT_ASSESSED = "relevance_not_assessed"
ANSWER_NOT_VALIDATED = "answer_not_validated"
NO_RELEVANT_REFERENCE = "no_relevant_reference"


class PlannerInputError(ValueError):
    """The inputs do not belong together (a programming or wiring error, not an evidence problem)."""


def linked_hypotheses(context: GroundedContext) -> list[LinkedHypothesis]:
    """Each Phase 9 finding exactly as reported: confirmed only if Phase 9 said exactly 'confirmed'."""
    return [
        LinkedHypothesis(
            hypothesis_id=item.hypothesis_id,
            status=item.status,
            is_confirmed=item.is_confirmed,
            basis="confirmed_finding" if item.is_confirmed else "inferred_hypothesis",
        )
        for item in context.hypotheses
    ]


@dataclass(frozen=True)
class PlannerInput:
    context: GroundedContext
    usable_references: tuple[ReferencePassage, ...]
    excluded_references: tuple[tuple[str, str], ...]     # (reference_id, label)
    relevance_source: str                                # "validated_phase_11d_answer" | "unavailable"
    configuration: ConfigurationContext
    insufficiency: tuple[InsufficiencyReason, ...]

    @property
    def ready(self) -> bool:
        return not self.insufficiency

    def summary(self) -> InputSummary:
        context = self.context
        return InputSummary(
            retriever=context.retriever,
            is_production_retriever=context.is_production_retriever,
            observed_evidence_ids=list(context.evidence_ids),
            unresolved_evidence_ids=list(context.unresolved_evidence_ids),
            hypotheses=linked_hypotheses(context),
            usable_reference_ids=[ref.reference_id for ref in self.usable_references],
            excluded_references=[ExcludedReference(reference_id=r, label=label) for r, label in self.excluded_references],
            relevance_source=self.relevance_source,  # type: ignore[arg-type]
            configuration_status=self.configuration.status,
            configuration_fact_ids=list(self.configuration.fact_ids),
            configuration_conflicts=[
                ConfigurationConflictSummary(
                    subject=c.subject, attribute=c.attribute, fact_ids=list(c.fact_ids), values=list(c.values)
                )
                for c in self.configuration.conflicts
            ],
        )


def _relevance_labels(
    context: GroundedContext, answer_result: GroundedAnswerResult | None
) -> tuple[dict[str, str], str, InsufficiencyReason | None]:
    """(labels, source, reason). Labels are empty unless an accepted, re-validated answer supplies them."""
    if answer_result is None:
        return {}, "unavailable", InsufficiencyReason(
            code=RELEVANCE_NOT_ASSESSED,
            message="No Phase 11D answer was supplied, so no retrieved passage has a relevance assessment.",
        )
    if not isinstance(answer_result, GroundedAnswerResult):
        raise PlannerInputError("answer_result must be a GroundedAnswerResult or None")
    if answer_result.context != context:
        raise PlannerInputError("the Phase 11D answer was generated from a different context")
    if answer_result.status != "answered" or not isinstance(answer_result.answer, dict):
        return {}, "unavailable", InsufficiencyReason(
            code=ANSWER_NOT_VALIDATED,
            message=f"The Phase 11D answer was not accepted (status {answer_result.status!r}), "
                    "so no relevance assessment can be trusted.",
        )
    # An accepted result is validated again here: a result built by hand must not smuggle in labels.
    if not validate_answer(answer_result.answer, context).passed:
        return {}, "unavailable", InsufficiencyReason(
            code=ANSWER_NOT_VALIDATED,
            message="The Phase 11D answer does not pass citation validation against this context.",
        )
    labels = {
        entry["reference_id"]: entry["assessment"]
        for entry in answer_result.answer.get("reference_relevance", [])
    }
    return labels, "validated_phase_11d_answer", None


def build_planner_input(
    context: GroundedContext,
    answer_result: GroundedAnswerResult | None,
    configuration: ConfigurationContext | None = None,
) -> PlannerInput:
    """Prepare and gate the planner input. Raises `PlannerInputError` only for mismatched inputs."""
    if not isinstance(context, GroundedContext):
        raise PlannerInputError("context must be a GroundedContext")
    configuration = configuration if configuration is not None else UNAVAILABLE
    if not isinstance(configuration, ConfigurationContext):
        raise PlannerInputError("configuration must be a ConfigurationContext")

    labels, source, relevance_problem = _relevance_labels(context, answer_result)

    usable: list[ReferencePassage] = []
    excluded: list[tuple[str, str]] = []
    for ref in context.references:
        label = labels.get(ref.reference_id, "unassessed")
        if label == "relevant":
            usable.append(ref)
        else:
            excluded.append((ref.reference_id, label))

    reasons: list[InsufficiencyReason] = []
    if not context.observed_evidence:
        reasons.append(InsufficiencyReason(
            code=NO_OBSERVED_EVIDENCE,
            message="The context contains no citable observed incident evidence; no evidence, no claim.",
        ))
    if not context.references:
        reasons.append(InsufficiencyReason(
            code=NO_REFERENCE_MATERIAL,
            message="Retrieval returned no reference passage, so no recommendation can be justified by reference knowledge.",
        ))
    else:
        if relevance_problem is not None:
            reasons.append(relevance_problem)
        elif not usable:
            reasons.append(InsufficiencyReason(
                code=NO_RELEVANT_REFERENCE,
                message="No retrieved passage was assessed as relevant to the incident hypothesis.",
            ))

    return PlannerInput(
        context=context,
        usable_references=tuple(usable),
        excluded_references=tuple(excluded),
        relevance_source=source,
        configuration=configuration,
        insufficiency=tuple(reasons),
    )
