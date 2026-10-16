"""
Phase 12 - parsing and grounding validation of a model-written remediation plan.

`parse_plan` turns a model response into a structurally valid `PlanDraft` or raises
`InvalidPlanError`. `validate_plan` then checks the draft against the prepared `PlannerInput` and returns
EVERY problem it finds (`PlanValidation.issues`), not just the first. Nothing is repaired or defaulted.

What is checked
---------------
Identifiers   every cited id exists in THIS input and sits in the right category: observed evidence ids only
              name incident events, reference ids only name passages labelled relevant, hypothesis ids name
              Phase 9 findings, configuration ids name usable configuration facts. A reference id used as
              evidence (or the reverse) is `category_mixing`; a passage labelled not_relevant, unclear or
              unassessed is `excluded_reference_cited`; a withheld conflicting configuration fact is
              `conflicting_configuration_cited`; anything else is an unknown id. Ids written inside the
              prose must also be known and listed in that proposal's own citations.
Grounding     every proposal cites at least one relevant reference; an incident-linked proposal (one that
              cites observed evidence) names the hypothesis it addresses; an "identified" affected area is
              backed by observed evidence or a configuration fact; every proposal lists side effects and
              implementation steps.
Honesty       when configuration is unavailable or conflicting every proposal must say it depends on
              configuration and carry an open question; a disruptive or not fully reversible proposal must
              list preconditions; confidence "high" needs a confirmed linked hypothesis and available
              configuration, "medium" needs a step stated by a cited reference.
Wording       no positive claim of exploitation, data access or compromise; nothing presented as confirmed
              unless Phase 9 confirmed it; no claim that a fix works or guarantees safety; a CWE, CAPEC or CVE
              identifier must come from the hypothesis text or a cited passage; a path or back-ticked
              command must occur in the material the model was shown.
Consistency   the right incident, an outcome that agrees with the proposals, no duplicate proposals.

What this does NOT prove: that a recommendation is correct, sufficient, safe or entailed by the cited
passage. See `remediation.text_checks` for the limits of the wording screens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from rag.citations import InvalidAnswerError, parse_answer_json
from remediation import text_checks
from remediation.contract import MAX_LIST_ITEMS, MAX_PROPOSALS, PlanDraft, ProposalDraft
from remediation.inputs import PlannerInput
from remediation.prompt import model_view

VALIDATION_NOTE = (
    "Validation checks that identifiers exist and sit in the right category, that proposals are honest "
    "about missing configuration and risk, and screens some high-risk wording. It does not prove that a "
    "recommendation is correct, sufficient or safe."
)


class InvalidPlanError(ValueError):
    """The model response is not a JSON object of the plan schema."""


@dataclass(frozen=True)
class PlanIssue:
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class PlanValidation:
    passed: bool
    issues: tuple[PlanIssue, ...]
    note: str = VALIDATION_NOTE

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(issue.code for issue in self.issues)


def parse_plan(response: object) -> PlanDraft:
    """Strict JSON parse plus structural validation. Raises `InvalidPlanError`; never repairs."""
    try:
        value = parse_answer_json(response)  # type: ignore[arg-type]
    except InvalidAnswerError as exc:
        raise InvalidPlanError(str(exc)) from exc
    try:
        return PlanDraft.model_validate(value)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']}"
            for error in exc.errors()[:8]
        ]
        more = f" (+{exc.error_count() - 8} more)" if exc.error_count() > 8 else ""
        raise InvalidPlanError("the plan does not match the schema: " + "; ".join(problems) + more) from exc


class _Collector:
    def __init__(self) -> None:
        self.issues: list[PlanIssue] = []

    def add(self, code: str, location: str, message: str) -> None:
        self.issues.append(PlanIssue(code, location, message))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return []


class _Facts:
    """What the prepared input makes citable, precomputed once."""

    def __init__(self, planner_input: PlannerInput) -> None:
        context = planner_input.context
        self.input = planner_input
        self.evidence = set(context.evidence_ids)
        self.unresolved = set(context.unresolved_evidence_ids)
        self.hypotheses = {item.hypothesis_id: item for item in context.hypotheses}
        self.usable = {ref.reference_id: ref for ref in planner_input.usable_references}
        self.excluded = dict(planner_input.excluded_references)
        self.config = {fact.fact_id for fact in planner_input.configuration.facts}
        self.conflicting_config = set(planner_input.configuration.conflicting_fact_ids)
        self.config_status = planner_input.configuration.status

        self.hypothesis_identifiers: set[str] = set()
        for item in context.hypotheses:
            for text in (item.vulnerability_type, item.root_cause, item.affected_component, item.rationale,
                         *item.evidence_needed_to_confirm):
                if text:
                    self.hypothesis_identifiers |= text_checks.identifiers(text)
        self.reference_identifiers = {
            ref_id: {ref.source_id.upper()} | text_checks.identifiers(ref.text) | text_checks.identifiers(ref.title)
            for ref_id, ref in self.usable.items()
        }
        self.global_identifiers = set(self.hypothesis_identifiers)
        for identifiers in self.reference_identifiers.values():
            self.global_identifiers |= identifiers
        self.corpus = "\n".join(_strings(model_view(planner_input)))

    def cited_identifiers(self, reference_ids: set[str]) -> set[str]:
        found: set[str] = set()
        for ref_id in reference_ids:
            found |= self.reference_identifiers.get(ref_id, set())
        return found


# ============================================================
# IDENTIFIER CHECKS
# ============================================================

def _check_ids(ids: list[str], kind: str, where: str, out: _Collector, facts: _Facts) -> set[str]:
    """Validate the ids of one citation field. `kind` is evidence, reference, hypothesis or configuration."""
    valid: set[str] = set()
    if len(ids) > MAX_LIST_ITEMS:
        out.add("too_many_items", where, f"more than {MAX_LIST_ITEMS} identifiers")
    if len(set(ids)) != len(ids):
        out.add("duplicate_citation", where, "the same identifier is cited more than once")
    for item in dict.fromkeys(ids):
        if kind == "evidence" and item in facts.evidence:
            valid.add(item)
        elif kind == "reference" and item in facts.usable:
            valid.add(item)
        elif kind == "hypothesis" and item in facts.hypotheses:
            valid.add(item)
        elif kind == "configuration" and item in facts.config:
            valid.add(item)
        elif item in facts.excluded:
            if kind == "reference":
                out.add("excluded_reference_cited", where,
                        f"{item!r} was labelled {facts.excluded[item]!r} and may not support a proposal")
            else:
                out.add("category_mixing", where, f"{item!r} is a reference passage, not a {kind} id")
        elif item in facts.conflicting_config and kind == "configuration":
            out.add("conflicting_configuration_cited", where,
                    f"{item!r} is part of a configuration conflict and was withheld")
        elif item in facts.unresolved and kind == "evidence":
            out.add("unknown_evidence_id", where, f"{item!r} has no description in the reconstruction and is not citable")
        elif item in facts.usable:
            out.add("category_mixing", where, f"{item!r} is a reference passage, not a {kind} id")
        elif item in facts.evidence:
            out.add("category_mixing", where, f"{item!r} is incident evidence, not a {kind} id")
        elif item in facts.hypotheses:
            out.add("category_mixing", where, f"{item!r} is a hypothesis, not a {kind} id")
        elif item in facts.config:
            out.add("category_mixing", where, f"{item!r} is a configuration fact, not a {kind} id")
        else:
            out.add(f"unknown_{kind}_id", where, f"{item!r} is not a {kind} id of this input")
    return valid


# ============================================================
# PROSE CHECKS
# ============================================================

def _check_text(
    text: str,
    where: str,
    out: _Collector,
    facts: _Facts,
    *,
    own_evidence: set[str],
    own_references: set[str],
    own_config: set[str],
    allowed_identifiers: set[str],
) -> None:
    for token in sorted(text_checks.evidence_tokens(text)):
        if token not in facts.evidence:
            out.add("unknown_evidence_id", where, f"the text mentions {token!r}, which is not a citable event of this input")
        elif token not in own_evidence:
            out.add("prose_citation_mismatch", where, f"the text mentions {token!r} but the item does not cite it")
    for token in sorted(text_checks.reference_tokens(text)):
        if token in facts.excluded:
            out.add("excluded_reference_cited", where, f"the text mentions {token!r}, labelled {facts.excluded[token]!r}")
        elif token not in facts.usable:
            out.add("unknown_reference_id", where, f"the text mentions {token!r}, which is not a usable passage of this input")
        elif token not in own_references:
            out.add("prose_citation_mismatch", where, f"the text mentions {token!r} but the item does not cite it")
    for token in sorted(text_checks.configuration_tokens(text)):
        if token in facts.conflicting_config:
            out.add("conflicting_configuration_cited", where, f"the text mentions the withheld conflicting fact {token!r}")
        elif token not in facts.config:
            out.add("unknown_configuration_id", where, f"the text mentions {token!r}, which is not a configuration fact of this input")
        elif token not in own_config:
            out.add("prose_citation_mismatch", where, f"the text mentions {token!r} but the item does not cite it")
    for identifier in sorted(text_checks.identifiers(text)):
        if identifier not in allowed_identifiers:
            out.add("unsupported_identifier", where,
                    f"{identifier} is not stated by the hypothesis or by a passage this item cites")
    for claim in text_checks.compromise_claims(text):
        out.add("unsupported_compromise_claim", where, f"asserts {claim!r}; no event in this input states that outcome")
    if not any(item.is_confirmed for item in facts.hypotheses.values()):
        for claim in text_checks.confirmation_claims(text):
            out.add("confirmation_claim", where, f"presents {claim!r} as confirmed, but the finding is only a hypothesis")
    for claim in text_checks.success_claims(text):
        out.add("remediation_success_claim", where, f"claims {claim!r}; whether a proposal works has not been tested")
    for literal in text_checks.ungrounded_literals(text, facts.corpus):
        out.add("ungrounded_literal", where,
                f"{literal!r} does not occur in the supplied material; paths, commands and settings may not be invented")


# ============================================================
# PROPOSAL AND PLAN RULES
# ============================================================

def _proposal_texts(index: int, p: ProposalDraft) -> list[tuple[str, str]]:
    base = f"proposals[{index}]"
    texts = [
        (f"{base}.title", p.title),
        (f"{base}.description", p.description),
        (f"{base}.problem_addressed", p.problem_addressed),
        (f"{base}.affected_area.description", p.affected_area.description),
        (f"{base}.rationale", p.rationale),
        (f"{base}.reference_support_limits", p.reference_support_limits),
        (f"{base}.severity.rationale", p.severity.rationale),
        (f"{base}.expected_security_effect", p.expected_security_effect),
        (f"{base}.validation.procedure", p.validation.procedure),
        (f"{base}.validation.expected_result", p.validation.expected_result),
        (f"{base}.confidence.rationale", p.confidence.rationale),
    ]
    for name in ("operational_side_effects", "preconditions", "dependencies", "unresolved_questions"):
        texts += [(f"{base}.{name}[{i}]", item) for i, item in enumerate(getattr(p, name))]
    texts += [(f"{base}.implementation_steps[{i}].text", s.text) for i, s in enumerate(p.implementation_steps)]
    return texts


def _check_proposal(index: int, p: ProposalDraft, out: _Collector, facts: _Facts) -> None:
    base = f"proposals[{index}]"
    evidence = _check_ids(p.observed_evidence_ids, "evidence", f"{base}.observed_evidence_ids", out, facts)
    hypotheses = _check_ids(p.hypothesis_ids, "hypothesis", f"{base}.hypothesis_ids", out, facts)
    config = _check_ids(p.configuration_fact_ids, "configuration", f"{base}.configuration_fact_ids", out, facts)
    references = _check_ids(p.reference_ids, "reference", f"{base}.reference_ids", out, facts)

    if not p.reference_ids:
        out.add("missing_citation", f"{base}.reference_ids", "a proposal must cite at least one relevant reference passage")
    if evidence and not hypotheses:
        out.add("missing_hypothesis_link", f"{base}.hypothesis_ids",
                "a proposal tied to observed evidence must name the hypothesis it addresses")
    if p.affected_area.state == "identified" and not (evidence or config):
        out.add("unsupported_affected_area", f"{base}.affected_area",
                "an identified affected area must be backed by observed evidence or a configuration fact")
    if not p.operational_side_effects:
        out.add("missing_side_effects", f"{base}.operational_side_effects",
                "list the operational side effects, or say that they cannot be assessed")
    if not p.implementation_steps:
        out.add("missing_steps", f"{base}.implementation_steps", "at least one implementation step is required")
    if len(p.implementation_steps) > MAX_LIST_ITEMS:
        out.add("too_many_items", f"{base}.implementation_steps", f"more than {MAX_LIST_ITEMS} steps")
    if facts.config_status != "available":
        if not p.configuration_dependent:
            out.add("configuration_dependency_missing", f"{base}.configuration_dependent",
                    f"configuration is {facts.config_status}; the proposal must say it depends on configuration details")
        if not p.unresolved_questions:
            out.add("missing_unresolved_question", f"{base}.unresolved_questions",
                    f"configuration is {facts.config_status}; name what must be learned before acting")
    if (p.disruption != "low" or p.reversibility != "reversible") and not p.preconditions:
        out.add("missing_precondition", f"{base}.preconditions",
                "a disruptive, unknown-impact or not fully reversible proposal must list its preconditions")

    cited_steps = any(step.origin == "cited_reference" for step in p.implementation_steps)
    if p.confidence.level == "high" and not (
        facts.config_status == "available" and any(facts.hypotheses[h].is_confirmed for h in hypotheses)
    ):
        out.add("confidence_exceeds_support", f"{base}.confidence.level",
                "high confidence needs a confirmed linked hypothesis and available configuration facts")
    if p.confidence.level == "medium" and not cited_steps:
        out.add("confidence_exceeds_support", f"{base}.confidence.level",
                "medium confidence needs at least one step stated by a cited reference")

    allowed = facts.hypothesis_identifiers | facts.cited_identifiers(references)
    for where, text in _proposal_texts(index, p):
        _check_text(text, where, out, facts, own_evidence=evidence, own_references=references,
                    own_config=config, allowed_identifiers=allowed)


def validate_plan(draft: PlanDraft, planner_input: PlannerInput) -> PlanValidation:
    """Check a structurally valid draft against `planner_input`; returns every issue found."""
    out = _Collector()
    facts = _Facts(planner_input)

    if draft.incident_id != planner_input.context.incident_id:
        out.add("wrong_incident", "incident_id", f"must be {planner_input.context.incident_id!r}")
    if draft.outcome == "proposals":
        if not draft.proposals:
            out.add("no_proposals", "proposals", "outcome 'proposals' needs at least one proposal")
    else:
        if draft.proposals:
            out.add("insufficient_with_proposals", "proposals", "outcome 'insufficient_evidence' must have no proposals")
        if not draft.insufficient_reason.strip():
            out.add("missing_reason", "insufficient_reason", "an insufficient_evidence outcome needs a reason")
    if len(draft.proposals) > MAX_PROPOSALS:
        out.add("too_many_items", "proposals", f"more than {MAX_PROPOSALS} proposals")
    for name in ("unresolved_questions", "limitations"):
        if len(getattr(draft, name)) > MAX_LIST_ITEMS:
            out.add("too_many_items", name, f"more than {MAX_LIST_ITEMS} items")

    seen: dict[str, int] = {}
    for index, proposal in enumerate(draft.proposals):
        _check_proposal(index, proposal, out, facts)
        key = json.dumps(proposal.model_dump(mode="json"), sort_keys=True)
        title = proposal.title.strip().casefold()
        for marker in (key, "title:" + title):
            if marker in seen:
                out.add("duplicate_proposal", f"proposals[{index}]", f"duplicates proposals[{seen[marker]}]")
                break
            seen[marker] = index

    plan_level = [("insufficient_reason", draft.insufficient_reason)] if draft.insufficient_reason.strip() else []
    plan_level += [(f"unresolved_questions[{i}]", t) for i, t in enumerate(draft.unresolved_questions)]
    plan_level += [(f"limitations[{i}]", t) for i, t in enumerate(draft.limitations)]
    for where, text in plan_level:
        _check_text(text, where, out, facts, own_evidence=facts.evidence, own_references=set(facts.usable),
                    own_config=facts.config, allowed_identifiers=facts.global_identifiers)
    return PlanValidation(not out.issues, tuple(out.issues))
