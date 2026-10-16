"""
Phase 12 - context-aware remediation planner (advisory only).

- `remediation.contract`       versioned plan contract (draft models for the LLM, output models for storage).
- `remediation.configuration`  configuration facts (kept apart from evidence, hypotheses and references).
- `remediation.inputs`         deterministic planner-input preparation and the sufficiency gate.
- `remediation.prompt`         system prompt, inline response schema and the model-facing view.
- `remediation.text_checks`    wording screens (compromise, confirmation, success claims, ungrounded literals).
- `remediation.grounding`      parsing and grounding validation of a model-written plan.
- `remediation.priority`       deterministic, documented priority ordering.
- `remediation.planner`        `plan_remediation` through the existing `LLMClient`.
- `remediation.report`         incident-level report built on the Phase 11D `rag` pipeline.

Nothing here executes a remediation, modifies a file or container, or launches a test. Submodules are not
imported here so that importing the package stays cheap.
"""
