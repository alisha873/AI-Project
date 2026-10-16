"""
Phase 13 - controlled post-remediation validation (Phase 13A: offline contract and safety foundation).

- `validation.contract`      the versioned `ValidationReport` contract and its status / reason-code model.
- `validation.scenario`      inert, reviewable scenario definitions and the scenario registry.
- `validation.scope`         the fail-closed scope policy (approved lab, independent identity, authorization).
- `validation.inputs`        the adapter from the Phase 12 plan result, and the remediation-application attestation.
- `validation.orchestrator`  `build_validation_report`: offline; can only yield `not_validated` or `blocked`.
- `validation.handoff`       the deterministic Phase 14 hand-off view of a report.

Phase 13A sends no traffic, runs no check, applies no remediation, calls no model and issues no Docker command.
It does not demonstrate that any remediation is effective. Submodules are not imported here so that importing
the package stays cheap.
"""
