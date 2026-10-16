# AutoShield — Project Specification

> Purpose: persistent, human-and-agent-readable description of the intended project scope.
> This is a **specification** (intent), not a status report. Implementation status lives in `PROJECT_STATUS.md`,
> which will be created after a read-only repository audit.

## 1. Overview

AutoShield is an evidence-grounded AI cybersecurity system for:

- post-incident forensic reconstruction
- attack-path reasoning
- root-cause analysis
- remediation planning
- controlled penetration-testing validation
- explainability
- residual-risk estimation

- Current development branch: `ph10-16_fixed`
- Developer-reported state (**unverified**): previously failing tests were fixed, the full test suite passed in the latest run, and the changes were committed and pushed to GitHub.

## 2. Phase Architecture

| Phase | Name |
|-------|------|
| 0 | Architecture and data contracts |
| 1 | Docker cyber range |
| 2 | Controlled incident generation |
| 3 | Incident package builder |
| 4 | Evidence normalization |
| 5 | Evidence Extraction Agent |
| 6 | Deterministic baseline attack graph |
| 7 | GNN-based attack graph reasoning |
| 8 | Incident Reconstruction Agent |
| 9 | Root Cause Agent |
| 10 | Security Knowledge Base |
| 11 | Dual-Vector Retrieval-Augmented Generation (RAG) |
| 12 | Context-aware Remediation Planner |
| 13 | Controlled, scenario-aware penetration-testing orchestrator |
| 14 | Judge Agent for remediation evaluation |
| 15 | SHAP-based explainability |
| 16 | Bayesian residual-risk estimation |
| 20 | Streamlit dashboard and end-to-end results presentation |

Phase numbering is established. Do not renumber phases. (Phases 17–19 are intentionally not part of this scope.)

### Reported status (NOT independently verified)

- Phases 0–9 are **reported** as implemented. This has not been verified against the current repository.
- Remaining assigned work: Phases 10–16 and Phase 20.
- Verified status must be recorded in `PROJECT_STATUS.md` after an audit; until then, treat all status claims as unconfirmed.

## 3. Intended Functionality (Phases 10–16, 20)

### Phase 10 — Security Knowledge Base
Organize security knowledge from suitable sources such as OWASP, CVE, CWE, CAPEC, MITRE ATT&CK and Docker security guidance. Establish:
- document ingestion
- provenance metadata
- chunking
- embeddings
- persistent retrieval storage

### Phase 11 — Dual-Vector RAG
Two complementary retrieval paths:
1. one specializing in vulnerability and weakness information
2. one specializing in attack techniques and behavior

Combine retrieved evidence with traceable source references for downstream remediation reasoning.

### Phase 12 — Context-aware Remediation Planner
Combine:
- the root-cause hypothesis
- supporting forensic evidence
- infrastructure/configuration context
- retrieved security knowledge

to produce structured, evidence-grounded remediation proposals. Each proposal includes affected components, rationale, proposed changes, expected effects and safety considerations.

### Phase 13 — Penetration-Testing Orchestrator
Controlled validation in the **authorized, isolated Docker cyber range**. Use scenario-specific validation modules and evidence to check whether the original attack objective remains achievable after remediation. Must not introduce uncontrolled attacks or expose public systems.

### Phase 14 — Judge Agent
Evaluate remediation proposals using **actual post-remediation validation evidence**. Produce an interpretable decision, rationale and confidence information. A remediation must not be accepted solely because an LLM says it is safe.

### Phase 15 — SHAP Explainability
Feature-attribution explanations for an **explicitly defined model and prediction**. SHAP values explain model behavior; they are not proof of causality or correctness.

### Phase 16 — Bayesian Residual-Risk Estimation
Estimate remaining security risk after remediation and represent predictive uncertainty. The model's inputs, outputs, assumptions and evaluation method must be defined. A successful validation test must not be equated with zero residual risk.

### Phase 20 — Streamlit Dashboard
Present actual end-to-end pipeline outputs:
- incident reconstruction
- root cause
- retrieved evidence
- proposed remediation
- validation results
- Judge decision
- explanations
- residual risk

The dashboard must consume **real pipeline outputs**, not fabricated demonstrations.

## 4. Core Research and Engineering Principles

1. Preserve evidence provenance.
2. Distinguish **observed activity** from **configured relationships** and **inferred hypotheses**.
3. No evidence → no claim.
4. Never give the model hidden evaluation ground truth as incident evidence.
5. Do not present correlation as conclusive attribution.
6. Distinguish validation success under tested conditions from a guarantee of universal security.
7. Prefer reproducible experiments, explicit data contracts, meaningful tests and measurable evaluation.
8. Do not claim novelty, patentability, production readiness or generalization without evidence.

## 5. Safety and Scope Boundaries

- All penetration-testing/attack activity is confined to the authorized, isolated Docker cyber range.
- No uncontrolled attacks; no targeting of public or third-party systems.

## 6. Items Requiring Verification

These are unconfirmed and must be resolved by a read-only audit (recorded in `PROJECT_STATUS.md`):

- Whether Phases 0–9 are actually implemented in the current repository, and to what degree.
- Whether the full test suite passes on `ph10-16_fixed`, and whether the latest commit is pushed to the remote.
- Whether any work for Phases 10–16 and 20 already exists, partially or fully.
- Existing data contracts, module layout, and public interfaces.
- Dependencies in use (e.g. `requirements.txt`) versus those needed for later phases.
- Whether local uncommitted changes exist (these may differ from the developer's report).
- Concrete choices still undefined: knowledge-base sources/versions, embedding model, vector store, LLM provider, scenario list, SHAP target model, Bayesian model design and its evaluation protocol.
