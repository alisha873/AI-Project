# Phase 15 — SHAP Explainability

## Implementation
- SHAP explainer module: risk/shap_explainer.py
- Canonical risk features: risk/features.py
- Model output contract: risk/model_contract.py
- Synthetic demonstration: risk/demo_shap.py
- SHAP tests: tests/test_shap_explainer.py and tests/test_shap_completion.py

## Verification from this run
- Targeted test exit code: 0
- Synthetic demo exit code: 0

## Scope and limitations
- Synthetic demo results are not real AutoShield security findings.
- SHAP explains the output of the supplied scoring function; it does not
  independently decide PASS or FAIL.
- Real-model integration is not confirmed by this test run.
- Production use requires a real, compatible risk/Judge scoring model,
  appropriate background data, and evaluation on representative inputs.
- SHAP contributions explain model output relative to a baseline; they
  do not establish causality.

## Completion status
- Explainer and targeted tests: PASS
- Synthetic demonstration: PASS
- End-to-end real model integration: PENDING until the real model is connected.
