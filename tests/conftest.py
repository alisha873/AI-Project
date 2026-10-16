"""
Shared pytest fixtures for AutoShield.

Test isolation for process-global in-memory ledgers
---------------------------------------------------
Two modules keep process-global, in-memory state that is meant to describe ONE process:

* `validation.seals` - the identity/report issuance records that back the Phase 13B.3 trust checks
  (`_IDENTITIES`, `_REPORTS`).
* `llm.rate_budget` - the shared per-minute token buckets (the module registry), ON by default for the Groq
  client.

If that state leaked between tests, test order (and the interpreter's hash seed, which affects set/dict
iteration in some code paths) could change outcomes: a report issued by one test could look "issued" to
another, or rate reservations from earlier tests could falsely exhaust a later test's budget. The autouse
fixture below gives every test an INDEPENDENT starting state and clears it again afterwards. It only ever
resets process-local trust/accounting records (never any persisted or production data), and it never runs in
the middle of a test, so a single test that exercises a multi-step identity or rate lifecycle is unaffected.
"""
from __future__ import annotations

import pytest

from llm.rate_budget import reset_rate_budgets


def _clear_process_global_ledgers() -> None:
    try:
        import validation.seals as seals

        seals._IDENTITIES.clear()
        seals._REPORTS.clear()
    except Exception:
        # The validation package may be absent in a reduced checkout; rate-budget isolation still applies.
        pass
    reset_rate_budgets()


@pytest.fixture(autouse=True)
def _isolate_process_global_ledgers():
    _clear_process_global_ledgers()
    yield
    _clear_process_global_ledgers()
