"""
Explicit live-execution opt-in. A CLI flag alone is not authorization.

Live execution requires BOTH:
  1. the operator-facing acknowledgement flag (`--i-accept-authorized-live-validation`), and
  2. the environment variable AUTOSHIELD_LIVE_VALIDATION=1.

Neither localhost, a private IP, a Docker hostname, nor `--live` by itself grants this. Reviewer
approvals and the scope policy are separate gates and are not replaced by this opt-in.
"""

from __future__ import annotations

import os
from typing import Mapping

LIVE_OPT_IN_ENV = "AUTOSHIELD_LIVE_VALIDATION"
LIVE_OPT_IN_VALUE = "1"


def live_opt_in_granted(*, acknowledgement: bool, environ: Mapping[str, str] | None = None) -> bool:
    """True only when the acknowledgement flag is set AND the env var is exactly `1`."""
    env = environ if environ is not None else os.environ
    return bool(acknowledgement) and env.get(LIVE_OPT_IN_ENV, "") == LIVE_OPT_IN_VALUE
