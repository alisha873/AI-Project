from pathlib import Path
from datetime import datetime
import re
import shutil

source = Path("remediation/text_checks.py")
tests = Path("tests/test_confirmation_negation_regression.py")

if not source.exists():
    raise SystemExit("ERROR: Run this from the AutoShield project root.")

original = source.read_text()
match = re.search(
    r"(?m)^def confirmation_claims\([^)]*\)\s*->\s*list\[str\]:",
    original,
)
if not match:
    raise SystemExit("ERROR: Could not locate confirmation_claims. No changes made.")

if "def _confirmation_negated(text, match):" in original:
    raise SystemExit("Patch helper already exists. No changes made.")

# Patch only the confirmation_claims function.
next_def = re.search(r"(?m)^def \w+\(", original[match.end():])
end = match.end() + next_def.start() if next_def else len(original)
block = original[match.start():end]

old_filter = "if not _negated(text, match.start())"
if block.count(old_filter) != 2:
    raise SystemExit(
        f"Expected 2 filters in confirmation_claims; found {block.count(old_filter)}. "
        "No changes made."
    )

block = block.replace(
    old_filter,
    "if not _confirmation_negated(text, match)",
)

helper = r'''
def _confirmation_negated(text: str, match: re.Match) -> bool:
    """Recognize explicit denials without suppressing later positive claims."""
    if _negated(text, match.start()):
        return True

    prefix = text[max(0, match.start() - 160):match.start()]

    # Example: "observed activity rather than confirmed exploitation"
    if re.search(r"\brather\s+than\s+$", prefix, re.IGNORECASE):
        return True

    # Example: "No server-side vulnerability or successful query
    # manipulation has been proven by the evidence."
    if (
        match.group(0).casefold() == "has been proven"
        and re.search(
            r"\bno\b"
            r"(?:(?!\b(?:but|however|although|though|yet|whereas|while|instead|rather)\b)"
            r"[^,.;:!?]){1,120}$",
            prefix,
            re.IGNORECASE,
        )
    ):
        return True

    return False


'''

# Insert helper immediately before confirmation_claims.
patched = original[:match.start()] + helper + block + original[end:]
compile(patched, str(source), "exec")

backup = source.with_name(
    source.name + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
)
shutil.copy2(source, backup)
source.write_text(patched)

regression_tests = '''from remediation.text_checks import confirmation_claims


def test_negated_proof_is_not_reported_as_confirmation():
    text = (
        "No server-side vulnerability or successful query manipulation "
        "has been proven by the evidence."
    )
    assert confirmation_claims(text) == []


def test_positive_proof_is_still_reported():
    claims = confirmation_claims("The vulnerability has been proven.")
    assert any("has been proven" in claim.lower() for claim in claims)


def test_rather_than_confirmed_exploitation_is_not_reported():
    text = (
        "Suspicious requests represent observed activity rather than "
        "confirmed exploitation."
    )
    assert confirmation_claims(text) == []


def test_positive_confirmation_after_denial_is_still_reported():
    text = "No vulnerability has been proven, but exploitation is confirmed."
    claims = confirmation_claims(text)
    assert any("exploitation" in claim.lower() for claim in claims)
'''

if not tests.exists():
    tests.write_text(regression_tests)
else:
    print(f"NOTE: Leaving existing {tests} unchanged.")

print(f"Patched: {source}")
print(f"Backup:  {backup}")
print("Next: run the regression tests.")
