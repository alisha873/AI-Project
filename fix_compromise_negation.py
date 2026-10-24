from pathlib import Path
from datetime import datetime
import re
import shutil

path = Path("remediation/text_checks.py")
test_path = Path("tests/test_compromise_negation_regression.py")

if not path.exists():
    raise SystemExit("ERROR: remediation/text_checks.py not found.")

source = path.read_text()

if "def _compromise_claim_negated(text: str, match: re.Match)" in source:
    raise SystemExit("Helper already exists; no changes made.")

start = re.search(r"(?m)^def compromise_claims\(text: str\)", source)
end = re.search(r"(?m)^def _confirmation_negated\(", source)

if not start or not end or start.start() >= end.start():
    raise SystemExit("ERROR: Could not safely locate compromise_claims. No changes made.")

block = source[start.start():end.start()]
old = "if not _negated(text, match.start(), match.end()) and not _hedged(text, match.start())"

if block.count(old) != 1:
    raise SystemExit("ERROR: Expected one compromise filter. No changes made.")

helper = r'''def _compromise_claim_negated(text: str, match: re.Match) -> bool:
    """Recognize explicit denials without suppressing claims after a contrast."""
    # Scope the check to text after the latest contrast. This prevents a
    # denial before "but" from hiding a positive claim after it.
    prefix = text[:match.start()]
    contrasts = list(re.finditer(
        r"\b(?:but|however|although|though|yet|whereas|while|instead)\b",
        prefix,
        re.IGNORECASE,
    ))
    scope_start = contrasts[-1].end() if contrasts else 0
    scoped_text = text[scope_start:]
    scoped_start = match.start() - scope_start
    scoped_end = match.end() - scope_start

    if _negated(scoped_text, scoped_start, scoped_end):
        return True

    scoped_prefix = scoped_text[max(0, scoped_start - 220):scoped_start]

    # Handles denials with a list, such as:
    # "There is no confirmation of A, successful exploitation, or database access."
    if re.search(
        r"\bno\s+(?:evidence|confirmation|indication|proof)\s+(?:of|that)\b"
        r"[^.!?;:]{0,180}$",
        scoped_prefix,
        re.IGNORECASE,
    ):
        return True

    return False


'''

updated_block = block.replace(
    old,
    "if not _compromise_claim_negated(text, match) and not _hedged(text, match.start())",
)

updated = source[:start.start()] + helper + updated_block + source[end.start():]
compile(updated, str(path), "exec")

tests = '''from remediation.text_checks import compromise_claims


def test_explicit_no_confirmation_does_not_claim_successful_exploitation():
    text = (
        "The observed evidence consists of SQL-injection-like request patterns. "
        "There is no confirmation of vulnerable server-side query construction, "
        "successful exploitation, or database access."
    )
    claims = compromise_claims(text)
    assert not any("successful exploitation" in c.lower() for c in claims)
    assert not any("database access" in c.lower() for c in claims)


def test_positive_successful_exploitation_is_still_detected():
    claims = compromise_claims("Successful exploitation occurred.")
    assert any("successful exploitation" in c.lower() for c in claims)


def test_positive_claim_after_but_is_not_suppressed_by_denial():
    text = (
        "There is no confirmation of successful exploitation, "
        "but successful exploitation occurred."
    )
    claims = compromise_claims(text)
    assert any("successful exploitation" in c.lower() for c in claims)
'''

if test_path.exists():
    raise SystemExit(f"ERROR: {test_path} already exists; no changes made.")

backup = path.with_name(
    path.name + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
)
shutil.copy2(path, backup)
path.write_text(updated)
test_path.write_text(tests)

print(f"Patched: {path}")
print(f"Backup:  {backup}")
print(f"Tests:   {test_path}")
