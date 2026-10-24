from pathlib import Path
from datetime import datetime
import re
import shutil

path = Path("remediation/text_checks.py")
text = path.read_text()

start = re.search(
    r"(?m)^def _confirmation_negated\(text: str, match: re\.Match\) -> bool:",
    text,
)
end = re.search(r"(?m)^def confirmation_claims\(", text)

if not start or not end or start.start() >= end.start():
    raise SystemExit(
        "ERROR: Could not find the helper and its boundary. No changes made."
    )

replacement = r'''def _confirmation_negated(text: str, match: re.Match) -> bool:
    """Recognize explicit denials without suppressing later positive claims."""

    # A contrast word starts a new claim scope. For example:
    # "No vulnerability has been proven, but exploitation is confirmed."
    # Only check negation within the scope containing this match.
    prefix = text[:match.start()]
    contrasts = list(re.finditer(
        r"\b(?:but|however|although|though|yet|whereas|while|instead)\b",
        prefix,
        re.IGNORECASE,
    ))
    scope_start = contrasts[-1].start() if contrasts else 0
    scoped_text = text[scope_start:]
    scoped_start = match.start() - scope_start
    scoped_prefix = scoped_text[max(0, scoped_start - 160):scoped_start]

    if _negated(scoped_text, scoped_start):
        return True

    # Example: "observed activity rather than confirmed exploitation"
    if re.search(r"\brather\s+than\s+$", scoped_prefix, re.IGNORECASE):
        return True

    # Example: "No server-side vulnerability or successful query
    # manipulation has been proven by the evidence."
    if (
        match.group(0).casefold() == "has been proven"
        and re.search(
            r"\bno\b"
            r"(?:(?!\b(?:but|however|although|though|yet|whereas|while|instead|rather)\b)"
            r"[^,.;:!?]){1,120}$",
            scoped_prefix,
            re.IGNORECASE,
        )
    ):
        return True

    return False


'''

backup = path.with_name(
    path.name + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
)
shutil.copy2(path, backup)

updated = text[:start.start()] + replacement + text[end.start():]
compile(updated, str(path), "exec")
path.write_text(updated)

print(f"Updated: {path}")
print(f"Backup:  {backup}")
