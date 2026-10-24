from pathlib import Path
from datetime import datetime
import shutil

path = Path("rag/citations.py")
if not path.exists():
    raise SystemExit("ERROR: rag/citations.py not found. No changes made.")

lines = path.read_text().splitlines(keepends=True)
updated = False

old = r"[^.;\n]{0,40}"
new = (
    r"(?:(?!\b(?:but|however|although|though|yet|whereas|while|instead)\b)"
    r"[^.;,\n]){0,40}"
)

for i, line in enumerate(lines):
    if (
        "root cause|vulnerability|sql injection|injection|exploitation|attack" in line
        and "confirmed" in line
        and old in line
    ):
        lines[i] = line.replace(old, new, 1)
        updated = True
        break

if not updated:
    raise SystemExit(
        "ERROR: Could not find the expected confirmation pattern. "
        "No files changed."
    )

content = "".join(lines)
compile(content, str(path), "exec")

backup = path.with_name(
    path.name + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S")
)
shutil.copy2(path, backup)
path.write_text(content)

print(f"Updated: {path}")
print(f"Backup:  {backup}")
