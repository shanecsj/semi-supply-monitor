"""Minimal .env loader.

No python-dotenv dependency - this is ~30 lines and the project is otherwise
dependency-light.

Two rules worth stating because they are the ones that surprise people:

1. **A real environment variable always wins over the file.** So a one-off
   `$env:OPENCODE_API_KEY = "..."` overrides `.env` for that shell, and CI
   secrets are never silently shadowed by a checked-out file.
2. Loading happens in `semimon/__init__.py`, before any submodule is imported.
   It has to: `classify.py` and `chat.py` read `os.environ` at module level for
   their model defaults, so a loader called later in `main()` would run after
   those values were already frozen.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """Nearest .env at or above `start` (default: the package's project root)."""
    here = (start or Path(__file__).resolve().parent).resolve()
    for directory in [here, *here.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv(path: Optional[Path] = None, override: bool = False) -> dict:
    """Load KEY=VALUE pairs into os.environ. Returns what was applied.

    Tolerates `export KEY=value`, quoted values, inline `#` comments, and blank
    lines. Silently does nothing when there is no .env, which is the normal case
    for anyone using real environment variables.
    """
    target = path or find_dotenv()
    if not target or not target.is_file():
        return {}

    applied: dict[str, str] = {}
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            # Strip an unquoted trailing comment, but never inside quotes.
            value = value.split(" #", 1)[0].rstrip()
        if not value or value.startswith("<"):
            continue          # unfilled placeholder like <your-key-here>
        if not override and key in os.environ:
            continue          # the real environment wins
        os.environ[key] = value
        applied[key] = value
    return applied
