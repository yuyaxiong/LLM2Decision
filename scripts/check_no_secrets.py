"""Fail if a real credential looks like it's about to be committed.

Why this exists: `.env.example`, `.gitignore` and `*.yaml.example` are *meant* to be committed,
so a key pasted into one of them goes public. That nearly happened once — `.env.example` briefly
held a live API key while sitting outside `.gitignore`. This script is the guard against a repeat.

Two independent checks:
  1. anything shaped like an `sk-...` token, anywhere outside the gitignored local files
  2. any `api_key` / `LLM2DECISION_API_KEY` value inside a committed example or config file that
     is neither empty nor an angle-bracket placeholder

Run locally with `python3 scripts/check_no_secrets.py`; CI runs it on every push and PR.
"""

from __future__ import annotations

import pathlib
import re
import sys

# Credential shapes worth failing on. Deliberately narrow: broad entropy heuristics produce
# false positives on hashes, and this repo is full of them (provenance records sha256 values).
_SK_TOKEN = re.compile(r"sk-[A-Za-z0-9._-]{20,}")
_API_KEY_LINE = re.compile(r"^\s*(?:LLM2DECISION_)?api_key\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)

# Files that are supposed to be committed, and therefore must only carry placeholders.
COMMITTED_CONFIG_GLOBS = ("*.example", "*.example.yaml", "*.yaml", "*.yml")

SKIP_DIR_PARTS = (".git", "__pycache__", "jevbench", "nimble", "kev", "results")
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".jsonl", ".lock"}

# Mirrors .gitignore: these files are *supposed* to hold a real key on a developer machine and
# are never committed, so scanning them would produce a false alarm locally (in CI they don't
# exist, being untracked). An explicit list rather than reimplementing gitignore semantics —
# if you add another credential-bearing file to .gitignore, add its name here too.
SKIP_NAMES = {
    ".env",
    ".env.local",
    "llm2decision.yaml",
    "llm2decision.local.yaml",
}


def _is_placeholder(value: str) -> bool:
    text = value.strip().strip("\"'")
    if not text:
        return True
    return text.startswith("<") and text.endswith(">")


def scan(root: pathlib.Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in path.parts for part in SKIP_DIR_PARTS):
            continue
        if path.name in SKIP_NAMES:
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        rel = path.relative_to(root)
        if _SK_TOKEN.search(text):
            findings.append(f"{rel}: contains an sk-... shaped token")

        if any(path.match(glob) for glob in COMMITTED_CONFIG_GLOBS):
            for number, line in enumerate(text.splitlines(), 1):
                match = _API_KEY_LINE.match(line)
                if match and not _is_placeholder(match.group(1)):
                    findings.append(
                        f"{rel}:{number}: api_key is not empty and not a <placeholder>"
                    )
    return findings


def main() -> int:
    findings = scan(pathlib.Path("."))
    if not findings:
        print("OK: no credentials found in files that would be committed")
        return 0
    print("Refusing to continue — these look like real credentials:\n")
    for finding in findings:
        print(f"  {finding}")
    print(
        "\nMove real keys to .env / .env.local (gitignored), or replace them with "
        "<your-...-key> placeholders in committed example files."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
