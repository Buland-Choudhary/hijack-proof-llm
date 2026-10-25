#!/usr/bin/env python3
"""Block commits that would leak secrets, personal data, or large files into this public repo.

Usage:
    python scripts/check_repo_safety.py            # check files staged for commit
    python scripts/check_repo_safety.py --all      # check every tracked file (used in CI)
    python scripts/check_repo_safety.py FILE...    # check specific files

Exit code 1 if anything is found. Installed as a git pre-commit hook by scripts/install_hooks.sh.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024

# (name, pattern). Patterns are deliberately specific to keep false positives rare.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("GitHub token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z]{30,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI-style API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b")),
    (
        "assigned secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token)\b\s*[:=]\s*"
            r"['\"][^'\"\s]{12,}['\"]"
        ),
    ),
]

PERSONAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Personal webmail addresses. The GitHub no-reply address and example.com are fine.
    (
        "personal email",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@(?:gmail|yahoo|hotmail|outlook|icloud|proton)\.[a-z]{2,}\b"
        ),
    ),
    ("phone number", re.compile(r"(?<![\w.-])\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?![\w-])")),
]

BLOCKED_NAMES = re.compile(r"(^|/)(\.env(\..*)?|id_rsa|id_ed25519|credentials\.json)$")

# Directories whose files must never be git-ignored.
SOURCE_DIRS = ["src", "tests", "scripts", "configs", "docs", "evalsets", "results"]

# Vendored public benchmark data that legitimately contains fake phone numbers or emails
# (e.g. IFEval prompt 3198 ends with a fixed example phone number). Only the personal-data
# patterns are skipped for these files; the secret patterns still apply.
PERSONAL_DATA_OK = {"src/hijack_proof/eval/ifeval/input_data.jsonl"}

# Files allowed to contain the patterns above (the checker and its own tests).
ALLOWLIST = {"scripts/check_repo_safety.py", "tests/test_repo_safety.py"}


def staged_files() -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def check_file(path: str) -> list[str]:
    """Return a list of human-readable problems found in one file."""
    problems: list[str] = []
    p = Path(path)
    if BLOCKED_NAMES.search(path):
        problems.append(f"{path}: file name is never allowed in this repo")
    if not p.is_file():
        return problems
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        limit_mb = MAX_FILE_BYTES // 2**20
        problems.append(f"{path}: {size / 1e6:.1f} MB is over the {limit_mb} MB limit")
        return problems
    if path in ALLOWLIST:
        return problems
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return problems  # binary files are covered by the size limit and .gitignore
    patterns = SECRET_PATTERNS + ([] if path in PERSONAL_DATA_OK else PERSONAL_PATTERNS)
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            problems.append(f"{path}:{line_no}: possible {name}")
    return problems


def ignored_source_files() -> list[str]:
    """Source files that .gitignore silently hides (they would never reach GitHub)."""
    out = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", *SOURCE_DIRS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [f for f in out.splitlines() if f and "__pycache__" not in f and not f.endswith(".pyc")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*")
    parser.add_argument("--all", action="store_true", help="check all tracked files")
    args = parser.parse_args(argv)

    files = args.files or (tracked_files() if args.all else staged_files())
    problems = [problem for f in files for problem in check_file(f)]
    if not args.files:
        problems += [f"{f}: source file is hidden by .gitignore" for f in ignored_source_files()]
    if problems:
        print("Repo safety check FAILED. Nothing was committed:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"Repo safety check passed ({len(files)} files).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
