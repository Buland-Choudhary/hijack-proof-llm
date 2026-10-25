#!/usr/bin/env python3
"""Regenerate and freeze the evaluation suites into evalsets/<version>/ (with a sha256 manifest).

Only run this when deliberately creating a NEW eval version (bump EVAL_VERSION first); frozen
sets must never change in place.
"""

import json
from pathlib import Path

from hijack_proof.eval.suites import EVAL_VERSION, freeze

if __name__ == "__main__":
    out = Path("evalsets") / EVAL_VERSION
    if (out / "MANIFEST.json").exists():
        raise SystemExit(
            f"{out} already exists and is frozen; bump EVAL_VERSION for a new version."
        )
    print(json.dumps(freeze(out), indent=2))
