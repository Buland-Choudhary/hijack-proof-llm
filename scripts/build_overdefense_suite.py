#!/usr/bin/env python3
"""Freeze the over-defense suite (harmless-instruction documents) into evalsets/<version>/.

One-time step; refuses to overwrite an existing frozen suite.
"""

import json
from pathlib import Path

from hijack_proof.data.overdefense import build_overdefense
from hijack_proof.eval import suites

if __name__ == "__main__":
    out = Path("evalsets") / suites.EVAL_VERSION
    extra = {"generator": "hijack_proof.data.overdefense", "seed": 4004, "n": 200}
    print(
        json.dumps(
            suites.add_suite(out, "overdefense", build_overdefense(200, 4004), extra), indent=2
        )
    )
