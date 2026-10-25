#!/usr/bin/env python3
"""Download Alpaca-cleaned and freeze the SecAlign-style injection suite into evalsets/<version>/.

One-time step; refuses to overwrite an existing frozen suite.
"""

import json
import subprocess
from pathlib import Path

from hijack_proof.eval import suites

if __name__ == "__main__":
    src = suites.ALPACA_SOURCE
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sSL", "-o", str(src), suites.ALPACA_URL], check=True)
    items, provenance = suites.build_alpaca_suite(src)
    out = Path("evalsets") / suites.EVAL_VERSION
    print(json.dumps(suites.add_suite(out, suites.ALPACA_SUITE, items, provenance), indent=2))
