#!/usr/bin/env python3
"""Build DPO pairs from attacker samples.

    redteam_pairs.py OUT.jsonl ATTACKS.jsonl [ATTACKS.jsonl ...]
"""

import json
import sys
from pathlib import Path

from hijack_proof.redteam import pairs

if __name__ == "__main__":
    out, sources = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    all_pairs = []
    for src in sources:
        tmp = out.with_suffix(".tmp")
        info = pairs.write_pairs(src, tmp)
        print(src, info)
        all_pairs += [json.loads(line) for line in tmp.read_text().splitlines()]
        tmp.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in all_pairs))
    print("pairs written:", len(all_pairs), "->", out)
