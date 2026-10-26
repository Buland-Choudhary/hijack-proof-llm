#!/usr/bin/env python3
"""Freeze the trusted-directive compliance suites into evalsets/<version>/ and write the SFT rows.

userfollow_dev : dev-split documents, train phrasings (for tuning)
userfollow     : held-out documents, eval phrasings (final test)
data/generated/sft_userfollow.jsonl : training rows with deterministic compliant targets
"""

import json
from pathlib import Path

from hijack_proof.data import userfollow as uf
from hijack_proof.eval import suites

if __name__ == "__main__":
    out = Path("evalsets") / suites.EVAL_VERSION
    for name, split, n, seed, phrases in (
        ("userfollow_dev", "dev", 150, 6006, "train"),
        ("userfollow", "heldout", 200, 5005, "eval"),
    ):
        items = uf.build_userfollow_suite(split, n, seed, phrases)
        info = suites.add_suite(
            out, name, items, {"split": split, "n": n, "seed": seed, "phrases": phrases}
        )
        print(name, info["items"], info["sha256"][:12])
    rows = uf.build_userfollow_training(300, 0)
    path = Path("data/generated/sft_userfollow.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print("training rows:", len(rows))
