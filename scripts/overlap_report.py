#!/usr/bin/env python3
"""8-gram overlap between training text and the frozen eval sets (contamination check).

For each eval item, the document text (the `data` field, plus the user's instruction) is split into
word 8-grams. An item is "overlapping" if at least `--threshold` of its 8-grams also occur in the
training text (SFT rows and the DPO/userfollow rows built from the same generators). Boilerplate
shared by all TaskForge templates is expected; the report also gives the share of eval items that
share a *long* run (>= 50% of 8-grams), which would indicate leaked documents.

    python scripts/overlap_report.py \
        --train data/generated/sft_qwen3-0.6b.jsonl data/generated/sft_userfollow.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

N = 8


def grams(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + N]) for i in range(len(words) - N + 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", nargs="+", type=Path, required=True)
    ap.add_argument("--evalsets", type=Path, default=Path("evalsets/v2"))
    ap.add_argument("--out", type=Path, default=Path("results/sft/overlap_8gram.json"))
    args = ap.parse_args()

    train: set[tuple[str, ...]] = set()
    n_rows = 0
    for path in args.train:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            train |= grams((r.get("data") or "") + " " + r["instruction"])
            n_rows += 1

    report = {
        "train_files": [str(p) for p in args.train],
        "train_rows": n_rows,
        "n": N,
        "suites": {},
    }
    for suite in ("core", "dev", "userfollow", "userfollow_dev", "overdefense", "secalign"):
        path = args.evalsets / f"{suite}.jsonl"
        if not path.exists():
            continue
        fracs, by_kind = [], {}
        for line in path.read_text().splitlines():
            item = json.loads(line)
            g = grams((item.get("data") or "") + " " + item["instruction"])
            frac = len(g & train) / len(g) if g else 0.0
            fracs.append(frac)
            kind = item.get("kind", "?") + (
                "/seen"
                if item.get("seen_family")
                else "/heldout"
                if item.get("seen_family") is False
                else ""
            )
            by_kind.setdefault(kind, []).append(frac)
        report["suites"][suite] = {
            "by_kind_mean_overlap": {
                k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())
            },
            "items": len(fracs),
            "mean_overlap": round(sum(fracs) / len(fracs), 4),
            "items_with_any_overlap": round(sum(f > 0 for f in fracs) / len(fracs), 4),
            "items_with_over_50pct_overlap": round(sum(f >= 0.5 for f in fracs) / len(fracs), 4),
            "max_overlap": round(max(fracs), 4),
        }
        print(suite, report["suites"][suite])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
