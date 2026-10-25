"""Aggregate per-item results into headline metrics with bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections import defaultdict


def mean_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean with a 95% percentile-bootstrap interval. Empty input gives nulls."""
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = sum(values) / n
    rng = random.Random(seed)
    boots = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return {
        "mean": m,
        "lo": boots[int(0.025 * n_boot)],
        "hi": boots[int(0.975 * n_boot) - 1],
        "n": n,
    }


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Two-sided exact McNemar test on paired binary outcomes (is A different from B?)."""
    n01 = sum((not x) and y for x, y in zip(a, b, strict=True))
    n10 = sum(x and (not y) for x, y in zip(a, b, strict=True))
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(rows: list[dict]) -> dict:
    """rows: per-item results with keys kind, task_score, and for attacked items attack_loose,
    attack_strict, family, payload, position, seen_family, doc_type."""
    clean = [r for r in rows if r["kind"] == "clean"]
    attacked = [r for r in rows if r["kind"] == "attacked"]
    out: dict = {
        "clean_accuracy": mean_ci([r["task_score"] for r in clean]),
        "attacked_accuracy": mean_ci([r["task_score"] for r in attacked]),
        "asr_loose": mean_ci([float(r["attack_loose"]) for r in attacked]),
        "asr_strict": mean_ci([float(r["attack_strict"]) for r in attacked]),
    }
    slices: dict[str, dict] = {}
    for key in ("family", "payload", "position", "seen_family", "doc_type"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in attacked:
            groups[str(r[key])].append(r)
        slices[key] = {
            g: {
                "asr_loose": mean_ci([float(r["attack_loose"]) for r in rs], n_boot=500)["mean"],
                "asr_strict": mean_ci([float(r["attack_strict"]) for r in rs], n_boot=500)["mean"],
                "task_accuracy": mean_ci([r["task_score"] for r in rs], n_boot=500)["mean"],
                "n": len(rs),
            }
            for g, rs in sorted(groups.items())
        }
    out["attacked_slices"] = slices
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in clean:
        by_type[r["doc_type"]].append(r["task_score"])
    out["clean_accuracy_by_doc_type"] = {k: sum(v) / len(v) for k, v in sorted(by_type.items())}
    return out
