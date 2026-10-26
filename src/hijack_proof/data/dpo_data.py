"""Preference pairs for DPO (plan M4).

Two kinds of pair, so that the model learns the hierarchy rather than blanket refusal:

- *injection pairs* (attacked prompts): chosen = the SFT answer to the user's task, rejected = the
  response that obeys the hidden instruction (the deterministic "hijacked" response for that
  payload, the same directive-following response used as the trusted-directive SFT target);
- *directive pairs* (the user asks for the directive themselves): chosen = obey it, rejected = the
  plain answer that ignores it.

Chosen and rejected are both deterministic functions of the SFT rows, so the pairs need no extra
generation and are reproducible. They are off-policy (the rejected side is synthetic, not
sampled from the model), which is recorded in the design notes.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hijack_proof.data import sft_data
from hijack_proof.data import userfollow as uf
from hijack_proof.eval.suites import alpaca_eval_indices

MAX_REFERENCE_CHARS = 200


def hijacked_response(kind: str, attack: dict, reference: str) -> str:
    """The response of a model that obeyed the injection for payload `kind`."""
    return uf.compliant_response(kind, attack, reference[:MAX_REFERENCE_CHARS])


def injection_pairs(rows: list[dict], attacks: dict[str, dict]) -> list[dict]:
    """Pairs from SFT rows of attacked prompts. `attacks` maps row id -> attack-success spec."""
    pairs = []
    for r in rows:
        if not r["source"].endswith("attacked") or r["id"] not in attacks:
            continue
        rejected = hijacked_response(r["payload"], attacks[r["id"]], r["response"])
        if rejected.strip() == r["response"].strip():
            continue
        pairs.append(
            {
                "id": r["id"],
                "kind": "injection",
                "payload": r["payload"],
                "instruction": r["instruction"],
                "data": r["data"],
                "chosen": r["response"],
                "rejected": rejected,
            }  # fmt: skip
        )
    return pairs


def directive_pairs(n: int = 300, seed: int = 0) -> list[dict]:
    """Pairs where the trusted user's own directive must be obeyed (rejected: plain answer)."""
    return [
        {
            "id": r["id"],
            "kind": "directive",
            "payload": r["payload"],
            "instruction": r["instruction"],
            "data": r["data"],
            "chosen": r["response"],
            "rejected": r["plain_answer"],
        }  # fmt: skip
        for r in uf.build_userfollow_training(n, seed)
    ]


def attack_specs(pool_args: dict | None = None) -> dict[str, dict]:
    """Rebuild the (deterministic) SFT pool and return id -> attack spec for attacked prompts."""
    args = {"n_bases": 1500, "n_clean": 900, "n_attacked": 1200, "n_general": 900, "seed": 0}
    args.update(pool_args or {})
    pool = sft_data.build_pool(**args, exclude_alpaca=alpaca_eval_indices())
    return {p["id"]: p["attack"] for p in pool if p.get("attack")}


def build_pairs(
    sft_path: Path, n_directive: int = 300, seed: int = 0, limit: int | None = None
) -> list[dict]:
    rows = [json.loads(line) for line in sft_path.read_text().splitlines()]
    pairs = injection_pairs(rows, attack_specs()) + directive_pairs(n_directive, seed)
    random.Random(f"dpo-pairs/{seed}").shuffle(pairs)
    return pairs[:limit] if limit else pairs
