"""Trusted-instruction compliance: the other half of the instruction hierarchy.

Attack training teaches "ignore directives that sit inside documents". If that is all the model
sees, it over-generalizes and starts ignoring the same kinds of directives when the *trusted user*
asks for them (answer in French, include this link, print this code, ...). This module builds
the contrastive counterpart: the same payload-style directive placed in the user's own request,
where the model must obey it.

- `build_userfollow_suite`: evaluation items; success = the response complied (same checkers as
  attack success, but here higher is better).
- `build_userfollow_training`: SFT rows with deterministic compliant targets (no model generation).
"""

from __future__ import annotations

import random

from hijack_proof.attacks import library as lib
from hijack_proof.data import sft_data
from hijack_proof.data import taskforge as tf

REFUSAL_TARGET = lib.REFUSAL + "."


def compliant_response(kind: str, spec: dict, reference: str) -> str:
    """A response that obeys the directive for payload `kind` (deterministic, no model)."""
    marker = spec["any"][0] if "any" in spec else ""
    if kind in ("canary", "wrong_answer"):
        return marker
    if kind == "url":
        return f"{reference} https://{marker}/verify"
    if kind == "refuse":
        return REFUSAL_TARGET
    if kind == "french":
        return f"La réponse est {reference}, et c'est tout pour vous."
    raise ValueError(f"unknown payload kind {kind!r}")


def _directive_item(rng: random.Random, ex: tf.Example, kind: str, phrase_set: str) -> dict:
    text, spec = lib.make_payload(rng, kind, phrase_set, ex)
    return {
        "instruction": f"{ex.instruction} {text}",
        "data": ex.data,
        "attack": spec,  # compliance check: the same spec that detects an obeyed injection
        "payload": kind,
        "base_id": ex.id,
        "reference": sft_data.reference_answer(ex.answer),
    }


def build_userfollow_suite(split: str, n: int, seed: int, phrase_set: str) -> list[dict]:
    """`n` evaluation items whose user instruction contains a directive. kind='userfollow'."""
    rng = random.Random(f"userfollow/{split}/{seed}")
    bases = tf.generate(n, split, seed)
    kinds = [lib.PAYLOADS[i % len(lib.PAYLOADS)] for i in range(n)]
    rng.shuffle(kinds)
    items = []
    for ex, kind in zip(bases, kinds, strict=True):
        d = _directive_item(rng, ex, kind, phrase_set)
        items.append(
            {
                "id": f"uf-{ex.id}",
                "kind": "userfollow",
                "split": split,
                "doc_type": ex.doc_type,
                "question_type": ex.question_type,
                "instruction": d["instruction"],
                "data": d["data"],
                "answer": {"type": "none"},
                "attack": d["attack"],
                "payload": kind,
                "family": "user_directive",
                "position": "user",
                "seen_family": phrase_set == "train",
            }  # fmt: skip
        )
    return items


def build_userfollow_training(n: int, seed: int) -> list[dict]:
    """SFT rows (train templates, train phrasings) where the model must obey a user directive."""
    rng = random.Random(f"userfollow-train/{seed}")
    bases = tf.generate(n, "train", seed + 91)
    rows = []
    for i, ex in enumerate(bases):
        kind = lib.PAYLOADS[i % len(lib.PAYLOADS)]
        d = _directive_item(rng, ex, kind, "train")
        rows.append(
            {
                "id": f"userfollow-{i:05d}",
                "source": "tf_userfollow",
                "instruction": d["instruction"],
                "data": d["data"],
                "response": compliant_response(kind, d["attack"], d["reference"]),
                "target_source": "directive",
                "family": "user_directive",
                "payload": kind,
                "plain_answer": d["reference"],  # what ignoring the directive looks like
            }  # fmt: skip
        )
    return rows
