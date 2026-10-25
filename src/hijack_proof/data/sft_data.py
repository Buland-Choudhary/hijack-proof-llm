"""Build SFT training data: clean + attacked TaskForge examples and general instructions.

Targets (plan D11): the base model answers the *clean* version of each prompt; an answer is used
as the target only if it is verified (TaskForge: the checker scores it 1.0; general prompts: it
finished with an end-of-turn token). Unverified TaskForge answers fall back to a concise
reference answer built from the gold answer. Unverified general answers are dropped. The attacked
version of a prompt gets the same target as its clean version: always answer the user's task.

Pipeline (resumable): pool -> targets (model generation, chunked) -> rows (jsonl).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hijack_proof.attacks import library as lib
from hijack_proof.data import taskforge as tf
from hijack_proof.eval import checkers

ALPACA_PATH = Path("data/raw/alpaca_data_cleaned.json")


def reference_answer(spec: dict) -> str:
    """A concise correct answer written from the gold answer (fallback target)."""
    kind = spec["type"]
    if kind == "contains":
        gold = spec["gold"]
        return str(gold[0] if isinstance(gold, list) else gold)
    if kind == "number":
        g = float(spec["gold"])
        return str(int(g)) if g == int(g) else f"{g:.1f}"
    if kind == "yesno":
        return spec["gold"].capitalize()
    if kind == "all_of":
        return ", ".join(spec["gold"])
    if kind == "json":
        return json.dumps(spec["gold"])
    raise ValueError(f"cannot write a reference answer for spec type {kind!r}")


def load_alpaca(exclude: set[int], path: Path = ALPACA_PATH) -> list[tuple[int, dict]]:
    rows = json.loads(path.read_text())
    return [(i, r) for i, r in enumerate(rows) if i not in exclude and r["instruction"].strip()]


def build_pool(
    n_bases: int = 1500, n_clean: int = 900, n_attacked: int = 1200, n_general: int = 900,
    seed: int = 0, exclude_alpaca: set[int] | None = None, alpaca_path: Path = ALPACA_PATH,
) -> list[dict]:  # fmt: skip
    """The prompts to train on. Each item: id, source, base_key, instruction, data (as shown to
    the model), clean_data (the document without an attack), answer spec (TaskForge) or None."""
    rng = random.Random(f"sft-pool/{seed}")
    bases = tf.generate(n_bases, "train", seed)
    pool: list[dict] = []

    def item(source, base_key, instruction, data, clean_data, answer, extra=None):
        return {
            "id": f"{source}-{len(pool):05d}", "source": source, "base_key": base_key,
            "instruction": instruction, "data": data, "clean_data": clean_data,
            "answer": answer, **(extra or {}),
        }  # fmt: skip

    for ex in rng.sample(bases, min(n_clean, len(bases))):
        pool.append(item("tf_clean", ex.id, ex.instruction, ex.data, ex.data, ex.answer))
    attacked_bases = [bases[rng.randrange(len(bases))] for _ in range(n_attacked)]
    for ex, atk in zip(
        attacked_bases, lib.build_training_attacks(attacked_bases, seed), strict=True
    ):
        extra = {"family": atk.family, "payload": atk.payload, "position": atk.position,
                 "attack": atk.attack}  # fmt: skip
        pool.append(item("tf_attacked", ex.id, ex.instruction, atk.data, ex.data, ex.answer, extra))

    if n_general:
        rows = load_alpaca(exclude_alpaca or set(), alpaca_path)
        with_input = [(i, r) for i, r in rows if 20 <= len(r["input"].strip()) <= 800]
        no_input = [(i, r) for i, r in rows if not r["input"].strip()]
        n_noinp = int(0.3 * n_general)
        n_inp = n_general - n_noinp
        for i, r in rng.sample(no_input, n_noinp):
            pool.append(
                item("gen_clean", f"alpaca-{i}", r["instruction"].strip(), None, None, None)
            )
        chosen = rng.sample(with_input, n_inp)
        half = n_inp // 2
        for i, r in chosen[:half]:
            pool.append(item("gen_clean", f"alpaca-{i}", r["instruction"].strip(),
                             r["input"].strip(), r["input"].strip(), None))  # fmt: skip
        faux = [
            tf.Example(f"alpaca-{i}", "train", "alpaca", "alpaca", "alpaca",
                       r["instruction"].strip(), r["input"].strip(), {"type": "none"})
            for i, r in chosen[half:]
        ]  # fmt: skip
        for (i, r), atk in zip(
            chosen[half:], lib.build_training_attacks(faux, seed + 7), strict=True
        ):
            extra = {"family": atk.family, "payload": atk.payload, "position": atk.position,
                     "attack": atk.attack}  # fmt: skip
            pool.append(item("gen_attacked", f"alpaca-{i}", r["instruction"].strip(), atk.data,
                             r["input"].strip(), None, extra))  # fmt: skip
    return pool


def target_prompts(pool: list[dict]) -> list[dict]:
    """One generation job per unique base prompt (clean version), in pool order."""
    seen: dict[str, dict] = {}
    for p in pool:
        seen.setdefault(
            p["base_key"],
            {"id": p["base_key"], "instruction": p["instruction"], "data": p["clean_data"],
             "answer": p["answer"], "general": p["answer"] is None},
        )  # fmt: skip
    return list(seen.values())


def make_target_rows(
    chunk: list[dict], generator, encoder, max_new_general: int = 192, max_new_tf: int = 64
):
    """Generate the base model's answer for each clean prompt and verify it."""
    from hijack_proof.chat import Prompt

    rows = []
    for general in (False, True):
        group = [c for c in chunk if c["general"] == general]
        if not group:
            continue
        prompts = [encoder.encode_prompt(Prompt(c["instruction"], c["data"])) for c in group]
        outs = generator.generate(
            prompts, max_new_tokens=max_new_general if general else max_new_tf
        )
        for c, ids in zip(group, outs, strict=True):
            finished = encoder.eos_token_id in ids
            text = encoder.decode_response(ids)
            if general:
                verified = finished and len(text.split()) >= 2
            else:
                verified = checkers.score_answer(text, c["answer"]) == 1.0
            rows.append({"id": c["id"], "response": text, "verified": bool(verified)})
    return rows


def assemble(pool: list[dict], targets: dict[str, dict]) -> list[dict]:
    """Final SFT rows. Attacked prompts reuse the target of their clean version."""
    out = []
    for p in pool:
        t = targets.get(p["base_key"])
        if t is None:
            continue
        if p["answer"] is not None:  # TaskForge: verified self answer, else reference answer
            if t["verified"]:
                response, how = t["response"], "self"
            else:
                response, how = reference_answer(p["answer"]), "reference"
        else:  # general: verified self answer only
            if not t["verified"]:
                continue
            response, how = t["response"], "self"
        out.append(
            {
                "id": p["id"],
                "source": p["source"],
                "instruction": p["instruction"],
                "data": p["data"],
                "response": response,
                "target_source": how,
                "family": p.get("family"),
                "payload": p.get("payload"),
            }  # fmt: skip
        )
    return out


def stats(rows: list[dict]) -> dict:
    from collections import Counter

    by = Counter((r["source"], r["target_source"]) for r in rows)
    return {
        "rows": len(rows),
        "by_source_and_target": {f"{s}/{t}": n for (s, t), n in sorted(by.items())},
        "attacked_share": sum(r["source"].endswith("attacked") for r in rows) / max(1, len(rows)),
    }
