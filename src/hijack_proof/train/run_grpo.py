"""GRPO command line: reinforce a SFT/DPO LoRA adapter with verifiable rewards (resumable).

    hp-grpo --model smollm2-360m --adapter runs/sft/sft360-lora-uf/adapter --run grpo360 \
            --task defense --steps 100
    hp-grpo ... --task sanity        # tiny format-learning task: reward must go up

Tasks: `defense` mixes attacked prompts (reward: answer the user, do not obey), clean prompts and
user-written directives (reward: obey) from the TaskForge *train* split with fresh seeds;
`sanity` rewards short correct answers. Outputs go to runs/grpo/<run>/ (+ adapter when done).
Exit code 3 means paused at the time budget: run the same command again.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.data import sft_data
from hijack_proof.data import taskforge as tf
from hijack_proof.data import userfollow as uf
from hijack_proof.train import grpo, sft

PAUSED = sft.PAUSED
KEYS = ("reward", "task", "obeyed", "format_violation")


def build_items(task: str, n: int, seed: int) -> list[dict]:
    """Training prompts with their reward specs. Never uses dev/heldout documents."""
    if task == "sanity":
        bases = tf.generate(n, "train", seed)
        return [
            {"kind": "sanity_short", "instruction": b.instruction, "data": b.data,
             "answer": b.answer, "attack": None}
            for b in bases
        ]  # fmt: skip
    pool = sft_data.build_pool(n_bases=n, n_clean=n // 4, n_attacked=n // 2, n_general=0, seed=seed)
    kinds = {"tf_clean": "clean", "tf_attacked": "attacked"}
    items = [
        {"kind": kinds[p["source"]], "instruction": p["instruction"], "data": p["data"],
         "answer": p["answer"], "attack": p.get("attack")}
        for p in pool
    ]  # fmt: skip
    rng = random.Random(f"grpo-directive/{seed}")
    for i, ex in enumerate(tf.generate(n // 4, "train", seed + 31)):
        d = uf._directive_item(rng, ex, tf_payload(i), "train")
        items.append({"kind": "directive", "instruction": d["instruction"], "data": d["data"],
                      "answer": {"type": "none"}, "attack": d["attack"]})  # fmt: skip
    random.Random(f"grpo-shuffle/{seed}").shuffle(items)
    return items


def tf_payload(i: int) -> str:
    from hijack_proof.attacks import library as lib

    return lib.PAYLOADS[i % len(lib.PAYLOADS)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, type=Path, help="starting LoRA adapter")
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", default="defense", choices=["defense", "sanity"])
    ap.add_argument("--format", default="input_role", choices=["input_role", "user_inline"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--beta-kl", type=float, default=0.0)
    ap.add_argument("--updates-per-rollout", type=int, default=1)
    ap.add_argument("--n-items", type=int, default=800, help="size of the prompt pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit-every", type=int, default=10)
    ap.add_argument("--ckpt-minutes", type=float, default=3.0)
    ap.add_argument("--max-minutes", type=float)
    args = ap.parse_args(argv)

    out_dir = Path("runs/grpo") / args.run
    if (out_dir / "DONE").exists():
        print(f"skip {args.run} (already done)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    if args.max_minutes is not None:
        own = time.time() + 60 * args.max_minutes
        deadline = own if deadline is None else min(deadline, own)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = Path("models") / args.model
    tok = AutoTokenizer.from_pretrained(base_dir)
    enc = ChatEncoder(tok, fmt=args.format)
    items = build_items(args.task, args.n_items, 4000 + args.seed)
    counts = {k: sum(it["kind"] == k for it in items) for k in sorted({i["kind"] for i in items})}
    print(f"prompt pool: {len(items)} items {counts}", flush=True)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    ref_adapter = None
    if args.beta_kl > 0:
        model.load_adapter(args.adapter, adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
        ref_adapter = "ref"
    cfg = grpo.GRPOConfig(
        lr=args.lr, steps=args.steps, group_size=args.group_size,
        prompts_per_step=args.prompts_per_step, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, beta_kl=args.beta_kl,
        updates_per_rollout=args.updates_per_rollout, seed=args.seed,
        ckpt_minutes=args.ckpt_minutes, audit_every=args.audit_every,
    )  # fmt: skip
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    status = grpo.train(
        model, items, lambda it: enc.encode_prompt(Prompt(it["instruction"], it["data"])),
        enc.decode_response, cfg, out_dir, enc.eos_token_id, pad_id, deadline,
        log=lambda m: print(f"{m} (+{time.time() - t0:.0f}s)", flush=True), ref_adapter=ref_adapter,
    )  # fmt: skip
    if status == "paused":
        print(f"PAUSED {args.run}: checkpoint saved; run the same command again")
        return PAUSED
    model.save_pretrained(out_dir / "adapter", selected_adapters=["default"])
    rows = [json.loads(ln) for ln in (out_dir / "metrics.jsonl").read_text().splitlines()]
    k = max(1, min(10, len(rows) // 4))
    first, last = rows[:k], rows[-k:]
    mean = lambda rs, key: sum(r[key] for r in rs) / len(rs)  # noqa: E731
    summary = {
        "run": args.run, "task": args.task, "base": args.model, "start_adapter": str(args.adapter),
        "config": cfg.__dict__, "steps": rows[-1]["step"], "seconds": rows[-1]["seconds"],
        "window": k,
        "first": {key: mean(first, key) for key in KEYS},
        "last": {key: mean(last, key) for key in KEYS},
    }  # fmt: skip
    Path("results/grpo").mkdir(parents=True, exist_ok=True)
    Path(f"results/grpo/train_{args.run}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "DONE").write_text("ok\n")
    r0, r1 = summary["first"]["reward"], summary["last"]["reward"]
    print(f"done: {args.run} reward {r0:.3f} -> {r1:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
