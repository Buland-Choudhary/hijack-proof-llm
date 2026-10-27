"""Train the red-team attacker with GRPO against a frozen defender, or sample attacks from it.

    hp-attacker train  --defender-gguf models/gguf/D0-q8_0.gguf --run atk-r1 --steps 40
    hp-attacker sample --adapter runs/redteam/atk-r1/adapter --defender-gguf G --n 200 --out a.jsonl

The attacker is SmolLM2-360M-Instruct with a fresh LoRA. The defender answers through llama-server
(GGUF), so each step costs one batch of attacker generations plus one batch of defender answers.
Everything is reproducible and resumable like the other trainers (exit code 3 = paused).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from hijack_proof import hw
from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.redteam import attacker as atk
from hijack_proof.train import grpo, sft

PAUSED = sft.PAUSED


def load_attacker(base_dir: Path, adapter: Path | None, lora_r: int, seed: int):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=hw.compute_dtype())
    if adapter is not None:
        return PeftModel.from_pretrained(model, adapter, is_trainable=True)
    torch.manual_seed(seed)
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                     target_modules="all-linear", task_type="CAUSAL_LM")  # fmt: skip
    return get_peft_model(model, cfg)


def make_scorer(args, tok, out_dir: Path):
    from hijack_proof.eval.generate import LlamaCppGenerator

    defender_enc = ChatEncoder(tok, fmt="input_role")
    generator = LlamaCppGenerator(args.defender_gguf, parallel=4, ctx=2048, port=args.port)
    scorer = atk.AttackScorer(
        generator, defender_enc,
        lambda instr, doc: defender_enc.encode_prompt(Prompt(instr, doc)),
        defender_enc.decode_response, seed=args.seed, log_path=out_dir / "attacks.jsonl",
    )  # fmt: skip
    return scorer, generator


def cmd_train(args) -> int:
    from transformers import AutoTokenizer

    out_dir = Path("runs/redteam") / args.run
    if (out_dir / "DONE").exists():
        print(f"skip {args.run} (already done)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    base_dir = Path("models") / args.model
    tok = AutoTokenizer.from_pretrained(base_dir)
    enc = ChatEncoder(tok, fmt="user_inline")
    items = atk.build_items(args.n_items, 7000 + args.seed)
    model = load_attacker(base_dir, None, args.lora_r, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer, generator = make_scorer(args, tok, out_dir)
    cfg = grpo.GRPOConfig(
        lr=args.lr, steps=args.steps, group_size=8, prompts_per_step=4, max_new_tokens=70,
        temperature=args.temperature, seed=args.seed, ckpt_minutes=3.0, audit_every=5,
    )  # fmt: skip
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    try:
        status = grpo.train(
            model, items, lambda it: enc.encode_prompt(Prompt(it["instruction"], None)),
            enc.decode_response, cfg, out_dir, enc.eos_token_id, pad_id, deadline,
            log=lambda m: print(f"{m} (+{time.time() - t0:.0f}s)", flush=True), score_fn=scorer,
        )  # fmt: skip
    finally:
        generator.close()
    if status == "paused":
        print(f"PAUSED {args.run}: checkpoint saved; run the same command again")
        return PAUSED
    model.save_pretrained(out_dir / "adapter")
    (out_dir / "DONE").write_text("ok\n")
    print(f"done: {args.run}")
    return 0


def cmd_sample(args) -> int:
    """Sample attacks from a trained attacker and test them on a defender."""
    from transformers import AutoTokenizer

    base_dir = Path("models") / args.model
    tok = AutoTokenizer.from_pretrained(base_dir)
    enc = ChatEncoder(tok, fmt="user_inline")
    items = atk.build_items(args.n, args.seed, split=args.split)
    model = load_attacker(base_dir, args.adapter, args.lora_r, 0) if args.adapter else None
    if model is None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=hw.compute_dtype())
    model.eval()
    cfg = grpo.GRPOConfig(group_size=1, max_new_tokens=70, temperature=args.temperature)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    prompts = [enc.encode_prompt(Prompt(it["instruction"], None)) for it in items]
    texts, fins = [], []
    for start in range(0, len(items), 8):
        rolls = grpo.sample_rollouts(
            model, prompts[start : start + 8], cfg, enc.eos_token_id, pad_id, args.seed + start
        )  # fmt: skip
        for group in rolls:
            resp, fin = group[0]
            texts.append(enc.decode_response(resp))
            fins.append(fin)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    args.log = out
    scorer, generator = make_scorer(args, tok, out.parent)
    scorer.log_path = out
    try:
        scorer(items, texts, fins)
    finally:
        generator.close()
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    valid = [r for r in rows if r["valid"]]
    summary = {
        "n": len(rows), "valid_rate": len(valid) / len(rows),
        "success_rate": sum(r["success"] for r in valid) / len(rows),
        "success_given_valid": sum(r["success"] for r in valid) / max(1, len(valid)),
        "defender": str(args.defender_gguf), "adapter": str(args.adapter),
    }  # fmt: skip
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "sample"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="smollm2-360m", help="attacker base under models/")
        p.add_argument("--defender-gguf", required=True, type=Path)
        p.add_argument("--port", type=int, default=8091)
        p.add_argument("--lora-r", type=int, default=16)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--temperature", type=float, default=0.7)
    t = sub.choices["train"]
    t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=40)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--n-items", type=int, default=400)
    s = sub.choices["sample"]
    s.add_argument("--adapter", type=Path, help="omit to sample from the untrained attacker")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--split", default="train")
    s.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return cmd_train(args) if args.cmd == "train" else cmd_sample(args)


if __name__ == "__main__":
    sys.exit(main())
