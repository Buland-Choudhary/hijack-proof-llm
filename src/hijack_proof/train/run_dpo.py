"""DPO command line: continue from an SFT LoRA adapter with preference pairs (resumable).

    hp-dpo --model qwen3-0.6b --adapter runs/sft/qwen06-lora-uf/adapter \
           --sft-data data/generated/sft_qwen3-0.6b.jsonl --run qwen06-dpo --beta 0.1

The SFT adapter is both the starting point and the frozen reference (its sequence log-probs are
cached before training). Outputs: runs/dpo/<run>/ (checkpoint, metrics, config) and, when
finished, runs/dpo/<run>/adapter. Exit code 3 means paused at the time budget: run it again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.data import dpo_data
from hijack_proof.train import dpo, sft

PAUSED = sft.PAUSED


def encode_pairs(pairs: list[dict], enc: ChatEncoder, max_len: int):
    out, dropped = [], 0
    for p in pairs:
        prompt = Prompt(p["instruction"], p["data"])
        chosen = enc.encode(prompt, p["chosen"])
        rejected = enc.encode(prompt, p["rejected"])
        if max(len(chosen[0]), len(rejected[0])) > max_len:
            dropped += 1
            continue
        out.append((chosen, rejected))
    return out, dropped


def load_reference(path: Path, n: int) -> list[tuple[float, float]] | None:
    if not path.exists():
        return None
    ref = [tuple(x) for x in json.loads(path.read_text())]
    return ref if len(ref) == n else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True, type=Path, help="SFT LoRA adapter to start from")
    ap.add_argument("--sft-data", required=True, type=Path)
    ap.add_argument("--run", required=True)
    ap.add_argument("--format", default="input_role", choices=["input_role", "user_inline"])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--sft-weight", type=float, default=0.0, help="RPO NLL term weight")
    ap.add_argument("--pairs-per-step", type=int, default=16)
    ap.add_argument("--micro-pairs", type=int, default=2)
    ap.add_argument("--n-directive", type=int, default=300)
    ap.add_argument(
        "--extra-pairs", type=Path, nargs="*", default=[], help="extra pair jsonl files"
    )
    ap.add_argument("--replay-n", type=int, help="keep only N of the standard pairs (replay)")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only the first N pairs (smoke tests)")
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--max-minutes", type=float)
    args = ap.parse_args(argv)

    out_dir = Path("runs/dpo") / args.run
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
    raw = dpo_data.build_pairs(
        args.sft_data, args.n_directive, args.seed, args.replay_n or args.limit
    )
    for path in args.extra_pairs:
        raw += [json.loads(line) for line in path.read_text().splitlines()]
    if args.extra_pairs:
        random.Random(f"dpo-mix/{args.seed}").shuffle(raw)
    pairs, dropped = encode_pairs(raw, enc, args.max_len)
    info = {
        "pairs": len(pairs), "dropped_too_long": dropped,
        "injection": sum(p["kind"] == "injection" for p in raw),
        "extra": sum(str(p["id"]).startswith("redteam-") for p in raw),
        "directive": sum(p["kind"] == "directive" for p in raw),
        "sft_data_sha256": hashlib.sha256(args.sft_data.read_bytes()).hexdigest(),
    }  # fmt: skip
    print(f"data: {info}", flush=True)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    log = lambda m: print(f"{m} (+{time.time() - t0:.0f}s)", flush=True)  # noqa: E731

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "reference_logps.json"
    ref = load_reference(ref_path, len(pairs))
    if ref is None:
        ref = dpo.reference_logps(model, pairs, pad_id, args.micro_pairs, log)
        ref_path.write_text(json.dumps(ref))
    cfg = dpo.DPOConfig(
        beta=args.beta, lr=args.lr, epochs=args.epochs, pairs_per_step=args.pairs_per_step,
        micro_pairs=args.micro_pairs, sft_weight=args.sft_weight, seed=args.seed,
        ckpt_minutes=args.ckpt_minutes,
    )  # fmt: skip
    status = dpo.train(model, pairs, ref, cfg, out_dir, pad_id, deadline, log)
    if status == "paused":
        print(f"PAUSED {args.run}: checkpoint saved; run the same command again")
        return PAUSED

    model.save_pretrained(out_dir / "adapter")
    metrics = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    summary = {
        "run": args.run, "base": args.model, "start_adapter": str(args.adapter),
        "config": cfg.__dict__, "data": info, "steps": metrics[-1]["step"],
        "first": metrics[0], "final": metrics[-1], "train_seconds": metrics[-1]["seconds"],
    }  # fmt: skip
    Path("results/dpo").mkdir(parents=True, exist_ok=True)
    Path(f"results/dpo/train_{args.run}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "DONE").write_text("ok\n")
    print(f"done: {args.run} acc {metrics[0]['acc']:.2f} -> {metrics[-1]['acc']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
