"""SFT command line: fine-tune a base model on an SFT jsonl file (resumable).

    hp-sft --model smollm2-360m --data data/generated/sft_smollm2-360m.jsonl --run sft360-full \
           --format input_role --lr 1e-5 --epochs 2
    hp-sft --model qwen3-0.6b --data ... --run sft06-lora --format input_role --lora-r 32 --lr 2e-4

Outputs: runs/sft/<run>/ (checkpoint, metrics.jsonl, config.json) and, when finished, either
runs/sft/<run>/adapter (LoRA) or a merged-free full model in models/hp-<run>/ so that
`hp-eval --model hp-<run>` works. Exit code 3 means paused at the time budget: run it again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from hijack_proof import hw
from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.train import sft

PAUSED = sft.PAUSED


def load_examples(
    path: Path | list[Path], enc: ChatEncoder, max_len: int, target_source: str = "all"
) -> tuple[list, dict]:
    paths = [path] if isinstance(path, Path) else list(path)
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    if target_source != "all":  # e.g. "self,directive": keep only rows with these target sources
        allowed = set(target_source.split(","))
        rows = [r for r in rows if r.get("target_source") in allowed]
    examples, dropped = [], 0
    for r in rows:
        ids, labels = enc.encode(Prompt(r["instruction"], r["data"]), r["response"])
        if len(ids) > max_len:
            dropped += 1
            continue
        examples.append((ids, labels))
    info = {
        "rows": len(rows), "kept": len(examples), "dropped_too_long": dropped,
        "tokens": sum(len(x) for x, _ in examples),
        "target_tokens": sum(sum(t != -100 for t in y) for _, y in examples),
        "data_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest(),
    }  # fmt: skip
    return examples, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="base model folder under models/")
    ap.add_argument("--data", required=True, type=Path, nargs="+", help="one or more jsonl files")
    ap.add_argument("--run", required=True, help="run name (folder under runs/sft/)")
    ap.add_argument("--format", default="input_role", choices=["input_role", "user_inline"])
    ap.add_argument("--lora-r", type=int, default=0, help="0 = full fine-tuning")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-tokens-per-batch", type=int, default=4096)
    ap.add_argument("--batches-per-step", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--limit", type=int, help="use only the first N training rows (smoke tests)")
    ap.add_argument(
        "--target-source",
        default="all",
        help="comma-separated target sources to train on: all, self, reference, directive",
    )
    args = ap.parse_args(argv)

    out_dir = Path("runs/sft") / args.run
    if (out_dir / "DONE").exists():
        print(f"skip {args.run} (already done)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    if args.max_minutes is not None:
        own = time.time() + 60 * args.max_minutes
        deadline = own if deadline is None else min(deadline, own)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = Path("models") / args.model
    tok = AutoTokenizer.from_pretrained(base_dir)
    enc = ChatEncoder(tok, fmt=args.format)
    if args.limit:
        limited = out_dir / "data_limited.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [ln for p in args.data for ln in p.read_text().splitlines(keepends=True)]
        limited.write_text("".join(lines[: args.limit]))
        args.data = [limited]
    examples, info = load_examples(args.data, enc, args.max_len, args.target_source)
    info["target_source"] = args.target_source
    print(f"data: {info}", flush=True)

    torch.manual_seed(args.seed)
    full = args.lora_r == 0
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, dtype=torch.float32 if full else hw.compute_dtype()
    )
    if not full:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                       target_modules="all-linear", task_type="CAUSAL_LM"),
        )  # fmt: skip
        model.print_trainable_parameters()
    cfg = sft.SFTConfig(
        lr=args.lr, epochs=args.epochs, max_tokens_per_batch=args.max_tokens_per_batch,
        batches_per_step=args.batches_per_step, warmup_steps=args.warmup_steps, seed=args.seed,
        ckpt_minutes=args.ckpt_minutes,
    )  # fmt: skip
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    status = sft.train(
        model, examples, cfg, out_dir, pad_id, deadline,
        log=lambda m: print(f"{m} (+{time.time() - t0:.0f}s)", flush=True),
    )  # fmt: skip
    if status == "paused":
        print(f"PAUSED {args.run}: checkpoint saved; run the same command again")
        return PAUSED

    if full:
        final_dir = Path("models") / f"hp-{args.run}"
        model.to(torch.bfloat16).save_pretrained(final_dir)
        tok.save_pretrained(final_dir)
        (final_dir / "SOURCE.json").write_text(
            json.dumps(
                {
                    "name": f"hp-{args.run}",
                    "source": "trained here",
                    "base": args.model,
                    "data_sha256": info["data_sha256"],
                    "format": args.format,
                },
                indent=2,
            )  # fmt: skip
        )
    else:
        model.save_pretrained(out_dir / "adapter")
    metrics = [json.loads(line) for line in (out_dir / "metrics.jsonl").read_text().splitlines()]
    summary = {
        "run": args.run, "base": args.model, "format": args.format, "lora_r": args.lora_r,
        "config": cfg.__dict__, "data": info, "steps": metrics[-1]["step"],
        "first_loss": metrics[0]["loss"], "final_loss": metrics[-1]["loss"],
        "train_seconds": metrics[-1]["seconds"],
    }  # fmt: skip
    Path("results/sft").mkdir(parents=True, exist_ok=True)
    Path(f"results/sft/train_{args.run}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "DONE").write_text("ok\n")
    print(f"done: {args.run} loss {summary['first_loss']:.3f} -> {summary['final_loss']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
