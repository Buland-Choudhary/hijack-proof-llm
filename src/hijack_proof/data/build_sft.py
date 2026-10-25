"""Build the SFT dataset for one base model (resumable).

    hp-sftdata --model qwen3-0.6b --out data/generated/sft_qwen3-0.6b.jsonl

Step 1 generates the base model's answers to the clean prompts in chunks (saved as it goes, so a
reset or a time budget only costs the current chunk); step 2 assembles the final rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from hijack_proof.chat import ChatEncoder
from hijack_proof.data import sft_data as sd
from hijack_proof.eval.run import PAUSED, load_rows, run_chunks
from hijack_proof.eval.suites import alpaca_eval_indices


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="folder name under models/ (the base model)")
    ap.add_argument("--gguf", help="generate with llama.cpp from this GGUF file instead of HF")
    ap.add_argument("--out", type=Path, help="default: data/generated/sft_<model>.jsonl")
    ap.add_argument("--n-bases", type=int, default=1500)
    ap.add_argument("--n-clean", type=int, default=900)
    ap.add_argument("--n-attacked", type=int, default=1200)
    ap.add_argument("--n-general", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or Path("data/generated") / f"sft_{args.model}.jsonl"
    if out.exists() and not args.force:
        print(f"skip {out} (already built; use --force to rebuild)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    if args.max_minutes is not None:
        own = time.time() + 60 * args.max_minutes
        deadline = own if deadline is None else min(deadline, own)

    pool = sd.build_pool(
        args.n_bases, args.n_clean, args.n_attacked, args.n_general, args.seed,
        exclude_alpaca=alpaca_eval_indices(),
    )  # fmt: skip
    prompts = sd.target_prompts(pool)
    print(f"pool: {len(pool)} training prompts, {len(prompts)} base prompts to answer", flush=True)

    model_dir = Path("models") / args.model
    if args.gguf:
        from transformers import AutoTokenizer

        from hijack_proof.eval.generate import LlamaCppGenerator

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        gen = LlamaCppGenerator(args.gguf)
    else:
        from hijack_proof.eval.generate import HFGenerator

        gen = HFGenerator(model_dir, batch_size=args.batch_size)
        tokenizer = gen.tokenizer
    enc = ChatEncoder(tokenizer, fmt="user_inline")  # the base model has never seen the input role
    rows_path = Path("runs/sftdata") / f"{args.model}__seed{args.seed}.jsonl"
    t0 = time.time()
    try:
        rows, finished = run_chunks(
            prompts, lambda chunk: sd.make_target_rows(chunk, gen, enc), rows_path,
            args.chunk_size, deadline,
            log=lambda m: print(f"{m} ({time.time() - t0:.0f}s)", flush=True),
        )  # fmt: skip
    finally:
        if hasattr(gen, "close"):
            gen.close()
    if not finished:
        print(f"PAUSED sft data for {args.model}: {len(rows)}/{len(prompts)} answers saved")
        return PAUSED
    targets = {r["id"]: r for r in load_rows(rows_path).values()}
    final = sd.assemble(pool, targets)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final))
    st = sd.stats(final)
    Path("results/sft").mkdir(parents=True, exist_ok=True)
    Path(f"results/sft/data_stats_{args.model}.json").write_text(json.dumps(st, indent=2) + "\n")
    print(json.dumps(st, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
