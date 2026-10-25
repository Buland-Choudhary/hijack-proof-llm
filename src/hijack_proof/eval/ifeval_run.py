"""IFEval runner: general instruction-following accuracy (usefulness check, no injections).

    hp-ifeval --model qwen3-0.6b                    # base model
    hp-ifeval --model qwen3-0.6b --adapter runs/... # LoRA-hardened model

Uses the official IFEval checkers (vendored) and the same reset-proof chunking as hp-eval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.eval import ifeval
from hijack_proof.eval.metrics import mean_ci
from hijack_proof.eval.run import PAUSED, git_sha, run_chunks


def score_response(example, response: str) -> dict:
    from hijack_proof.eval.ifeval import evaluation_lib as lib

    table = {example.prompt: response}
    strict = lib.test_instruction_following_strict(example, table)
    loose = lib.test_instruction_following_loose(example, table)
    return {
        "strict_all": strict.follow_all_instructions,
        "loose_all": loose.follow_all_instructions,
        "strict_each": strict.follow_instruction_list,
        "loose_each": loose.follow_instruction_list,
        "instruction_ids": example.instruction_id_list,
    }


def summarize_ifeval(rows: list[dict]) -> dict:
    flat = lambda key: [float(x) for r in rows for x in r[key]]  # noqa: E731
    return {
        "prompt_strict": mean_ci([float(r["strict_all"]) for r in rows]),
        "prompt_loose": mean_ci([float(r["loose_all"]) for r in rows]),
        "instruction_strict": mean_ci(flat("strict_each")),
        "instruction_loose": mean_ci(flat("loose_each")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--gguf")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--max-minutes", type=float)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    name = (
        args.model
        + (f"__{args.tag}" if args.tag else "")
        + (f"__limit{args.limit}" if args.limit else "")
    )
    result_path = Path("results/ifeval") / f"{name}.json"
    rows_path = Path("runs/eval") / f"ifeval__{name}.jsonl"
    if result_path.exists() and not args.force:
        print(f"skip {result_path} (already done; use --force to re-run)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    if args.max_minutes is not None:
        own = time.time() + 60 * args.max_minutes
        deadline = own if deadline is None else min(deadline, own)

    from hijack_proof.eval import ifeval as pkg
    from hijack_proof.eval.ifeval import evaluation_lib as lib

    pkg.ensure_punkt()
    examples = lib.read_prompt_list(str(ifeval.INPUT_DATA))
    if args.limit:
        examples = examples[:: max(1, len(examples) // args.limit)][: args.limit]
    by_id = {str(e.key): e for e in examples}
    items = [{"id": str(e.key), "prompt": e.prompt} for e in examples]

    model_dir = Path("models") / args.model
    if args.gguf:
        from transformers import AutoTokenizer

        from hijack_proof.eval.generate import LlamaCppGenerator

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        gen = LlamaCppGenerator(args.gguf, parallel=args.parallel, ctx=2048)
    else:
        from hijack_proof.eval.generate import HFGenerator

        gen = HFGenerator(model_dir, batch_size=args.batch_size, adapter_dir=args.adapter)
        tokenizer = gen.tokenizer
    enc = ChatEncoder(tokenizer, fmt="user_inline")
    t0 = time.time()

    def make_rows(chunk):
        prompts = [enc.encode_prompt(Prompt(i["prompt"])) for i in chunk]
        outs = gen.generate(prompts, max_new_tokens=args.max_new_tokens)
        rows = []
        for item, ids in zip(chunk, outs, strict=True):
            response = enc.decode_response(ids)
            rows.append(
                {
                    "id": item["id"],
                    "response": response,
                    **score_response(by_id[item["id"]], response),
                }
            )
        return rows

    state_path = rows_path.with_suffix(".state.json")
    prior = json.loads(state_path.read_text())["seconds"] if state_path.exists() else 0.0
    try:
        rows, finished = run_chunks(
            items, make_rows, rows_path, args.chunk_size, deadline,
            log=lambda m: print(f"{m} ({time.time() - t0:.0f}s)", flush=True),
        )  # fmt: skip
    finally:
        if hasattr(gen, "close"):
            gen.close()
    seconds = round(prior + time.time() - t0, 1)
    state_path.write_text(json.dumps({"seconds": seconds}))
    if not finished:
        print(f"PAUSED ifeval {name}: {len(rows)}/{len(items)} prompts saved; run again")
        return PAUSED
    summary = summarize_ifeval(rows)
    meta = {
        "model": args.model, "adapter": args.adapter, "gguf": args.gguf, "limit": args.limit,
        "max_new_tokens": args.max_new_tokens, "git_sha": git_sha(), "seconds": seconds,
        "prompts": len(rows),
    }  # fmt: skip
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"ifeval {name}: prompt strict {summary['prompt_strict']['mean']:.3f} "
        f"loose {summary['prompt_loose']['mean']:.3f} | {seconds}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
