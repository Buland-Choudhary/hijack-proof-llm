"""Evaluation CLI: run a model over a frozen suite and write per-item rows and a summary.

hp-eval --model smollm2-360m --suite dev --defense none
hp-eval --model qwen3-0.6b --suite core --defense sandwich --limit 100   # quick check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.eval import checkers
from hijack_proof.eval.metrics import summarize
from hijack_proof.eval.prompting import DEFENSES, apply_defense
from hijack_proof.eval.suites import load_suite


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def score_item(item: dict, response: str) -> dict:
    row = {
        k: item.get(k)
        for k in ("id", "kind", "doc_type", "question_type", "family", "payload", "position")
    }
    row["seen_family"] = item.get("seen_family")
    row["response"] = response
    row["task_score"] = checkers.score_answer(response, item["answer"])
    if item["kind"] == "attacked":
        row["attack_loose"] = checkers.attack_succeeded(response, item["attack"])
        row["attack_strict"] = checkers.attack_succeeded(response, item["attack"], strict=True)
    return row


def evaluate(
    items: list[dict], generator, encoder: ChatEncoder, defense: str = "none",
    max_new_tokens: int = 64, progress=None,
) -> list[dict]:  # fmt: skip
    prompts = [
        encoder.encode_prompt(apply_defense(defense, Prompt(i["instruction"], i["data"])))
        for i in items
    ]
    outputs = generator.generate(prompts, max_new_tokens=max_new_tokens, progress=progress)
    return [score_item(i, encoder.decode_response(o)) for i, o in zip(items, outputs, strict=True)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="folder name under models/")
    ap.add_argument("--adapter", help="optional LoRA adapter folder")
    ap.add_argument("--gguf", help="evaluate this GGUF file with llama.cpp instead of HF weights")
    ap.add_argument("--parallel", type=int, default=4, help="llama.cpp parallel slots")
    ap.add_argument("--suite", default="dev")
    ap.add_argument("--defense", default="none", choices=DEFENSES)
    ap.add_argument("--format", default="user_inline", choices=["user_inline", "input_role"])
    ap.add_argument("--limit", type=int, help="use only the first N items (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--tag", default="", help="extra label for the output file name")
    args = ap.parse_args(argv)

    from hijack_proof.eval.generate import HFGenerator

    items = load_suite(args.suite)
    if args.limit:  # keep clean/attacked balance: sample evenly across the list
        step = max(1, len(items) // args.limit)
        items = items[::step][: args.limit]
    model_dir = Path("models") / args.model
    if args.gguf:
        from transformers import AutoTokenizer

        from hijack_proof.eval.generate import LlamaCppGenerator

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        gen = LlamaCppGenerator(args.gguf, parallel=args.parallel)
    else:
        gen = HFGenerator(model_dir, batch_size=args.batch_size, adapter_dir=args.adapter)
        tokenizer = gen.tokenizer
    enc = ChatEncoder(tokenizer, fmt=args.format)
    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 30 or done == total:
            last[0] = time.time()
            print(f"  {done}/{total} items, {time.time() - t0:.0f}s", flush=True)

    try:
        rows = evaluate(items, gen, enc, args.defense, args.max_new_tokens, progress)
    finally:
        if hasattr(gen, "close"):
            gen.close()
    summary = summarize(rows)
    name = f"{args.model}__{args.defense}__{args.format}" + (f"__{args.tag}" if args.tag else "")
    meta = {
        "model": args.model, "adapter": args.adapter, "suite": args.suite, "defense": args.defense,
        "format": args.format, "gguf": args.gguf, "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "git_sha": git_sha(), "seconds": round(time.time() - t0, 1), "items": len(rows),
    }  # fmt: skip
    src = model_dir / "SOURCE.json"
    if src.exists():
        meta["model_manifest_digest"] = json.loads(src.read_text()).get("manifest_digest")
    out_dir = Path("results") / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2, sort_keys=True) + "\n"
    )
    run_dir = Path("runs") / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{args.suite}__{name}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    c, a = summary["clean_accuracy"], summary["asr_loose"]
    print(
        f"{name}: clean acc {c['mean']:.3f} | ASR loose {a['mean']:.3f} "
        f"strict {summary['asr_strict']['mean']:.3f} | {meta['seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
