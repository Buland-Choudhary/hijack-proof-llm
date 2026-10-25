"""Evaluation CLI: run a model over a frozen suite and write per-item rows and a summary.

hp-eval --model smollm2-360m --suite dev --defense none
hp-eval --model qwen3-0.6b --suite core --defense sandwich --limit 100   # quick check

Reset-proof: per-item rows are appended to runs/eval/<suite>__<name>.jsonl after every chunk, a
re-run skips finished items, and a finished run (results/<suite>/<name>.json exists) is skipped
unless --force. A time budget (--max-minutes, or the HP_DEADLINE epoch set by hp-queue) pauses the
run cleanly with exit code 3; run the same command again to continue.
"""

from __future__ import annotations

import argparse
import json
import os
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
    has_gold = item["answer"]["type"] != "none"
    row["task_score"] = checkers.score_answer(response, item["answer"]) if has_gold else None
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


PAUSED = 3  # exit code: time budget used up, run the same command again to continue


def load_rows(path: Path) -> dict[str, dict]:
    """Rows already saved by an interrupted run (a half-written last line is ignored)."""
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["id"]] = row
    return rows


def run_chunks(
    items: list[dict], make_rows, rows_path: Path, chunk_size: int = 64,
    deadline: float | None = None, log=print,
) -> tuple[list[dict], bool]:  # fmt: skip
    """Process `items` in chunks, appending each chunk's rows (dicts with an "id") to `rows_path`.

    Finished items are skipped on re-run. Returns (rows in item order, finished).
    """
    done = load_rows(rows_path)
    todo = [i for i in items if i["id"] not in done]
    if done:
        log(f"resuming: {len(done)} items already done, {len(todo)} to go")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), chunk_size):
        # always finish at least one chunk per call, otherwise a short budget could never progress
        if start > 0 and deadline is not None and time.time() >= deadline:
            break
        new_rows = make_rows(todo[start : start + chunk_size])
        with rows_path.open("a") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        done.update({r["id"]: r for r in new_rows})
        log(f"  {len(done)}/{len(items)} items")
    ordered = [done[i["id"]] for i in items if i["id"] in done]
    return ordered, len(ordered) == len(items)


def evaluate_resumable(
    items: list[dict], generator, encoder: ChatEncoder, defense: str, max_new_tokens: int,
    rows_path: Path, chunk_size: int = 64, deadline: float | None = None, log=print,
) -> tuple[list[dict], bool]:  # fmt: skip
    """Evaluate in chunks, appending rows to `rows_path`. Returns (rows in item order, finished)."""
    return run_chunks(
        items, lambda chunk: evaluate(chunk, generator, encoder, defense, max_new_tokens),
        rows_path, chunk_size, deadline, log,
    )  # fmt: skip


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
    ap.add_argument("--chunk-size", type=int, default=64, help="items per saved chunk")
    ap.add_argument("--max-minutes", type=float, help="pause (exit 3) after this many minutes")
    ap.add_argument("--force", action="store_true", help="re-run even if results already exist")
    args = ap.parse_args(argv)

    name = f"{args.model}__{args.defense}__{args.format}" + (f"__{args.tag}" if args.tag else "")
    if args.limit:
        name += f"__limit{args.limit}"
    if args.max_new_tokens != 64:
        name += f"__mnt{args.max_new_tokens}"
    result_path = Path("results") / args.suite / f"{name}.json"
    rows_path = Path("runs") / "eval" / f"{args.suite}__{name}.jsonl"
    if result_path.exists() and not args.force:
        print(f"skip {result_path} (already done; use --force to re-run)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    if args.max_minutes is not None:
        own = time.time() + 60 * args.max_minutes
        deadline = own if deadline is None else min(deadline, own)

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
    state_path = rows_path.with_suffix(".state.json")
    prior_seconds = json.loads(state_path.read_text())["seconds"] if state_path.exists() else 0.0
    try:
        rows, finished = evaluate_resumable(
            items, gen, enc, args.defense, args.max_new_tokens, rows_path,
            chunk_size=args.chunk_size, deadline=deadline,
            log=lambda m, **kw: print(f"{m} ({time.time() - t0:.0f}s)", flush=True),
        )  # fmt: skip
    finally:
        if hasattr(gen, "close"):
            gen.close()
    seconds = round(prior_seconds + time.time() - t0, 1)
    state_path.write_text(json.dumps({"seconds": seconds}))
    if not finished:
        print(f"PAUSED {name}: {len(rows)}/{len(items)} items saved; run the same command again")
        return PAUSED
    summary = summarize(rows)
    meta = {
        "model": args.model, "adapter": args.adapter, "suite": args.suite, "defense": args.defense,
        "format": args.format, "gguf": args.gguf, "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "git_sha": git_sha(), "seconds": seconds, "items": len(rows),
    }  # fmt: skip
    src = model_dir / "SOURCE.json"
    if src.exists():
        meta["model_manifest_digest"] = json.loads(src.read_text()).get("manifest_digest")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"meta": meta, "summary": summary}, indent=2, sort_keys=True) + "\n"
    )

    def fmt(m):
        return "n/a" if m["mean"] is None else f"{m['mean']:.3f}"

    print(
        f"{name}: clean acc {fmt(summary['clean_accuracy'])} | ASR loose "
        f"{fmt(summary['asr_loose'])} strict {fmt(summary['asr_strict'])} | {seconds}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
