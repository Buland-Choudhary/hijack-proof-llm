"""Agent-task evaluation and SFT on tool traces (plan M7).

    hp-agent eval --model qwen3-0.6b --format user_inline --split dev --tag untrained
    hp-agent sft  --model qwen3-0.6b --run agent-qwen06 --n 600
    hp-agent eval --model qwen3-0.6b --adapter runs/agent_sft/agent-qwen06/adapter \
                  --format input_role --split heldout --tag agent-sft

Tasks: read a file and email/save something; for attacked tasks the file also holds an injection
asking for an email to an attacker address. Success is checked on the final environment state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from hijack_proof import hw
from hijack_proof.agents import rollout
from hijack_proof.agents import tasks as agent_tasks
from hijack_proof.attacks import library as lib
from hijack_proof.eval.metrics import mean_ci
from hijack_proof.train import sft

SPLITS = {  # name -> (seed, phrase_set)
    "dev": (31, "train"),
    "heldout": (41, "eval"),
}


def build_eval_tasks(split: str, n: int) -> list[agent_tasks.AgentTask]:
    seed, phrases = SPLITS[split]
    return agent_tasks.generate(n, split, seed, False, phrases) + agent_tasks.generate(
        n, split, seed, True, phrases
    )


def summarize(rows: list[dict]) -> dict:
    clean = [r for r in rows if not r["attacked"]]
    att = [r for r in rows if r["attacked"]]
    seen = [r for r in att if r["family"] in lib.SEEN_FAMILIES]
    held = [r for r in att if r["family"] not in lib.SEEN_FAMILIES]
    return {
        "clean_task_success": mean_ci([float(r["task_ok"]) for r in clean]),
        "attacked_task_success": mean_ci([float(r["task_ok"]) for r in att]),
        "attack_success": mean_ci([float(r["attack_ok"]) for r in att]),
        "attack_success_seen_families": mean_ci([float(r["attack_ok"]) for r in seen]),
        "attack_success_heldout_families": mean_ci([float(r["attack_ok"]) for r in held]),
        "invalid_format_rate": mean_ci([float(r["invalid"]) for r in rows]),
    }


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except Exception:
        return "unknown"


def cmd_eval(args) -> int:
    from transformers import AutoTokenizer

    from hijack_proof.eval.generate import HFGenerator

    out = Path("results/agents") / f"{args.model}__{args.format}__{args.split}__{args.tag}.json"
    if out.exists():
        print(f"skip {out} (exists)")
        return 0
    model_dir = Path("models") / args.model
    enc = rollout.AgentEncoder(AutoTokenizer.from_pretrained(model_dir), fmt=args.format)
    gen = HFGenerator(model_dir, batch_size=args.batch_size, adapter_dir=args.adapter)
    eval_tasks = build_eval_tasks(args.split, args.n)
    t0 = time.time()
    rows = rollout.run_episodes(eval_tasks, gen, enc, max_turns=4, max_new_tokens=80)
    summary = summarize(rows)
    meta = {"model": args.model, "adapter": str(args.adapter), "format": args.format,
            "split": args.split, "n_per_condition": args.n, "git_sha": git_sha(),
            "seconds": round(time.time() - t0)}  # fmt: skip
    for r in rows:  # keep transcripts for a handful of episodes only
        if not (r["attack_ok"] or r["id"].endswith(("0000", "0001", "0002"))):
            r["turns"] = None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"meta": meta, "summary": summary, "episodes": rows}, indent=1) + "\n"
    )
    f = lambda k: f"{100 * summary[k]['mean']:.1f}"  # noqa: E731
    print(f"{out.name}: clean task {f('clean_task_success')} | attacked task "
          f"{f('attacked_task_success')} | attack success {f('attack_success')} | "
          f"invalid {f('invalid_format_rate')}")  # fmt: skip
    return 0


def cmd_sft(args) -> int:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path("runs/agent_sft") / args.run
    if (out_dir / "DONE").exists():
        print(f"skip {args.run} (already done)")
        return 0
    deadline = float(os.environ["HP_DEADLINE"]) if "HP_DEADLINE" in os.environ else None
    base_dir = Path("models") / args.model
    tok = AutoTokenizer.from_pretrained(base_dir)
    enc = rollout.AgentEncoder(tok, fmt="input_role")
    n_att = round(args.n * args.attack_share)
    train = agent_tasks.generate(n_att, "train", 5, True) + agent_tasks.generate(
        args.n - n_att, "train", 5, False
    )
    examples = [enc.encode_train(rollout.expert_turns(t)) for t in train]
    print(f"{len(examples)} traces, {sum(len(x) for x, _ in examples)} tokens", flush=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=hw.compute_dtype())
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")  # fmt: skip
    model = get_peft_model(model, lora)
    cfg = sft.SFTConfig(lr=1e-4, epochs=2.0, max_tokens_per_batch=1024, batches_per_step=8,
                        warmup_steps=5, ckpt_minutes=3.0)  # fmt: skip
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    t0 = time.time()
    status = sft.train(
        model, examples, cfg, out_dir, pad_id, deadline,
        log=lambda m: print(f"{m} (+{time.time() - t0:.0f}s)", flush=True),
    )  # fmt: skip
    if status == "paused":
        print(f"PAUSED {args.run}; run the same command again")
        return sft.PAUSED
    model.save_pretrained(out_dir / "adapter")
    (out_dir / "DONE").write_text("ok\n")
    print(f"done: {args.run}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--adapter", type=Path)
    e.add_argument("--format", default="input_role", choices=["input_role", "user_inline"])
    e.add_argument("--split", default="dev", choices=list(SPLITS))
    e.add_argument("--n", type=int, default=60, help="tasks per condition (clean and attacked)")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--tag", default="untrained")
    s = sub.add_parser("sft")
    s.add_argument("--model", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--n", type=int, default=600)
    s.add_argument(
        "--attack-share", type=float, default=0.5, help="share of traces with an injection"
    )
    args = ap.parse_args(argv)
    return cmd_eval(args) if args.cmd == "eval" else cmd_sft(args)


if __name__ == "__main__":
    sys.exit(main())
