# CLAUDE.md: rules for any Claude session working on this repo

## Start of every session
1. Read `PLAN.md` (scope, milestones) and `PROGRESS.md` (where we are, what's next).
2. Check `runs/` for unfinished jobs; resume them from their last checkpoint (`--resume`) before
   starting new work.
3. Pick the next unchecked item in the current milestone and work through it.

## Golden rules
- **Never invent numbers.** Every number in README, docs or results must come from a results file
  produced by a real run (with git SHA, model hash, data hash, config). If a run didn't happen,
  write "TBD".
- **Public repo:** never commit secrets, tokens, `.env` files, model weights, raw datasets, or
  personal info (no résumé, phone numbers, personal emails). Run `scripts/check_repo_safety.py`
  (installed as a pre-commit hook) before every commit.
- **Benign attacks only.** Injection payloads are canary strings, wrong answers, language switches,
  URLs on `example.com`, or fake tool calls in the simulated environment. Never harmful content.
- Eval sets in `evalsets/` are frozen. Never train or tune on them. Use the dev split for tuning.
- Scope changes go into the Decision Log in `PLAN.md`. Ask the owner before big scope changes.
- Work in tier order (plan §11): finish Tier 1 and tag `v1.0` before starting Tier 2 (GRPO, red
  teaming, agents).
- Dev tuning uses the dev suite only. Never choose defenses, hyperparameters or prompts on core.

## Git
- Identity (set once per clone):
  `git config user.name "Buland Choudhary"`
  `git config user.email "101297124+Buland-Choudhary@users.noreply.github.com"`
- Solo repo: commit straight to `main`. Small commits, pushed right away, at least every
  ~30–60 minutes of work. Run tests (`pytest -q`) and lint before pushing.
- Commit messages: imperative and specific, e.g. `Add TaskForge invoice templates`.
- Before pushing from a shallow clone: `git fetch origin main` and rebase if needed.

## Environment notes
- **Cloud environment:** CPU only (4 vCPU, AMX bf16, 15 GB RAM). Hugging Face is blocked, so
  models come from Docker Hub `ai/` via the model fetcher. PyPI, GitHub and Docker Hub are reachable.
- **Local / GPU machines:** the same code runs. Device is auto-detected; HF fallback for models.
- **Long jobs (plan §9, D12):** the cloud machine can be reset while a session is idle, which kills
  detached jobs and scheduled check-ins (this happened on 2026-10-25).
  - every job must be resumable at a fine grain: evals append per-item outputs and skip finished
    items on restart; training checkpoints every ~15 minutes with `--resume`
  - run one heavy CPU job at a time through the job queue; parallel jobs slow each other 2x or more
  - in the cloud, keep jobs in chunks of <=30 minutes and check in at least every ~10 minutes
  - multi-hour unattended runs belong on the owner's own machine (same code, same commands)
  - after any restart, run `uptime` and check `runs/` before assuming a job is still alive
  - never use `pkill -f` / `pgrep -f` with text that also appears in your own command (it matches
    and kills the shell running it); use `pgrep -x <name>` and kill by PID
- **The cloud VM's CPU can change after a restart** (AMX bf16 earlier, plain AVX-512 later; torch got
  3-5x slower and the native llama.cpp build crashed with `Illegal instruction`). After a restart check
  `lscpu` and rebuild llama.cpp with `rm -rf third_party/llama.cpp/build && scripts/setup_llama_cpp.sh`.
- Keep disk use in check: delete old checkpoints, keep the best and the last.

## Where things go
| Path | What | In git? |
|---|---|---|
| `src/hijack_proof/` | package code | yes |
| `configs/` | run configs (YAML) | yes |
| `tests/` | pytest; tiny random models only, no large downloads | yes |
| `evalsets/` | frozen eval sets + sha256 manifest | yes |
| `results/` | summary metrics, tables, charts | yes |
| `docs/` | report, cards, design notes per milestone | yes |
| `models/`, `data/raw/`, `runs/`, `mlruns/` | weights, downloads, checkpoints, tracking | **no** |

## End of every working block
- Update `PROGRESS.md`: what was done, results (with paths), what's next, any blockers.
- Tick finished checkboxes in `PLAN.md`.
- Commit and push.

## Commands
- setup: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev,demo]"` then `scripts/install_hooks.sh`
- test / lint: `.venv/bin/pytest` and `.venv/bin/ruff check . && .venv/bin/ruff format .`
- fetch models: `.venv/bin/hp-fetch smollm2-360m qwen3-0.6b` (`--list` shows all)
- llama.cpp: `scripts/setup_llama_cpp.sh`, `scripts/to_gguf.sh models/<name> q8_0`,
  merge an adapter and quantize: `scripts/make_gguf.sh <base> <adapter dir> <name>`
- eval: `.venv/bin/hp-eval --model <name> [--adapter <dir>] --suite dev|core|userfollow|overdefense
  --format input_role|user_inline --defense none|sandwich|...` (add `--gguf <file>` for llama.cpp);
  `hp-ifeval`, `hp-report`, `hp-charts`
- data: `hp-sftdata --model <name>`, `scripts/build_userfollow.py` (needs `data/raw/alpaca_data_cleaned.json`,
  download command in the README)
- training: `hp-sft`, `hp-dpo`, `hp-grpo`, `hp-attacker train|sample`, `hp-agent sft|eval`, `hp-merge`
- serving: `hp-serve --model <tokenizer folder> --gguf <file>`, `hp-demo --endpoint <url>`
- long runs: put one command per line in `scripts/queues/<name>.txt` and run
  `scripts/run_queue_until_done.sh scripts/queues/<name>.txt` (resumes after interruptions)
