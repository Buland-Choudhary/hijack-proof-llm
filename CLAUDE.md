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
- **Long jobs:**
  - launch detached: `setsid nohup <cmd> > runs/<id>/log.txt 2>&1 &`
  - checkpoint every ~15 minutes, and make every job resumable
  - don't use harness background commands for anything that may take more than ~1.5 hours,
    because they get killed at the time limit
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
Filled in as they're built (M0–M2): setup, test, lint, fetch models, eval, train.
