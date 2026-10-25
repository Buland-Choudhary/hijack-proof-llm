# Progress log

## Current status
- **Plan:** v1.1 (revised 2026-10-25; see PLAN.md decisions D6-D14).
- **Milestone:** M2 (benchmark and baselines), in progress. M0 and M1 done.
- **Next up (in order):** dev sweep (via `hp-queue`), core baselines, detector run, baseline table +
  README findings.
- **Blockers:** none.

## Session log

### 2026-10-25, cloud session 1
- Checked the cloud machine: 4 vCPU Xeon with AMX/AVX-512, 15 GB RAM, no GPU, Docker available.
- Network: PyPI, GitHub (raw files and clones) and Docker Hub are reachable. Hugging Face,
  download.pytorch.org, Kaggle and arXiv are blocked.
- Found model weights on Docker Hub `ai/` (SmolLM2 135M/360M/1.7B and Qwen3 0.6B/1.7B/4B as
  safetensors) and checked that downloads work.
- Measured CPU speed with random-weight models (torch 2.14, bf16):
  - matmul: 2.27 TFLOPS (fp32: 0.40)
  - SmolLM2-360M shape: full fine-tune ~290 tokens/s; sampling ~150 new tokens/s (16 in parallel)
  - Qwen3-0.6B shape: full fine-tune ~150 tokens/s; sampling ~80 new tokens/s (16 in parallel)
- Checked that these data sources are reachable: Alpaca-cleaned, AlpacaFarm eval outputs, IFEval,
  BIPIA, Tensor Trust, InjecAgent, AgentDojo, SecAlign repo, llama.cpp.
- Wrote PLAN.md, CLAUDE.md, README skeleton, .gitignore, LICENSE.

### 2026-10-25, cloud session 1 (after plan approval)
- M0 done: packaging (src layout, uv lock), safety pre-commit hook + CI, tests, ruff.
- M1: `hp-fetch` downloads SmolLM2 135M/360M and Qwen3 0.6B/1.7B/4B from Docker Hub with sha256 checks.
  Generation smoke tests pass for the three small models. `ChatEncoder` (secure `input` role, special-token
  sanitization) matches the official chat templates. Fixed a .gitignore bug that hid `src/hijack_proof/models/`.
- M2 so far: deterministic answer/attack checkers; TaskForge generator (12 document types, 18 question
  types, splits by template, 2 doc types held out entirely). Oracle answers score 1.0 in tests.
- M1 finished: llama.cpp built; Qwen3 0.6B/1.7B/4B converted to GGUF (Q8_0, 0.6B also Q4_K_M).
  Single-stream CPU generation (4 threads): 0.6B Q8_0 56 tok/s, 0.6B Q4_K_M 40, 1.7B Q8_0 22, 4B Q8_0 9.6
  (`results/bench/llama_cpp_cpu_throughput.md`). Cloud disk allowance is tight (~39 GB): the 1.7B and 4B
  safetensors were deleted after conversion (re-fetch with `hp-fetch` if needed; GGUFs are kept).
- M2: attack library done: 11 single families + multi-injection (4 families held out: authority,
  multilingual, formatting, obfuscation), 5 harmless payload kinds (canary, wrong answer, link on
  example.com, refuse, answer in French), train vs eval phrasings, 3 positions. Loose and strict
  (response-starts-with) attack-success checks. Tests: clean oracle answers never trigger a false
  attack hit; role-spoofing cannot create real role markers after sanitization.
- M2: eval harness done. `hp-eval` runs a model over a frozen suite with greedy decoding; metrics have
  95% bootstrap CIs, McNemar test, per-family/payload/position slices. Prompting-defense baselines:
  none, delimiters, warning, sandwich, spotlight. Frozen sets in `evalsets/v1/` (core: 450 documents,
  each clean + 2 attacked = 1350 items from held-out templates; dev: 450 items), sha256 manifest,
  regeneration test fails if generators change (bump versions instead). Smoke run on SmolLM2-135M works
  (60-item sample: not a result, only a pipeline check).
- M2: llama.cpp backend (`hp-eval --gguf`) for the 1.7B/4B baselines; detector baseline code
  (SmolLM2-135M sequence classifier trained on TaskForge train docs with train-only attacks; reports
  detection rate by family and false-alarm rate) with tests. `build_training_attacks` is the shared
  builder for training data (seen families + train phrasings only).
- First real baseline result (`results/core/smollm2-135m__none__user_inline.json`): SmolLM2-135M with no
  defense: clean accuracy 0.251, loose ASR 0.262, strict ASR 0.237 (1350 items, 933 s while other CPU
  work was running). The model is too weak for ASR to mean much; 360M and 0.6B are the informative ones.

### 2026-10-25, cloud machine reset
- The cloud machine was reset while the session was idle (uptime was 2 minutes on return). The
  detached baseline matrix died after 1 of 15 runs, and the scheduled check-in never fired. Code and
  results already pushed to GitHub were safe.
- Lesson: unattended multi-hour cloud jobs are unreliable. Plan revised to v1.1: Tier 1 / Tier 2
  split, per-item resumable evals, job queue with one heavy job at a time, <=30-minute chunks.
- Found that eval v1 dev contained held-out families and eval phrasings (leak into "unseen attack"
  claims if used for tuning). Fix is eval v2 (core unchanged).

### 2026-10-25, after plan v1.1
- Plan v1.1 pushed (tiers, pre-registered metrics, execution policy, decisions D6-D14).
- Reset-proof evals: `hp-eval` appends per-item rows after each chunk, resumes, skips finished runs,
  pauses with exit code 3 at its time budget (always completing at least one chunk). `hp-queue run
  <file> --max-minutes N` runs a file of such commands one at a time. Tested with a simulated crash,
  a half-written line and a time-out pause, plus a real-model pause/resume check.
- Eval v2 frozen in `evalsets/v2/`: dev now uses seen families and train phrasings only; core is
  byte-identical to v1 (tested), so the SmolLM2-135M core result stays valid. `evalsets/v1/` is kept
  for history.
- External evals: SecAlign-style injection suite (208 held-out Alpaca-cleaned prompts x 4 attack types,
  ASR only; frozen in `evalsets/v2/secalign.jsonl`; deviation recorded as D15 because the AlpacaFarm
  prompts are only on Hugging Face). IFEval via `hp-ifeval`: official Google checkers vendored (Apache-2.0,
  `src/hijack_proof/eval/ifeval/NOTICE.md`), 541 prompts, max 320 new tokens, strict/loose prompt- and
  instruction-level accuracy with bootstrap CIs. NLTK's own downloader refuses to run through the sandbox
  proxy (SSRF protection), so the punkt_tab data is fetched directly into `data/raw/nltk_data`.
