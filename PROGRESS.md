# Progress log

## Current status
- **Plan:** v1.1 (revised 2026-10-25; see PLAN.md decisions D6-D15).
- **Milestone:** M2 (benchmark and baselines) is DONE. Next: M3 (SFT).
- **Next up (in order):** (1) generate SFT data for SmolLM2-360M and Qwen3-0.6B with `hp-sftdata`
  (resumable, needs the CPU), (2) over-defense suite + 8-gram overlap report, (3) smoke-run `hp-sft`,
  (4) train 360M LoRA / full and 0.6B LoRA, evaluate on dev then core, ablations.
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
- Dev sweep running via `scripts/run_queue_until_done.sh scripts/queues/dev_sweep.txt` (12 jobs, 25-minute
  resumable chunks, log in runs/queue/dev_sweep_2.log). First dev results (SmolLM2-360M, no defense):
  clean acc 0.533, loose ASR 0.377; sandwich: clean acc 0.533, loose ASR 0.330.
- M3 code written while waiting: from-scratch SFT trainer (`train/sft.py`, tests incl. exact resume),
  `hp-sft` runner, SFT data builder `hp-sftdata`. Not yet run on real data (needs the CPU).
- Lesson: a `pkill -f` pattern that matches its own command line kills the shell that runs it.
- Dev sweep finished (12 jobs). Chosen baseline defense for both SmolLM2-360M and Qwen3-0.6B: sandwich
  (lowest loose ASR with <=3 points clean-accuracy loss; spotlight blocks more attacks but costs 13-20
  points of clean accuracy). Tables: `results/baselines/dev_table.md`.

### 2026-10-25, M2 finished
- Core baselines done (results/core/*.json), IFEval for 360M and 0.6B (results/ifeval), detector
  (results/core/detector__det135.json). Headline: untrained models obey hidden instructions in 26-59% of
  attacks; sandwich cuts 0.6B from 45.3% to 33.8%; held-out styles are weaker on untrained models
  (0.6B: 39.0% vs 48.5%); detector flags 97.1% with a 9.3% false-alarm rate.
- Charts: results/baselines/asr_by_model.png and asr_by_family.png (`hp-charts`); tables via `hp-report`.
- Incidents: llama-server OOM-killed by its default 8 GB prompt cache (fixed with --cache-ram 512), one
  cold-disk server start timeout (limit raised to 900 s), 2-hour limit on a background wrapper
  (resumed). All runs resumed without losing items.

### 2026-10-25, first SFT run (attempt 1) and diagnosis
- Attempt 1: SmolLM2-360M, LoRA r=32, lr 2e-4, 2 epochs, input role, all 2,742 rows (54% verified self
  answers, 46% short reference answers for TaskForge). Dev result: loose ASR 37.7% -> 0.3%, attacked-task
  accuracy 22.7% -> 42.0%, BUT clean accuracy 53.3% -> 42.7% (gate: <=3 points) and over-defense accuracy
  89.0% -> 80.5%. Fails the usefulness gate; ASR gate passed by a wide margin.
- Diagnosis (dev, per question type): the model now answers in bare words ("No", "4") and loses its
  sentence-style reasoning; yes/no and counting questions regress most (yesno_compare 11 -> 3 of 13).
  Cause: the reference-answer fallback (bare gold answers) differs in style from the model's own answers.
- Attempt 2 (planned, tuned on dev only): train on verified self answers only (`--target-source self`).
- Attempt 2 (verified self answers only, `--target-source self`, same hyperparameters): dev loose ASR
  37.7% -> 2.3%; clean accuracy 53.3% -> 49.3% (80 -> 74 of 150 correct, McNemar p=0.15, not significant;
  attempt 1 was 80 -> 64, p=0.04); over-defense accuracy 89.0% -> 87.5%. Misses the 3-point gate by 1 point.
- Attempt 3 (queued): same as attempt 2 with lr 1e-4. The trainer also got a memory guard after RSS crept to
  14 GB and the OOM killer ended attempt 2 at step 49 (resumed from its step-36 checkpoint).
- Attempt 3 (verified self answers only, lr 1e-4): dev loose ASR 37.7% -> 4.3% (13/300 obeyed), clean accuracy
  53.3% -> 50.7% (76 vs 80 of 150, McNemar p=0.42), over-defense 89.0% -> 88.0%. Meets both gates (>=50%
  relative ASR drop, <=3 points clean loss), so it is the chosen SmolLM2-360M recipe: LoRA r=32, lr 1e-4,
  2 epochs, verified self answers only, input role. Next: one core evaluation + IFEval of this adapter
  (tag sft360-final), then the same recipe on Qwen3-0.6B (micro-batch 1024 tokens because of its large vocabulary).

### 2026-10-26, final-test result and IFEval finding for SmolLM2-360M
- Core (frozen final test), chosen 360M recipe (`sft360-lora-self-lr1e4`, tag sft360-final): loose ASR 44.4% ->
  11.3% (held-out families 33.0% -> 8.7%), clean accuracy 41.3% -> 43.3%, task accuracy under attack 21.7% ->
  38.2%. Weakest payload: fake answer (36% obeyed).
- IFEval (strict prompt level): 29.6% -> 20.5% (-9.1 points), over the 3-point target. Per instruction type
  (strict, base -> trained): language 23 -> 4 of 31, detectable_format 111 -> 86 of 157, keywords 70 -> 56 of 163,
  length constraints unchanged. Response length is unchanged (150 vs 147 words), so this is not shorter answers.
- Explanation: training only showed injected directives being ignored (including "answer in French" style
  payloads), and the model over-generalized to ignoring such directives when the TRUSTED user asks for them.
  Fix to try (attempt 4, one-directional training was the flaw): add contrastive training rows where the same
  kinds of directives sit in the user turn and the model obeys them, and add a `userfollow` eval suite that
  measures obedience to trusted user directives.
- Qwen3-0.6B SFT (`qwen06-lora-self`: LoRA r=32, lr 1e-4, 2 epochs, verified self answers only, input role,
  micro-batch 1024 tokens): loss 0.317 -> 0.114. Dev: obeyed injections 146 -> 1 of 300 (0.3%), clean accuracy
  61.3% -> 64.0% (92 -> 96 of 150, McNemar p=0.48), over-defense accuracy 91.0% -> 90.5%. Passes all dev gates
  on the first attempt. Final-test evaluation waits for the userfollow/IFEval diagnostics (queue
  `scripts/queues/m3_userfollow.txt`).
- Over-generalization confirmed on `userfollow_dev` (trusted directive in the USER turn; higher is better):
  SmolLM2-360M untrained 27.3% compliance (40 canary / 23 french / 33 refuse / 0 url / 40 fake answer), attempt-3
  trained 6.0% (3 / 0 / 7 / 0 / 20). Attack-only training makes the model ignore directives from the trusted user
  too. Attempt 4 (300 contrastive rows where the user asks and the model complies, plus attempt-3 data) is training.
- Attempt 4 (`sft360-lora-uf`: attempt-3 data + 300 trusted-directive rows, same hyperparameters), dev:
  trusted-directive compliance 6.0% -> 49.3% (untrained 27.3%), loose ASR 13.7% (attempt 3: 4.3%; untrained 37.7%),
  clean accuracy 50.7% (unchanged), over-defense accuracy 90.5% (attempt 3: 88.0%, untrained 89.0%). Meets
  all three gates (D16). IFEval for this model is running. Plan: use this recipe for both models.
- IFEval (strict prompt level) for SmolLM2-360M: untrained 29.6%, attempt 3 (attack-only) 20.5%, attempt 4
  (with trusted-directive rows) 25.9%. The fix recovers 5.4 of the 9.1 lost points; the remaining 3.7-point gap
  is within the interval width (~+/-3.8 points at n=541).
- Qwen3-0.6B trusted-directive compliance (`userfollow_dev`): untrained 60.0% (canary 67, french 23, refuse 97,
  url 57, fake answer 57), attack-only SFT (`qwen06-lora-self`) 0.0% on every payload. The over-generalization is
  total for the main model, so the attempt-4 recipe (self answers + directive rows) is queued for Qwen3-0.6B
  (`scripts/queues/m3_qwen_uf.txt`).
