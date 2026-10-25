# Progress log

## Current status
- **Milestone:** M2 (benchmark and baselines), in progress. M0 done; M1 done (llama.cpp built; GGUF conversion and CPU throughput benchmark in results/bench).
- **Next up:** llama.cpp generation backend (for 1.7B/4B), then run the baseline matrix (5 models x prompting defenses) and the detector baseline.
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
