# Progress log

## Current status
- **Milestone:** M2 (benchmark and baselines), in progress. M0 done; M1 mostly done (llama.cpp build and bigger-model downloads were still running).
- **Next up:** attack library (families, payloads, success checkers), then the eval harness and baselines.
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
