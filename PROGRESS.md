# Progress log

## Current status
- **Milestone:** M0 (bootstrap). Plan v1 written, waiting for owner approval.
- **Next up:** after approval, finish M0 (pyproject, safety hook, CI), then M1 (model fetcher).
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
