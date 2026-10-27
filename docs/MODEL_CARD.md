# Model card: hijack-proof LoRA adapters (v1.0)

**What:** LoRA adapters (r=32, all linear layers) for Qwen3-0.6B and SmolLM2-360M-Instruct trained to answer the
user's request and ignore instructions hidden in separately supplied untrusted text. A merged, quantized
Qwen3-0.6B build (GGUF Q8_0 / Q4_K_M) is described in the report. **Weights are not stored in this repository.**

**Intended use:** research and demonstration of prompt-injection defense with a separate `input` role. Not a
security boundary: do not rely on it to protect a production system.

**Required prompt format:** system / user / `input` roles as built by `hijack_proof.chat.ChatEncoder`, with the
untrusted text tokenized with special tokens disabled. Without that format the defense was not trained or
evaluated.

**Training data:** generated TaskForge documents (train templates), clean and with attacks from seen families;
Alpaca-cleaned general instructions (also with injected attacks); trusted-directive rows. See the
[data card](DATA_CARD.md). Targets are the base model's own verified answers (or a short reference answer).

**Results:** [report](report.md) section 5. Qwen3-0.6B: 3.9% attack success (2.7-5.2) vs 45.3% untrained on the
frozen test; SmolLM2-360M + SFT + DPO: 20.0% (17.3-22.6) vs 44.4%.

**Known limitations:** lower IFEval score than the base model (Qwen 47.3 vs 54.9, 360M 25.9 vs 29.6); synthetic
benchmark; scripted attacks only; English; single seed. Base-model licenses apply to the adapters.
