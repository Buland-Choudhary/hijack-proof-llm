# Training small LLMs to ignore instructions hidden in data: report (v1.0)

Every number below comes from a file in [`results/`](../results/) produced by a real run. Intervals are 95%
bootstrap confidence intervals. One training seed per model: differences smaller than the intervals are
trends, not findings.

## 1. Problem and threat model
An assistant reads untrusted text (a document, a web page, a tool result) while serving a trusted user. An
attacker hides an instruction in that text. **Attack success** means the model obeyed the hidden instruction
instead of the user. All payloads are benign by design: a canary code to print, a wrong answer to give, a link
on an `example.com` host, a refusal, or a reply in French. The attacker can only change the document.

## 2. Benchmark (TaskForge + attack library)
- **TaskForge:** synthetic documents of 12 types (invoices, tables, emails, ...) with 18 question types and
  programmatic gold answers, so correctness is checked by code, not by a judge model.
- **Attacks:** 11 families plus multi-injection. Four are held out from all training data (authority
  impersonation, foreign-language wrappers, hidden formatting, obfuscation) and eval phrasings differ from
  training phrasings.
- **Frozen sets** in [`evalsets/v2/`](../evalsets/v2/MANIFEST.json) with sha256 manifests: `core` (the final
  test, 1,350 items = 450 held-out-template documents, each clean and with 2 attacks), `dev` (tuning only),
  `userfollow` (does the model still obey the *user*), `overdefense` (ordinary documents containing
  instruction-like text), plus a SecAlign-style Alpaca suite. IFEval is run with the vendored official checkers.
- A train/eval 8-gram overlap check found no copied documents ([note](notes/M3-training-data-overlap.md)).

## 3. Baselines (no training)
Untrained models obey hidden instructions in 26-59% of attacks; bigger models are hijacked more, not less
(Qwen3-1.7B 58%, Qwen3-4B 59% vs Qwen3-0.6B 45%), because they follow instructions better. The best of
four prompting defenses ("sandwich", repeating the task after the document) lowers Qwen3-0.6B from 45% to 34%.
A fine-tuned detector flags 97.1% of attacked documents but also blocks 9.3% of clean ones. Details:
[`results/baselines/`](../results/baselines/core_table.md) and the [M2 note](notes/M2-benchmark-and-baselines.md).

## 4. Method (all training code written from scratch in PyTorch)
1. **Secure format.** The document goes in a separate `input` role and is tokenized with special tokens
   disabled, so typed role markers stay plain text (unit-tested).
2. **SFT** (LoRA r=32, lr 1e-4, 2 epochs) on clean and attacked prompts. The target is always the answer to the
   *user's* task: the model's own checker-verified answer where possible.
3. **Trusted-directive rows.** The first SFT recipes taught the model to ignore directive-like text so well that
   it stopped obeying the user asking for the same thing (0% compliance on the dev suite). Rows where the
   *user* writes such a directive and the model must comply fixed this (decision D16 in the plan).
4. **DPO** written from scratch: chosen = the user's answer, rejected = the hijacked response; cached
   reference log-probs; optional NLL term.

Compute: SFT of the 0.6B model took 12.5 minutes on 4 CPU cores (`results/sft/train_qwen06-lora-uf.json`).

## 5. Results (frozen final test)
| Model | Clean accuracy | Obeys injection | Trusted directive | IFEval (strict) |
|---|---|---|---|---|
| Qwen3-0.6B, untrained | 57.6% | 45.3% (42.1-48.6) | 67.0% | 54.9% |
| Qwen3-0.6B + SFT | 60.9% (56.2-65.3) | **3.9% (2.7-5.2)** | 73.0% | 47.3% |
| SmolLM2-360M, untrained | 41.3% | 44.4% (41.2-47.8) | 39.0% | 29.6% |
| SmolLM2-360M + SFT | 42.7% (38.2-47.1) | 29.4% (26.4-32.3) | 50.0% | 25.9% |
| SmolLM2-360M + SFT + DPO | 42.7% (38.2-47.1) | 20.0% (17.3-22.6) | 54.0% | not run |

- Qwen3-0.6B: 3.9% attack success; on held-out styles 3.0% vs 4.3% on seen styles.
- SmolLM2-360M learns only part of the defense; DPO helps there.
- **DPO did not help Qwen3-0.6B** (dev split): its SFT adapter was already at 1% attack success and two DPO
  variants (0-0.3%) cost 9-11 points of clean accuracy. See the DPO sections of [PROGRESS.md](../PROGRESS.md).
- **Costs:** IFEval drops 7.6 points (Qwen) and 3.7 (360M). On the over-defense suite, Qwen SFT keeps 90.5%
  (untrained: 91.0%); DPO on the 360M model lowers it from 90.5% to 83.5%.
- Input role vs inline placement (360M, dev): 13.7% vs 17.7% attack success, intervals overlap.

## 6. Quantization
Merged and converted to GGUF, re-evaluated on the same frozen test through llama.cpp: Q8_0 (639 MB) 3.7%
attack success / 60.2% clean; Q4_K_M (397 MB) 4.6% / 58.4%. The defense survives 4-bit quantization within
the intervals.

## 7. Limitations (read these before citing the numbers)
- **Synthetic benchmark.** TaskForge documents are template-generated; real documents are messier. The
  overlap check shows shared boilerplate between train and test templates.
- **Fixed attacks.** The attacks are scripted. There is no adaptive attacker in v1.0 (planned: Tier 2 red teaming),
  so 3.9% is not a security guarantee.
- **Benign payloads only**, by design; no claim about harmful content.
- **Small models, one seed**, English only, short documents (<=1,024 tokens in training).
- **Usefulness cost:** instruction-following (IFEval) gets worse; the defense is not free.
- Qwen's "trusted directive" score on the final suite (73.0%) is well below its dev score (92.7%): the
  trusted-directive behavior generalizes to new phrasings less well than attack resistance does.
- Only the Qwen SFT model was quantized and served; the 360M models were not.

## 8. Reproduce
Commands are in [`CLAUDE.md`](../CLAUDE.md) and the queue files in
[`scripts/queues/`](../scripts/queues/) (e.g. `m3_qwen_uf.txt`, `m4_dpo360_v2.txt`, `m8_quant_qwen.txt`).
Models come from Docker Hub `ai/` images or Hugging Face; weights and generated data are not in the repo.
