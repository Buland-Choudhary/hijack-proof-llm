# M3-M4 design note: explain it in two minutes

**Goal.** A small model that answers the *user* and treats text inside a document as data, not commands.

**Secure format.** Untrusted text sits in its own `input` chat role, tokenized with special tokens off, so an
attacker cannot type a real `<|im_start|>system`. The model is trained only in this format.

**SFT.** Mix clean prompts, attacked prompts and general instructions. The target is always the correct answer to
the user's task, preferably the model's own answer that a checker verified (so we do not change its writing
style). Written from scratch: loss only on response tokens, normalized per token; batches grouped by length
(no sequence packing, so no attention leakage); LoRA; atomic checkpoints and exact resume because the cloud
machine can be reset at any time.

**What went wrong first (and the fix).** Attempt 1 used short reference answers as targets and lost accuracy
(answer style changed). Attempt 2 used only the model's own verified answers. Later recipes blocked attacks but
the model also stopped obeying the *user* asking for the same thing (0% on the dev "trusted directive" suite):
it had learned "ignore directives" instead of "ignore directives in the data". Adding rows where the user writes
the directive and the model must follow it fixed that. This became a third gate for choosing the recipe.

**DPO.** Pairs of responses to the same prompt: chosen = the user's answer, rejected = the hijacked response
(and for user-written directives the opposite way round). Loss as in Rafailov et al.; reference log-probs are
computed once and cached; an optional NLL term on the chosen response limits drift. Variant 1 (few directive
pairs) cut attack success on the 360M model to 2% on dev but broke directive following and over-defense. Variant 2
(more directive pairs, NLL term, lower lr) kept a balance and was the one run on the frozen test (29.4% -> 20.0%).

**What did not work.** DPO on Qwen3-0.6B: SFT already reached ~1% attack success on dev, and both DPO variants
cost 9-11 points of clean accuracy, so Qwen ships the SFT adapter. Likely cause: DPO lowers the likelihood of
both responses. Not tested further.

**Honest caveats.** Synthetic documents, scripted attackers, one seed; IFEval gets worse; the confidence
intervals for the format ablation overlap. See `docs/report.md`.
