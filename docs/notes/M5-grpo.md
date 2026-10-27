# M5 design note: GRPO with verifiable rewards, in two minutes

**Idea.** For each prompt, sample a group of 8 answers, score each with code, and push the model toward the
answers that scored above the group average. No value network, no judge model.

**Reward (all checked by code).** `task_score - 1.0 * obeyed_the_injection - 0.2 * format_violation` on
attacked prompts; `task_score - 0.2 * format_violation` on clean prompts; on prompts where the *user* writes a
directive, `1.0 * followed - 0.2 * format_violation`. A format violation is an empty or unfinished answer
(hit the length cap). Because the penalty for obeying (-1.0) is larger than the reward for the correct answer
(+1.0 is the maximum), hijacked answers always rank below ignoring the injection.

**Trainer details.** Advantage = reward minus the group mean (no std division, as in Dr. GRPO); groups with
identical rewards are skipped because they carry no gradient; the clipped objective is summed over response
tokens and normalized by the step's total token count (DAPO-style); an optional k3 KL penalty to a frozen
copy of the starting adapter exists but the reported run used none. With one update per rollout batch the
importance ratio is 1, so the clip only matters when `updates_per_rollout > 1` (unit-tested in
`tests/test_grpo.py`). Prompts and sampling seeds of a step depend only on (seed, step), so a run that was
reset resumes the interrupted step exactly.

**Sanity check first.** A tiny task (reward 1 for an answer of at most three words) must show reward rising
before any real run: mean reward 0.278 over the first 7 steps -> 0.821 over the last 7 (30 steps).

**Result (SmolLM2-360M, dev split, then frozen test, see PROGRESS.md and the README).** GRPO on top of
SFT+DPO mainly improved obeying the *user's* directives, the behavior DPO had damaged, and lowered attack
success slightly. It did not recover clean accuracy.

**Reward-hacking audit.** Every 10 steps the trainer logs empty-answer rate, unfinished rate, mean length,
verbatim document copying and refusals on clean prompts (`runs/grpo/<run>/metrics.jsonl`, samples in
`samples.jsonl`). In the 360M run there were no empty answers and no refusals on clean prompts; verbatim
copying appeared in isolated batches and unfinished answers in 0-19% of samples.

**Not done.** GRPO on Qwen3-0.6B: the CPU in the cloud session became about five times slower after a machine
restart (rollouts took 2 minutes per step) and Qwen's SFT adapter already scores near 1.0 on most training
groups (little signal), so it was stopped; the code is the same and ready for a faster machine. No KL
ablation and no G=4 vs G=8 ablation were run. No llama.cpp sampler comparison: the policy changes every step,
so each step would need an adapter merge and GGUF conversion.
