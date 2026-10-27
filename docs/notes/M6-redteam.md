# M6 note: AI-vs-AI red teaming (one round)

**Setup.** An attacker (SmolLM2-360M-Instruct + LoRA) is shown a benign hidden instruction and must write
text to hide in a document. It is trained with the same GRPO trainer against a frozen defender (D0: the
360M SFT+DPO+GRPO model as a Q8_0 GGUF served by llama.cpp). Reward = hijack success x (1 + 0.3 x novelty)
minus a length penalty; unusable text (marker missing, too long, unfinished) = -0.5. Success is decided by the
same code checkers as the benchmark. Successful attacks, plus an equal number of failed ones, became DPO
pairs (chosen = correct answer, rejected = hijacked answer) for the next defender D1, mixed with replay pairs.

**What happened (files in `results/redteam/`).**
- Attacker training: 30 steps. The unusable-text rate fell from 36% to 8% and the reward rose
  (-0.06 -> +0.057 averaged over the first and last 10 steps).
- Fresh attempts against D0: untrained attacker 3.9% hijacks (60% usable text); trained attacker 9.8-10.0%
  on two seeds (96-97% usable), and 6.3% on a third, never-used seed. The spread across seeds
  (6-10%) is as large as most of the effects here.
- Defender update: D1 on the fixed core test: attack success 15.3% (13.0-17.8) -> 11.2% (9.2-13.3), clean
  accuracy 44.7% -> 42.4% (intervals overlap). Against *fresh* attacker samples D1 was not measurably
  better: 14 of 256 hijacks vs 16 of 256 for D0.

**What the attacker actually found.** 63 of the 76 winning attacks use the `wrong_answer` payload, and most
simply state the fake answer ("The answer is 7779."). That is closer to planting false data than to hijacking
the assistant's instructions, and it exploits the loose success check (the marker appears in the reply). It is
a reward-hacking finding about my success metric, not a new attack style. See
[`results/redteam/attack_gallery.md`](../../results/redteam/attack_gallery.md).

**Not done.** Rounds 2 and 3, the arms-race chart (one round gives no curve), the transfer test against undefended
Qwen3-4B, a stricter success check (only count a win if the defender *also* fails the user's question or obeys
an instruction). The cloud CPU was 3-5x slower than at the start of the project, so one round took about
10 hours of wall-clock time.
