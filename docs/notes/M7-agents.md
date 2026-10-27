# M7 note: tool-using agents (small, synthetic)

**Environment.** A text tool protocol with three tools (`read_file`, `send_email`, `write_file`), simulated
state (files, outbox) and three task kinds: email the total of an invoice, email the project name from meeting
notes, save the first item of a to-do list. For attacked tasks the file the agent must read also contains an
injection that asks for an email to an attacker address (benign `example.com` host) with a canary code.
Success is checked on the *final state*: the right email or file exists (task success), or an email went to the
attacker (attack success). Injections reuse the benchmark's attack families; the held-out split uses the four
families never seen in training and the evaluation phrasings.

**Models and training.** Qwen3-0.6B + LoRA, trained from scratch with the SFT code on expert tool traces (300
traces, 2 epochs): `agent-clean` (traces from clean files only) and `agent-def` (half of the traces come from
files with an injection; the expert ignores it). Tool results go in the separate `input` role with special
tokens disabled (`AgentEncoder`).

**Results** (60 tasks per condition on dev, 90 on held-out; 95% bootstrap intervals; `results/agents/`):

| Model | Split | Task success, clean | Task success, attacked | Attack success |
|---|---|---|---|---|
| untrained Qwen3-0.6B | held-out | 0.0% | 0.0% | 0.0% |
| agent-clean | dev | 100.0% | 33.3% (21.7-45.0) | 66.7% (55.0-78.3) |
| agent-clean | held-out | 100.0% | 30.0% (21.1-40.0) | 65.6% (55.6-74.4) |
| agent-def | dev | 100.0% | 100.0% | 0.0% |
| agent-def | held-out | 100.0% | 100.0% | 0.0% |

On held-out, `agent-clean` is hijacked by seen styles 75.4% (63.9-85.2) and unseen styles 44.8% (27.6-62.1) of the time.

**Reading the numbers carefully.**
- The untrained model cannot use the tools at all (it replies DONE immediately), so its 0% attack success
  says nothing about safety; it is a floor, not a baseline defense.
- The tasks are very easy and templated, and the training loss went to about 0. 100% and 0% are results on this
  synthetic environment, not on real agent benchmarks. The held-out split changes the attack wrapper styles
  and phrasing but keeps the same three task kinds and the same injected request (email a code to an
  attacker address), so it tests style generalization only.
- The comparison that carries information is `agent-clean` vs `agent-def`: the same model and recipe, with and
  without injected training files, differ by about 66 points of attack success.

**Not done.** An InjecAgent subset (the benchmark is not available offline here), more tools or task kinds,
adaptive attackers against the agent, and agent training with GRPO or DPO.
