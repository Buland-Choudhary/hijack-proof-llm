"""Multi-turn chat encoding and batched episode rollouts for the agent tasks (plan M7)."""

from __future__ import annotations

from hijack_proof.agents import env as agent_env
from hijack_proof.agents import tasks as agent_tasks
from hijack_proof.chat import IGNORE_INDEX, ChatEncoder

Turn = tuple[str, str]  # (role, text), role in user | assistant | tool


class AgentEncoder:
    """Like `ChatEncoder`, for several turns. Tool results (and earlier assistant text) are
    tokenized with special tokens disabled, so a spoofed role marker inside a file stays plain text.

    `input_role`: tool results get their own `input` role. `user_inline`: tool results arrive as a
    user message prefixed with "Tool result:" (what an off-the-shelf model sees)."""

    def __init__(self, tokenizer, fmt: str = "input_role", system: str = agent_env.SYSTEM):
        self.chat = ChatEncoder(tokenizer, fmt=fmt)
        self.tok, self.fmt, self.system = tokenizer, fmt, system
        self.prefix = self.chat.assistant_prefix
        self.opener = "<|im_start|>assistant\n"
        self.think = self.prefix[len(self.opener) :]  # empty think block for Qwen3, else ""
        self.eos_token_id = self.chat.eos_token_id

    def _segments(self, turns: list[Turn]) -> list[tuple[str, bool, bool]]:
        """(text, untrusted, labelled) pieces. The last piece opens a new assistant turn."""
        end = "<|im_end|>"
        segs = [(f"<|im_start|>system\n{self.system}{end}\n", False, False)]
        for role, text in turns:
            if role == "user":
                segs.append((f"<|im_start|>user\n{text}{end}\n", False, False))
            elif role == "tool":
                head = (
                    "<|im_start|>input\n"
                    if self.fmt == "input_role"
                    else "<|im_start|>user\nTool result:\n"
                )
                segs += [(head, False, False), (text, True, False), (f"{end}\n", False, False)]
            else:
                segs += [(self.opener + self.think, False, False), (text, True, True),
                         (end, False, True), ("\n", False, False)]  # fmt: skip
        return segs

    def _tok(self, text: str, untrusted: bool) -> list[int]:
        return self.tok(text, add_special_tokens=False, split_special_tokens=untrusted)["input_ids"]

    def encode_prompt(self, turns: list[Turn]) -> list[int]:
        ids: list[int] = []
        for text, untrusted, _ in self._segments(turns):
            ids += self._tok(text, untrusted)
        return ids + self._tok(self.prefix, False)

    def encode_train(self, turns: list[Turn]) -> tuple[list[int], list[int]]:
        """Ids and labels with loss on every assistant turn's text and end-of-turn token only."""
        ids: list[int] = []
        labels: list[int] = []
        for text, untrusted, labelled in self._segments(turns):
            piece = self._tok(text, untrusted)
            ids += piece
            labels += piece if labelled else [IGNORE_INDEX] * len(piece)
        return ids, labels

    def decode(self, ids: list[int]) -> str:
        return self.chat.decode_response(ids)


def run_episodes(tasks: list[agent_tasks.AgentTask], generator, enc: AgentEncoder,
                 max_turns: int = 4, max_new_tokens: int = 80) -> list[dict]:  # fmt: skip
    """Run all tasks in lockstep (one batched generation per turn). Returns one result per task."""
    envs = [t.env() for t in tasks]
    turns = [[("user", t.user)] for t in tasks]
    state = [{"done": False, "invalid": False, "steps": 0} for _ in tasks]
    for _ in range(max_turns):
        active = [i for i, s in enumerate(state) if not s["done"]]
        if not active:
            break
        outs = generator.generate([enc.encode_prompt(turns[i]) for i in active],
                                  max_new_tokens=max_new_tokens)  # fmt: skip
        for i, ids in zip(active, outs, strict=True):
            text = enc.decode(ids)
            turns[i].append(("assistant", text))
            state[i]["steps"] += 1
            action = agent_env.parse_action(text)
            if action is None:
                state[i].update(done=True, invalid=True)
            elif action[0] == "done":
                state[i]["done"] = True
            else:
                turns[i].append(("tool", envs[i].execute(*action)))
    return [
        {"id": t.id, "kind": t.kind, "attacked": t.attack is not None,
         "family": t.attack["family"] if t.attack else None,
         "task_ok": agent_tasks.task_success(t, e), "attack_ok": agent_tasks.attack_success(t, e),
         "invalid": s["invalid"], "finished": s["done"], "steps": s["steps"], "turns": ts}
        for t, e, s, ts in zip(tasks, envs, state, turns, strict=True)
    ]  # fmt: skip


def expert_turns(task: agent_tasks.AgentTask) -> list[tuple[str, str]]:
    """The expert run as chat turns (user, assistant, tool, ..., assistant DONE)."""
    turns = [("user", task.user)]
    for line, result in agent_tasks.expert_trajectory(task):
        turns.append(("assistant", line))
        if result:
            turns.append(("tool", result))
    return turns
