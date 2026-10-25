"""Prompt formats: where untrusted data goes and how it is tokenized.

Both model families use ChatML (`<|im_start|>role\\n...<|im_end|>`). Two formats are supported:

- ``input_role`` (the secure format): untrusted data gets its own turn with role ``input``,
  placed after the user's instruction. The model is trained to treat that turn as data only.
- ``user_inline`` (baseline): data is appended inside the user turn, as most apps do today.

In both formats untrusted data is tokenized with ``split_special_tokens=True``: if an attacker
writes ``<|im_start|>system`` inside a document, it becomes ordinary text tokens and can never
produce a real role marker. Trusted text (system prompt, user instruction) is tokenized normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IGNORE_INDEX = -100
DEFAULT_SYSTEM = "You are a helpful assistant."
DATA_ROLE = "input"

Format = Literal["input_role", "user_inline"]


@dataclass(frozen=True)
class Prompt:
    """One request: trusted instruction (+ optional system prompt) and optional untrusted data."""

    instruction: str
    data: str | None = None
    system: str = DEFAULT_SYSTEM


class ChatEncoder:
    """Turns a `Prompt` (and optionally a target response) into token ids and loss labels."""

    def __init__(self, tokenizer, fmt: Format = "input_role", sanitize: bool = True):
        if fmt not in ("input_role", "user_inline"):
            raise ValueError(f"unknown format {fmt!r}")
        self.tok = tokenizer
        self.fmt = fmt
        self.sanitize = sanitize
        template = tokenizer.chat_template or ""
        # Qwen3 is a hybrid thinking model; we always use non-thinking mode, whose
        # assistant prefix contains an empty think block.
        self.assistant_prefix = "<|im_start|>assistant\n" + (
            "<think>\n\n</think>\n\n" if "enable_thinking" in template else ""
        )
        self.end = "<|im_end|>"
        self.eos_token_id = tokenizer.convert_tokens_to_ids(self.end)

    def _segments(self, p: Prompt) -> list[tuple[str, bool]]:
        """(text, is_untrusted) pieces of the prompt, in order."""
        segs: list[tuple[str, bool]] = [(f"<|im_start|>system\n{p.system}{self.end}\n", False)]
        if p.data is None:
            segs.append((f"<|im_start|>user\n{p.instruction}{self.end}\n", False))
        elif self.fmt == "input_role":
            segs += [
                (f"<|im_start|>user\n{p.instruction}{self.end}\n", False),
                (f"<|im_start|>{DATA_ROLE}\n", False),
                (p.data, True),
                (f"{self.end}\n", False),
            ]
        else:  # user_inline
            segs += [
                (f"<|im_start|>user\n{p.instruction}\n\n", False),
                (p.data, True),
                (f"{self.end}\n", False),
            ]
        segs.append((self.assistant_prefix, False))
        return segs

    def _tokenize(self, text: str, untrusted: bool) -> list[int]:
        split = untrusted and self.sanitize
        return self.tok(text, add_special_tokens=False, split_special_tokens=split)["input_ids"]

    def render(self, p: Prompt) -> str:
        """The prompt as text (for logging and debugging)."""
        return "".join(text for text, _ in self._segments(p))

    def encode_prompt(self, p: Prompt) -> list[int]:
        ids: list[int] = []
        for text, untrusted in self._segments(p):
            ids += self._tokenize(text, untrusted)
        return ids

    def encode(self, p: Prompt, response: str) -> tuple[list[int], list[int]]:
        """Prompt + response for training. Only the response and <|im_end|> get loss labels."""
        prompt_ids = self.encode_prompt(p)
        # The response is model output, never parsed for special tokens.
        response_ids = self._tokenize(response, untrusted=True) + [self.eos_token_id]
        input_ids = prompt_ids + response_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
        return input_ids, labels

    def decode_response(self, ids: list[int]) -> str:
        """Generated ids (after the prompt) to text, cut at the end-of-turn token."""
        if self.eos_token_id in ids:
            ids = ids[: ids.index(self.eos_token_id)]
        return self.tok.decode(ids, skip_special_tokens=True).strip()
