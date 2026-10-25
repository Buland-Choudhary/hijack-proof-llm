"""Batched greedy generation backends. Input and output are token ids, so the caller controls
exactly how untrusted text is tokenized (see chat.ChatEncoder)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch


class HFGenerator:
    """Greedy batched generation with a Hugging Face causal LM (CPU bf16 by default)."""

    def __init__(
        self,
        model_dir: str | Path,
        batch_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        adapter_dir: str | Path | None = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype)
        if adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
        self.model = model.eval()
        self.batch_size = batch_size
        self.pad_id = self.tokenizer.pad_token_id or 0
        self.eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[int]]:
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        outputs: list[list[int]] = [[] for _ in prompts]
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            width = max(len(prompts[i]) for i in idx)
            ids = torch.full((len(idx), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for row, i in enumerate(idx):  # left-pad so generation continues from the prompt end
                p = prompts[i]
                ids[row, width - len(p) :] = torch.tensor(p)
                mask[row, width - len(p) :] = 1
            out = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=self.eos_id, pad_token_id=self.pad_id,
            )  # fmt: skip
            for row, i in enumerate(idx):
                outputs[i] = out[row, width:].tolist()
            if progress:
                progress(min(start + self.batch_size, len(order)), len(order))
        return outputs
