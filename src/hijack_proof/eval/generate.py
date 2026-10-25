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


class LlamaCppGenerator:
    """Greedy generation through a local llama-server (GGUF models, parallel slots).

    Prompts are sent as token ids, so the caller's tokenization of untrusted text is preserved.
    The server stops at the model's end-of-turn token on its own.
    """

    def __init__(
        self,
        gguf: str | Path,
        parallel: int = 4,
        ctx: int = 4096,
        port: int = 8089,
        threads: int = 4,
        startup_seconds: int = 900,
        server_bin: str | Path = "third_party/llama.cpp/build/bin/llama-server",
    ):
        import os
        import subprocess
        import time

        import requests

        self.parallel = parallel
        self.url = f"http://127.0.0.1:{port}"
        bin_dir = str(Path(server_bin).parent.resolve())
        env = {**os.environ, "LD_LIBRARY_PATH": bin_dir}
        self.proc = subprocess.Popen(
            [str(server_bin), "-m", str(gguf), "-c", str(ctx * parallel), "-np", str(parallel),
             "-t", str(threads), "--port", str(port), "--host", "127.0.0.1", "--no-webui",
             "--cache-ram", "512"],  # default 8 GiB prompt cache caused an out-of-memory kill
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
        )  # fmt: skip
        for _ in range(startup_seconds):
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited during startup")
            try:
                if requests.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(1)
        self.close()
        raise RuntimeError(f"llama-server did not become ready in {startup_seconds}s")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=20)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _one(self, prompt: list[int], max_new_tokens: int) -> list[int]:
        import requests

        r = requests.post(
            f"{self.url}/completion",
            json={"prompt": prompt, "n_predict": max_new_tokens, "temperature": 0.0,
                  "top_k": 1, "cache_prompt": True, "return_tokens": True},
            timeout=1800,
        )  # fmt: skip
        r.raise_for_status()
        return r.json()["tokens"]

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int = 64,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[int]]:
        from concurrent.futures import ThreadPoolExecutor

        done = 0
        results: list[list[int]] = [[] for _ in prompts]

        def work(i: int) -> None:
            nonlocal done
            results[i] = self._one(prompts[i], max_new_tokens)
            done += 1
            if progress:
                progress(done, len(prompts))

        with ThreadPoolExecutor(self.parallel) as pool:
            list(pool.map(work, range(len(prompts))))
        return results
