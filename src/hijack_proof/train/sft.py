"""Supervised fine-tuning (SFT), written from scratch in PyTorch.

Design choices (plan §8, Stage 1):
- loss only on response tokens (labels are -100 elsewhere), normalized by the number of target
  tokens in each optimizer step, so long and short examples are weighted per token;
- length-bucketed batches under a token budget (no sequence packing, hence no attention leakage
  between examples);
- full fine-tuning keeps fp32 master weights and runs the forward/backward in bf16 autocast;
  LoRA keeps the frozen base in bf16 with fp32 adapters;
- cosine learning-rate schedule with linear warmup;
- reset-proof: an atomic checkpoint every few minutes, exact resume (the step plan is a pure
  function of the seed), and a clean pause (exit code 3) when HP_DEADLINE passes.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from hijack_proof.chat import IGNORE_INDEX

PAUSED = 3


@dataclass
class SFTConfig:
    lr: float = 2e-5
    epochs: float = 1.0
    max_tokens_per_batch: int = 4096  # padded tokens per micro-batch
    batches_per_step: int = 4  # micro-batches accumulated per optimizer step
    warmup_steps: int = 10
    weight_decay: float = 0.0
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 0
    ckpt_minutes: float = 15.0
    log_every: int = 5
    max_rss_gb: float = 11.0  # checkpoint and pause (restartable) if the process grows past this


Example = tuple[list[int], list[int]]  # (input_ids, labels)


def build_batches(lengths: list[int], max_tokens: int, rng: random.Random) -> list[list[int]]:
    """Group example indices into batches of similar length, each within the padded-token budget."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    window = 256  # sort inside windows of shuffled examples: similar lengths, still random mixing
    for start in range(0, len(order), window):
        chunk = sorted(order[start : start + window], key=lambda i: lengths[i])
        batch: list[int] = []
        for i in chunk:
            width = max([lengths[j] for j in batch] + [lengths[i]])
            if batch and width * (len(batch) + 1) > max_tokens:
                batches.append(batch)
                batch = []
            batch.append(i)
        if batch:
            batches.append(batch)
    rng.shuffle(batches)
    return batches


def make_steps(lengths: list[int], cfg: SFTConfig) -> list[list[list[int]]]:
    """The full training plan: a list of optimizer steps, each a list of micro-batches."""
    steps: list[list[list[int]]] = []
    per_epoch = None
    for epoch in range(math.ceil(cfg.epochs)):
        rng = random.Random(f"{cfg.seed}/epoch{epoch}")
        batches = build_batches(lengths, cfg.max_tokens_per_batch, rng)
        epoch_steps = [
            batches[i : i + cfg.batches_per_step]
            for i in range(0, len(batches), cfg.batches_per_step)
        ]
        per_epoch = per_epoch or len(epoch_steps)
        steps += epoch_steps
    return steps[: max(1, round(cfg.epochs * per_epoch))]


def collate(examples: list[Example], idxs: list[int], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(examples[i][0]) for i in idxs)
    ids = torch.full((len(idxs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(idxs), width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((len(idxs), width), dtype=torch.long)
    for row, i in enumerate(idxs):
        x, y = examples[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def lr_at(step: int, total: int, cfg: SFTConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def token_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of next-token cross-entropy over target positions, and the number of target tokens."""
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss, int((shift_labels != IGNORE_INDEX).sum())


def rss_gb() -> float:
    """Resident memory of this process in GB (Linux)."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def release_memory() -> None:
    """Hand freed heap memory back to the OS. glibc fragments badly under the varying tensor
    shapes of variable-length batches, which otherwise lets RSS creep until the container's
    memory limit kills the process."""
    gc.collect()
    with contextlib.suppress(OSError):
        ctypes.CDLL("libc.so.6").malloc_trim(0)


def save_checkpoint(path: Path, model, opt, step: int) -> None:
    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "params": trainable, "opt": opt.state_dict()}, tmp)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written checkpoint


def load_checkpoint(path: Path, model, opt) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in ck["params"].items():
            params[name].copy_(value)
    opt.load_state_dict(ck["opt"])
    return ck["step"]


def train(
    model, examples: list[Example], cfg: SFTConfig, out_dir: Path, pad_id: int,
    deadline: float | None = None, log=print,
) -> str:  # fmt: skip
    """Run (or resume) training. Returns "done" or "paused"."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    lengths = [len(x) for x, _ in examples]
    steps = make_steps(lengths, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    ckpt_path = out_dir / "ckpt.pt"
    step = load_checkpoint(ckpt_path, model, opt) if ckpt_path.exists() else 0
    if step:
        log(f"resumed from checkpoint at step {step}/{len(steps)}")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics = (out_dir / "metrics.jsonl").open("a")
    model.train()
    last_ckpt, t0 = time.time(), time.time()
    status = "done"
    steps_this_call = 0
    while step < len(steps):
        # always do at least one step per call, otherwise a restart could never make progress
        if deadline is not None and time.time() >= deadline and steps_this_call > 0:
            status = "paused"
            break
        if steps_this_call > 0 and rss_gb() > cfg.max_rss_gb:
            log(
                f"memory guard: RSS {rss_gb():.1f} GB > {cfg.max_rss_gb} GB; "
                "checkpointing for a clean restart"
            )
            status = "paused"
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step, len(steps), cfg)
        micro = steps[step]
        n_target = sum(
            sum(1 for t in examples[i][1][1:] if t != IGNORE_INDEX) for b in micro for i in b
        )
        total_loss = 0.0
        for batch in micro:
            data = collate(examples, batch, pad_id)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(
                    input_ids=data["input_ids"], attention_mask=data["attention_mask"]
                ).logits
            loss_sum, _ = token_loss_sum(logits, data["labels"])
            (loss_sum / max(1, n_target)).backward()
            total_loss += loss_sum.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        steps_this_call += 1
        release_memory()
        row = {
            "step": step,
            "loss": total_loss / max(1, n_target),
            "lr": opt.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "target_tokens": n_target,
            "seconds": round(time.time() - t0, 1),
        }
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if step % cfg.log_every == 0 or step == len(steps):
            log(
                f"step {step}/{len(steps)} loss {row['loss']:.4f} lr {row['lr']:.2e} "
                f"{row['seconds']:.0f}s"
            )
        if time.time() - last_ckpt >= cfg.ckpt_minutes * 60:
            save_checkpoint(ckpt_path, model, opt, step)
            last_ckpt = time.time()
    save_checkpoint(ckpt_path, model, opt, step)
    metrics.close()
    return status
