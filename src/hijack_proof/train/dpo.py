"""Direct Preference Optimization (Rafailov et al. 2023), written from scratch.

For a prompt x with a preferred response y_w and a rejected response y_l:

    L = -log sigmoid(beta * [(log pi(y_w|x) - log pi_ref(y_w|x))
                             - (log pi(y_l|x) - log pi_ref(y_l|x))])

where log pi(y|x) is the SUM of token log-probabilities over the response (including the
end-of-turn token). The reference policy is the SFT model; its sequence log-probs are computed
once before training and cached (they never change), so only the trained adapter is needed in
memory. An optional `sft_weight` adds the length-normalized negative log-likelihood of the chosen
response (RPO-style) to keep the policy from drifting.

Training reuses the SFT utilities: atomic checkpoints with exact resume, a memory guard, and
deadline pauses.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from hijack_proof import hw
from hijack_proof.chat import IGNORE_INDEX
from hijack_proof.train import sft

Example = tuple[list[int], list[int]]  # (input_ids, labels)


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 2e-5
    epochs: float = 1.0
    pairs_per_step: int = 16
    micro_pairs: int = 2  # pairs per forward pass (2 sequences each)
    warmup_steps: int = 5
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    sft_weight: float = 0.0
    seed: int = 0
    ckpt_minutes: float = 15.0
    log_every: int = 2
    max_rss_gb: float = 11.0


def sequence_logps(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum of next-token log-probs over the labelled (response) positions of each sequence: [B]."""
    with hw.autocast():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    logits = logits[:, :-1].float()
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE_INDEX
    logp = (
        torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    )
    return (logp * mask).sum(dim=-1)


def dpo_loss(
    pol_chosen: torch.Tensor, pol_rejected: torch.Tensor,
    ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # fmt: skip
    """Returns (mean loss, mean reward margin, preference accuracy)."""
    margin = beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
    loss = -F.logsigmoid(margin).mean()
    return loss, margin.mean(), (margin > 0).float().mean()


def pair_batch(pairs: list[tuple[Example, Example]], idxs: list[int], pad_id: int) -> dict:
    """One padded batch holding the chosen sequences first, then the rejected ones."""
    seqs = [pairs[i][0] for i in idxs] + [pairs[i][1] for i in idxs]
    return sft.collate(seqs, list(range(len(seqs))), pad_id)


@torch.inference_mode()
def reference_logps(
    model, pairs, pad_id: int, micro_pairs: int = 2, log=print
) -> list[tuple[float, float]]:
    """Cache (chosen, rejected) sequence log-probs of the frozen reference policy."""
    model.eval()
    order = sorted(range(len(pairs)), key=lambda i: -max(len(pairs[i][0][0]), len(pairs[i][1][0])))
    out: list[tuple[float, float]] = [(0.0, 0.0)] * len(pairs)
    for start in range(0, len(order), micro_pairs):
        idx = order[start : start + micro_pairs]
        lp = sequence_logps(model, pair_batch(pairs, idx, pad_id))
        for k, i in enumerate(idx):
            out[i] = (lp[k].item(), lp[len(idx) + k].item())
        if (start // micro_pairs) % 25 == 0:
            log(f"  reference log-probs {min(start + micro_pairs, len(order))}/{len(order)}")
        sft.release_memory()
    return out


def make_steps(n_pairs: int, cfg: DPOConfig) -> list[list[int]]:
    steps: list[list[int]] = []
    for epoch in range(math.ceil(cfg.epochs)):
        order = list(range(n_pairs))
        random.Random(f"{cfg.seed}/dpo-epoch{epoch}").shuffle(order)
        steps += [order[i : i + cfg.pairs_per_step] for i in range(0, n_pairs, cfg.pairs_per_step)]
    per_epoch = math.ceil(n_pairs / cfg.pairs_per_step)
    return steps[: max(1, round(cfg.epochs * per_epoch))]


def train(
    model, pairs, ref: list[tuple[float, float]], cfg: DPOConfig, out_dir: Path, pad_id: int,
    deadline: float | None = None, log=print,
) -> str:  # fmt: skip
    """Train (or resume) on cached reference log-probs. Returns "done" or "paused"."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    steps = make_steps(len(pairs), cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    ckpt_path = out_dir / "ckpt.pt"
    step = sft.load_checkpoint(ckpt_path, model, opt) if ckpt_path.exists() else 0
    if step:
        log(f"resumed from checkpoint at step {step}/{len(steps)}")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics = (out_dir / "metrics.jsonl").open("a")
    model.train()
    last_ckpt, t0, status, done_this_call = time.time(), time.time(), "done", 0
    sft_cfg = sft.SFTConfig(lr=cfg.lr, warmup_steps=cfg.warmup_steps, min_lr_ratio=cfg.min_lr_ratio)
    while step < len(steps):
        if deadline is not None and time.time() >= deadline and done_this_call > 0:
            status = "paused"
            break
        if done_this_call > 0 and sft.rss_gb() > cfg.max_rss_gb:
            log(f"memory guard: RSS {sft.rss_gb():.1f} GB > {cfg.max_rss_gb} GB; checkpointing")
            status = "paused"
            break
        for group in opt.param_groups:
            group["lr"] = sft.lr_at(step, len(steps), sft_cfg)
        group_idx = steps[step]
        tot = {"loss": 0.0, "margin": 0.0, "acc": 0.0}
        for start in range(0, len(group_idx), cfg.micro_pairs):
            idx = group_idx[start : start + cfg.micro_pairs]
            batch = pair_batch(pairs, idx, pad_id)
            lp = sequence_logps(model, batch)
            n = len(idx)
            rc = torch.tensor([ref[i][0] for i in idx])
            rr = torch.tensor([ref[i][1] for i in idx])
            loss, margin, acc = dpo_loss(lp[:n], lp[n:], rc, rr, cfg.beta)
            if cfg.sft_weight:
                n_tok = (batch["labels"][:n, 1:] != IGNORE_INDEX).sum(dim=-1).clamp(min=1)
                loss = loss + cfg.sft_weight * (-(lp[:n] / n_tok)).mean()
            (loss * n / len(group_idx)).backward()
            for k, v in (("loss", loss), ("margin", margin), ("acc", acc)):
                tot[k] += v.item() * n / len(group_idx)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        done_this_call += 1
        sft.release_memory()
        row = {
            "step": step,
            **{k: round(v, 5) for k, v in tot.items()},
            "lr": opt.param_groups[0]["lr"],
            "grad_norm": grad_norm,
            "seconds": round(time.time() - t0, 1),
        }
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        if step % cfg.log_every == 0 or step == len(steps):
            log(
                f"step {step}/{len(steps)} loss {row['loss']:.4f} margin {row['margin']:.3f} "
                f"acc {row['acc']:.2f} {row['seconds']:.0f}s"
            )
        if time.time() - last_ckpt >= cfg.ckpt_minutes * 60:
            sft.save_checkpoint(ckpt_path, model, opt, step)
            last_ckpt = time.time()
    sft.save_checkpoint(ckpt_path, model, opt, step)
    metrics.close()
    return status
