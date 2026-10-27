"""Group Relative Policy Optimization (Shao et al. 2024), written from scratch.

For each prompt the policy samples a group of G responses; a response's advantage is its reward
minus the group mean (Dr. GRPO: no division by the group std, which biases toward easy or hard
prompts). Groups whose rewards are all equal carry no signal and are skipped. The loss is the
clipped policy-gradient objective, summed over response tokens and normalized by the total
number of response tokens in the step (DAPO-style token-level normalization), plus an
optional k3 KL penalty to a frozen reference adapter:

    L = -mean_tokens[ min(r * A, clip(r, 1 - eps, 1 + eps) * A) ] + beta * mean_tokens[k3],
    r = pi(token) / pi_old(token),      k3 = exp(ref - logp) - (ref - logp) - 1

With one optimizer update per rollout batch, r = 1 and the gradient is the plain policy gradient;
`updates_per_rollout > 1` makes later updates off-policy so that the clipping matters.

Reset-proof like the SFT trainer: the prompts and the sampling seed of a step are functions of
(seed, step), so a resumed run repeats the interrupted step exactly.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from hijack_proof.train import rewards as rw
from hijack_proof.train import sft


@dataclass
class GRPOConfig:
    lr: float = 1e-5
    steps: int = 100
    group_size: int = 8
    prompts_per_step: int = 4
    max_new_tokens: int = 96
    temperature: float = 1.0
    clip_eps: float = 0.2
    beta_kl: float = 0.0
    updates_per_rollout: int = 1
    micro_batch: int = 4  # sequences per forward/backward pass
    warmup_steps: int = 3
    grad_clip: float = 1.0
    seed: int = 0
    ckpt_minutes: float = 3.0
    audit_every: int = 10
    max_rss_gb: float = 11.0


def group_advantages(group_rewards: list[list[float]]) -> list[list[float]]:
    """Reward minus the group mean; all-zero for groups with identical rewards."""
    out = []
    for rs in group_rewards:
        mean = sum(rs) / len(rs)
        out.append([r - mean if max(rs) - min(rs) > 1e-9 else 0.0 for r in rs])
    return out


def token_logps(model, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-token log-prob of each next token: [B, T-1] (position t scores token t+1)."""
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=mask).logits
    logits = logits[:, :-1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def policy_loss_per_token(
    logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, clip_eps: float,
    ref_logp: torch.Tensor | None = None, beta: float = 0.0,
) -> torch.Tensor:  # fmt: skip
    """Per-token loss [B, T-1]. `adv` is [B, 1] and broadcast over tokens."""
    ratio = torch.exp(logp - old_logp)
    loss = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    if ref_logp is not None and beta > 0:
        delta = ref_logp - logp
        loss = loss + beta * (torch.exp(delta) - delta - 1)
    return loss


@torch.inference_mode()
def sample_rollouts(
    model, prompts: list[list[int]], cfg: GRPOConfig, eos_id: int, pad_id: int, seed: int
) -> list[list[tuple[list[int], bool]]]:  # fmt: skip
    """G sampled responses per prompt: [(response_ids incl. end token, finished)]."""
    model.eval()
    width = max(len(p) for p in prompts)
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    for row, p in enumerate(prompts):  # left-pad
        ids[row, width - len(p) :] = torch.tensor(p)
        mask[row, width - len(p) :] = 1
    torch.manual_seed(seed)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model.generate(
            input_ids=ids, attention_mask=mask, max_new_tokens=cfg.max_new_tokens,
            do_sample=True, temperature=cfg.temperature, top_p=1.0, top_k=0,
            num_return_sequences=cfg.group_size, eos_token_id=eos_id, pad_token_id=pad_id,
        )  # fmt: skip
    groups: list[list[tuple[list[int], bool]]] = []
    for g in range(len(prompts)):
        group = []
        for k in range(cfg.group_size):
            resp = out[g * cfg.group_size + k, width:].tolist()
            if eos_id in resp:
                group.append((resp[: resp.index(eos_id) + 1], True))
            else:
                group.append((resp, False))
        groups.append(group)
    model.train()
    return groups


def build_batch(prompt: list[int], resp: list[int], pad_id: int, width: int):
    seq = prompt + resp
    ids = torch.full((width,), pad_id, dtype=torch.long)
    mask = torch.zeros(width, dtype=torch.long)
    ids[: len(seq)] = torch.tensor(seq)
    mask[: len(seq)] = 1
    resp_mask = torch.zeros(width - 1)
    resp_mask[len(prompt) - 1 : len(seq) - 1] = (
        1.0  # positions whose *next* token is a response token
    )
    return ids, mask, resp_mask


def train(
    model, items: list[dict], encode_prompt, decode, cfg: GRPOConfig, out_dir: Path,
    eos_id: int, pad_id: int, deadline: float | None = None, log=print,
    ref_adapter: str | None = None,
) -> str:  # fmt: skip
    """Run (or resume) GRPO on `items` (instruction, data, kind, answer, attack)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.0)
    ckpt_path = out_dir / "ckpt.pt"
    step = sft.load_checkpoint(ckpt_path, model, opt) if ckpt_path.exists() else 0
    if step:
        log(f"resumed from checkpoint at step {step}/{cfg.steps}")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics = (out_dir / "metrics.jsonl").open("a")
    samples = (out_dir / "samples.jsonl").open("a")
    prompt_ids = [encode_prompt(it) for it in items]
    last_ckpt, t0, done_this_call, status = time.time(), time.time(), 0, "done"
    sft_cfg = sft.SFTConfig(lr=cfg.lr, warmup_steps=cfg.warmup_steps, min_lr_ratio=1.0)

    while step < cfg.steps:
        if deadline is not None and time.time() >= deadline and done_this_call > 0:
            status = "paused"
            break
        if done_this_call > 0 and sft.rss_gb() > cfg.max_rss_gb:
            log(f"memory guard: RSS {sft.rss_gb():.1f} GB > {cfg.max_rss_gb} GB; checkpointing")
            status = "paused"
            break
        for group in opt.param_groups:
            group["lr"] = sft.lr_at(step, cfg.steps, sft_cfg)
        rng = random.Random(f"{cfg.seed}/grpo/{step}")
        chosen = rng.sample(range(len(items)), cfg.prompts_per_step)
        t_roll = time.time()
        roll_seed = cfg.seed * 100003 + step
        rolls = sample_rollouts(
            model, [prompt_ids[i] for i in chosen], cfg, eos_id, pad_id, roll_seed
        )
        t_roll = time.time() - t_roll

        flat_items, flat_text, flat_fin, scored = [], [], [], []
        for i, group in zip(chosen, rolls, strict=True):
            scores = []
            for resp, fin in group:
                text = decode(resp)
                scores.append(rw.reward(items[i], text, fin))
                flat_items.append(items[i])
                flat_text.append(text)
                flat_fin.append(fin)
            scored.append(scores)
        advs = group_advantages([[s["total"] for s in g] for g in scored])

        seqs = [  # (prompt, response, advantage) for responses that carry signal
            (prompt_ids[i], resp, a)
            for i, group, adv_g in zip(chosen, rolls, advs, strict=True)
            for (resp, _), a in zip(group, adv_g, strict=True)
            if abs(a) > 1e-9
        ]
        n_tok = sum(len(r) for _, r, _ in seqs)
        stats = {"kl": 0.0, "clipfrac": 0.0}
        grad_norm = 0.0
        if seqs:
            order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][0]) + len(seqs[k][1]))
            micro = [order[k : k + cfg.micro_batch] for k in range(0, len(order), cfg.micro_batch)]
            old: dict[int, torch.Tensor] = {}
            built = {}
            for mb in micro:
                width = max(len(seqs[k][0]) + len(seqs[k][1]) for k in mb)
                parts = [build_batch(seqs[k][0], seqs[k][1], pad_id, width) for k in mb]
                built[tuple(mb)] = tuple(torch.stack(x) for x in zip(*parts, strict=True))
            for u in range(cfg.updates_per_rollout):
                tot_kl = tot_clip = 0.0
                for mb in micro:
                    ids, mask, rmask = built[tuple(mb)]
                    adv = torch.tensor([[seqs[k][2]] for k in mb])
                    if cfg.updates_per_rollout > 1 and u == 0:
                        with torch.no_grad():
                            old[tuple(mb)] = token_logps(model, ids, mask)
                    ref = None
                    if ref_adapter and cfg.beta_kl > 0:
                        with torch.no_grad():
                            model.set_adapter(ref_adapter)
                            ref = token_logps(model, ids, mask)
                            model.set_adapter("default")
                    logp = token_logps(model, ids, mask)
                    old_lp = old.get(tuple(mb), logp.detach())
                    per_tok = policy_loss_per_token(
                        logp, old_lp, adv, cfg.clip_eps, ref, cfg.beta_kl
                    )
                    (per_tok * rmask).sum().div(max(1, n_tok)).backward()
                    with torch.no_grad():
                        r = torch.exp(logp - old_lp)
                        tot_clip += (((r - 1).abs() > cfg.clip_eps).float() * rmask).sum().item()
                        if ref is not None:
                            d = ref - logp
                            tot_kl += ((torch.exp(d) - d - 1) * rmask).sum().item()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
                opt.step()
                opt.zero_grad(set_to_none=True)
                stats = {"kl": tot_kl / max(1, n_tok), "clipfrac": tot_clip / max(1, n_tok)}
        step += 1
        done_this_call += 1
        sft.release_memory()

        flat = [s for g in scored for s in g]
        n = len(flat)
        row = {
            "step": step, "reward": sum(s["total"] for s in flat) / n,
            "task": sum(s["task"] for s in flat) / n, "obeyed": sum(s["obeyed"] for s in flat) / n,
            "format_violation": sum(s["format_violation"] for s in flat) / n,
            "signal_share": len(seqs) / n, "resp_tokens": n_tok / max(1, len(seqs)),
            "grad_norm": grad_norm, **stats, "rollout_seconds": round(t_roll, 1),
            "seconds": round(time.time() - t0, 1),
        }  # fmt: skip
        if step % cfg.audit_every == 0 or step == 1:
            row["audit"] = rw.audit(flat_items, flat_text, flat_fin)
            triples = list(zip(flat_items, flat_text, flat, strict=True))
            for it, text, sc in triples[:: cfg.group_size]:
                rec = {"step": step, "kind": it["kind"], "response": text, "reward": sc["total"]}
                samples.write(json.dumps(rec, ensure_ascii=False) + "\n")
            samples.flush()
        metrics.write(json.dumps(row) + "\n")
        metrics.flush()
        log(
            f"step {step}/{cfg.steps} reward {row['reward']:.3f} task {row['task']:.2f} "
            f"obeyed {row['obeyed']:.2f} signal {row['signal_share']:.2f} "
            f"roll {t_roll:.0f}s total {row['seconds']:.0f}s"
        )
        if time.time() - last_ckpt >= cfg.ckpt_minutes * 60:
            sft.save_checkpoint(ckpt_path, model, opt, step)
            last_ckpt = time.time()
    sft.save_checkpoint(ckpt_path, model, opt, step)
    metrics.close()
    samples.close()
    return status
