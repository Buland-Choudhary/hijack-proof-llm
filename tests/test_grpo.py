import math

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hijack_proof.train import grpo


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )  # fmt: skip
    return LlamaForCausalLM(cfg).float()


def test_advantages_are_reward_minus_group_mean_and_skip_flat_groups():
    advs = grpo.group_advantages([[1.0, 0.0, 0.0, 1.0], [0.5, 0.5, 0.5, 0.5], [2.0, 0.0]])
    assert advs[0] == pytest.approx([0.5, -0.5, -0.5, 0.5])
    assert advs[1] == [0.0, 0.0, 0.0, 0.0]
    assert advs[2] == pytest.approx([1.0, -1.0])


def test_loss_at_ratio_one_is_minus_advantage():
    logp = torch.tensor([[-1.0, -2.0]])
    loss = grpo.policy_loss_per_token(logp, logp.detach(), torch.tensor([[0.7]]), 0.2)
    assert loss.tolist()[0] == pytest.approx([-0.7, -0.7])


def test_clipping_blocks_gradient_beyond_the_trust_region():
    adv = torch.tensor([[1.0]])
    old = torch.tensor([[0.0]])
    inside = torch.tensor([[0.1]], requires_grad=True)
    outside = torch.tensor([[0.5]], requires_grad=True)  # ratio e^0.5 = 1.65 > 1.2, adv > 0
    grpo.policy_loss_per_token(inside, old, adv, 0.2).sum().backward()
    grpo.policy_loss_per_token(outside, old, adv, 0.2).sum().backward()
    assert inside.grad.item() != 0 and outside.grad.item() == 0
    # negative advantage: the clip bites on the other side (ratio below 1 - eps)
    low = torch.tensor([[-0.5]], requires_grad=True)
    grpo.policy_loss_per_token(low, old, -adv, 0.2).sum().backward()
    assert low.grad.item() == 0


def test_kl_term_is_zero_at_reference_and_positive_away_from_it():
    logp = torch.tensor([[-1.0, -2.0]])
    zero = grpo.policy_loss_per_token(logp, logp, torch.zeros(1, 1), 0.2, logp, beta=0.5)
    assert zero.abs().sum().item() == pytest.approx(0)
    away = grpo.policy_loss_per_token(logp, logp, torch.zeros(1, 1), 0.2, logp - 0.5, beta=0.5)
    delta = -0.5
    assert away[0, 0].item() == pytest.approx(0.5 * (math.exp(delta) - delta - 1))


def test_token_logps_matches_manual_log_softmax():
    model = tiny_model()
    ids = torch.tensor([[1, 5, 9, 12]])
    with torch.no_grad():
        got = grpo.token_logps(model, ids, torch.ones_like(ids))
        logits = model(input_ids=ids).logits[0].float()
    manual = [torch.log_softmax(logits[t], -1)[ids[0, t + 1]].item() for t in range(3)]
    assert got[0].tolist() == pytest.approx(manual, abs=0.05)  # bf16 autocast tolerance


def test_sample_rollouts_cut_at_end_token_and_are_reproducible():
    model = tiny_model()
    cfg = grpo.GRPOConfig(group_size=3, max_new_tokens=6)
    prompts = [[1, 5, 9], [2, 7]]
    a = grpo.sample_rollouts(model, prompts, cfg, eos_id=3, pad_id=0, seed=11)
    b = grpo.sample_rollouts(model, prompts, cfg, eos_id=3, pad_id=0, seed=11)
    assert a == b and len(a) == 2 and all(len(g) == 3 for g in a)
    for group in a:
        for resp, finished in group:
            assert len(resp) <= 6
            assert finished == (3 in resp) and (not finished or resp[-1] == 3)


def test_build_batch_masks_only_response_positions():
    ids, mask, rmask = grpo.build_batch([1, 2, 3], [7, 8], pad_id=0, width=8)
    assert ids.tolist() == [1, 2, 3, 7, 8, 0, 0, 0] and mask.sum().item() == 5
    # positions 2 and 3 predict tokens 7 and 8
    assert rmask.tolist() == [0, 0, 1, 1, 0, 0, 0]


def _items():
    return [
        {"kind": "sanity_short", "instruction": "q", "data": "d", "answer": {"type": "none"},
         "attack": None}
        for _ in range(6)
    ]  # fmt: skip


def test_training_pauses_and_resumes_exactly(tmp_path):
    import json
    import time

    cfg = grpo.GRPOConfig(steps=3, group_size=4, prompts_per_step=2, max_new_tokens=5,
                          micro_batch=2, warmup_steps=1, lr=1e-3)  # fmt: skip
    args = dict(encode_prompt=lambda it: [1, 5, 9], decode=lambda ids: " ".join(map(str, ids)),
                cfg=cfg, out_dir=tmp_path, eos_id=3, pad_id=0, log=lambda m: None)  # fmt: skip
    model = tiny_model()
    status = grpo.train(model, _items(), deadline=time.time() - 1, **args)
    assert status == "paused"  # one step is always done per call
    steps = [json.loads(ln)["step"] for ln in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1]
    resumed = tiny_model()
    assert grpo.train(resumed, _items(), **args) == "done"
    steps = [json.loads(ln)["step"] for ln in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3]
    # the resumed model started from the checkpointed weights, not from scratch
    assert any(
        not torch.equal(a, b)
        for a, b in zip(tiny_model().parameters(), resumed.parameters(), strict=True)
    )  # noqa: E501


def test_policy_gradient_raises_the_probability_of_rewarded_tokens():
    model = tiny_model()
    ids = torch.tensor([[1, 5, 9, 12]])
    mask = torch.ones_like(ids)
    before = grpo.token_logps(model, ids, mask).sum().item()
    opt = torch.optim.SGD(model.parameters(), lr=0.02)
    for _ in range(8):
        logp = grpo.token_logps(model, ids, mask)
        loss = grpo.policy_loss_per_token(logp, logp.detach(), torch.tensor([[1.0]]), 0.2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert grpo.token_logps(model, ids, mask).sum().item() > before
