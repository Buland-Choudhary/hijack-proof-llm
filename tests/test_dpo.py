import json
import math
import random

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM

from hijack_proof.chat import IGNORE_INDEX
from hijack_proof.train import dpo


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
    )  # fmt: skip
    return LlamaForCausalLM(cfg).float()


def test_loss_matches_the_formula_by_hand():
    pc, pr = torch.tensor([-10.0, -8.0]), torch.tensor([-12.0, -7.0])
    rc, rr = torch.tensor([-11.0, -9.0]), torch.tensor([-11.5, -8.0])
    beta = 0.5
    margins = [beta * ((-10 + 11) - (-12 + 11.5)), beta * ((-8 + 9) - (-7 + 8))]
    expected = sum(-math.log(1 / (1 + math.exp(-m))) for m in margins) / 2
    loss, margin, acc = dpo.dpo_loss(pc, pr, rc, rr, beta)
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert margin.item() == pytest.approx(sum(margins) / 2, rel=1e-6)
    assert acc.item() == pytest.approx(0.5)  # margins are +0.75 and 0


def test_loss_is_log2_when_policy_equals_reference():
    x = torch.randn(4)
    loss, margin, _ = dpo.dpo_loss(x, x - 1, x, x - 1, beta=0.3)
    assert loss.item() == pytest.approx(math.log(2), rel=1e-6) and margin.item() == pytest.approx(0)


def test_sequence_logps_equals_manual_sum_over_response_tokens():
    model = tiny_model()
    ids = [1, 5, 9, 12, 7, 2]
    labels = [IGNORE_INDEX] * 3 + [12, 7, 2]
    batch = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, 6, dtype=torch.long),
             "labels": torch.tensor([labels])}  # fmt: skip
    with torch.no_grad():
        got = dpo.sequence_logps(model, batch)[0].item()
        logits = model(input_ids=batch["input_ids"]).logits[0].float()
        manual = sum(F.log_softmax(logits[t - 1], -1)[ids[t]].item() for t in (3, 4, 5))
    assert got == pytest.approx(manual, rel=1e-2, abs=1e-2)  # bf16 autocast tolerance


def make_pairs(n=24, seed=0):
    """Chosen = copy the prompt tokens, rejected = a fixed wrong token run. Easy to prefer."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        toks = [rng.randrange(3, 60) for _ in range(4)]
        prompt = [1] + toks + [2]
        chosen = (prompt + toks + [2], [IGNORE_INDEX] * len(prompt) + toks + [2])
        rejected = (
            prompt + [60, 61, 62, 63, 2],
            [IGNORE_INDEX] * len(prompt) + [60, 61, 62, 63, 2],
        )
        out.append((chosen, rejected))
    return out


def test_training_increases_the_preference_margin(tmp_path):
    pairs = make_pairs()
    model = tiny_model()
    ref = dpo.reference_logps(model, pairs, 0, log=lambda m: None)
    cfg = dpo.DPOConfig(
        beta=0.5, lr=3e-3, epochs=6, pairs_per_step=8, micro_pairs=2, warmup_steps=1,
        log_every=1000, max_rss_gb=64,
    )  # fmt: skip
    assert dpo.train(model, pairs, ref, cfg, tmp_path, 0, log=lambda m: None) == "done"
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["margin"] > rows[0]["margin"] + 0.5
    assert rows[-1]["acc"] >= 0.9 and rows[-1]["loss"] < rows[0]["loss"]


def test_pause_and_resume_gives_the_same_weights(tmp_path):
    pairs = make_pairs(16)
    base = dict(beta=0.3, lr=1e-3, epochs=3, pairs_per_step=4, micro_pairs=2, warmup_steps=1,
                log_every=1000, max_rss_gb=64)  # fmt: skip
    a = tiny_model()
    ref = dpo.reference_logps(a, pairs, 0, log=lambda m: None)
    dpo.train(a, pairs, ref, dpo.DPOConfig(**base), tmp_path / "full", 0, log=lambda m: None)
    b = tiny_model()
    calls = 0
    while True:
        calls += 1
        guarded = dpo.DPOConfig(**{**base, "max_rss_gb": 1e-6})
        status = dpo.train(b, pairs, ref, guarded, tmp_path / "part", 0, log=lambda m: None)
        if status == "done":
            break
        assert calls < 100
    assert calls > 2
    for (n, x), (_, y) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.allclose(x, y, atol=1e-6), n
