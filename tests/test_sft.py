import json
import random

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from hijack_proof.chat import IGNORE_INDEX
from hijack_proof.train import sft


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).float()


def copy_task(n=48, seed=0):
    """Prompt = 4 random tokens, response = the same 4 tokens: easy to learn."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        toks = [rng.randrange(3, 60) for _ in range(4)]
        ids = [1] + toks + [2] + toks + [2]
        labels = [IGNORE_INDEX] * 6 + toks + [2]
        out.append((ids, labels))
    return out


def test_batches_respect_token_budget_and_cover_everything():
    lengths = [random.Random(1).randint(5, 60) for _ in range(300)]
    lengths = [5 + (i * 7) % 55 for i in range(300)]
    batches = sft.build_batches(lengths, 200, random.Random(0))
    assert sorted(i for b in batches for i in b) == list(range(300))
    for b in batches:
        assert max(lengths[i] for i in b) * len(b) <= 200 or len(b) == 1


def test_step_plan_is_deterministic_and_epochs_scale():
    lengths = [10] * 100
    cfg = sft.SFTConfig(epochs=2.0, max_tokens_per_batch=100, batches_per_step=2, seed=3)
    a, b = sft.make_steps(lengths, cfg), sft.make_steps(lengths, cfg)
    assert a == b
    one = sft.make_steps(
        lengths, sft.SFTConfig(epochs=1.0, max_tokens_per_batch=100, batches_per_step=2, seed=3)
    )
    assert len(a) == 2 * len(one)
    half = sft.make_steps(
        lengths, sft.SFTConfig(epochs=0.5, max_tokens_per_batch=100, batches_per_step=2, seed=3)
    )
    assert len(half) == round(0.5 * len(one))


def test_lr_schedule_warmup_then_cosine_to_floor():
    cfg = sft.SFTConfig(lr=1.0, warmup_steps=4, min_lr_ratio=0.1)
    assert sft.lr_at(0, 20, cfg) == pytest.approx(0.25)
    assert sft.lr_at(3, 20, cfg) == pytest.approx(1.0)
    assert sft.lr_at(19, 20, cfg) == pytest.approx(0.1, abs=0.01)
    assert sft.lr_at(4, 20, cfg) > sft.lr_at(12, 20, cfg) > sft.lr_at(19, 20, cfg)


def test_loss_matches_manual_cross_entropy_on_target_tokens_only():
    logits = torch.randn(2, 5, 7)
    labels = torch.tensor(
        [[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5], [IGNORE_INDEX, 1, 2, IGNORE_INDEX, IGNORE_INDEX]]
    )
    loss, n = sft.token_loss_sum(logits, labels)
    assert n == 5  # after the one-token shift: row 0 has 3 target tokens, row 1 has 2
    manual = 0.0
    for r in range(2):
        for t in range(4):
            y = labels[r, t + 1].item()
            if y != IGNORE_INDEX:
                manual += -torch.log_softmax(logits[r, t], -1)[y].item()
    assert loss.item() == pytest.approx(manual, rel=1e-5)


def test_training_learns_the_copy_task():
    model, data = tiny_model(), copy_task()
    cfg = sft.SFTConfig(
        lr=3e-3,
        epochs=30,
        max_tokens_per_batch=64,
        batches_per_step=1,
        warmup_steps=2,
        log_every=1000,
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        status = sft.train(model, data, cfg, Path(d), pad_id=0, log=lambda m: None)
        losses = [
            json.loads(line)["loss"]
            for line in (Path(d) / "metrics.jsonl").read_text().splitlines()
        ]
    assert status == "done"
    assert losses[-1] < 0.5 * losses[0]


def test_resume_matches_uninterrupted_training(tmp_path):
    data = copy_task(32)
    cfg = sft.SFTConfig(
        lr=1e-3,
        epochs=3,
        max_tokens_per_batch=64,
        batches_per_step=1,
        warmup_steps=1,
        log_every=1000,
    )
    full = tiny_model()
    sft.train(full, data, cfg, tmp_path / "full", pad_id=0, log=lambda m: None)

    # interrupted run: stop after 5 steps via a deadline in the past, then resume to completion
    part = tiny_model()

    sft.train(
        part,
        data,
        sft.SFTConfig(**{**cfg.__dict__, "epochs": 3}),
        tmp_path / "part",
        pad_id=0,
        deadline=0.0,
        log=lambda m: None,
    )  # step 0 always runs, then pauses
    assert (tmp_path / "part" / "ckpt.pt").exists()
    resumed = tiny_model()
    sft.train(resumed, data, cfg, tmp_path / "part", pad_id=0, log=lambda m: None)
    for (n, a), (_, b) in zip(full.named_parameters(), resumed.named_parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), n


def test_lora_only_trains_adapters():
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(tiny_model(), LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


def test_load_examples_masks_prompt_and_drops_long_rows(tmp_path):
    from hijack_proof.chat import ChatEncoder
    from hijack_proof.train.run_sft import load_examples
    from tests.test_eval_harness import FakeTokenizer

    rows = [
        {"instruction": "Who sent it?", "data": "From: Ana", "response": "Ana"},
        {"instruction": "Who?", "data": " ".join(["word"] * 400), "response": "x"},
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    examples, info = load_examples(path, ChatEncoder(FakeTokenizer()), max_len=200)
    assert info["rows"] == 2 and info["kept"] == 1 and info["dropped_too_long"] == 1
    ids, labels = examples[0]
    assert len(ids) == len(labels) and labels[-1] == ids[-1]
    n_target = sum(t != IGNORE_INDEX for t in labels)
    assert n_target == info["target_tokens"] == 2  # the response token and the end-of-turn token


def test_target_source_filter_keeps_only_self_answers(tmp_path):
    from hijack_proof.chat import ChatEncoder
    from hijack_proof.train.run_sft import load_examples
    from tests.test_eval_harness import FakeTokenizer

    rows = [
        {"instruction": "q1", "data": "d", "response": "a", "target_source": "self"},
        {"instruction": "q2", "data": "d", "response": "b", "target_source": "reference"},
        {"instruction": "q3", "data": None, "response": "c", "target_source": "self"},
    ]
    path = tmp_path / "sft.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    enc = ChatEncoder(FakeTokenizer())
    assert load_examples(path, enc, 200)[1]["kept"] == 3
    assert load_examples(path, enc, 200, "self")[1]["kept"] == 2
    assert load_examples(path, enc, 200, "reference")[1]["kept"] == 1


def test_memory_guard_pauses_and_resumes_to_the_same_weights(tmp_path):
    data = copy_task(24)
    base = dict(
        lr=1e-3,
        epochs=2,
        max_tokens_per_batch=64,
        batches_per_step=1,
        warmup_steps=1,
        log_every=1000,
    )
    full = tiny_model()
    sft.train(full, data, sft.SFTConfig(**base), tmp_path / "full", pad_id=0, log=lambda m: None)

    guarded = tiny_model()
    cfg = sft.SFTConfig(**base, max_rss_gb=1e-6)  # always over the limit: one step per call
    calls = 0
    while True:
        calls += 1
        status = sft.train(guarded, data, cfg, tmp_path / "guard", pad_id=0, log=lambda m: None)
        if status == "done":
            break
        assert calls < 200
    assert calls > 3
    for (n, a), (_, b) in zip(full.named_parameters(), guarded.named_parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), n


def test_rss_is_reported():
    assert 0.01 < sft.rss_gb() < 64
