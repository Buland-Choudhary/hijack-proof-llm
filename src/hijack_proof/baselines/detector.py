"""Detector baseline: a small fine-tuned classifier that flags documents containing injections.

This is the common non-LLM defense (block suspicious inputs). It is compared with trained
defended models on both attack success and false alarms on clean documents.

    python -m hijack_proof.baselines.detector --base smollm2-135m --name det135 --n 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hijack_proof.attacks.library import build_training_attacks
from hijack_proof.data import taskforge as tf


def load_classifier(model_dir: str | Path, dtype=torch.float32):
    # fp32 master weights; the forward/backward runs in bf16 autocast (see train / predict)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id or 0
    tok.padding_side = "right"
    return tok, model


def make_training_data(n: int, seed: int) -> tuple[list[str], list[int]]:
    """n/2 clean documents and n/2 attacked documents (train templates, train attacks)."""
    examples = tf.generate(n // 2, "train", seed)
    attacked = build_training_attacks(tf.generate(n // 2, "train", seed + 1), seed)
    texts = [e.data for e in examples] + [a.data for a in attacked]
    labels = [0] * len(examples) + [1] * len(attacked)
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    return [texts[i] for i in order], [labels[i] for i in order]


def train(tok, model, texts, labels, epochs=1, lr=2e-5, batch_size=8, max_len=512, log=print):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps = epochs * ((len(texts) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * (1 - s / steps)
    )
    step, t0 = 0, time.time()
    for _ in range(epochs):
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len)  # fmt: skip
            y = torch.tensor(labels[i : i + batch_size])
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = model(**enc).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == steps:
                log(f"step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s")
    model.eval()


@torch.inference_mode()
def predict(tok, model, texts, batch_size=16, max_len=512) -> list[float]:
    """Probability that each document contains an injection."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    probs = [0.0] * len(texts)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len)  # fmt: skip
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(**enc).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1].tolist()
        for j, pj in zip(idx, p, strict=True):
            probs[j] = pj
    return probs


def evaluate_detector(tok, model, items: list[dict], threshold: float = 0.5) -> dict:
    flags = [p >= threshold for p in predict(tok, model, [i["data"] for i in items])]
    clean = [f for i, f in zip(items, flags, strict=True) if i["kind"] == "clean"]
    out = {"false_positive_rate": sum(clean) / len(clean), "n_clean": len(clean), "by_family": {}}
    fam: dict[str, list[bool]] = {}
    for i, f in zip(items, flags, strict=True):
        if i["kind"] == "attacked":
            fam.setdefault(i["family"], []).append(f)
    att = [f for v in fam.values() for f in v]
    out["detection_rate"] = sum(att) / len(att)
    seen = [
        f for i, f in zip(items, flags, strict=True) if i["kind"] == "attacked" and i["seen_family"]
    ]
    unseen = [
        f
        for i, f in zip(items, flags, strict=True)
        if i["kind"] == "attacked" and not i["seen_family"]
    ]
    out["detection_rate_seen_families"] = sum(seen) / len(seen)
    out["detection_rate_heldout_families"] = sum(unseen) / len(unseen)
    out["by_family"] = {k: sum(v) / len(v) for k, v in sorted(fam.items())}
    out["flags"] = {i["id"]: bool(f) for i, f in zip(items, flags, strict=True)}
    return out


def main(argv=None) -> int:
    from hijack_proof.eval.suites import load_suite

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="smollm2-135m")
    ap.add_argument("--name", default="detector")
    ap.add_argument("--n", type=int, default=3000, help="training documents (half attacked)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--suite", default="core")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    tok, model = load_classifier(Path("models") / args.base)
    texts, labels = make_training_data(args.n, args.seed)
    train(tok, model, texts, labels, epochs=args.epochs, log=lambda m: print(m, flush=True))
    res = evaluate_detector(tok, model, load_suite(args.suite))
    flags = res.pop("flags")
    out_dir = Path("results") / args.suite
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"base": args.base, "train_docs": args.n, "epochs": args.epochs, "seed": args.seed}
    (out_dir / f"detector__{args.name}.json").write_text(
        json.dumps({"meta": meta, "summary": res}, indent=2, sort_keys=True) + "\n"
    )
    Path("runs/eval").mkdir(parents=True, exist_ok=True)
    Path(f"runs/eval/{args.suite}__detector__{args.name}__flags.json").write_text(json.dumps(flags))
    print(
        f"FPR {res['false_positive_rate']:.3f} | detection {res['detection_rate']:.3f} "
        f"(seen {res['detection_rate_seen_families']:.3f}, "
        f"held-out {res['detection_rate_heldout_families']:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
