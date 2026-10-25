"""Evaluation suites: clean and attacked items built from TaskForge and the attack library.

Each base document appears once clean and (attacks_per_base) times attacked, so task accuracy
under attack can be compared with clean accuracy on the same documents (paired design).
Suites are written once to ``evalsets/<version>/`` and frozen with a sha256 manifest.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from hijack_proof.attacks import library as lib
from hijack_proof.data import taskforge as tf

EVAL_VERSION = "v2"  # v2: dev uses seen families + train phrasings only; core == v1 core


@dataclass(frozen=True)
class SuiteConfig:
    split: str  # TaskForge split the base documents come from
    n_bases: int
    attacks_per_base: int
    seed: int
    families: tuple[str, ...] | None = None  # None = every family, including held-out ones
    phrase_set: str = "eval"  # "train" or "eval" payload phrasings


SUITES = {
    # Final numbers: held-out templates, eval phrasings. Never used for tuning.
    "core": SuiteConfig("heldout", 450, 2, 1001),
    # Tuning: dev templates, only attack families and phrasings that training may use, so tuning
    # on dev can never leak knowledge of the held-out families or eval phrasings.
    "dev": SuiteConfig(
        "dev", 150, 2, 2002, families=tuple(lib.SEEN_FAMILIES + [lib.MULTI]), phrase_set="train"
    ),
}


def _balanced(rng: random.Random, options: list[str], n: int) -> list[str]:
    seq = [options[i % len(options)] for i in range(n)]
    rng.shuffle(seq)
    return seq


def build_suite(name: str) -> list[dict]:
    cfg = SUITES[name]
    rng = random.Random(f"suite/{name}/{cfg.seed}")
    bases = tf.generate(cfg.n_bases, cfg.split, cfg.seed)
    items: list[dict] = []
    for ex in bases:
        items.append(
            {
                "id": ex.id,
                "kind": "clean",
                "split": ex.split,
                "doc_type": ex.doc_type,
                "question_type": ex.question_type,
                "instruction": ex.instruction,
                "data": ex.data,
                "answer": ex.answer,
            }  # fmt: skip
        )
    n_att = cfg.n_bases * cfg.attacks_per_base
    families = _balanced(rng, list(cfg.families or lib.ALL_FAMILY_NAMES), n_att)
    payloads = _balanced(rng, list(lib.PAYLOADS), n_att)
    positions = _balanced(rng, list(lib.POSITIONS), n_att)
    for k in range(n_att):
        base = bases[k % cfg.n_bases]
        atk = lib.build_attack(rng, base, families[k], payloads[k], positions[k], cfg.phrase_set)
        d = atk.to_dict()
        d["id"] = f"{base.id}#a{k // cfg.n_bases}/{families[k]}/{payloads[k]}"
        d["kind"] = "attacked"
        d["seen_family"] = families[k] in lib.SEEN_FAMILIES or families[k] == lib.MULTI
        items.append(d)
    return items


def to_jsonl(items: list[dict]) -> str:
    return "".join(json.dumps(i, sort_keys=True, ensure_ascii=False) + "\n" for i in items)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def freeze(out_dir: Path) -> dict:
    """Write all suites + MANIFEST.json to `out_dir` and return the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": EVAL_VERSION, "taskforge_version": tf.TEMPLATE_VERSION, "suites": {}}
    for name in SUITES:
        text = to_jsonl(build_suite(name))
        (out_dir / f"{name}.jsonl").write_text(text)
        manifest["suites"][name] = {
            "sha256": sha256_text(text),
            "items": text.count("\n"),
            **{
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in SUITES[name].__dict__.items()
            },
        }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_suite(name: str, root: Path = Path("evalsets") / EVAL_VERSION) -> list[dict]:
    """Load a frozen suite, verifying its checksum against the manifest."""
    text = (root / f"{name}.jsonl").read_text()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if sha256_text(text) != manifest["suites"][name]["sha256"]:
        raise RuntimeError(f"frozen suite {name!r} was modified (checksum mismatch)")
    return [json.loads(line) for line in text.splitlines()]
