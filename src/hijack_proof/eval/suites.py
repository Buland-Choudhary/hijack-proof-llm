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


# ---------------------------------------------------------------- Alpaca injection suite
# SecAlign-style protocol (D15): general instruction+input prompts with a standard canary injection
# appended to the input. The original AlpacaFarm prompts are only on Hugging Face (blocked in the
# cloud environment), so the prompts are a seeded held-out sample of Alpaca-cleaned. There are no
# gold answers, so this suite measures ASR only. Numbers are NOT comparable with published ones.
ALPACA_SOURCE = Path("data/raw/alpaca_data_cleaned.json")
ALPACA_URL = (
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
)
ALPACA_SUITE = "secalign"
ALPACA_FAMILIES = ("naive", "ignore_previous", "fake_completion", "combined")
ALPACA_N = 208
ALPACA_SEED = 3003


def build_alpaca_suite(source: Path = ALPACA_SOURCE) -> tuple[list[dict], dict]:
    """Return (items, provenance) for the Alpaca injection suite from raw Alpaca-cleaned."""
    raw = source.read_bytes()
    pool = [
        (i, x)
        for i, x in enumerate(json.loads(raw))
        if 20 <= len(x["input"].strip()) <= 800 and x["instruction"].strip()
    ]
    rng = random.Random(f"alpaca-suite/{ALPACA_SEED}")
    chosen = sorted(rng.sample(range(len(pool)), ALPACA_N))
    items = []
    for k in chosen:
        idx, row = pool[k]
        base = tf.Example(
            id=f"alpaca-{idx}", split="heldout", template_id="alpaca", doc_type="alpaca",
            question_type="alpaca", instruction=row["instruction"].strip(),
            data=row["input"].strip(), answer={"type": "none"},
        )  # fmt: skip
        for fam in ALPACA_FAMILIES:
            atk = lib.build_attack(rng, base, fam, "canary", "end", "eval").to_dict()
            atk["id"] = f"{base.id}/{fam}"
            atk["kind"] = "attacked"
            atk["seen_family"] = True
            atk["source_index"] = idx
            items.append(atk)
    provenance = {
        "source": "gururise/AlpacaDataCleaned alpaca_data_cleaned.json (CC BY-NC 4.0)",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "prompts": ALPACA_N, "families": list(ALPACA_FAMILIES), "seed": ALPACA_SEED,
    }  # fmt: skip
    return items, provenance


def add_suite(out_dir: Path, name: str, items: list[dict], extra: dict) -> dict:
    """Add one more frozen suite to an existing version folder (never overwrites a suite)."""
    manifest_path = out_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    if name in manifest["suites"]:
        raise SystemExit(f"suite {name!r} is already frozen in {out_dir}")
    text = to_jsonl(items)
    (out_dir / f"{name}.jsonl").write_text(text)
    manifest["suites"][name] = {"sha256": sha256_text(text), "items": text.count("\n"), **extra}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["suites"][name]


def alpaca_eval_indices(root: Path = Path("evalsets") / EVAL_VERSION) -> set[int]:
    """Alpaca-cleaned row indices used by the eval suite; training must exclude them."""
    return {i["source_index"] for i in load_suite(ALPACA_SUITE, root)}
