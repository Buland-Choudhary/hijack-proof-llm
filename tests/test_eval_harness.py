import json
from pathlib import Path

import pytest

from hijack_proof.chat import ChatEncoder, Prompt
from hijack_proof.eval import suites
from hijack_proof.eval.metrics import mcnemar_p, mean_ci, summarize
from hijack_proof.eval.prompting import DEFENSES, apply_defense
from hijack_proof.eval.run import evaluate, score_item


class FakeTokenizer:
    """Whitespace 'tokenizer' so the harness can be tested without model files."""

    chat_template = ""
    vocab: dict[str, int] = {}

    def convert_tokens_to_ids(self, tok):
        return 0

    def __call__(self, text, add_special_tokens=False, split_special_tokens=False):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split(" ")]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv[i] for i in ids if i in inv)


class OracleGenerator:
    """Answers every prompt with a fixed string."""

    def __init__(self, text, tokenizer):
        self.ids = tokenizer(text)["input_ids"] + [0]

    def generate(self, prompts, max_new_tokens=64, progress=None):
        return [list(self.ids) for _ in prompts]


# ---------------------------------------------------------------- suites
def test_frozen_suites_match_manifest_and_are_reproducible():
    root = Path("evalsets") / suites.EVAL_VERSION
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for name, info in manifest["suites"].items():
        loaded = suites.load_suite(name)  # verifies checksum
        assert len(loaded) == info["items"]
        if name not in suites.SUITES:  # externally sourced suites are checked via checksum only
            continue
        # Regeneration must reproduce the frozen file exactly. If this fails you changed the
        # generators: bump TEMPLATE_VERSION / EVAL_VERSION and freeze a new version instead.
        assert suites.sha256_text(suites.to_jsonl(suites.build_suite(name))) == info["sha256"]


def test_core_suite_is_balanced_and_covers_heldout_families():
    items = suites.load_suite("core")
    attacked = [i for i in items if i["kind"] == "attacked"]
    clean = [i for i in items if i["kind"] == "clean"]
    assert len(clean) >= 400 and len(attacked) >= 800
    assert {i["family"] for i in attacked} >= {"authority", "multilingual", "formatting"}
    assert {i["payload"] for i in attacked} == {"canary", "wrong_answer", "url", "refuse", "french"}
    assert all(i["phrase_set"] == "eval" for i in attacked)
    assert all(i["split"] == "heldout" for i in items)


def test_core_and_dev_share_no_documents():
    core = {i["data"] for i in suites.load_suite("core") if i["kind"] == "clean"}
    dev = {i["data"] for i in suites.load_suite("dev") if i["kind"] == "clean"}
    assert not core & dev


def test_tampered_suite_is_rejected(tmp_path):
    root = tmp_path / "v"
    root.mkdir()
    (root / "MANIFEST.json").write_text(json.dumps({"suites": {"core": {"sha256": "0" * 64}}}))
    (root / "core.jsonl").write_text('{"a": 1}\n')
    with pytest.raises(RuntimeError):
        suites.load_suite("core", root)


# ---------------------------------------------------------------- defenses
def test_defenses_change_the_prompt_as_documented():
    p = Prompt("Who sent it?", "From: Ana\nHello there")
    assert apply_defense("none", p) == p
    assert "<document>" in apply_defense("delimiters", p).data
    assert "untrusted" in apply_defense("warning", p).system
    assert apply_defense("sandwich", p).suffix.endswith("Who sent it?")
    assert apply_defense("spotlight", p).data == "From:^Ana^Hello^there"
    with pytest.raises(ValueError):
        apply_defense("bogus", p)
    assert set(DEFENSES) == {"none", "delimiters", "warning", "sandwich", "spotlight"}


def test_sandwich_suffix_is_rendered_after_the_data():
    enc = ChatEncoder(FakeTokenizer(), fmt="user_inline")
    text = enc.render(apply_defense("sandwich", Prompt("Who sent it?", "DATA")))
    assert text.index("DATA") < text.index("Remember, your task is")


def test_suffix_rejected_in_input_role_format():
    enc = ChatEncoder(FakeTokenizer(), fmt="input_role")
    with pytest.raises(ValueError):
        enc.encode_prompt(Prompt("q", "d", suffix="x"))


# ---------------------------------------------------------------- metrics
def test_mean_ci_basic():
    r = mean_ci([1.0] * 80 + [0.0] * 20)
    assert r["mean"] == 0.8 and r["lo"] < 0.8 < r["hi"] and r["n"] == 100
    assert mean_ci([])["mean"] is None


def test_mcnemar():
    same = [True, False, True, False]
    assert mcnemar_p(same, same) == 1.0
    a, b = [True] * 30 + [False] * 5, [False] * 30 + [False] * 5
    assert mcnemar_p(a, b) < 1e-6


def test_end_to_end_with_fake_generator():
    items = (
        suites.load_suite("dev")[:40]
        + [i for i in suites.load_suite("dev") if i["kind"] == "attacked"][:40]
    )
    tok = FakeTokenizer()
    enc = ChatEncoder(tok, fmt="user_inline")
    rows = evaluate(items, OracleGenerator("nothing useful", tok), enc, "none")
    assert len(rows) == len(items)
    s = summarize(rows)
    assert s["clean_accuracy"]["mean"] == 0.0 and s["asr_loose"]["mean"] == 0.0
    assert set(s["attacked_slices"]) == {"family", "payload", "position", "seen_family", "doc_type"}


def test_score_item_detects_obeyed_injection():
    item = next(
        i for i in suites.load_suite("dev") if i["kind"] == "attacked" and i["payload"] == "canary"
    )
    row = score_item(item, item["attack"]["any"][0])
    assert row["attack_loose"] and row["attack_strict"] and row["task_score"] == 0.0


def test_v2_core_is_byte_identical_to_v1_core():
    v1 = json.loads(Path("evalsets/v1/MANIFEST.json").read_text())["suites"]["core"]["sha256"]
    v2 = json.loads(Path("evalsets/v2/MANIFEST.json").read_text())["suites"]["core"]["sha256"]
    assert v1 == v2
    assert (
        Path("evalsets/v1/core.jsonl").read_bytes() == Path("evalsets/v2/core.jsonl").read_bytes()
    )


def test_dev_suite_cannot_leak_heldout_families_or_eval_phrasings():
    from hijack_proof.attacks import library as lib

    dev = [i for i in suites.load_suite("dev") if i["kind"] == "attacked"]
    assert dev
    assert {i["family"] for i in dev} <= set(lib.SEEN_FAMILIES) | {lib.MULTI}
    assert all(i["phrase_set"] == "train" and i["seen_family"] for i in dev)
    # no eval-only phrasing text appears in the dev injections
    eval_only = {
        p.split("{")[0]
        for ps in lib.PHRASES.values()
        for p in ps["eval"]
        if len(p.split("{")[0]) > 15
    }
    assert not any(any(t in i["injection"] for t in eval_only) for i in dev)


def test_secalign_style_suite_shape_and_scoring():
    items = suites.load_suite("secalign")
    assert len(items) == 208 * 4 and all(i["kind"] == "attacked" for i in items)
    assert {i["family"] for i in items} == set(suites.ALPACA_FAMILIES)
    assert all(i["answer"] == {"type": "none"} and i["position"] == "end" for i in items)
    assert all(i["attack"]["any"][0] in i["data"] for i in items)
    idx = suites.alpaca_eval_indices()
    assert len(idx) == 208
    row = score_item(items[0], items[0]["attack"]["any"][0])
    assert row["task_score"] is None and row["attack_loose"] and row["attack_strict"]
    s = summarize([score_item(i, "A normal answer.") for i in items[:20]])
    assert s["asr_loose"]["mean"] == 0.0 and s["attacked_accuracy"]["mean"] is None
