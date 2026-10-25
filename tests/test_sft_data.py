import json

import pytest

from hijack_proof.data import sft_data as sd
from hijack_proof.data import taskforge as tf
from hijack_proof.eval import checkers, suites

ALPACA = suites.ALPACA_SOURCE


@pytest.fixture(scope="module")
def pool():
    if not ALPACA.exists():
        pytest.skip("alpaca_data_cleaned.json not downloaded (run scripts/build_secalign_suite.py)")
    return sd.build_pool(300, 100, 120, 90, seed=1, exclude_alpaca=suites.alpaca_eval_indices())


def test_reference_answers_score_one_for_every_question_type():
    for split in tf.SPLITS:
        for ex in tf.generate(300, split, 5):
            assert checkers.score_answer(sd.reference_answer(ex.answer), ex.answer) == 1.0, ex


def test_pool_excludes_eval_prompts_and_documents(pool):
    banned = suites.alpaca_eval_indices()
    for p in pool:
        if p["base_key"].startswith("alpaca-"):
            assert int(p["base_key"].split("-")[1]) not in banned
    eval_docs = {
        i["data"] for s in ("core", "dev") for i in suites.load_suite(s) if i["kind"] == "clean"
    }
    assert not any(p["clean_data"] in eval_docs for p in pool if p["clean_data"])


def test_pool_uses_only_seen_families_and_train_phrasings(pool):
    from hijack_proof.attacks import library as lib

    fams = {p["family"] for p in pool if p.get("family")}
    assert fams <= set(lib.SEEN_FAMILIES) | {lib.MULTI}
    eval_only = [
        t.split("{")[0]
        for ps in lib.PHRASES.values()
        for t in ps["eval"]
        if len(t.split("{")[0]) > 15
    ]
    assert not any(t in (p["data"] or "") for p in pool for t in eval_only)


def test_attacked_prompt_reuses_clean_prompt_target(pool):
    attacked = [p for p in pool if p["source"] == "tf_attacked"]
    assert attacked and all(p["data"] != p["clean_data"] for p in attacked)
    jobs = {j["id"]: j for j in sd.target_prompts(pool)}
    for p in attacked:
        assert (
            jobs[p["base_key"]]["data"] == p["clean_data"]
        )  # target is generated on the clean doc


def test_assemble_uses_verified_self_answers_else_reference_else_drops(pool):
    targets = {}
    for j in sd.target_prompts(pool):
        targets[j["id"]] = {"id": j["id"], "response": "SELF " + j["id"], "verified": False}
    rows = sd.assemble(pool, targets)
    assert all(r["target_source"] == "reference" for r in rows)  # nothing verified
    assert not any(
        r["source"].startswith("gen") for r in rows
    )  # unverified general answers dropped
    for j in sd.target_prompts(pool):
        targets[j["id"]]["verified"] = True
    rows = sd.assemble(pool, targets)
    assert all(r["target_source"] == "self" and r["response"].startswith("SELF") for r in rows)
    assert len(rows) == len(pool)
    assert json.dumps(sd.stats(rows))


def test_make_target_rows_verifies_with_the_checker(pool):
    from hijack_proof.chat import ChatEncoder
    from tests.test_eval_harness import FakeTokenizer

    class Gen:
        def __init__(self, texts, tok):
            self.ids = [tok(t)["input_ids"] + [0] for t in texts]
            self.i = 0

        def generate(self, prompts, max_new_tokens=64, progress=None):
            out = [self.ids[(self.i + k) % len(self.ids)] for k in range(len(prompts))]
            self.i += len(prompts)
            return out

    tok = FakeTokenizer()
    enc = ChatEncoder(tok, fmt="user_inline")
    jobs = [j for j in sd.target_prompts(pool) if not j["general"]][:2]
    good = sd.reference_answer(jobs[0]["answer"])
    rows = sd.make_target_rows(jobs, Gen([good, "no idea"], tok), enc)
    assert [r["verified"] for r in rows] == [True, False]
