import json

from hijack_proof.attacks import library as lib
from hijack_proof.data import userfollow as uf
from hijack_proof.eval import checkers
from hijack_proof.eval.metrics import summarize
from hijack_proof.eval.run import score_item


def test_compliant_responses_satisfy_their_own_checker():
    for item in uf.build_userfollow_suite("dev", 100, seed=1, phrase_set="train"):
        row = score_item(item, uf.compliant_response(item["payload"], item["attack"], "Ana Ruiz"))
        assert row["attack_loose"], item["payload"]


def test_non_compliant_answers_do_not_count():
    items = uf.build_userfollow_suite("dev", 50, seed=2, phrase_set="train")
    assert not any(score_item(i, "Ana Ruiz")["attack_loose"] for i in items)


def test_suite_is_deterministic_and_balanced_over_payloads():
    a = uf.build_userfollow_suite("heldout", 100, seed=3, phrase_set="eval")
    assert a == uf.build_userfollow_suite("heldout", 100, seed=3, phrase_set="eval")
    assert {i["payload"] for i in a} == set(lib.PAYLOADS)
    assert all(i["kind"] == "userfollow" and i["data"] and "<|im_" not in i["data"] for i in a)


def test_eval_suites_use_different_phrasings_than_training():
    train = {
        i["instruction"] for i in uf.build_userfollow_training(200, 0)
    } and uf.build_userfollow_training(200, 0)
    ev = uf.build_userfollow_suite("heldout", 200, seed=4, phrase_set="eval")
    eval_only = [
        t.split("{")[0]
        for ps in lib.PHRASES.values()
        for t in ps["eval"]
        if len(t.split("{")[0]) > 15
    ]
    assert not any(any(t in r["instruction"] for t in eval_only) for r in train)
    assert any(any(t in i["instruction"] for t in eval_only) for i in ev)


def test_training_rows_have_directive_targets_and_clean_documents():
    rows = uf.build_userfollow_training(60, 0)
    assert {r["target_source"] for r in rows} == {"directive"}
    assert all("<|im_" not in (r["data"] or "") for r in rows)
    assert len({r["payload"] for r in rows}) == len(lib.PAYLOADS)


def test_summary_reports_compliance_separately_from_attack_success():
    items = uf.build_userfollow_suite("dev", 40, seed=5, phrase_set="train")
    rows = [score_item(i, uf.compliant_response(i["payload"], i["attack"], "x")) for i in items]
    s = summarize(rows)
    assert s["userfollow_compliance"]["mean"] == 1.0
    assert s["asr_loose"]["mean"] is None  # no attacked items in this suite
    assert json.dumps(s)
    assert checkers.attack_succeeded("HACKED-0000", {"type": "contains", "any": ["HACKED-0000"]})


def test_frozen_suites_match_their_generators():
    from hijack_proof.eval import suites

    for name, split, n, seed, phrases in (
        ("userfollow_dev", "dev", 150, 6006, "train"),
        ("userfollow", "heldout", 200, 5005, "eval"),
    ):
        frozen = suites.load_suite(name)  # verifies the checksum
        assert suites.to_jsonl(frozen) == suites.to_jsonl(
            uf.build_userfollow_suite(split, n, seed, phrases)
        )
