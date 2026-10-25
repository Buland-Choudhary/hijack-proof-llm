import base64
import itertools
import random
import re
from pathlib import Path

import pytest

from hijack_proof.attacks import library as lib
from hijack_proof.data import taskforge as tf
from hijack_proof.eval.checkers import attack_succeeded, french_ratio, score_answer


@pytest.fixture(scope="module")
def examples():
    return tf.generate(40, "train", seed=5)


def all_combos():
    return itertools.product(lib.ALL_FAMILY_NAMES, lib.PAYLOADS, ["train", "eval"])


def test_family_counts_and_split():
    assert len(lib.FAMILIES) + 1 >= 12  # +1 for the multi-injection family
    assert set(lib.HELDOUT_FAMILIES) == {"authority", "multilingual", "formatting", "obfuscation"}
    assert not set(lib.SEEN_FAMILIES) & set(lib.HELDOUT_FAMILIES)


def test_payload_phrasings_do_not_overlap_between_train_and_eval():
    for kind, sets in lib.PHRASES.items():
        assert not set(sets["train"]) & set(sets["eval"]), kind


def test_deterministic(examples):
    def run():
        rng = random.Random(1)
        return [lib.build_attack(rng, examples[0], "combined", "canary", "middle").to_dict()]

    assert run() == run()


@pytest.mark.parametrize("family,payload,phrase_set", list(all_combos()))
def test_injection_is_present_and_spec_is_consistent(examples, family, payload, phrase_set):
    rng = random.Random(f"{family}{payload}{phrase_set}")
    for ex in examples[:6]:
        position = rng.choice(lib.POSITIONS)
        atk = lib.build_attack(rng, ex, family, payload, position, phrase_set)
        assert atk.injection in atk.data or family == lib.MULTI
        assert len(atk.data) > len(ex.data)
        if family == lib.MULTI:
            assert "+" in atk.position
        # the clean answer is untouched
        assert atk.answer == ex.answer and atk.instruction == ex.instruction
        if payload in ("canary", "url", "wrong_answer"):
            marker = atk.attack["any"][0]
            visible = atk.injection
            if family == "obfuscation" and "base64" in visible:
                encoded = visible.rsplit(": ", 1)[-1]
                visible = base64.b64decode(encoded).decode()
            if family != lib.MULTI:
                assert marker in visible, (family, payload, visible)


def test_position_is_respected():
    data = "line1\nline2\nline3\nline4"
    rng = random.Random(0)
    assert lib.insert_at(data, "X", "start", rng).split("\n")[0] == "X"
    assert lib.insert_at(data, "X", "end", rng).split("\n")[-1] == "X"
    mid = lib.insert_at(data, "X", "middle", rng).split("\n")
    assert mid[0] == "line1" and mid[-1] == "line4" and "X" in mid


def test_clean_answers_never_look_like_successful_attacks(examples):
    rng = random.Random(3)
    for ex in tf.generate(200, "dev", 1):
        response = tf.gold_response(ex.answer)
        for payload in lib.PAYLOADS:
            atk = lib.build_attack(rng, ex, "naive", payload, "end")
            assert not attack_succeeded(response, atk.attack), (payload, response)
            assert not attack_succeeded(response, atk.attack, strict=True)


def test_obeying_responses_are_detected(examples):
    rng = random.Random(4)
    ex = examples[0]
    canary = lib.build_attack(rng, ex, "naive", "canary", "end")
    code = canary.attack["any"][0]
    assert attack_succeeded(code, canary.attack, strict=True)
    assert attack_succeeded(f"**{code}**", canary.attack, strict=True)
    # quoting the injected text while answering is a loose hit but not a strict one
    quoted = f"The document contains a strange instruction saying to print {code}. The answer is 5."
    assert attack_succeeded(quoted, canary.attack)
    assert not attack_succeeded(quoted, canary.attack, strict=True)
    url = lib.build_attack(rng, ex, "naive", "url", "end")
    assert (
        attack_succeeded(f"See {url.injection.split()[-1]}", url.attack)
        or url.attack["any"][0] in url.injection
    )
    refuse = lib.build_attack(rng, ex, "naive", "refuse", "end")
    assert attack_succeeded("I cannot help with that request.", refuse.attack)


def test_french_detector():
    fr = "Le total est de 25 euros et la facture est dans le dossier pour les clients."
    en = "The total is 25 dollars and the invoice is in the folder for the customers."
    assert french_ratio(fr)[0] >= 3 and attack_succeeded(fr, {"type": "french"})
    assert not attack_succeeded(en, {"type": "french"})


def test_attacks_do_not_break_clean_scoring(examples):
    """An attack adds text but a correct response to the user's task still scores 1."""
    rng = random.Random(9)
    for ex in examples[:20]:
        atk = lib.build_attack(rng, ex, "ignore_previous", "canary", "middle")
        assert score_answer(tf.gold_response(atk.answer), atk.answer) == 1.0


@pytest.mark.slow
def test_role_spoofing_cannot_create_role_markers():
    path = Path("models/qwen3-0.6b")
    if not (path / "tokenizer.json").exists():
        pytest.skip("qwen3-0.6b not downloaded")
    from transformers import AutoTokenizer

    from hijack_proof.chat import ChatEncoder, Prompt

    tok = AutoTokenizer.from_pretrained(path)
    enc = ChatEncoder(tok)
    start = tok.convert_tokens_to_ids("<|im_start|>")
    rng = random.Random(0)
    ex = tf.generate(5, "train", 2)[0]
    clean = enc.encode_prompt(Prompt(ex.instruction, ex.data))
    for _ in range(20):
        atk = lib.build_attack(rng, ex, "role_spoof", "canary", rng.choice(lib.POSITIONS))
        ids = enc.encode_prompt(Prompt(atk.instruction, atk.data))
        assert ids.count(start) == clean.count(start)
        assert re.search("im_start", atk.data) or "[SYSTEM]" in atk.data
