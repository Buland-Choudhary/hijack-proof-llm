from collections import Counter

import pytest

from hijack_proof.data import taskforge as tf
from hijack_proof.eval.checkers import score_answer


@pytest.fixture(scope="module")
def sets():
    return {s: tf.generate(600, s, seed=1) for s in tf.SPLITS}


def test_deterministic():
    assert tf.to_jsonl(tf.generate(50, "train", 7)) == tf.to_jsonl(tf.generate(50, "train", 7))
    assert tf.to_jsonl(tf.generate(50, "train", 7)) != tf.to_jsonl(tf.generate(50, "train", 8))


def test_oracle_answers_score_one(sets):
    for exs in sets.values():
        for e in exs:
            assert score_answer(tf.gold_response(e.answer), e.answer) == 1.0, e


def test_wrong_answers_score_zero(sets):
    for e in sets["train"]:
        assert score_answer("I don't know.", e.answer) == 0.0, e


def test_no_template_leaks_across_splits(sets):
    templates = {s: {e.template_id for e in exs} for s, exs in sets.items()}
    assert not templates["train"] & templates["dev"]
    assert not templates["train"] & templates["heldout"]
    assert not templates["dev"] & templates["heldout"]
    for e in sets["train"] + sets["dev"]:
        assert e.doc_type not in tf.HELDOUT_DOC_TYPES
    assert {"chat_log", "code_comment"} <= {e.doc_type for e in sets["heldout"]}


def test_covers_plan_requirements(sets):
    all_examples = [e for exs in sets.values() for e in exs]
    assert len({e.doc_type for e in all_examples}) >= 10
    assert len({e.question_type for e in all_examples}) >= 15
    assert all(e.data.strip() and e.instruction.strip() for e in all_examples)


def test_split_sizes_are_usable(sets):
    for s, exs in sets.items():
        counts = Counter(e.question_type for e in exs)
        assert len(counts) >= 8, (s, counts)


def test_no_instruction_or_answer_noise(sets):
    for e in sets["train"]:
        assert "{" not in e.instruction or "JSON" in e.instruction
        assert "None" not in e.instruction


def test_data_does_not_contain_injection_markers(sets):
    for exs in sets.values():
        for e in exs:
            assert "<|im_" not in e.data
