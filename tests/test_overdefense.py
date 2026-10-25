from hijack_proof.data import overdefense as od
from hijack_proof.data import taskforge as tf
from hijack_proof.eval.checkers import score_answer


def test_deterministic_and_balanced():
    a, b = od.build_overdefense(100, seed=1), od.build_overdefense(100, seed=1)
    assert a == b and a != od.build_overdefense(100, seed=2)
    assert {i["doc_type"] for i in a} == {"howto", "email_request", "recipe_steps", "notice"}
    assert all(i["kind"] == "clean" and "<|im_" not in i["data"] for i in a)


def test_gold_answers_score_one_and_wrong_ones_zero():
    for item in od.build_overdefense(200, seed=3):
        spec = item["answer"]
        assert score_answer(tf.gold_response(spec), spec) == 1.0, item
        assert score_answer("I cannot help with that request.", spec) == 0.0, item


def test_frozen_overdefense_suite_matches_the_generator():
    from hijack_proof.eval import suites

    frozen = suites.load_suite("overdefense")  # verifies the checksum
    assert len(frozen) == 200
    assert suites.to_jsonl(frozen) == suites.to_jsonl(od.build_overdefense(200, 4004))
