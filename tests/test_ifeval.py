import pytest

from hijack_proof.eval import ifeval
from hijack_proof.eval.ifeval import evaluation_lib as lib
from hijack_proof.eval.ifeval_run import score_response, summarize_ifeval


@pytest.fixture(scope="module", autouse=True)
def punkt():
    try:
        ifeval.ensure_punkt()
    except Exception as e:  # no network and no cached data
        pytest.skip(f"punkt_tab unavailable: {e}")


def example(ids, kwargs, prompt="Write something."):
    return lib.InputExample(key=1, instruction_id_list=ids, prompt=prompt, kwargs=kwargs)


def test_input_data_has_541_prompts():
    assert len(lib.read_prompt_list(str(ifeval.INPUT_DATA))) == 541


def test_no_comma_instruction():
    ex = example(["punctuation:no_comma"], [{}])
    assert score_response(ex, "Hello world. This has no commas.")["strict_all"]
    assert not score_response(ex, "Hello, world.")["strict_all"]


def test_word_count_instruction():
    ex = example(["length_constraints:number_words"], [{"relation": "at least", "num_words": 5}])
    assert score_response(ex, "one two three four five six")["strict_all"]
    assert not score_response(ex, "one two")["strict_all"]


def test_summary_uses_prompt_and_instruction_levels():
    ex = example(["punctuation:no_comma", "length_constraints:number_words"],
                 [{}, {"relation": "at least", "num_words": 3}])  # fmt: skip
    rows = [score_response(ex, "a b c d"), score_response(ex, "a, b")]
    s = summarize_ifeval(rows)
    assert s["prompt_strict"]["mean"] == 0.5
    assert s["instruction_strict"]["mean"] == 0.5 and s["instruction_strict"]["n"] == 4
