from hijack_proof.eval import report


def ci(mean):
    return {"mean": mean, "lo": max(0, mean - 0.02), "hi": min(1, mean + 0.02), "n": 100}


def result(model, defense, clean, asr, fmt="user_inline"):
    return {
        "meta": {"model": model, "defense": defense, "format": fmt},
        "summary": {
            "clean_accuracy": ci(clean), "attacked_accuracy": ci(clean - 0.1),
            "asr_loose": ci(asr), "asr_strict": ci(asr - 0.05),
            "attacked_slices": {
                "seen_family": {"True": {"asr_loose": asr}, "False": {"asr_loose": asr}}
            },
        },
    }  # fmt: skip


def test_choose_picks_lowest_asr_within_clean_budget():
    rs = [
        result("m", "none", 0.60, 0.50),
        result("m", "sandwich", 0.59, 0.30),  # allowed, best
        result("m", "warning", 0.40, 0.05),  # too costly in clean accuracy
        result("m", "delimiters", 0.60, 0.45),
        result("m", "none", 0.60, 0.01, fmt="input_role"),  # not a prompting defense, ignored
    ]
    assert report.choose_defense(rs, "m")["chosen_defense"] == "sandwich"


def test_choose_falls_back_to_none_when_nothing_helps():
    rs = [result("m", "none", 0.60, 0.20), result("m", "warning", 0.60, 0.30),
          result("m", "sandwich", 0.20, 0.00)]  # fmt: skip
    assert report.choose_defense(rs, "m")["chosen_defense"] == "none"


def test_markdown_table_has_a_row_per_result():
    text = report.markdown_table([result("a", "none", 0.5, 0.4), result("b", "warning", 0.6, 0.2)])
    lines = text.strip().splitlines()
    assert len(lines) == 4 and "| a | none | user_inline | 50.0 (48.0-52.0) |" in lines[2]
