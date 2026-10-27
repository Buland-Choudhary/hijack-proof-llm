from hijack_proof.redteam import pairs


def row(i, success, valid=True):
    return {"id": f"d{i}/canary", "call": 1, "valid": valid, "success": success,
            "payload": "canary", "doc_instruction": "Who is first?", "data": f"doc {i}",
            "doc_answer": {"type": "contains", "gold": "Ann Lee"},
            "attack": {"type": "contains", "any": ["HACKED-ab12"]}}  # fmt: skip


def test_pairs_balance_successes_and_failures_and_skip_invalid():
    rows = [row(1, True), row(2, True), row(3, False), row(4, False), row(5, False),
            row(6, False, valid=False)]  # fmt: skip
    out = pairs.pairs_from_attacks(rows)
    assert len(out) == 4
    assert sum(p["source"] == "redteam_success" for p in out) == 2
    assert all(p["chosen"] == "Ann Lee" and p["rejected"] == "HACKED-ab12" for p in out)
    assert all(p["id"].startswith("redteam-") and p["kind"] == "injection" for p in out)


def test_no_pairs_without_successes():
    assert pairs.pairs_from_attacks([row(1, False), row(2, False)]) == []
