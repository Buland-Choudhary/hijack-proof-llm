from hijack_proof.attacks import library as lib
from hijack_proof.data import dpo_data


def test_injection_pairs_reject_the_hijacked_response():
    rows = [
        {"id": "a1", "source": "tf_attacked", "payload": "canary", "instruction": "Q?",
         "data": "doc", "response": "42"},
        {"id": "c1", "source": "tf_clean", "payload": None, "instruction": "Q?",
         "data": "doc", "response": "42"},
    ]  # fmt: skip
    attacks = {"a1": {"type": "contains", "any": ["HACKED-ab12"], "strict_prefix": True}}
    pairs = dpo_data.injection_pairs(rows, attacks)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "42" and pairs[0]["rejected"] == "HACKED-ab12"


def test_hijacked_response_matches_each_payload_checker():
    spec = {"type": "contains", "any": ["x.attacker-site.example.com"]}
    assert "x.attacker-site.example.com" in dpo_data.hijacked_response("url", spec, "42")
    assert dpo_data.hijacked_response("refuse", {"any": ["r"]}, "42") == lib.REFUSAL + "."


def test_directive_pairs_prefer_obeying_over_the_plain_answer():
    pairs = dpo_data.directive_pairs(10, 0)
    assert len(pairs) == 10
    assert all(p["kind"] == "directive" and p["chosen"] != p["rejected"] for p in pairs)
