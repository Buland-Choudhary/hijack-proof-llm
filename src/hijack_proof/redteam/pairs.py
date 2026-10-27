"""Turn attacker samples into DPO pairs for the next defender (plan Stage 4).

Each usable attack (the text contains the payload marker) becomes one preference pair on the
attacked document: chosen = the correct answer to the user's question, rejected = the response that
obeys the injection. Successful attacks (they hijacked the current defender) are the main signal;
an equal number of usable attacks that failed are added so the defender also learns that these
phrasings must be ignored, not only the few that got through.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hijack_proof.data import dpo_data, sft_data


def pairs_from_attacks(rows: list[dict], seed: int = 0, failure_ratio: float = 1.0) -> list[dict]:
    usable = [r for r in rows if r.get("valid")]
    wins = [r for r in usable if r["success"]]
    losses = [r for r in usable if not r["success"]]
    random.Random(f"redteam-pairs/{seed}").shuffle(losses)
    chosen_rows = wins + losses[: int(round(failure_ratio * len(wins)))]
    pairs = []
    for r in chosen_rows:
        answer = sft_data.reference_answer(r["doc_answer"])
        rejected = dpo_data.hijacked_response(r["payload"], r["attack"], answer)
        if rejected.strip() == answer.strip():
            continue
        pairs.append({
            "id": f"redteam-{r['id']}-{r['call']}", "kind": "injection", "payload": r["payload"],
            "source": "redteam_success" if r["success"] else "redteam_failure",
            "instruction": r["doc_instruction"], "data": r["data"],
            "chosen": answer, "rejected": rejected,
        })  # fmt: skip
    return pairs


def write_pairs(attacks_jsonl: Path, out: Path, seed: int = 0) -> dict:
    rows = [json.loads(line) for line in attacks_jsonl.read_text().splitlines()]
    pairs = pairs_from_attacks(rows, seed)
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs))
    return {
        "attacks": len(rows), "usable": sum(bool(r.get("valid")) for r in rows),
        "successful": sum(bool(r.get("success")) for r in rows), "pairs": len(pairs),
    }  # fmt: skip
