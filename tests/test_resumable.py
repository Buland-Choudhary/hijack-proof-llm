import sys
import time

import pytest

from hijack_proof import queue as hpq
from hijack_proof.chat import ChatEncoder
from hijack_proof.eval import suites
from hijack_proof.eval.run import evaluate_resumable, load_rows
from tests.test_eval_harness import FakeTokenizer, OracleGenerator


class CrashingGenerator(OracleGenerator):
    """Works for the first `ok_calls` calls, then raises (simulates a killed job)."""

    def __init__(self, text, tok, ok_calls):
        super().__init__(text, tok)
        self.calls, self.ok_calls = 0, ok_calls

    def generate(self, prompts, max_new_tokens=64, progress=None):
        self.calls += 1
        if self.calls > self.ok_calls:
            raise RuntimeError("machine reset")
        return super().generate(prompts, max_new_tokens, progress)


@pytest.fixture
def items():
    return suites.load_suite("dev")[:50]


def run(items, gen, path, **kw):
    enc = ChatEncoder(FakeTokenizer(), fmt="user_inline")
    return evaluate_resumable(items, gen, enc, "none", 16, path, log=lambda m: None, **kw)


def test_resume_after_crash_matches_uninterrupted_run(tmp_path, items):
    tok = FakeTokenizer()
    reference, finished = run(
        items, OracleGenerator("answer", tok), tmp_path / "ref.jsonl", chunk_size=10
    )
    assert finished
    path = tmp_path / "rows.jsonl"
    with pytest.raises(RuntimeError):
        run(items, CrashingGenerator("answer", tok, ok_calls=2), path, chunk_size=10)
    assert len(load_rows(path)) == 20  # two finished chunks were saved before the crash
    rows, finished = run(items, OracleGenerator("answer", tok), path, chunk_size=10)
    assert finished and rows == reference
    assert len(path.read_text().splitlines()) == 50  # nothing was recomputed or duplicated


def test_truncated_last_line_is_ignored(tmp_path, items):
    tok = FakeTokenizer()
    path = tmp_path / "rows.jsonl"
    run(items[:10], OracleGenerator("answer", tok), path, chunk_size=10)
    with path.open("a") as f:
        f.write('{"id": "half-written", "kin')  # killed mid-write
    assert len(load_rows(path)) == 10
    rows, finished = run(items[:20], OracleGenerator("answer", tok), path, chunk_size=10)
    assert finished and len(rows) == 20


def test_deadline_pauses_cleanly_and_resumes(tmp_path, items):
    tok = FakeTokenizer()
    path = tmp_path / "rows.jsonl"
    rows, finished = run(
        items, OracleGenerator("answer", tok), path, chunk_size=10, deadline=time.time() - 1
    )
    assert not finished and len(rows) == 10  # one chunk is always completed, so progress is made
    rows, finished = run(items, OracleGenerator("answer", tok), path, chunk_size=10)
    assert finished and len(rows) == 50


# ---------------------------------------------------------------- queue
def write_queue(tmp_path, lines):
    q = tmp_path / "q.txt"
    q.write_text("\n".join(lines) + "\n")
    return q


def test_queue_runs_in_order_and_skips_finished(tmp_path, capsys):
    out = tmp_path / "out.txt"
    q = write_queue(tmp_path, ["# comment", f"echo a >> {out}", "", f"echo b >> {out}"])
    assert hpq.run_queue(q, None, tmp_path / "state") == 0
    assert out.read_text() == "a\nb\n"
    assert hpq.run_queue(q, None, tmp_path / "state") == 0  # everything already done
    assert out.read_text() == "a\nb\n"


def test_queue_pauses_on_exit_3_and_continues_there(tmp_path):
    out, flag = tmp_path / "out.txt", tmp_path / "flag"
    pause_once = f"test -f {flag} && echo second-pass >> {out} || (touch {flag}; exit 3)"
    q = write_queue(tmp_path, [f"echo a >> {out}", pause_once, f"echo c >> {out}"])
    assert hpq.run_queue(q, None, tmp_path / "state") == hpq.PAUSED
    assert out.read_text() == "a\n"
    assert hpq.run_queue(q, None, tmp_path / "state") == 0
    assert out.read_text() == "a\nsecond-pass\nc\n"


def test_queue_stops_on_failure(tmp_path):
    out = tmp_path / "out.txt"
    q = write_queue(tmp_path, ["exit 1", f"echo never >> {out}"])
    assert hpq.run_queue(q, None, tmp_path / "state") == 1
    assert not out.exists()


def test_queue_exports_deadline_to_jobs(tmp_path):
    out = tmp_path / "out.txt"
    q = write_queue(
        tmp_path, [f"{sys.executable} -c \"import os; print(os.environ['HP_DEADLINE'])\" > {out}"]
    )
    assert hpq.run_queue(q, 5, tmp_path / "state") == 0
    assert float(out.read_text()) > time.time()
