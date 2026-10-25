"""Sequential job queue for long, resumable jobs (one heavy CPU job at a time).

    hp-queue run scripts/queues/baselines_dev.txt --max-minutes 25

A queue file has one shell command per line (blank lines and # comments ignored). Commands run in
order. Each must be resumable and honor the HP_DEADLINE environment variable (epoch seconds), as
hp-eval does: when time is up a job exits with code 3 ("paused"), the queue stops with exit code 3,
and running the same queue again continues where it left off. Finished commands are recorded in
runs/queue/<queue name>.done and skipped next time.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PAUSED = 3


def read_commands(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def run_queue(path: Path, max_minutes: float | None, state_dir: Path = Path("runs/queue")) -> int:
    commands = read_commands(path)
    state_dir.mkdir(parents=True, exist_ok=True)
    done_path = state_dir / f"{path.stem}.done"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    if max_minutes is not None:
        env["HP_DEADLINE"] = str(time.time() + 60 * max_minutes)
    for n, cmd in enumerate(commands, 1):
        if cmd in done:
            print(f"[{n}/{len(commands)}] done earlier: {cmd}")
            continue
        if max_minutes is not None and time.time() >= float(env["HP_DEADLINE"]):
            print(f"PAUSED before [{n}/{len(commands)}]: time budget used up")
            return PAUSED
        print(f"[{n}/{len(commands)}] {time.strftime('%H:%M:%S')} {cmd}", flush=True)
        code = subprocess.run(cmd, shell=True, env=env).returncode
        if code == PAUSED:
            print(f"PAUSED during [{n}/{len(commands)}]: run the queue again to continue")
            return PAUSED
        if code != 0:
            print(f"FAILED (exit {code}): {cmd}")
            return 1
        with done_path.open("a") as f:
            f.write(cmd + "\n")
    print(f"queue {path.name}: all {len(commands)} jobs done")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run (or continue) a queue file")
    run.add_argument("file", type=Path)
    run.add_argument("--max-minutes", type=float, help="stop cleanly after this many minutes")
    args = ap.parse_args(argv)
    return run_queue(args.file, args.max_minutes)


if __name__ == "__main__":
    sys.exit(main())
