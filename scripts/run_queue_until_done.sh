#!/usr/bin/env bash
# Run a queue file in <=25-minute chunks until it finishes (each chunk is resumable).
# Usage: scripts/run_queue_until_done.sh scripts/queues/<name>.txt
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
while true; do
  .venv/bin/hp-queue run "$1" --max-minutes 25
  code=$?
  [ "$code" -ne 3 ] && exit "$code"
  echo "--- chunk finished at $(date -u +%H:%M:%S), continuing ---"
done
