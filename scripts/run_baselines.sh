#!/usr/bin/env bash
# Run the undefended + prompting-defense baseline matrix on the frozen core suite.
# Resumable: a (model, defense) pair is skipped if its result file already exists.
# Usage: scripts/run_baselines.sh [suite]   (default: core)
set -uo pipefail
suite="${1:-core}"
cd "$(git rev-parse --show-toplevel)"
for model in smollm2-135m smollm2-360m qwen3-0.6b; do
  for defense in none sandwich warning delimiters spotlight; do
    out="results/$suite/${model}__${defense}__user_inline.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    echo "$(date -u +%H:%M:%S) start $model $defense"
    .venv/bin/hp-eval --model "$model" --suite "$suite" --defense "$defense" --batch-size 16 2>&1 \
      | grep -v -i "warn\|Loading weights" || echo "FAILED $model $defense"
  done
done
echo "$(date -u +%H:%M:%S) ALL DONE"
