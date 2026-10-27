#!/usr/bin/env bash
# Merge a LoRA adapter into its base model and produce a Q8_0 GGUF.
# Usage: make_gguf.sh <base model folder> <adapter dir> <name>   -> models/gguf/<name>-q8_0.gguf
set -euo pipefail
base="$1"; adapter="$2"; name="$3"
.venv/bin/hp-merge --model "$base" --adapter "$adapter" --name "$name"
.venv/bin/python third_party/llama.cpp/convert_hf_to_gguf.py "models/$name" \
  --outfile "models/gguf/$name-f16.gguf" --outtype f16 > /dev/null
LD_LIBRARY_PATH=third_party/llama.cpp/build/bin third_party/llama.cpp/build/bin/llama-quantize \
  "models/gguf/$name-f16.gguf" "models/gguf/$name-q8_0.gguf" q8_0 > /dev/null
rm -f "models/gguf/$name-f16.gguf"
ls -la "models/gguf/$name-q8_0.gguf"
