#!/usr/bin/env bash
# Convert a Hugging Face-format model folder to GGUF.  Usage: to_gguf.sh models/qwen3-0.6b q8_0
# For q4_k_m, convert to f16 first and run llama-quantize (see docs/notes/M1).
set -euo pipefail
src="$1"; type="${2:-q8_0}"
mkdir -p models/gguf
python third_party/llama.cpp/convert_hf_to_gguf.py "$src" \
  --outfile "models/gguf/$(basename "$src")-$type.gguf" --outtype "$type"
