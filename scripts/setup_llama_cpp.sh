#!/usr/bin/env bash
# Clone and build llama.cpp (CPU, native AVX-512/AMX) under third_party/ (git-ignored).
set -euo pipefail
mkdir -p third_party
[ -d third_party/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp third_party/llama.cpp
cd third_party/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build -j"$(nproc)" --target llama-cli llama-server llama-bench llama-quantize llama-perplexity
