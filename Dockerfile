# CPU image: llama.cpp server backend + hp-serve (secure chat format) + Gradio demo.
# Models are not baked in. Mount a folder with the GGUF and the merged model's tokenizer:
#   docker build -t hijack-proof-llm .
#   docker run -p 8000:8000 -p 7860:7860 -v "$PWD/models:/app/models" hijack-proof-llm
# NOTE: written for the project but not built in the cloud environment that produced the results.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY scripts/setup_llama_cpp.sh scripts/
RUN bash scripts/setup_llama_cpp.sh

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /app/third_party/llama.cpp/build/bin third_party/llama.cpp/build/bin
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir ".[demo]"
ENV MODEL=hp-qwen06-sft GGUF=models/gguf/hp-qwen06-sft-q4_k_m.gguf
EXPOSE 8000 7860
CMD ["sh", "-c", "hp-serve --model $MODEL --gguf $GGUF --host 0.0.0.0 --port 8000 & sleep 20 && hp-demo --endpoint http://127.0.0.1:8000 --host 0.0.0.0"]
