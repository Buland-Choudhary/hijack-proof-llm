"""A small OpenAI-style chat server that applies the secure format.

    hp-serve --model hp-qwen06-sft --gguf models/gguf/hp-qwen06-sft-q4_k_m.gguf --port 8000

POST /v1/chat/completions accepts `messages` with the roles `system`, `user` (the trusted request)
and `input` (untrusted text: a document, a web page, a tool result; `tool` is accepted as an alias).
The untrusted text is tokenized with special tokens disabled, so a typed `<|im_start|>system` stays
plain text and cannot forge a role. Greedy decoding, no streaming.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hijack_proof.chat import DEFAULT_SYSTEM, ChatEncoder, Prompt

UNTRUSTED_ROLES = ("input", "tool")


def parse_request(body: dict) -> tuple[Prompt, int]:
    """OpenAI-style request body -> (Prompt, max_tokens). Raises ValueError on bad input."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("`messages` must be a non-empty list")
    system, users, data = DEFAULT_SYSTEM, [], []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if not isinstance(content, str):
            raise ValueError("every message needs a string `content`")
        if role == "system":
            system = content
        elif role == "user":
            users.append(content)
        elif role in UNTRUSTED_ROLES:
            data.append(content)
        else:
            raise ValueError(f"unsupported role {role!r}; use system, user, input")
    if not users:
        raise ValueError("at least one `user` message is required")
    max_tokens = int(body.get("max_tokens", 256))
    if not 1 <= max_tokens <= 1024:
        raise ValueError("`max_tokens` must be between 1 and 1024")
    prompt = Prompt("\n\n".join(users), "\n\n".join(data) if data else None, system)
    return prompt, max_tokens


def make_handler(encoder: ChatEncoder, generator, model_name: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": model_name})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                prompt, max_tokens = parse_request(json.loads(self.rfile.read(length)))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            ids = encoder.encode_prompt(prompt)
            out = generator.generate([ids], max_new_tokens=max_tokens)[0]
            finished = encoder.eos_token_id in out
            self._send(200, {
                "object": "chat.completion", "created": int(time.time()), "model": model_name,
                "choices": [{
                    "index": 0, "finish_reason": "stop" if finished else "length",
                    "message": {"role": "assistant", "content": encoder.decode_response(out)},
                }],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out)},
            })  # fmt: skip

        def log_message(self, *args):  # keep the console quiet
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="folder under models/ (tokenizer)")
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from hijack_proof.eval.generate import LlamaCppGenerator

    encoder = ChatEncoder(AutoTokenizer.from_pretrained(Path("models") / args.model))
    with LlamaCppGenerator(args.gguf, parallel=2) as generator:
        server = ThreadingHTTPServer(
            (args.host, args.port), make_handler(encoder, generator, args.model)
        )
        print(
            f"serving {args.model} on http://{args.host}:{args.port}/v1/chat/completions",
            flush=True,
        )
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
