import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from hijack_proof.serve import server as serve


def test_parse_request_maps_roles():
    prompt, max_tokens = serve.parse_request({
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Summarize this."},
            {"role": "input", "content": "doc A"},
            {"role": "tool", "content": "doc B"},
        ],
        "max_tokens": 50,
    })  # fmt: skip
    assert prompt.system == "Be brief." and prompt.instruction == "Summarize this."
    assert prompt.data == "doc A\n\ndoc B" and max_tokens == 50


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {"messages": [{"role": "input", "content": "x"}]},
        {"messages": [{"role": "assistant", "content": "x"}]},
        {"messages": [{"role": "user", "content": 5}]},
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0},
    ],
)
def test_parse_request_rejects_bad_input(body):
    with pytest.raises(ValueError):
        serve.parse_request(body)


class FakeEncoder:
    eos_token_id = 9

    def encode_prompt(self, p):
        self.seen = p
        return [1, 2, 3]

    def decode_response(self, ids):
        return "hello"


class FakeGenerator:
    def generate(self, prompts, max_new_tokens=64):
        return [[5, 6, 9]]


def test_http_roundtrip():
    enc = FakeEncoder()
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve.make_handler(enc, FakeGenerator(), "m"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        body = {"messages": [{"role": "user", "content": "hi"}, {"role": "input", "content": "d"}]}
        conn.request("POST", "/v1/chat/completions", json.dumps(body))
        resp = conn.getresponse()
        out = json.loads(resp.read())
        assert resp.status == 200 and out["choices"][0]["message"]["content"] == "hello"
        assert out["choices"][0]["finish_reason"] == "stop" and enc.seen.data == "d"
        conn.request("POST", "/v1/chat/completions", json.dumps({"messages": []}))
        assert conn.getresponse().status == 400
    finally:
        server.shutdown()
