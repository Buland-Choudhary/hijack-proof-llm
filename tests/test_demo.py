from hijack_proof.serve import demo


class FakeResponse:
    status_code = 200

    def json(self):
        return {"choices": [{"message": {"content": "$113"}}]}


def test_ask_sends_the_document_as_untrusted_input(monkeypatch):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(url=url, body=json)
        return FakeResponse()

    monkeypatch.setattr(demo.requests, "post", fake_post)
    assert demo.ask("http://x", "Total?", "doc text") == "$113"
    roles = [m["role"] for m in sent["body"]["messages"]]
    assert roles == ["user", "input"] and sent["url"] == "http://x/v1/chat/completions"
    demo.ask("http://x", "Total?", "  ")
    assert [m["role"] for m in sent["body"]["messages"]] == ["user"]
