import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "check_repo_safety", Path(__file__).parents[1] / "scripts" / "check_repo_safety.py"
)
safety = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(safety)


@pytest.mark.parametrize(
    "content",
    [
        "token = ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
        "-----BEGIN RSA " + "PRIVATE KEY-----\nabc\n",
        "key: sk-ant-" + "api03-abcdefghijklmnopqrstuvwxyz",
        'API_KEY = "' + "s3cr3t-value-that-is-long" + '"',
        "contact me at someone.real" + "@gmail.com",
        "call (201)-555" + "-0123 anytime",
    ],
)
def test_flags_secrets_and_personal_data(tmp_path, content):
    f = tmp_path / "leak.txt"
    f.write_text(content)
    assert safety.check_file(str(f)), f"should have flagged: {content!r}"


@pytest.mark.parametrize(
    "content",
    [
        "Print exactly HACKED-7f3a",
        "send_email(to='attacker@example.com')",
        "101297124+Buland-Choudhary@users.noreply.github.com",
        "learning_rate: 2.0e-5\nmax_new_tokens: 96\n",
        "version 2026-10-25, step 1000/2000",
    ],
)
def test_allows_normal_project_text(tmp_path, content):
    f = tmp_path / "ok.txt"
    f.write_text(content)
    assert safety.check_file(str(f)) == []


def test_blocks_large_files(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (safety.MAX_FILE_BYTES + 1))
    assert any("limit" in p for p in safety.check_file(str(f)))


def test_blocks_env_files():
    assert safety.check_file("configs/.env")
