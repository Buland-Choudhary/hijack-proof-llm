import hashlib
from pathlib import Path

import pytest

from hijack_proof.models import fetch

MODELS_DIR = Path("models")


def test_wanted_files():
    assert fetch.wanted("model.safetensors")
    assert fetch.wanted("model-00001-of-00002.safetensors")
    assert fetch.wanted("tokenizer.json")
    assert fetch.wanted("LICENSE")
    assert not fetch.wanted("training_args.bin")  # pickled file: never downloaded
    assert not fetch.wanted("trainer_state.json")


def test_sha256_file(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    assert fetch.sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_registry_entries_are_complete():
    for name, spec in fetch.MODELS.items():
        assert spec.oci_repo.startswith("ai/"), name
        assert spec.oci_tag.endswith("safetensors"), name
        assert "/" in spec.hf_repo, name


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        fetch.fetch("not-a-model")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["smollm2-135m", "smollm2-360m", "qwen3-0.6b"])
def test_fetched_model_matches_recorded_digests(name):
    source = MODELS_DIR / name / "SOURCE.json"
    if not source.exists():
        pytest.skip(f"{name} not downloaded")
    import json

    files = json.loads(source.read_text())["files"]
    for rel, digest in files.items():
        if rel.endswith(".safetensors"):
            continue  # large files are verified at download time
        assert "sha256:" + fetch.sha256_file(MODELS_DIR / name / rel) == digest
