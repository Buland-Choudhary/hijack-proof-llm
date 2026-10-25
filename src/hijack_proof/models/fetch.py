"""Download model weights with checksum verification.

Primary source is Docker Hub's `ai/` model registry: models are OCI artifacts whose layers are the
raw Hugging Face files (config.json, tokenizer.json, model.safetensors, ...), each annotated with
its file path and addressed by sha256 digest. This works where huggingface.co is blocked.
Fallback is the Hugging Face Hub (needs `huggingface_hub` and network access to it).

Every fetched model gets a SOURCE.json recording exactly which artifact was downloaded, so results
can cite a model by manifest digest.

Usage:
    hp-fetch --list
    hp-fetch smollm2-360m qwen3-0.6b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

REGISTRY = "https://registry-1.docker.io"
AUTH_URL = "https://auth.docker.io/token"
MANIFEST_ACCEPT = "application/vnd.oci.image.manifest.v1+json"
FILEPATH_ANNOTATION = "org.cncf.model.filepath"
DEFAULT_DEST = Path("models")

# Files needed to load a model with transformers. Everything else in the artifact
# (training logs, pickled training args, ...) is skipped.
WANTED_SUFFIXES = (".safetensors", ".json", ".txt", ".jinja", ".model")
WANTED_NAMES = {"LICENSE", "README.md"}
SKIPPED_NAMES = {
    "all_results.json",
    "eval_results.json",
    "train_results.json",
    "trainer_state.json",
}


@dataclass(frozen=True)
class ModelSpec:
    oci_repo: str
    oci_tag: str
    hf_repo: str


MODELS: dict[str, ModelSpec] = {
    "smollm2-135m": ModelSpec(
        "ai/smollm2", "135m-safetensors", "HuggingFaceTB/SmolLM2-135M-Instruct"
    ),
    "smollm2-360m": ModelSpec(
        "ai/smollm2", "360m-safetensors", "HuggingFaceTB/SmolLM2-360M-Instruct"
    ),
    "smollm2-1.7b": ModelSpec(
        "ai/smollm2", "1.7b-safetensors", "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    ),
    "qwen3-0.6b": ModelSpec("ai/qwen3", "0.6b-safetensors", "Qwen/Qwen3-0.6B"),
    "qwen3-1.7b": ModelSpec("ai/qwen3", "1.7b-safetensors", "Qwen/Qwen3-1.7B"),
    "qwen3-4b-instruct": ModelSpec(
        "ai/qwen3", "4b-instruct-2507-safetensors", "Qwen/Qwen3-4B-Instruct-2507"
    ),
}


def wanted(filepath: str) -> bool:
    name = Path(filepath).name
    if name in SKIPPED_NAMES:
        return False
    return name in WANTED_NAMES or name.endswith(WANTED_SUFFIXES)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


RETRY_STATUS = {429, 500, 502, 503, 504}


def _get(url: str, retries: int = 5, **kwargs) -> requests.Response:
    """GET with backoff on rate limits (Docker Hub allows ~100 anonymous requests/hour per IP)."""
    for attempt in range(retries + 1):
        r = requests.get(url, **kwargs)
        if r.status_code not in RETRY_STATUS or attempt == retries:
            r.raise_for_status()
            return r
        wait = float(r.headers.get("Retry-After", 0)) or min(10 * 3**attempt, 300)
        print(
            f"HTTP {r.status_code} from {url.split('/v2/')[-1][:60]}; retrying in {wait:.0f}s",
            file=sys.stderr,
        )
        r.close()
        time.sleep(wait)
    raise AssertionError("unreachable")


def _token(repo: str) -> str:
    r = _get(
        AUTH_URL,
        params={"service": "registry.docker.io", "scope": f"repository:{repo}:pull"},
        timeout=30,
    )
    return r.json()["token"]


def get_manifest(repo: str, tag: str) -> tuple[dict, str]:
    """Return (manifest, manifest_digest) for an OCI model artifact."""
    r = _get(
        f"{REGISTRY}/v2/{repo}/manifests/{tag}",
        headers={"Authorization": f"Bearer {_token(repo)}", "Accept": MANIFEST_ACCEPT},
        timeout=30,
    )
    digest = (
        r.headers.get("Docker-Content-Digest") or "sha256:" + hashlib.sha256(r.content).hexdigest()
    )
    return r.json(), digest


def download_blob(repo: str, digest: str, size: int, dest: Path) -> None:
    """Download one blob to `dest`, resuming a partial download, and verify its sha256."""
    if (
        dest.exists()
        and dest.stat().st_size == size
        and sha256_file(dest) == digest.split(":", 1)[1]
    ):
        return
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    headers = {"Authorization": f"Bearer {_token(repo)}"}
    if have:
        headers["Range"] = f"bytes={have}-"
    # requests drops the Authorization header when the registry redirects to its CDN host.
    with _get(
        f"{REGISTRY}/v2/{repo}/blobs/{digest}", headers=headers, stream=True, timeout=60
    ) as r:
        if have and r.status_code != 206:  # server ignored the range request: start over
            have = 0
        mode = "ab" if have else "wb"
        with (
            part.open(mode) as f,
            tqdm(
                total=size, initial=have, unit="B", unit_scale=True, desc=dest.name, leave=False
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    actual = sha256_file(part)
    if actual != digest.split(":", 1)[1]:
        part.unlink()
        raise RuntimeError(
            f"checksum mismatch for {dest.name}: expected {digest}, got sha256:{actual}"
        )
    part.rename(dest)


def fetch_oci(name: str, spec: ModelSpec, dest_root: Path) -> Path:
    manifest, manifest_digest = get_manifest(spec.oci_repo, spec.oci_tag)
    out = dest_root / name
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for layer in manifest.get("layers", []):
        filepath = (layer.get("annotations") or {}).get(FILEPATH_ANNOTATION)
        if not filepath or not wanted(filepath):
            continue
        target = out / filepath
        if out.resolve() not in target.resolve().parents:  # never write outside the model folder
            raise RuntimeError(f"unsafe file path in manifest: {filepath}")
        target.parent.mkdir(parents=True, exist_ok=True)
        download_blob(spec.oci_repo, layer["digest"], layer["size"], target)
        files[filepath] = layer["digest"]
    if not any(f.endswith(".safetensors") for f in files):
        raise RuntimeError(f"{spec.oci_repo}:{spec.oci_tag} has no safetensors weights")
    source = {
        "name": name,
        "source": "docker-hub-oci",
        "ref": f"{spec.oci_repo}:{spec.oci_tag}",
        "manifest_digest": manifest_digest,
        "upstream_hf_repo": spec.hf_repo,
        "files": files,
    }
    (out / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    return out


def fetch_hf(name: str, spec: ModelSpec, dest_root: Path) -> Path:
    from huggingface_hub import snapshot_download  # optional dependency

    out = dest_root / name
    path = snapshot_download(
        spec.hf_repo,
        local_dir=out,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*", "README.md"],
    )
    (out / "SOURCE.json").write_text(
        json.dumps({"name": name, "source": "huggingface", "ref": spec.hf_repo}, indent=2) + "\n"
    )
    return Path(path)


def fetch(name: str, dest_root: Path = DEFAULT_DEST) -> Path:
    """Fetch a model by short name. Tries Docker Hub first, then Hugging Face."""
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(MODELS)}")
    spec = MODELS[name]
    try:
        return fetch_oci(name, spec, dest_root)
    except (requests.RequestException, RuntimeError) as oci_error:
        print(f"Docker Hub fetch failed ({oci_error}); trying Hugging Face", file=sys.stderr)
        return fetch_hf(name, spec, dest_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download model weights with checksum checks.")
    parser.add_argument("models", nargs="*", help="short model names (see --list)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--list", action="store_true", help="list known models")
    args = parser.parse_args(argv)
    if args.list or not args.models:
        for name, spec in MODELS.items():
            print(f"{name:20s} {spec.oci_repo}:{spec.oci_tag}  (HF: {spec.hf_repo})")
        return 0
    for name in args.models:
        path = fetch(name, args.dest)
        print(f"{name}: ready at {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
