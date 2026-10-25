"""IFEval: verifiable instruction-following benchmark (official checkers vendored, see NOTICE.md).

The checkers need NLTK's punkt_tab tokenizer data. NLTK's own downloader refuses to run through
the sandbox proxy, so `ensure_punkt` fetches the data file directly.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import nltk
import requests

NLTK_DIR = Path("data/raw/nltk_data")
PUNKT_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
)
INPUT_DATA = Path(__file__).parent / "input_data.jsonl"


def ensure_punkt() -> None:
    """Make NLTK's punkt_tab data available (downloaded once into data/raw/nltk_data)."""
    target = NLTK_DIR / "tokenizers"
    if not (target / "punkt_tab").exists():
        target.mkdir(parents=True, exist_ok=True)
        r = requests.get(PUNKT_URL, timeout=120)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():  # refuse paths that would escape the target folder
                if not (target / name).resolve().is_relative_to(target.resolve()):
                    raise RuntimeError(f"unsafe path in punkt archive: {name}")
            z.extractall(target)
    resolved = str(NLTK_DIR.resolve())
    if resolved not in nltk.data.path:
        nltk.data.path.insert(0, resolved)
