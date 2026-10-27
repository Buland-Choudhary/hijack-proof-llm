"""Hardware-dependent numeric settings.

bfloat16 is fast only on CPUs with AMX or AVX512-BF16. On other CPUs (the cloud VM's CPU changed
during this project) bf16 matmuls ran about 4x slower than fp32 (measured: 53 vs 230 GFLOP/s on 2
threads), so training and generation fall back to fp32 there. Override with HP_DTYPE=bf16|fp32.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def has_fast_bf16() -> bool:
    forced = os.environ.get("HP_DTYPE", "").lower()
    if forced in ("bf16", "fp32"):
        return forced == "bf16"
    try:
        with open("/proc/cpuinfo") as f:
            flags = next((line for line in f if line.startswith("flags")), "")
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def compute_dtype() -> torch.dtype:
    return torch.bfloat16 if has_fast_bf16() else torch.float32


def autocast():
    """CPU autocast to bf16 where that is fast, otherwise a no-op context."""
    return torch.autocast("cpu", dtype=torch.bfloat16, enabled=has_fast_bf16())
