"""Merge a LoRA adapter into its base model and save a plain Hugging Face folder.

    hp-merge --model qwen3-0.6b --adapter runs/sft/qwen06-lora-uf/adapter --name hp-qwen06-sft

Writes models/<name>/ (bf16 weights, tokenizer, SOURCE.json). Used before GGUF conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="base model folder under models/")
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--name", required=True, help="output folder name under models/")
    args = ap.parse_args(argv)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir, out_dir = Path("models") / args.model, Path("models") / args.name
    model = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base_dir).save_pretrained(out_dir)
    (out_dir / "SOURCE.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "source": "trained here",
                "base": args.model,
                "adapter": str(args.adapter),
            },
            indent=2,
        )  # fmt: skip
    )
    print(f"merged {args.adapter} into {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
