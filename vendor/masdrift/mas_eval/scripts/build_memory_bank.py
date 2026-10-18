#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_eval import TextEmbedder, build_memory_bank
from mas_eval.simple_judge import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a simple embedded memory bank from JSONL items.")
    parser.add_argument("--input", required=True, help="Input JSONL. Each line should contain id/text/label/rationale/evidence.")
    parser.add_argument("--output", required=True, help="Output memory_bank.json path.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedder = TextEmbedder(model_name=args.embedding_model, dim=args.embedding_dim)
    bank = build_memory_bank(
        input_jsonl=Path(args.input),
        output_json=Path(args.output),
        embedder=embedder,
        batch_size=args.batch_size,
    )
    print(f"[mas-eval] memory_bank={args.output}")
    print(f"[mas-eval] items={len(bank.get('items', []))} model={bank.get('embedding_model')} dim={bank.get('embedding_dim')}")


if __name__ == "__main__":
    main()
