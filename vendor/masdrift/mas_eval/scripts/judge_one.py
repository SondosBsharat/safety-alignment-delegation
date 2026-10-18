#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mas_eval import GatewayLLMClient, LLMConfig, TextEmbedder, judge_one, load_memory_bank, retrieve_memory
from mas_eval.simple_judge import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL, build_judge_messages, item_to_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge one item with retrieved memory examples.")
    parser.add_argument("--input", required=True, help="Input .json or text file.")
    parser.add_argument("--memory-bank", required=True, help="Path to memory_bank.json built by build_memory_bank.py.")
    parser.add_argument("--criterion", default="", help="One-sentence judgment criterion.")
    parser.add_argument("--criterion-file", default="", help="Optional file containing the judgment criterion.")
    parser.add_argument("--output", default="", help="Optional output JSON path.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_EMBEDDING_DIM)
    parser.add_argument("--provider", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-endpoint", default="")
    parser.add_argument("--api-id", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--print-prompt-only", action="store_true", help="Print retrieved prompt and do not call the LLM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    criterion = _load_criterion(args)
    data = _load_input(Path(args.input))
    memory_bank = load_memory_bank(Path(args.memory_bank))
    embedder = TextEmbedder(model_name=args.embedding_model, dim=args.embedding_dim)

    if args.print_prompt_only:
        data_text = item_to_text(data)
        hits = retrieve_memory(query_text=data_text, memory_bank=memory_bank, embedder=embedder, top_k=args.top_k)
        print(json.dumps(build_judge_messages(data_text=data_text, criterion=criterion, memory_hits=hits), ensure_ascii=False, indent=2))
        return

    llm_config = LLMConfig.from_env()
    if args.provider:
        llm_config.provider = args.provider
    if args.model:
        llm_config.model = args.model
    if args.api_base:
        llm_config.api_base = args.api_base
    if args.api_endpoint:
        llm_config.api_endpoint = args.api_endpoint
    if args.api_id:
        llm_config.api_id = args.api_id
    if args.api_key:
        llm_config.api_key = args.api_key
    if args.max_tokens > 0:
        llm_config.max_tokens = args.max_tokens

    result = judge_one(
        data=data,
        criterion=criterion,
        memory_bank=memory_bank,
        embedder=embedder,
        llm_client=GatewayLLMClient(llm_config),
        top_k=args.top_k,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output + "\n", encoding="utf-8")
        print(f"[mas-eval] result={args.output}")
    else:
        print(output)


def _load_criterion(args: argparse.Namespace) -> str:
    if args.criterion_file:
        text = Path(args.criterion_file).read_text(encoding="utf-8").strip()
    else:
        text = str(args.criterion or "").strip()
    if not text:
        raise SystemExit("--criterion or --criterion-file is required")
    return text


def _load_input(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return text


if __name__ == "__main__":
    main()
