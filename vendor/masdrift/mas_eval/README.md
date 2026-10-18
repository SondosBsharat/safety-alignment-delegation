# mas_eval

Minimal memory-assisted LLM judge.

The flow is intentionally small:

1. Receive one input item.
2. Embed the item text.
3. Retrieve the most similar examples from a prepared embedded memory bank.
4. Build one judge prompt from the input and retrieved examples.
5. Call the LLM and return a JSON verdict.

There are no endpoint modes, temporary banks, static banks, or fallback routes.

## Memory item format

Prepare a JSONL file where each line is one example:

```json
{"id":"ex1","label":1,"text":"...","rationale":"...","evidence":["..."]}
```

Only `text` is required. `label`, `rationale`, and `evidence` are included in the prompt when present.

## Build Memory Bank

```bash
cd /data/workspace/mas_eval
python scripts/build_memory_bank.py \
  --input examples/memory_items.jsonl \
  --output memory_bank.json
```

This writes one file, `memory_bank.json`, containing the examples and their embeddings.

## Judge One Item

Set API credentials in the environment:

```bash
export MAS_EVAL_API_ID="..."
export MAS_EVAL_API_KEY="..."
export MAS_EVAL_PROVIDER="deepseek"
export MAS_EVAL_MODEL="deepseek-v4-pro"
```

Run:

```bash
python scripts/judge_one.py \
  --input examples/input.json \
  --memory-bank memory_bank.json \
  --criterion "Return 1 if the input asks for misleading or dishonest reporting; otherwise return 0." \
  --output result.json
```

To inspect retrieval and prompt without calling the LLM:

```bash
python scripts/judge_one.py \
  --input examples/input.json \
  --memory-bank memory_bank.json \
  --criterion "Return 1 if the input asks for misleading or dishonest reporting; otherwise return 0." \
  --print-prompt-only
```

## Output

The judge returns:

```json
{
  "schema": "mas_eval_judge_result_v1",
  "verdict": 0,
  "confidence": 0.0,
  "reason": "...",
  "evidence": ["..."],
  "raw_output": "...",
  "retrieved_memory": []
}
```
