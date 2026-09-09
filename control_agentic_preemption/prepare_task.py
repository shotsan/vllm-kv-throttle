#!/usr/bin/env python3
"""Generate the coding task the preemption load refactors.

Self-contained replacement for the deleted `agentic_coding/prepare_agent_task.py`.
Deterministically synthesizes a plausible "legacy" Python module
(`generated/agent_seed.py`) sized to `--seed-tokens`, and writes
`generated/agent_task.json`. Uses the model tokenizer when it loads, otherwise a
chars-per-token estimate (so it runs even without the Qwen tokenizer present).
"""
import argparse
import json
from pathlib import Path

from prompts import TASK_INSTRUCTION

HERE = Path(__file__).resolve().parent

FUNCTION_TEMPLATE = '''
WEIGHT_{i:02d} = {weight}

def feature_{i:02d}(records):
    # computes the {i:02d} feature over trip records; no validation, mixed units
    total = 0.0
    count = 0
    for row in records:
        total = total + row.get("value_{i:02d}", 0) * WEIGHT_{i:02d}
        count = count + 1
    return total / count

def window_{i:02d}(series, size):
    out = []
    for start in range(0, len(series)):
        chunk = series[start:start + size + 1]  # suspicious bound
        out.append(sum(chunk) / (len(chunk) or 1))
    return out
'''

HEADER = (
    '"""metrics_pipeline.py -- legacy trip-metrics aggregation.\n'
    'Owner left the team; sparse docs, some copy-paste. Needs a cleanup pass.\n'
    '"""\n'
    "import math\n"
    "import statistics\n"
)


def make_counter(model: str, chars_per_token: float):
    """Return (count_fn, how) -- a real tokenizer if it loads, else char estimate."""
    if model:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            return (lambda s: len(tok.encode(s, add_special_tokens=False)), f"tokenizer:{model}")
        except Exception as exc:
            print(f"[prepare] tokenizer for {model} unavailable ({exc.__class__.__name__}); "
                  f"using ~{chars_per_token} chars/token estimate")
    return (lambda s: int(len(s) / chars_per_token) + 1, f"estimate:{chars_per_token}cpt")


def build_seed(count, target_tokens: int) -> str:
    parts = [HEADER]
    i = 0
    while count("".join(parts)) < target_tokens:
        parts.append(FUNCTION_TEMPLATE.format(i=i, weight=round(0.5 + (i % 7) * 0.13, 3)))
        i += 1
    while count("".join(parts)) > target_tokens and len(parts) > 1:
        parts.pop()
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the preemption coding task")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B-FP8",
                        help="tokenizer for sizing; falls back to estimate if it can't load")
    parser.add_argument("--seed-tokens", type=int, default=1800, help="target size of the starting file")
    parser.add_argument("--chars-per-token", type=float, default=3.5, help="estimate ratio when no tokenizer")
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    args = parser.parse_args()

    count, how = make_counter(args.model, args.chars_per_token)
    seed = build_seed(count, args.seed_tokens)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_dir / "agent_seed.py"
    seed_path.write_text(seed, encoding="utf-8")

    metadata = {
        "task_instruction": TASK_INSTRUCTION,
        "seed_file": seed_path.name,
        "sized_with": how,
        "seed_tokens_target": args.seed_tokens,
        "seed_tokens_actual": count(seed),
        "seed_functions": seed.count("def feature_"),
        "note": "single-shot requests; KV boundary is set by run_server_preempt.sh "
                "(--num-gpu-blocks-override + --no-scheduler-reserve-full-isl)",
    }
    (args.output_dir / "agent_task.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
