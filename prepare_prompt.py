#!/usr/bin/env python3
import argparse
import ast
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a token-bounded LooGLE prompt")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--target-tokens", type=int, default=1751)
    parser.add_argument("--output", type=Path, default=Path("generated/prompt.txt"))
    args = parser.parse_args()

    source = Path("benchmark-source/LooGLE-testdata/longdep_qa.jsonl")
    rows = [json.loads(line) for line in source.open(encoding="utf-8") if line.strip()]
    record = max(rows, key=lambda row: len(row["input"]))
    qa_pairs = ast.literal_eval(record["qa_pairs"])
    question = qa_pairs[0]["Q"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    marker = "Request 000000.\n"
    suffix = f"\n\nQuestion: {question}\nAnswer briefly using the document:"
    fixed_tokens = tokenizer.encode(marker + suffix, add_special_tokens=False)
    budget = args.target_tokens - len(fixed_tokens)
    if budget <= 0:
        raise ValueError("target token count is too small for the instruction")
    document_tokens = tokenizer.encode(record["input"], add_special_tokens=False)[:budget]
    prompt = marker + tokenizer.decode(document_tokens, skip_special_tokens=True) + suffix
    while len(tokenizer.encode(prompt, add_special_tokens=False)) > args.target_tokens:
        document_tokens.pop()
        prompt = marker + tokenizer.decode(document_tokens, skip_special_tokens=True) + suffix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prompt, encoding="utf-8")
    metadata = {
        "source": str(source), "title": record["title"], "question": question,
        "source_chars": len(record["input"]), "source_words": len(record["input"].split()),
        "prepared_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
        "model_tokenizer": args.model,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
