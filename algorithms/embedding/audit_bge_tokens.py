from __future__ import annotations

import argparse
import json
from pathlib import Path


def audit(corpus_path: str | Path, model_name: str = "BAAI/bge-m3", max_length: int = 8192) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    total = 0
    over = 0
    max_seen = 0
    for line in Path(corpus_path).open("r", encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        n = len(tokenizer.encode(str(obj["content"]), add_special_tokens=True))
        total += 1
        max_seen = max(max_seen, n)
        if n > max_length:
            over += 1
    return {"chunks": total, "over_limit": over, "max_tokens": max_seen, "max_length": max_length}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default="BAAI/bge-m3")
    p.add_argument("--max-length", type=int, default=8192)
    a = p.parse_args()
    print(json.dumps(audit(a.corpus, a.model, a.max_length), ensure_ascii=False, indent=2))
