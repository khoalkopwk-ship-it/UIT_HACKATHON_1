from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(x["id"]): x for x in data}
    if not isinstance(data, dict):
        raise ValueError("JSON phải là object hoặc list")
    return data


def _answers(obj: dict) -> list[str]:
    for key in ("answer", "answers", "ground_truth", "relevant"):
        value = obj.get(key)
        if isinstance(value, list):
            return [str(x) for x in value]
    return []


def evaluate(prediction_path: str | Path, ground_truth_path: str | Path) -> float:
    pred = _load(prediction_path)
    truth = _load(ground_truth_path)
    if set(pred) != set(truth):
        raise ValueError("Prediction và ground truth không cùng query id")

    hits = 0
    total = len(truth)
    if total == 0:
        return 0.0

    for qid, truth_obj in truth.items():
        gold = set(_answers(truth_obj))
        predicted = [str(x) for x in pred[qid].get("answer", [])]
        if gold and any(x in gold for x in predicted[:5]):
            hits += 1

    return hits / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Recall@5 cho LegalIR")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--ground-truth", required=True)
    args = parser.parse_args()

    score = evaluate(args.prediction, args.ground_truth)
    print(f"Recall@5: {score:.4f}")


if __name__ == "__main__":
    main()
