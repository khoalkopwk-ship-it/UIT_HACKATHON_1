from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any

from algorithms.retrieval.hybrid_search import HybridLegalRetriever


def load_public_questions(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if isinstance(data, list):
        converted: dict[str, dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict) or "id" not in item:
                raise ValueError("Mỗi query dạng list phải có id")
            converted[str(item["id"])] = item
        data = converted

    if not isinstance(data, dict):
        raise ValueError("File public/train phải là JSON object hoặc list")

    for qid, obj in data.items():
        if not isinstance(obj, dict):
            raise ValueError(f"Query {qid} không phải object")
        question = obj.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Query {qid} thiếu question hợp lệ")
    return data


def validate_submission(
    submission: dict[str, Any],
    public_data: dict[str, Any],
    max_answers: int = 5,
    include_question: bool = True,
) -> None:
    if set(submission) != set(public_data):
        raise ValueError("Submission thiếu hoặc thừa query id")

    required = {"question", "answer"} if include_question else {"answer"}
    for qid, obj in submission.items():
        if not isinstance(obj, dict) or set(obj.keys()) != required:
            raise ValueError(
                f"Query {qid}: output phải có đúng các field {sorted(required)}"
            )
        if include_question and obj["question"] != public_data[qid]["question"]:
            raise ValueError(f"Query {qid}: question bị thay đổi")
        answers = obj["answer"]
        if not isinstance(answers, list) or len(answers) > max_answers:
            raise ValueError(f"Query {qid}: answer phải là list tối đa {max_answers} phần tử")
        normalized = [str(x) for x in answers]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Query {qid}: document_id bị trùng")
        if any(not x.strip() for x in normalized):
            raise ValueError(f"Query {qid}: document_id rỗng")


def generate_submission(
    public_path: str | Path,
    index_dir: str | Path,
    output_path: str | Path,
    model_name: str = "BAAI/bge-m3",
    query_batch_size: int = 8,
    bm25_top_k: int = 200,
    dense_top_k: int = 200,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    top_k_documents: int = 5,
    max_length: int = 8192,
    include_question: bool = True,
    create_zip: bool = False,
) -> dict[str, dict[str, Any]]:
    public_data = load_public_questions(public_path)
    if top_k_documents < 1 or top_k_documents > 5:
        raise ValueError("top_k_documents phải nằm trong 1..5")
    if query_batch_size < 1:
        raise ValueError("query_batch_size phải > 0")

    retriever = HybridLegalRetriever(
        index_dir=index_dir,
        model_name=model_name,
        bge_batch_size=query_batch_size,
        max_length=max_length,
    )

    qids = list(public_data.keys())
    submission: dict[str, dict[str, Any]] = {}

    for start in range(0, len(qids), query_batch_size):
        batch_ids = qids[start : start + query_batch_size]
        questions = [public_data[qid]["question"].strip() for qid in batch_ids]

        answers = retriever.retrieve_batch(
            questions,
            bm25_top_k=bm25_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
            top_k_documents=top_k_documents,
        )

        for qid, doc_ids in zip(batch_ids, answers):
            item: dict[str, Any] = {
                "answer": [str(x) for x in doc_ids[:top_k_documents]]
            }
            if include_question:
                item = {
                    "question": public_data[qid]["question"],
                    "answer": item["answer"],
                }
            submission[str(qid)] = item

        done = min(start + query_batch_size, len(qids))
        print(f"Processed {done}/{len(qids)} questions")

    validate_submission(
        submission,
        public_data,
        max_answers=5,
        include_question=include_question,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)

    if create_zip:
        zip_path = output_path.with_suffix(".zip")
        with zipfile.ZipFile(
            zip_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            zf.write(output_path, arcname="submission.json")
        print(f"Created: {zip_path}")

    print(f"DONE: {output_path.resolve()}")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BM25 + BGE-M3 + RRF -> Top 5 -> submission.json"
    )
    parser.add_argument("--public", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output", default="output/submission.json")
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--bm25-top-k", type=int, default=200)
    parser.add_argument("--dense-top-k", type=int, default=200)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--top-k-documents", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    generate_submission(
        public_path=args.public,
        index_dir=args.index_dir,
        output_path=args.output,
        model_name=args.model,
        query_batch_size=args.query_batch_size,
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        rrf_k=args.rrf_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        top_k_documents=args.top_k_documents,
        max_length=args.max_length,
        include_question=not args.answer_only,
        create_zip=args.zip,
    )


if __name__ == "__main__":
    main()
