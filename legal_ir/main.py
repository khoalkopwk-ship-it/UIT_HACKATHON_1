from __future__ import annotations

import argparse

from algorithms.chunking.preprocess_chunking_legal import main as chunking_main
from algorithms.retrieval.hybrid_search import build_indexes
from algorithms.generation.rag import generate_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LegalIR: preprocessing -> BM25/BGE-M3 -> RRF -> submission"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_chunk = sub.add_parser("preprocess", help="Chạy chunking module hiện có")
    p_chunk.add_argument("-i", "--input-dir", required=True)
    p_chunk.add_argument("-o", "--output-dir", required=True)
    p_chunk.add_argument("--pattern", default="context_*.json")
    p_chunk.add_argument("--output-mode", choices=("json", "jsonl"), default="json")
    p_chunk.add_argument("--jsonl-shard-size", type=int, default=5000)

    p_index = sub.add_parser("build-index", help="Build BM25 + BGE-M3 index")
    p_index.add_argument("--chunks", required=True)
    p_index.add_argument("--index-dir", default="indexes/legalir_bge_m3")
    p_index.add_argument("--model", default="BAAI/bge-m3")
    p_index.add_argument("--bge-batch-size", type=int, default=4)
    p_index.add_argument("--max-length", type=int, default=8192)
    p_index.add_argument("--bm25-k1", type=float, default=1.2)
    p_index.add_argument("--bm25-b", type=float, default=0.75)

    p_submit = sub.add_parser("submit", help="Sinh submission")
    p_submit.add_argument("--public", required=True)
    p_submit.add_argument("--index-dir", default="indexes/legalir_bge_m3")
    p_submit.add_argument("--output", default="output/submission.json")
    p_submit.add_argument("--model", default="BAAI/bge-m3")
    p_submit.add_argument("--query-batch-size", type=int, default=8)
    p_submit.add_argument("--bm25-top-k", type=int, default=200)
    p_submit.add_argument("--dense-top-k", type=int, default=200)
    p_submit.add_argument("--rrf-k", type=int, default=60)
    p_submit.add_argument("--bm25-weight", type=float, default=1.0)
    p_submit.add_argument("--dense-weight", type=float, default=1.0)
    p_submit.add_argument("--top-k-documents", type=int, default=5)
    p_submit.add_argument("--max-length", type=int, default=8192)
    p_submit.add_argument("--answer-only", action="store_true")
    p_submit.add_argument("--zip", action="store_true")

    args = parser.parse_args()

    if args.command == "preprocess":
        # Reuse the project's existing preprocessing implementation.
        argv = [
            "preprocess_chunking_legal.py",
            "-i", args.input_dir,
            "-o", args.output_dir,
            "--pattern", args.pattern,
            "--output-mode", args.output_mode,
            "--jsonl-shard-size", str(args.jsonl_shard_size),
        ]
        import sys
        old = sys.argv
        try:
            sys.argv = argv
            chunking_main()
        finally:
            sys.argv = old

    elif args.command == "build-index":
        build_indexes(
            chunks=args.chunks,
            index_dir=args.index_dir,
            model_name=args.model,
            bge_batch_size=args.bge_batch_size,
            max_length=args.max_length,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
        )

    elif args.command == "submit":
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
