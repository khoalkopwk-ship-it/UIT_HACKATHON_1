from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from algorithms.embedding.embedder import BGE_M3_Embedder
from algorithms.retrieval.bm25 import BM25Index, build_bm25_index
from algorithms.retrieval.dense_search import DenseRetriever


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def iter_chunk_objects(chunks_input: str | Path) -> Iterator[dict[str, Any]]:
    root = Path(chunks_input)
    if root.is_dir() and (root / "chunks").is_dir():
        root = root / "chunks"

    if root.is_file():
        files = [root]
    else:
        jsonl = sorted(root.rglob("*.jsonl"))
        # Preprocessing may place non-chunk reports (for example
        # reports/summary.json) below the chunks directory. Prefer the
        # project's chunk naming convention so those files are not indexed.
        named_chunk_json_files = sorted(root.rglob("context_*.json"))
        json_files = named_chunk_json_files or sorted(root.rglob("*.json"))
        if jsonl and json_files:
            raise ValueError("Không trộn JSON và JSONL trong cùng corpus")
        files = jsonl or json_files

    if not files:
        raise FileNotFoundError(f"Không tìm thấy chunks tại: {root}")

    for path in files:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(f"JSONL lỗi: {path}:{line_no}")
                    yield obj
        else:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                yield data
            elif isinstance(data, list):
                for obj in data:
                    if not isinstance(obj, dict):
                        raise ValueError(f"JSON list lỗi: {path}")
                    yield obj
            else:
                raise ValueError(f"JSON root không hợp lệ: {path}")


def materialize_corpus(chunks_input: str | Path, corpus_path: str | Path) -> dict:
    corpus_path = Path(corpus_path)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    chunk_id_occurrences: dict[str, int] = defaultdict(int)
    sources: set[str] = set()
    count = 0
    renamed_duplicate_count = 0

    with corpus_path.open("w", encoding="utf-8") as out:
        for obj in iter_chunk_objects(chunks_input):
            chunk_id = obj.get("chunk_id")
            source = obj.get("source")
            content = obj.get("content")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Chunk thiếu chunk_id")
            if not isinstance(source, dict) or source.get("id") is None:
                raise ValueError(f"Chunk {chunk_id} thiếu source.id")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Chunk {chunk_id} thiếu content")

            original_chunk_id = chunk_id.strip()
            chunk_id_occurrences[original_chunk_id] += 1
            occurrence = chunk_id_occurrences[original_chunk_id]
            chunk_id = original_chunk_id
            if occurrence > 1:
                chunk_id = f"{original_chunk_id}__dup_{occurrence}"
                while chunk_id in seen:
                    occurrence += 1
                    chunk_id_occurrences[original_chunk_id] = occurrence
                    chunk_id = f"{original_chunk_id}__dup_{occurrence}"
                renamed_duplicate_count += 1
            source_id = str(source["id"])
            seen.add(chunk_id)
            sources.add(source_id)

            out.write(
                json.dumps(
                    {
                        "index": count,
                        "chunk_id": chunk_id,
                        "source_id": source_id,
                        "content": content.strip(),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1

    if count == 0:
        raise ValueError("Không có chunk nào để build index")

    meta = {
        "chunk_count": count,
        "source_document_count": len(sources),
        "renamed_duplicate_chunk_ids": renamed_duplicate_count,
        "sha256": sha256_file(corpus_path),
    }
    with (corpus_path.parent / "corpus_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _iter_corpus(corpus_path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_dense_index(
    corpus_path: str | Path,
    output_dir: str | Path,
    model_name: str = "BAAI/bge-m3",
    batch_size: int = 4,
    max_length: int = 8192,
    use_fp16: bool | None = None,
) -> dict:
    total_count = sum(1 for _ in _iter_corpus(corpus_path))
    if total_count == 0:
        raise ValueError("Corpus rỗng")

    encoder = BGE_M3_Embedder(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stream the corpus instead of keeping every JSON object and its content
    # in RAM. Large preprocessed corpora can otherwise exhaust system memory
    # before BGE-M3 finishes encoding.
    texts = (str(record["content"]) for record in _iter_corpus(corpus_path))
    meta = encoder.encode_corpus_to_npy(
        texts,
        output_dir / "dense_embeddings.npy",
        total_count,
    )
    meta.update(
        {
            "corpus_sha256": sha256_file(corpus_path),
            "type": "BGE-M3 dense",
        }
    )
    with (output_dir / "dense_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def build_indexes(
    chunks: str | Path,
    index_dir: str | Path,
    model_name: str = "BAAI/bge-m3",
    bge_batch_size: int = 4,
    max_length: int = 8192,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = index_dir / "corpus.jsonl"

    print("[1/3] Chuẩn hóa chunks -> corpus.jsonl")
    corpus_meta = materialize_corpus(chunks, corpus_path)
    print(
        f"      chunks={corpus_meta['chunk_count']:,}, "
        f"documents={corpus_meta['source_document_count']:,}"
    )

    print("[2/3] Build BM25 index")
    bm25_meta = build_bm25_index(corpus_path, index_dir, bm25_k1, bm25_b)
    print(
        f"      docs={bm25_meta['doc_count']:,}, "
        f"vocab={bm25_meta['vocabulary_size']:,}"
    )
    del bm25_meta
    gc.collect()

    print("[3/3] Build BGE-M3 dense index")
    dense_meta = build_dense_index(
        corpus_path,
        index_dir,
        model_name=model_name,
        batch_size=bge_batch_size,
        max_length=max_length,
    )
    print(
        f"      docs={dense_meta['doc_count']:,}, "
        f"dim={dense_meta['dimension']}, "
        f"device={dense_meta['device']}"
    )
    print(f"DONE: {index_dir.resolve()}")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[int, float]]],
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    if rrf_k < 0:
        raise ValueError("rrf_k phải >= 0")
    weights = list(weights or [1.0] * len(rankings))
    if len(weights) != len(rankings):
        raise ValueError("weights không khớp số ranking")

    fused: dict[int, float] = defaultdict(float)
    best_rank: dict[int, int] = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[int] = set()
        for rank, (idx, _) in enumerate(ranking, 1):
            if idx in seen:
                continue
            seen.add(idx)
            fused[idx] += float(weight) / (rrf_k + rank)
            best_rank[idx] = min(best_rank.get(idx, rank), rank)

    return sorted(
        fused.items(),
        key=lambda x: (-x[1], best_rank.get(x[0], 10**9), x[0]),
    )


def aggregate_to_documents(
    fused: Sequence[tuple[int, float]],
    chunk_metadata: Sequence[tuple[str, str]],
    top_k: int = 5,
) -> list[str]:
    best: dict[str, tuple[float, int]] = {}
    for fused_rank, (chunk_index, score) in enumerate(fused, 1):
        chunk_id, source_id = chunk_metadata[chunk_index]
        current = best.get(source_id)
        if current is None or score > current[0]:
            best[source_id] = (float(score), fused_rank)

    ranked = sorted(best.items(), key=lambda x: (-x[1][0], x[1][1], x[0]))
    return [source_id for source_id, _ in ranked[:top_k]]


class HybridLegalRetriever:
    def __init__(
        self,
        index_dir: str | Path,
        model_name: str = "BAAI/bge-m3",
        bge_batch_size: int = 4,
        max_length: int = 8192,
        use_fp16: bool | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.corpus_path = self.index_dir / "corpus.jsonl"
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Thiếu {self.corpus_path}")

        records = list(_iter_corpus(self.corpus_path))
        self.chunk_metadata = [
            (str(r["chunk_id"]), str(r["source_id"])) for r in records
        ]
        corpus_hash = sha256_file(self.corpus_path)

        self.bm25 = BM25Index.load(self.index_dir / "bm25_index.pkl")
        with (self.index_dir / "bm25_meta.json").open(encoding="utf-8") as f:
            bm25_meta = json.load(f)
        with (self.index_dir / "dense_meta.json").open(encoding="utf-8") as f:
            dense_meta = json.load(f)

        if bm25_meta.get("corpus_sha256") not in (None, corpus_hash):
            raise ValueError("BM25 index không khớp corpus.jsonl")
        if dense_meta.get("corpus_sha256") != corpus_hash:
            raise ValueError("Dense index không khớp corpus.jsonl")
        if int(bm25_meta["doc_count"]) != len(self.chunk_metadata):
            raise ValueError("BM25 doc_count không khớp corpus")
        if int(dense_meta["doc_count"]) != len(self.chunk_metadata):
            raise ValueError("Dense doc_count không khớp corpus")
        if dense_meta.get("model") != model_name:
            raise ValueError(
                f"Dense index dùng {dense_meta.get('model')}, không phải {model_name}"
            )

        self.encoder = BGE_M3_Embedder(
            model_name=model_name,
            batch_size=bge_batch_size,
            max_length=max_length,
            use_fp16=use_fp16,
        )
        self.dense = DenseRetriever(self.index_dir, self.encoder)

    def retrieve_batch(
        self,
        queries: Sequence[str],
        bm25_top_k: int = 200,
        dense_top_k: int = 200,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        top_k_documents: int = 5,
    ) -> list[list[str]]:
        bm25_rankings = [self.bm25.retrieve(q, bm25_top_k) for q in queries]
        dense_rankings = self.dense.retrieve_batch(queries, dense_top_k)

        result: list[list[str]] = []
        for bm25_rank, dense_rank in zip(bm25_rankings, dense_rankings):
            fused = reciprocal_rank_fusion(
                [bm25_rank, dense_rank],
                rrf_k=rrf_k,
                weights=[bm25_weight, dense_weight],
            )
            result.append(
                aggregate_to_documents(
                    fused, self.chunk_metadata, top_k=top_k_documents
                )
            )
        return result


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 + BGE-M3 indexes for LegalIR"
    )
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--bge-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    args = parser.parse_args()
    build_indexes(
        chunks=args.chunks,
        index_dir=args.index_dir,
        model_name=args.model,
        bge_batch_size=args.bge_batch_size,
        max_length=args.max_length,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )


if __name__ == "__main__":
    _cli()
