from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from algorithms.embedding.embedder import BGE_M3_Embedder


class DenseRetriever:
    def __init__(self, index_dir: str | Path, encoder: BGE_M3_Embedder) -> None:
        self.index_dir = Path(index_dir)
        self.encoder = encoder
        with (self.index_dir / "dense_meta.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.embeddings = np.load(
            self.index_dir / "dense_embeddings.npy", mmap_mode="r"
        )
        if self.embeddings.ndim != 2:
            raise ValueError("dense_embeddings.npy phải là ma trận 2D")
        if self.embeddings.shape[0] != int(self.meta["doc_count"]):
            raise ValueError("dense_meta doc_count không khớp embeddings")

    @staticmethod
    def _topk(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if len(scores) == 0 or top_k <= 0:
            return []
        k = min(top_k, len(scores))
        if k == len(scores):
            indices = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k - 1)[:k]
            indices = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in indices]

    def retrieve_batch(
        self, queries: Sequence[str], top_k: int = 200
    ) -> list[list[tuple[int, float]]]:
        if not queries:
            return []
        vectors = self.encoder.encode(queries)
        if vectors.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query/document embedding dimension không khớp")
        scores = vectors @ self.embeddings.T
        return [self._topk(row, top_k) for row in scores]
