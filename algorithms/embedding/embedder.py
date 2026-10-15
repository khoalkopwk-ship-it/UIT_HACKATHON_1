from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class BGE_M3_Embedder:
    """Small wrapper around BAAI/bge-m3.

    The model automatically uses CUDA when PyTorch detects an NVIDIA GPU.
    On an RTX 3060, FP16 is enabled by default and batch size 4 is a safe
    starting point for max_length=8192.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 4,
        max_length: int = 8192,
        use_fp16: bool | None = None,
    ) -> None:
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Thiếu dependency. Hãy chạy: python -m pip install -r requirements.txt"
            ) from exc

        if batch_size <= 0:
            raise ValueError("batch_size phải > 0")
        if max_length <= 0:
            raise ValueError("max_length phải > 0")

        if use_fp16 is None:
            use_fp16 = bool(torch.cuda.is_available())

        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.use_fp16 = bool(use_fp16)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=self.use_fp16,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        output = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        dense = np.asarray(output["dense_vecs"], dtype=np.float32)
        if dense.ndim != 2:
            raise ValueError(f"dense_vecs phải là ma trận 2D, shape={dense.shape}")
        return l2_normalize(dense)

    def encode_corpus_to_npy(
        self,
        texts: Iterable[str],
        output_path: str | Path,
        total_count: int,
    ) -> dict:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        memmap = None
        dimension = None
        written = 0
        batch: list[str] = []

        def flush() -> None:
            nonlocal memmap, dimension, written, batch
            if not batch:
                return
            vectors = self.encode(batch)
            if dimension is None:
                dimension = int(vectors.shape[1])
                memmap = np.lib.format.open_memmap(
                    output_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=(total_count, dimension),
                )
            if vectors.shape[1] != dimension:
                raise ValueError("Embedding dimension thay đổi giữa các batch")
            memmap[written : written + len(batch)] = vectors
            written += len(batch)
            batch = []

        for text in texts:
            batch.append(text)
            if len(batch) >= self.batch_size:
                flush()
        flush()

        if memmap is None or dimension is None or written != total_count:
            raise ValueError(
                f"Không ghi đủ embeddings: written={written}, expected={total_count}"
            )
        memmap.flush()
        del memmap

        return {
            "model": self.model_name,
            "doc_count": total_count,
            "dimension": dimension,
            "max_length": self.max_length,
            "normalized": True,
            "device": self.device,
            "fp16": self.use_fp16,
        }


# Backward-compatible name used by the original baseline.
BGEM3DenseEncoder = BGE_M3_Embedder
