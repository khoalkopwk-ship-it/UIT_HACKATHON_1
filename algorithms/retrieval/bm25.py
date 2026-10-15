from __future__ import annotations

import json
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOKEN_RE = re.compile(
    r"(?:[0-9A-Za-zÀ-ỹĐđ]+(?:[-.][0-9A-Za-zÀ-ỹĐđ]+)*/"
    r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-.][0-9A-Za-zÀ-ỹĐđ]+)*(?:/"
    r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-.][0-9A-Za-zÀ-ỹĐđ]+)*)*)"
    r"|(?:[0-9A-Za-zÀ-ỹĐđ]+)",
    re.UNICODE,
)
SPLIT_CODE_RE = re.compile(r"[/.-]+")


def normalize_for_retrieval(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize_bm25(text: str) -> list[str]:
    normalized = normalize_for_retrieval(text)
    output: list[str] = []
    for match in TOKEN_RE.finditer(normalized):
        token = match.group(0)
        output.append(token)
        if any(sep in token for sep in ("/", "-", ".")):
            output.extend(p for p in SPLIT_CODE_RE.split(token) if p)
    return output


class BM25Index:
    def __init__(
        self,
        doc_lengths: np.ndarray,
        postings: dict[str, tuple[np.ndarray, np.ndarray]],
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.doc_lengths = np.asarray(doc_lengths, dtype=np.float32)
        self.postings = postings
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = int(len(self.doc_lengths))
        self.avgdl = float(self.doc_lengths.mean()) if self.n_docs else 0.0

    def retrieve(self, query: str, top_k: int = 200) -> list[tuple[int, float]]:
        if self.n_docs == 0 or top_k <= 0 or self.avgdl <= 0:
            return []

        terms = list(dict.fromkeys(tokenize_bm25(query)))
        scores = np.zeros(self.n_docs, dtype=np.float32)
        touched = np.zeros(self.n_docs, dtype=bool)

        for term in terms:
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_ids, term_freqs = posting
            df = len(doc_ids)
            idf = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            dls = self.doc_lengths[doc_ids]
            denom = term_freqs + self.k1 * (
                1.0 - self.b + self.b * dls / self.avgdl
            )
            scores[doc_ids] += (
                idf * (term_freqs * (self.k1 + 1.0)) / denom
            ).astype(np.float32)
            touched[doc_ids] = True

        candidates = np.flatnonzero(touched)
        if len(candidates) == 0:
            return []

        k = min(top_k, len(candidates))
        values = scores[candidates]
        if k == len(candidates):
            order = np.argsort(-values)
        else:
            part = np.argpartition(-values, k - 1)[:k]
            order = part[np.argsort(-values[part])]

        return [(int(candidates[i]), float(values[i])) for i in order]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, BM25Index):
            raise TypeError("File không chứa BM25Index hợp lệ")
        return obj


def build_bm25_index(
    corpus_path: str | Path,
    output_dir: str | Path,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths: list[int] = []
    postings_tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)

    with Path(corpus_path).open("r", encoding="utf-8") as f:
        for expected_index, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record["index"]) != expected_index:
                raise ValueError("Corpus index không liên tục")
            tokens = tokenize_bm25(str(record["content"]))
            lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                postings_tmp[term].append((expected_index, int(freq)))

    postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for term, pairs in postings_tmp.items():
        docs = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
        tfs = np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs))
        postings[term] = (docs, tfs)

    index = BM25Index(np.asarray(lengths, dtype=np.float32), postings, k1, b)
    index_path = output_dir / "bm25_index.pkl"
    index.save(index_path)

    meta = {
        "type": "BM25",
        "k1": k1,
        "b": b,
        "doc_count": index.n_docs,
        "avgdl": index.avgdl,
        "vocabulary_size": len(postings),
        "corpus_sha256": __import__("hashlib").sha256(Path(corpus_path).read_bytes()).hexdigest(),
    }
    with (output_dir / "bm25_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta
