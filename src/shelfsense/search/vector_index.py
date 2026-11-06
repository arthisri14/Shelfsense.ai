"""Exact-cosine vector index in NumPy, plus a dependency-free hashing embedder.

HashingEmbedder captures SPELLING (character n-grams), not meaning: it catches typos such as
'buter' -> butter. Meaning across languages comes from the aliases in the product text.
A real multilingual embedding model can replace it later: any object with `dim` and
`embed(list[str]) -> ndarray (n, dim), L2-normalised` will work.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from shelfsense.nlp.text import normalize, words


class HashingEmbedder:
    def __init__(self, dim: int = 512, ngram_sizes: tuple[int, ...] = (3, 4)):
        self.dim = dim
        self.ngram_sizes = ngram_sizes

    def _features(self, text: str):
        for w in words(normalize(text)):
            yield "w:" + w, 1.0
            padded = f"<{w}>"
            for n in self.ngram_sizes:
                for i in range(len(padded) - n + 1):
                    yield "g:" + padded[i:i + n], 0.5

    def embed(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = zlib.crc32(feat.encode("utf-8"))      # stable across runs (unlike hash())
                sign = 1.0 if (h >> 16) & 1 == 0 else -1.0
                mat[r, h % self.dim] += sign * weight
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class VectorIndex:
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingEmbedder()
        self._ids: list[int] = []
        self._mat = np.zeros((0, self.embedder.dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._ids)

    def build(self, docs: dict[int, str]) -> "VectorIndex":
        self._ids = sorted(docs)
        if self._ids:
            self._mat = self.embedder.embed([docs[i] for i in self._ids])
        else:
            self._mat = np.zeros((0, self.embedder.dim), dtype=np.float32)
        return self

    def upsert(self, doc_id: int, text: str) -> None:
        vec = self.embedder.embed([text])[0]
        if doc_id in self._ids:
            self._mat[self._ids.index(doc_id)] = vec
        else:
            self._ids.append(doc_id)
            self._mat = np.vstack([self._mat, vec[None, :]])

    def remove(self, doc_id: int) -> bool:
        if doc_id not in self._ids:
            return False
        row = self._ids.index(doc_id)
        self._ids.pop(row)
        self._mat = np.delete(self._mat, row, axis=0)
        return True

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        q = self.embedder.embed([query])[0]
        if not q.any():
            return []
        sims = self._mat @ q
        order = np.lexsort((np.array(self._ids), -sims))[:k]    # best score first, ties by id
        return [(self._ids[i], float(sims[i])) for i in order]

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, ids=np.array(self._ids, dtype=np.int64), mat=self._mat)

    @classmethod
    def load(cls, path, embedder=None) -> "VectorIndex":
        idx = cls(embedder)
        data = np.load(Path(path), allow_pickle=False)     # never unpickle
        mat = data["mat"]
        if mat.ndim != 2 or mat.shape[1] != idx.embedder.dim:
            raise ValueError("saved index does not match the embedder dimension")
        idx._ids = data["ids"].tolist()
        idx._mat = mat
        return idx