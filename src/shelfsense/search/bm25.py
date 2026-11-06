"""Okapi BM25 over an inverted index. The analyzer (text -> terms) is pluggable."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from shelfsense.nlp.text import normalize, words


def default_analyzer(text: str) -> list[str]:
    return words(normalize(text))


def bpe_analyzer(tokenizer):
    """Terms are BPE token ids of each word (no leading space), for subword matching."""
    def analyze(text: str) -> list[str]:
        out: list[str] = []
        for w in words(normalize(text)):
            out.extend(f"t{i}" for i in tokenizer.encode_word(w))
        return out
    return analyze


class BM25Index:
    def __init__(self, analyzer=None, k1: float = 1.5, b: float = 0.75):
        self.analyzer = analyzer or default_analyzer
        self.k1, self.b = k1, b
        self._ids: list[int] = []
        self._lens: list[int] = []
        self._post: dict[str, list[tuple[int, int]]] = {}
        self._avgdl = 1.0

    def build(self, docs: dict[int, str]) -> "BM25Index":
        self._ids, self._lens, self._post = [], [], {}
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, (doc_id, text) in enumerate(sorted(docs.items())):
            terms = self.analyzer(text)
            self._ids.append(doc_id)
            self._lens.append(len(terms))
            for term, tf in Counter(terms).items():
                post[term].append((idx, tf))
        self._post = dict(post)
        self._avgdl = max(sum(self._lens) / len(self._lens), 1e-9) if self._lens else 1.0
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if not self._ids:
            return []
        n = len(self._ids)
        scores: dict[int, float] = defaultdict(float)
        for term in set(self.analyzer(query)):
            postings = self._post.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for idx, tf in postings:
                norm = tf + self.k1 * (1 - self.b + self.b * self._lens[idx] / self._avgdl)
                scores[idx] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], self._ids[kv[0]]))[:k]
        return [(self._ids[i], s) for i, s in ranked]