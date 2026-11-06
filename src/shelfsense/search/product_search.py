"""Customer-safe product search: hybrid BM25 + vector retrieval plus an 'is it really there?' check.

Reads ONLY v_customer_catalog (no cost/floor/margin). Stock and price are read live at query
time, never stored in the index, so an 'available' claim always reflects the database.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher

from shelfsense.nlp.text import normalize, words
from shelfsense.search.bm25 import BM25Index
from shelfsense.search.hybrid import rrf_fuse
from shelfsense.search.vector_index import HashingEmbedder, VectorIndex

# Words that carry no product meaning (English, transliterated Tamil/Hindi, and their scripts).
FILLER = frozenset({
    "a", "an", "the", "is", "are", "do", "does", "you", "have", "has", "any", "available",
    "availability", "stock", "in", "instock", "need", "want", "please", "pls", "give", "me", "i",
    "my", "for", "of", "and", "or", "with", "can", "get", "there", "got", "ok", "sir", "madam",
    "bro", "boss", "anna", "akka", "irukka", "irukku", "kidaikkuma", "kidaikuma", "venum",
    "vendum", "bhaiya", "bhai", "hai", "hain", "kya", "milega", "milegi", "chahiye",
    "இருக்கா", "இருக்கிறதா", "அண்ணா", "வேணும்", "கிடைக்குமா",
    "है", "क्या", "भैया", "मिलेगा", "चाहिए",
})
FUZZY_MIN_LEN = 4
FUZZY_RATIO = 0.84
POOL = 20        # candidates taken from each retriever
CHECK_TOP = 5    # candidates inspected for the coverage check


@dataclass(frozen=True)
class Hit:
    product_id: int
    name: str
    brand: str
    pack_size: str
    selling_price: float
    in_stock: bool
    bm25: float      # raw BM25 score (0.0 if not retrieved by BM25)
    cosine: float    # raw cosine score (0.0 if not retrieved by the vector index)
    fused: float     # reciprocal-rank-fusion score


@dataclass(frozen=True)
class SearchResult:
    query: str
    status: str                  # found | partial | not_found | empty
    coverage: float              # share of the query's content words matched by the best candidate
    unmatched: tuple[str, ...]   # content words no candidate covered
    hits: tuple[Hit, ...]


class ProductSearch:
    def __init__(self, conn: sqlite3.Connection, store_id: int, embedder=None):
        self.conn = conn
        self.store_id = store_id
        self._embedder = embedder or HashingEmbedder()
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the indexes from the database (takes milliseconds for a small catalog)."""
        rows = self.conn.execute(
            "SELECT product_id, name, brand, category, pack_size "
            "FROM v_customer_catalog WHERE store_id = ?", (self.store_id,)).fetchall()
        aliases: dict[int, list[str]] = {}
        try:
            for r in self.conn.execute(
                    "SELECT product_id, alias FROM product_aliases ORDER BY product_id, alias"):
                aliases.setdefault(r["product_id"], []).append(r["alias"])
        except sqlite3.OperationalError:
            pass    # older database without the aliases table
        docs = {}
        for r in rows:
            parts = [r["brand"], r["name"], r["pack_size"], r["category"],
                     *aliases.get(r["product_id"], [])]
            docs[r["product_id"]] = " ".join(parts)
        self._doc_words = {pid: set(words(normalize(t))) for pid, t in docs.items()}
        self.bm25 = BM25Index().build(docs)
        self.vec = VectorIndex(self._embedder).build(docs)

    @staticmethod
    def _content_words(query: str) -> list[str]:
        seen: list[str] = []
        for w in words(normalize(query)):
            if w not in FILLER and w not in seen:
                seen.append(w)
        return seen

    @staticmethod
    def _word_present(word: str, doc_words: set[str]) -> bool:
        if word in doc_words:
            return True
        if len(word) >= FUZZY_MIN_LEN and not word.isdigit():
            for d in doc_words:
                if len(d) >= FUZZY_MIN_LEN and SequenceMatcher(None, word, d).ratio() >= FUZZY_RATIO:
                    return True
        return False

    def search(self, query: str, k: int = 5) -> SearchResult:
        content = self._content_words(query)
        if not content:
            return SearchResult(query, "empty", 0.0, (), ())

        bm_list = self.bm25.search(query, k=POOL)
        vc_list = self.vec.search(query, k=POOL)
        bm_scores, vc_scores = dict(bm_list), dict(vc_list)
        fused = rrf_fuse([[d for d, _ in bm_list], [d for d, _ in vc_list]])
        if not fused:
            return SearchResult(query, "not_found", 0.0, tuple(content), ())

        # Does ANY top candidate really contain what was asked for?
        best_cov, best_missing = -1.0, tuple(content)
        for pid, _ in fused[:max(k, CHECK_TOP)]:
            present = [w for w in content if self._word_present(w, self._doc_words[pid])]
            cov = len(present) / len(content)
            if cov > best_cov:
                best_cov = cov
                best_missing = tuple(w for w in content if w not in present)
        status = "found" if best_cov >= 1.0 else ("partial" if best_cov > 0 else "not_found")

        top = fused[:k]
        ids = [pid for pid, _ in top]
        marks = ",".join("?" * len(ids))
        live = {r["product_id"]: r for r in self.conn.execute(
            f"SELECT product_id, name, brand, pack_size, selling_price, in_stock "
            f"FROM v_customer_catalog WHERE store_id = ? AND product_id IN ({marks})",
            (self.store_id, *ids))}
        hits = []
        for pid, score in top:
            r = live.get(pid)
            if r is None:
                continue
            hits.append(Hit(pid, r["name"], r["brand"], r["pack_size"], r["selling_price"],
                            bool(r["in_stock"]), round(bm_scores.get(pid, 0.0), 4),
                            round(vc_scores.get(pid, 0.0), 4), round(score, 5)))
        return SearchResult(query, status, round(max(best_cov, 0.0), 3), best_missing, tuple(hits))