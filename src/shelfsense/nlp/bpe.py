"""Byte-level BPE tokenizer trained from scratch. Lossless: decode(encode(x)) == x.

Because it works on UTF-8 bytes, any language (and emoji) round-trips with no <unk>.
Id layout: [special tokens][256 byte tokens][learned merges...].
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from shelfsense.nlp.text import pieces

SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>", "<sep>")


def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class BPETokenizer:
    def __init__(self, merges: list | None = None):
        self.n_special = len(SPECIALS)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in (merges or [])]
        self._rebuild()

    def _rebuild(self) -> None:
        self._base = self.n_special + 256
        self._rank = {pair: i for i, pair in enumerate(self.merges)}
        self._tok: list[bytes] = [b""] * self.n_special + [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self._tok.append(self._tok[a] + self._tok[b])
        self._cache: dict[str, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return self._base + len(self.merges)

    def special_id(self, name: str) -> int:
        return SPECIALS.index(name)

    # ------------------------------------------------------------------ training
    def train(self, texts, vocab_size: int = 1000, min_freq: int = 2) -> "BPETokenizer":
        base = self.n_special + 256
        if vocab_size < base:
            raise ValueError(f"vocab_size must be at least {base}")
        counts: Counter = Counter()
        for text in texts:
            for p in pieces(text):
                counts[tuple(self.n_special + b for b in p.encode("utf-8"))] += 1
        words = dict(counts)
        merges: list[tuple[int, int]] = []
        while len(merges) < vocab_size - base:
            pair_counts: Counter = Counter()
            for w, c in words.items():
                for pair in zip(w, w[1:]):
                    pair_counts[pair] += c
            if not pair_counts:
                break
            # most frequent pair; ties broken by smallest ids, so training is deterministic
            best, freq = max(pair_counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if freq < min_freq:
                break
            new_id = base + len(merges)
            merges.append(best)
            merged: dict[tuple[int, ...], int] = {}
            for w, c in words.items():
                nw = _merge_word(w, best, new_id)
                merged[nw] = merged.get(nw, 0) + c
            words = merged
        self.merges = merges
        self._rebuild()
        return self

    # ------------------------------------------------------------------ encoding
    def _encode_piece(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return list(cached)
        ids = [self.n_special + b for b in piece.encode("utf-8")]
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self._rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i:best_i + 2] = [self._base + best_rank]
        self._cache[piece] = tuple(ids)
        return ids

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        if add_special:
            ids.append(self.special_id("<bos>"))
        for p in pieces(text):
            ids.extend(self._encode_piece(p))
        if add_special:
            ids.append(self.special_id("<eos>"))
        return ids

    def encode_word(self, word: str) -> list[int]:
        """Encode a bare word with no leading space (used for search analyzers)."""
        return self._encode_piece(word)

    def decode(self, ids, skip_special: bool = True) -> str:
        chunks: list[bytes] = []
        for i in ids:
            if i < 0 or i >= self.vocab_size:
                raise ValueError(f"token id {i} is out of range")
            if i < self.n_special:
                if not skip_special:
                    chunks.append(SPECIALS[i].encode("utf-8"))
                continue
            chunks.append(self._tok[i])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def token_str(self, token_id: int) -> str:
        """Readable form of one token (for debugging)."""
        if token_id < self.n_special:
            return SPECIALS[token_id]
        return self._tok[token_id].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "specials": list(SPECIALS), "merges": [list(m) for m in self.merges]}
        p.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("specials") != list(SPECIALS):
            raise ValueError("tokenizer file uses a different set of special tokens")
        return cls(data["merges"])