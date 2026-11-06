"""Unicode-aware text helpers shared by the tokenizer and the search code.

Python's \\w does not treat combining marks (Tamil/Hindi vowel signs) as word characters,
so regex-based splitting can cut those words mid-word. We classify characters by Unicode
category instead.
"""
from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """NFKC + casefold. Used for search. NOT applied inside the BPE tokenizer (it stays lossless)."""
    return unicodedata.normalize("NFKC", text).casefold()


def _char_class(ch: str) -> str:
    cat = unicodedata.category(ch)
    if cat[0] in ("L", "M"):      # letters and combining marks stay together
        return "a"
    if cat[0] == "N":
        return "d"
    if ch.isspace():
        return "s"
    return "p"


def _runs(text: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, list[str]]] = []
    for ch in text:
        c = _char_class(ch)
        if runs and runs[-1][0] == c:
            runs[-1][1].append(ch)
        else:
            runs.append((c, [ch]))
    return [(c, "".join(chars)) for c, chars in runs]


def words(text: str) -> list[str]:
    """Letter runs and digit runs only. '5kg' -> ['5', 'kg']; Tamil/Hindi words stay whole."""
    return [s for c, s in _runs(text) if c in ("a", "d")]


def pieces(text: str) -> list[str]:
    """Lossless split used by BPE: "".join(pieces(t)) == t. A space attaches to the next word."""
    runs = _runs(text)
    out: list[str] = []
    carry = ""
    for i, (cls, s) in enumerate(runs):
        if carry:
            s = carry + s
            carry = ""
        if cls == "s" and i + 1 < len(runs) and runs[i + 1][0] in ("a", "d"):
            head, carry = s[:-1], s[-1]
            if head:
                out.append(head)
        else:
            out.append(s)
    return out