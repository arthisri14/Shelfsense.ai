"""A rule-based parser: the honest baseline the mini-GPT has to beat.

It is given every advantage a developer reasonably would have: the alias list and the words
that appear in the training templates. It does NOT know the held-out phrasings.
"""
from __future__ import annotations

from shelfsense.nlp.schema import build_target
from shelfsense.nlp.synth import ITEMS, static_template_words
from shelfsense.nlp.text import _runs, normalize

UNIT_MAP = {"kg": "kg", "kilo": "kg", "g": "g", "gm": "g", "ml": "ml", "l": "l",
            "litre": "l", "liter": "l"}
SEPARATORS = {"and", "aur", "&", "மற்றும்", "और"}
GREET = {"hi", "hello", "hey", "vanakkam", "namaste", "good", "morning", "evening",
         "வணக்கம்", "नमस्ते"}
PRICE_W = {"price", "rate", "evlo", "vila", "kitne", "much", "விலை", "कितने"}
AVAIL_W = {"have", "available", "irukka", "irukku", "stock", "kidaikkuma", "milega", "mil", "hai",
           "got", "இருக்கா", "இருக்கிறதா", "है", "मिलेगा"}
ORDER_W = {"give", "want", "send", "need", "venum", "vendum", "chahiye", "bhej", "anuppunga",
           "kudunga", "do", "வேணும்", "चाहिए", "reserve"}
BARGAIN_W = {"reduce", "kam", "kammi"}
FILLER = (static_template_words(holdout=False) | GREET | PRICE_W | AVAIL_W | ORDER_W | BARGAIN_W
          | {"packet", "packets", "pack", "pkt"})

ALIAS: dict[str, str] = {}
for _canon, _forms in ITEMS.items():
    ALIAS[_canon] = _canon
    for _f in _forms:
        ALIAS[normalize(_f)] = _canon


def _tokens(text: str) -> list[str]:
    toks: list[str] = []
    for cls, s in _runs(normalize(text)):
        if cls == "s":
            continue
        if cls == "p":
            if "," in s or "&" in s or "+" in s:
                toks.append(",")
            continue
        toks.append(s)
    return toks


def _resolve(ws: list[str]) -> str:
    phrase = " ".join(ws)
    if phrase in ALIAS:
        return ALIAS[phrase]
    for n in range(len(ws) - 1, 0, -1):          # a known alias hidden inside extra words
        for i in range(len(ws) - n + 1):
            sub = " ".join(ws[i:i + n])
            if sub in ALIAS:
                return ALIAS[sub]
    return phrase                                 # unknown item: copy what the customer wrote


def parse(text: str) -> dict:
    toks = _tokens(text)
    tokset = set(toks)
    chunks: list[list[str]] = []
    cur: list[str] = []
    for t in toks:
        if t == "," or t in SEPARATORS:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(t)
    if cur:
        chunks.append(cur)

    items: list[dict] = []
    offers: list[int] = []
    for ch in chunks:
        size = None
        rest: list[str] = []
        i = 0
        while i < len(ch):
            t = ch[i]
            if t.isascii() and t.isdigit() and i + 1 < len(ch) and ch[i + 1] in UNIT_MAP:
                size = f"{int(t)}{UNIT_MAP[ch[i + 1]]}"
                i += 2
                continue
            rest.append(t)
            i += 1
        qty = None
        name_words: list[str] = []
        for t in rest:
            if t.isascii() and t.isdigit():
                v = int(t)
                if v >= 10:
                    offers.append(v)
                elif qty is None and v >= 1:
                    qty = v
                continue
            if t in FILLER:
                continue
            name_words.append(t)
        if not name_words:
            continue
        item = {"name": _resolve(name_words), "qty": qty or 1}
        if size:
            item["size"] = size
        items.append(item)

    if not items:
        return build_target("greeting" if tokset & GREET else "other", [])
    offer = offers[0] if offers else None
    if offer is not None or tokset & BARGAIN_W:
        intent = "bargain"
    elif tokset & PRICE_W:
        intent = "price"
    elif tokset & AVAIL_W:
        intent = "availability"
    elif tokset & ORDER_W:
        intent = "order"
    else:
        intent = "availability"
    return build_target(intent, items, offer if intent == "bargain" else None)