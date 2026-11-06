"""The JSON contract between the parser and the rest of ShelfSense."""
from __future__ import annotations

import json
import re

INTENTS = ("availability", "price", "bargain", "order", "greeting", "other")
_SIZE_RE = re.compile(r"\d+(kg|g|ml|l)")


def build_target(intent: str, items: list[dict], offer: int | None = None) -> dict:
    obj = {"intent": intent, "items": items}
    if offer is not None:
        obj["offer"] = offer
    return obj


def dumps_target(obj: dict) -> str:
    """Compact, fixed-order JSON: this exact string is what the GPT learns to write."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate(obj) -> str | None:
    """Return None if obj matches the contract, otherwise a short reason."""
    if not isinstance(obj, dict):
        return "not an object"
    extra = set(obj) - {"intent", "items", "offer"}
    if extra:
        return f"unexpected keys {sorted(extra)}"
    if obj.get("intent") not in INTENTS:
        return "bad intent"
    items = obj.get("items")
    if not isinstance(items, list):
        return "items must be a list"
    for it in items:
        if not isinstance(it, dict):
            return "item is not an object"
        if set(it) - {"name", "qty", "size"}:
            return "unexpected item keys"
        name = it.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 60:
            return "bad item name"
        if not _is_int(it.get("qty")) or not 1 <= it["qty"] <= 99:
            return "bad qty"
        if "size" in it and not (isinstance(it["size"], str) and _SIZE_RE.fullmatch(it["size"])):
            return "bad size"
    if "offer" in obj and (not _is_int(obj["offer"]) or not 1 <= obj["offer"] <= 100000):
        return "bad offer"
    return None


def canonical(obj: dict) -> dict:
    """Rebuild a validated object with fixed key order and trimmed names."""
    items = []
    for it in obj["items"]:
        c = {"name": it["name"].strip(), "qty": it["qty"]}
        if "size" in it:
            c["size"] = it["size"]
        items.append(c)
    return build_target(obj["intent"], items, obj.get("offer"))