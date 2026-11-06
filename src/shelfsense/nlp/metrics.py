"""Scoring for the query parser. Invalid outputs count as wrong everywhere."""
from __future__ import annotations


def _key(obj: dict):
    return (obj["intent"], obj.get("offer"),
            tuple(sorted((i["name"], i["qty"], i.get("size")) for i in obj["items"])))


def score_split(preds: list[dict | None], golds: list[dict]) -> dict:
    n = len(golds)
    valid = intent = exact = offer = 0
    tp = fp = fn = 0
    qty_ok = size_ok = gold_items = 0
    for p, g in zip(preds, golds):
        gnames = {i["name"]: i for i in g["items"]}
        gold_items += len(gnames)
        if p is None:
            fn += len(gnames)
            continue
        valid += 1
        pnames = {i["name"]: i for i in p["items"]}
        intent += p["intent"] == g["intent"]
        offer += p.get("offer") == g.get("offer")
        exact += _key(p) == _key(g)
        both = set(pnames) & set(gnames)
        tp += len(both)
        fp += len(set(pnames) - set(gnames))
        fn += len(set(gnames) - set(pnames))
        for name in both:
            qty_ok += pnames[name]["qty"] == gnames[name]["qty"]
            size_ok += pnames[name].get("size") == gnames[name].get("size")
    denom = 2 * tp + fp + fn
    return {
        "n": n,
        "valid_rate": round(valid / n, 3),
        "intent_acc": round(intent / n, 3),
        "name_f1": round(2 * tp / denom, 3) if denom else 1.0,
        "qty_acc": round(qty_ok / gold_items, 3) if gold_items else 1.0,
        "size_acc": round(size_ok / gold_items, 3) if gold_items else 1.0,
        "offer_acc": round(offer / n, 3),
        "exact_match": round(exact / n, 3),
    }