"""Compare the rule-based parser, the mini-GPT, and the GPT+fallback hybrid on held-out data."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from shelfsense.nlp import baseline
from shelfsense.nlp.decode import MiniGPTParser
from shelfsense.nlp.metrics import score_split
from shelfsense.nlp.synth import make_splits

SPLITS = ["test_clean", "test_noisy", "test_new_phrasing", "test_unseen_items"]
METRICS = ["valid_rate", "intent_acc", "name_f1", "qty_acc", "size_acc", "offer_acc", "exact_match"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/parser_v0")
    ap.add_argument("--show-errors", type=int, default=4)
    args = ap.parse_args()

    meta = json.loads((Path(args.model) / "meta.json").read_text(encoding="utf-8"))
    parser = MiniGPTParser.load(args.model)
    splits = make_splits(meta["seed"], meta["n_train"], meta["n_val"], meta["n_test"])

    report = {}
    for name in SPLITS:
        exs = splits[name]
        texts = [e["text"] for e in exs]
        golds = [e["obj"] for e in exs]
        t0 = time.time()
        outs = parser.parse_many(texts)
        ms = (time.time() - t0) / len(texts) * 1000
        rules = [baseline.parse(t) for t in texts]
        gpt = [o.obj for o in outs]
        hybrid = [o.obj if o.obj is not None else r for o, r in zip(outs, rules)]
        scores = {"rules": score_split(rules, golds), "gpt": score_split(gpt, golds),
                  "hybrid": score_split(hybrid, golds)}
        report[name] = scores

        print(f"\n=== {name} (n={len(exs)}) | gpt speed {ms:.0f} ms/query | "
              f"gpt fallback rate {1 - scores['gpt']['valid_rate']:.1%} ===")
        print(f"{'metric':<12}{'rules':>8}{'gpt':>8}{'hybrid':>8}")
        for m in METRICS:
            print(f"{m:<12}{scores['rules'][m]:>8}{scores['gpt'][m]:>8}{scores['hybrid'][m]:>8}")

        shown = 0
        for e, o in zip(exs, outs):
            if shown >= args.show_errors:
                break
            if o.obj is None or o.obj != e["obj"]:
                print(f"  GPT miss: {e['text']!r}\n    gold: {e['target']}\n    gpt : {o.raw}")
                shown += 1

    out = Path("data/processed/eval_parser.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport saved to {out}")


if __name__ == "__main__":
    main()