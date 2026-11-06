"""Look at the synthetic data before training on it."""
from __future__ import annotations

import sys
from collections import Counter

from shelfsense.nlp.synth import make_splits


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    splits = make_splits(n_train=3000, n_val=100, n_test=100)
    for name, exs in splits.items():
        print(f"\n=== {name} ({len(exs)} examples) ===")
        print("intents:", dict(Counter(e["intent"] for e in exs)))
        print("avg chars:", round(sum(len(e["text"]) for e in exs) / len(exs), 1))
        for e in exs[:4]:
            print(f"  {e['text']!r}\n    -> {e['target']}")


if __name__ == "__main__":
    main()