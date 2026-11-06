"""Train the from-scratch BPE tokenizer on the store's own text and show what it learned."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from shelfsense.db.connection import get_connection
from shelfsense.nlp.bpe import BPETokenizer

OUT = "models/tokenizer_v0.json"
SAMPLES = ["anna rice 5kg irukka?", "பால் இருக்கா?", "bhaiya doodh hai kya?", "millet noodles 🙂"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    conn = get_connection(os.getenv("DB_PATH", "data/shelfsense.db"))
    corpus = [r[0] for r in conn.execute("SELECT raw_text FROM demand_signals")]
    corpus += [f"{r['brand']} {r['name']} {r['pack_size']}"
               for r in conn.execute("SELECT brand, name, pack_size FROM products")]
    corpus += [r[0] for r in conn.execute("SELECT alias FROM product_aliases")]

    tok = BPETokenizer().train(corpus, vocab_size=800, min_freq=2)
    tok.save(OUT)

    n_bytes = sum(len(t.encode("utf-8")) for t in corpus)
    n_tokens = sum(len(tok.encode(t)) for t in corpus)
    print(f"corpus lines: {len(corpus)} | vocab: {tok.vocab_size} | merges: {len(tok.merges)}")
    print(f"bytes per token on the corpus: {n_bytes / n_tokens:.2f} (1.00 = no compression)")
    first = [tok.token_str(tok._base + i) for i in range(min(15, len(tok.merges)))]
    print("first merges:", first)
    print("\nsample encodings (round trip must be exact):")
    for s in SAMPLES:
        ids = tok.encode(s)
        print(f"  {s!r} -> {len(ids)} tokens: {[tok.token_str(i) for i in ids]}"
              f" | round trip ok: {tok.decode(ids) == s}")
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()