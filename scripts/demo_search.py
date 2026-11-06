"""Try the product search with English, typos, code-mixed text, Tamil and Hindi. Read-only."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from shelfsense.db.connection import get_connection
from shelfsense.search.product_search import ProductSearch

QUERIES = [
    "aavin milk 500ml", "buter", "anna rice 5kg irukka?", "paal venum", "பால் இருக்கா?",
    "bhaiya doodh hai kya?", "thakkali", "millet noodles", "peanut butter",
    "do you have quinoa", "do you have?",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Tamil/Hindi output on Windows
    load_dotenv()
    conn = get_connection(os.getenv("DB_PATH", "data/shelfsense.db"))
    ps = ProductSearch(conn, store_id=1)
    for q in QUERIES:
        r = ps.search(q, k=3)
        print(f"\n> {q}\n  status={r.status} coverage={r.coverage} unmatched={list(r.unmatched)}")
        for h in r.hits:
            stock = "in stock" if h.in_stock else "OUT"
            print(f"    {h.name} ({h.brand}, {h.pack_size}) Rs {h.selling_price:g} [{stock}]"
                  f"  bm25={h.bm25} cos={h.cosine}")


if __name__ == "__main__":
    main()