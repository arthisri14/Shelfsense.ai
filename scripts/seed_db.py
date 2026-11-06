"""Create the database and fill it with synthetic store data.

Run from the project root:  python scripts/seed_db.py --reset
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from shelfsense.db.connection import get_connection, init_db
from shelfsense.simulator.generate import generate
from shelfsense.search.aliases import seed_aliases


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("DB_PATH", "data/shelfsense.db"))
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--end-date", type=date.fromisoformat, default=None, help="YYYY-MM-DD")
    ap.add_argument("--reset", action="store_true", help="delete the existing database first")
    args = ap.parse_args()

    db = Path(args.db)
    if args.reset:
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
    elif db.exists():
        raise SystemExit(f"{db} already exists. Use --reset to rebuild it.")

    conn = get_connection(str(db))
    init_db(conn)
    counts = generate(conn, end_date=args.end_date, days=args.days, seed=args.seed)
    n_alias = seed_aliases(conn)
    print(f"Database ready: {db}")
    for table, n in counts.items():
        print(f"  {table:24} {n:>8,}")
    print(f"  {'product_aliases':24} {n_alias:>8,}")


if __name__ == "__main__":
    main()