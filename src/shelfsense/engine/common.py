"""Small helpers shared by the engine modules."""
from __future__ import annotations

import math
import sqlite3
from datetime import date


def ceil_half(x: float) -> float:
    """Round UP to the nearest 0.5 rupee. Rounds first to kill float noise (24.500000001)."""
    return math.ceil(round(x * 2, 6)) / 2


def data_as_of(conn: sqlite3.Connection) -> date:
    """The latest day that has sales data. Used as 'today' for the simulated store."""
    row = conn.execute("SELECT MAX(date(sold_at)) FROM sales").fetchone()
    if row[0] is None:
        raise LookupError("No sales data found. Run scripts/seed_db.py first.")
    return date.fromisoformat(row[0])