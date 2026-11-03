"""Owner insights built from customer demand signals."""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta


def unmet_demand(conn: sqlite3.Connection, store_id: int, as_of: date,
                 days: int = 30, top_n: int = 5) -> list[dict]:
    """Most-requested items the store could not supply (not stocked, or out of stock)."""
    since = (as_of - timedelta(days=days - 1)).isoformat()
    rows = conn.execute(
        "SELECT extracted_item, signal_type, matched_product_id, COUNT(*) AS asks, "
        "       COUNT(DISTINCT customer_id) AS known_customers "
        "FROM demand_signals "
        "WHERE store_id = ? AND date(created_at) >= ? AND date(created_at) <= ? "
        "GROUP BY extracted_item, signal_type, matched_product_id "
        "ORDER BY asks DESC, extracted_item LIMIT ?",
        (store_id, since, as_of.isoformat(), top_n))
    return [dict(r) for r in rows]