"""Stock lookups, FEFO deduction, customer-safe search and substitutes."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class InsufficientStock(Exception):
    pass


@dataclass(frozen=True)
class BatchInfo:
    batch_id: int
    qty: int
    expiry_date: str | None


@dataclass(frozen=True)
class StockInfo:
    store_id: int
    product_id: int
    qty_on_hand: int
    batches: tuple[BatchInfo, ...]   # FEFO order: earliest expiry first


_FEFO = "ORDER BY (expiry_date IS NULL), expiry_date, id"


def get_stock(conn: sqlite3.Connection, store_id: int, product_id: int) -> StockInfo:
    """Staff-level view of stock. Deliberately excludes cost."""
    rows = conn.execute(
        f"SELECT id, qty, expiry_date FROM stock_batches "
        f"WHERE store_id = ? AND product_id = ? AND qty > 0 {_FEFO}",
        (store_id, product_id)).fetchall()
    batches = tuple(BatchInfo(r["id"], r["qty"], r["expiry_date"]) for r in rows)
    return StockInfo(store_id, product_id, sum(b.qty for b in batches), batches)


def consume_stock(conn: sqlite3.Connection, store_id: int, product_id: int,
                  qty: int) -> list[tuple[int, int]]:
    """Remove qty units FEFO. Returns [(batch_id, units_taken), ...].

    All-or-nothing: raises InsufficientStock without changing anything if stock is short.
    (Single-writer SQLite is fine for now; we harden concurrency in a later phase.)
    """
    qty = int(qty)
    if qty <= 0:
        raise ValueError("qty must be positive")
    rows = conn.execute(
        f"SELECT id, qty FROM stock_batches "
        f"WHERE store_id = ? AND product_id = ? AND qty > 0 {_FEFO}",
        (store_id, product_id)).fetchall()
    available = sum(r["qty"] for r in rows)
    if available < qty:
        raise InsufficientStock(f"requested {qty}, only {available} in stock")
    taken: list[tuple[int, int]] = []
    remaining = qty
    try:
        for r in rows:
            take = min(r["qty"], remaining)
            conn.execute("UPDATE stock_batches SET qty = qty - ? WHERE id = ?", (take, r["id"]))
            taken.append((r["id"], take))
            remaining -= take
            if remaining == 0:
                break
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return taken


def search_products(conn: sqlite3.Connection, store_id: int, text: str,
                    limit: int = 10) -> list[dict]:
    """Simple keyword search over the CUSTOMER view (no cost/floor columns exist there).

    Placeholder: Phase 3 replaces this with our own BM25 index.
    """
    tokens = []
    for t in text.lower().split():
        t = t.replace("%", "").replace("_", "")
        if t:
            tokens.append(t)
    if not tokens:
        return []
    clauses, params = [], [store_id]
    for t in tokens:
        clauses.append("(lower(name) LIKE ? OR lower(brand) LIKE ? "
                       "OR lower(category) LIKE ? OR lower(pack_size) LIKE ?)")
        params += [f"%{t}%"] * 4
    sql = ("SELECT product_id, name, brand, pack_size, selling_price, in_stock "
           "FROM v_customer_catalog WHERE store_id = ? AND " + " AND ".join(clauses) +
           " ORDER BY in_stock DESC, name LIMIT ?")
    params.append(limit)
    return [dict(r) for r in conn.execute(sql, params)]


def find_substitutes(conn: sqlite3.Connection, store_id: int, product_id: int,
                     limit: int = 3) -> list[dict]:
    """In-stock products from the same substitute group, closest price first."""
    target = conn.execute(
        "SELECT p.substitute_group, sp.selling_price FROM products p "
        "JOIN store_products sp ON sp.product_id = p.id AND sp.store_id = ? WHERE p.id = ?",
        (store_id, product_id)).fetchone()
    if target is None:
        return []
    rows = conn.execute(
        "SELECT product_id, name, brand, pack_size, selling_price, 'same_group' AS match_type "
        "FROM v_customer_catalog "
        "WHERE store_id = ? AND in_stock = 1 AND substitute_group = ? AND product_id != ? "
        "ORDER BY ABS(selling_price - ?) LIMIT ?",
        (store_id, target["substitute_group"], product_id, target["selling_price"], limit))
    return [dict(r) for r in rows]