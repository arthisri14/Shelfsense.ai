"""Run sanity queries against the database and print the results."""
from __future__ import annotations

import os
from datetime import date, timedelta

from dotenv import load_dotenv

from shelfsense.db.connection import get_connection


def show(conn, title, sql, params=()):
    print(f"\n=== {title} ===")
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        print("(no rows)")
        return
    print(" | ".join(rows[0].keys()))
    for r in rows:
        print(" | ".join(str(r[k]) for k in r.keys()))


def main() -> None:
    load_dotenv()
    conn = get_connection(os.getenv("DB_PATH", "data/shelfsense.db"))
    store = 1
    last = conn.execute("SELECT MAX(date(sold_at)) FROM sales").fetchone()[0]
    today = date.fromisoformat(last)
    d30 = (today - timedelta(days=29)).isoformat()

    show(conn, "Top 8 sellers, last 30 days (store 1)", """
        SELECT p.name, p.brand, p.pack_size, SUM(i.qty) AS units, ROUND(SUM(i.qty*i.unit_price)) AS revenue
        FROM sale_items i JOIN sales s ON s.id=i.sale_id JOIN products p ON p.id=i.product_id
        WHERE s.store_id=? AND date(s.sold_at)>=? GROUP BY p.id ORDER BY units DESC LIMIT 8""",
        (store, d30))

    show(conn, "Average baskets by weekday (store 1)", """
        SELECT CASE strftime('%w', sold_at) WHEN '0' THEN 'Sun' WHEN '1' THEN 'Mon' WHEN '2' THEN 'Tue'
               WHEN '3' THEN 'Wed' WHEN '4' THEN 'Thu' WHEN '5' THEN 'Fri' ELSE 'Sat' END AS weekday,
               ROUND(COUNT(*) * 1.0 / COUNT(DISTINCT date(sold_at)), 1) AS avg_baskets
        FROM sales WHERE store_id=? GROUP BY strftime('%w', sold_at)
        ORDER BY strftime('%w', sold_at)""", (store,))

    show(conn, "Out of stock now (store 1)", """
        SELECT p.name, p.brand, p.pack_size FROM v_stock v JOIN products p ON p.id=v.product_id
        WHERE v.store_id=? AND v.qty_on_hand=0""", (store,))

    show(conn, "Stock expiring within 3 days (store 1)", """
        SELECT p.name, p.brand, b.qty, b.expiry_date FROM stock_batches b
        JOIN products p ON p.id=b.product_id
        WHERE b.store_id=? AND b.qty>0 AND b.expiry_date<=date(?, '+3 day')
        ORDER BY b.expiry_date""", (store, today.isoformat()))

    show(conn, "Top unmet demand, last 90 days (store 1)", """
        SELECT extracted_item, signal_type, COUNT(*) AS asks FROM demand_signals
        WHERE store_id=? GROUP BY extracted_item, signal_type ORDER BY asks DESC LIMIT 8""", (store,))

    show(conn, "Price floor check: lowest allowed price vs cost (5 thinnest margins)", """
        SELECT p.name, sp.selling_price, sp.floor_price, sc.unit_cost,
               ROUND(sp.floor_price - sc.unit_cost, 2) AS floor_minus_cost
        FROM store_products sp JOIN products p ON p.id=sp.product_id
        JOIN supplier_products sc ON sc.product_id=sp.product_id
        WHERE sp.store_id=? ORDER BY floor_minus_cost ASC LIMIT 5""", (store,))

    show(conn, "What the CUSTOMER agent can see (v_customer_catalog sample)", """
        SELECT name, brand, pack_size, selling_price, in_stock FROM v_customer_catalog
        WHERE store_id=? AND name LIKE 'Milk%'""", (store,))


if __name__ == "__main__":
    main()