"""Show the engine working on the Phase 1 database. Read-only: changes nothing."""
from __future__ import annotations

import os

from dotenv import load_dotenv

from shelfsense.db.connection import get_connection
from shelfsense.engine.brief import build_owner_brief, format_brief
from shelfsense.engine.common import data_as_of
from shelfsense.engine.forecast import evaluate_forecasts, forecast_demand
from shelfsense.engine.pricing import get_price_rules, negotiate
from shelfsense.engine.stock import find_substitutes, search_products


def main() -> None:
    load_dotenv()
    conn = get_connection(os.getenv("DB_PATH", "data/shelfsense.db"))
    store = 1
    as_of = data_as_of(conn)

    print(format_brief(build_owner_brief(conn, store, as_of)))

    print("\n=== Forecast: Sona Masoori Rice (next 7 days) ===")
    hit = search_products(conn, store, "sona masoori")[0]
    fc = forecast_demand(conn, store, hit["product_id"], as_of, horizon_days=7)
    print(f"daily: {fc.daily_mean}")
    print(f"total ~{fc.total_mean} units (80% range {fc.low} to {fc.high})")

    print("\n=== Backtest: last 14 days, store 1 ===")
    print(evaluate_forecasts(conn, store, as_of))

    print(f"\n=== Bargaining demo: {hit['name']} {hit['pack_size']} ===")
    rules = get_price_rules(conn, store, hit["product_id"])
    print(f"list price: {rules.selling_price}")   # owner view; floor is never shown to customers
    for rnd, frac in enumerate([0.80, 0.85, 0.90], start=1):
        offer = round(rules.selling_price * frac, 1)
        res = negotiate(conn, store, hit["product_id"], offer, rnd)
        print(f"  round {rnd}: customer offers {offer} -> {res.decision} at {res.price} ({res.code})")
    offer = round(rules.selling_price * 0.985, 1)
    res = negotiate(conn, store, hit["product_id"], offer, 2)
    print(f"  round 2 with offer {offer} -> {res.decision} at {res.price} ({res.code})")

    print("\n=== Substitutes for the first out-of-stock product ===")
    oos = conn.execute("SELECT v.product_id, p.name, p.brand FROM v_stock v "
                       "JOIN products p ON p.id = v.product_id "
                       "WHERE v.store_id = ? AND v.qty_on_hand = 0 ORDER BY v.product_id",
                       (store,)).fetchall()
    if not oos:
        print("Nothing is out of stock right now.")
    else:
        o = oos[0]
        print(f"Out of stock: {o['name']} ({o['brand']})")
        subs = find_substitutes(conn, store, o["product_id"])
        print(subs if subs else "No same-group substitute is in stock.")


if __name__ == "__main__":
    main()