"""Expiry-risk analysis and markdown suggestions.

Items are assumed sellable through their expiry date; stock sells FEFO. For each batch we
estimate how many units will sell before it expires and flag the rest as 'at risk'.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import groupby

from shelfsense.engine.common import ceil_half
from shelfsense.engine.forecast import forecast_demand


@dataclass(frozen=True)
class MarkdownSuggestion:
    batch_id: int
    product_id: int
    name: str
    brand: str
    pack_size: str
    category: str
    qty: int
    expiry_date: str
    days_left: int
    expected_sold: float
    at_risk_qty: int
    selling_price: float
    suggested_price: float
    discount_pct: float
    value_at_risk: float      # at_risk_qty * batch cost
    action: str               # remove_from_shelf | markdown | consider_alternatives | monitor
    note: str


def suggest_markdowns(conn: sqlite3.Connection, store_id: int, as_of: date,
                      within_days: int = 3) -> list[MarkdownSuggestion]:
    within_days = max(1, within_days)
    limit_date = (as_of + timedelta(days=within_days)).isoformat()
    rows = conn.execute(
        "SELECT b.id AS batch_id, b.product_id, b.qty, b.unit_cost, b.expiry_date, "
        "       p.name, p.brand, p.pack_size, p.category, sp.selling_price, sp.floor_price, "
        "       COALESCE(dp.max_discount_pct, 0) AS max_disc "
        "FROM stock_batches b "
        "JOIN products p ON p.id = b.product_id "
        "JOIN store_products sp ON sp.store_id = b.store_id AND sp.product_id = b.product_id "
        "LEFT JOIN discount_policy dp ON dp.store_id = b.store_id AND dp.category = p.category "
        "WHERE b.store_id = ? AND b.qty > 0 AND b.expiry_date IS NOT NULL "
        "  AND b.expiry_date <= ? "
        "ORDER BY b.product_id, b.expiry_date, b.id", (store_id, limit_date)).fetchall()

    out: list[MarkdownSuggestion] = []
    for pid, group in groupby(rows, key=lambda r: r["product_id"]):
        fc = forecast_demand(conn, store_id, pid, as_of, horizon_days=within_days)
        cum = [0.0]
        for v in fc.daily_mean:
            cum.append(cum[-1] + v)
        sold_by_earlier = 0.0   # units earlier-expiring batches are expected to sell first

        for r in group:
            qty = r["qty"]
            sell, floor = r["selling_price"], r["floor_price"]
            days_left = (date.fromisoformat(r["expiry_date"]) - as_of).days
            base = dict(batch_id=r["batch_id"], product_id=pid, name=r["name"],
                        brand=r["brand"], pack_size=r["pack_size"], category=r["category"],
                        qty=qty, expiry_date=r["expiry_date"], days_left=days_left,
                        selling_price=sell)

            if days_left <= 0:
                out.append(MarkdownSuggestion(
                    **base, expected_sold=0.0, at_risk_qty=qty, suggested_price=sell,
                    discount_pct=0.0, value_at_risk=round(qty * r["unit_cost"], 2),
                    action="remove_from_shelf",
                    note="Expired: take off the shelf, do not sell."))
                continue

            d = min(days_left, within_days)
            expected_sold = min(float(qty), max(0.0, cum[d] - sold_by_earlier))
            sold_by_earlier += expected_sold
            at_risk = max(0, int(round(qty - expected_sold)))

            if at_risk == 0:
                out.append(MarkdownSuggestion(
                    **base, expected_sold=round(expected_sold, 1), at_risk_qty=0,
                    suggested_price=sell, discount_pct=0.0, value_at_risk=0.0,
                    action="monitor", note="Expected to sell through before expiry."))
                continue

            risk_ratio = at_risk / qty
            urgency = 1.0 if days_left <= 1 else (0.7 if days_left == 2 else 0.5)
            wanted_pct = 40.0 * risk_ratio * urgency
            pct = min(r["max_disc"], wanted_pct)
            price = max(floor, ceil_half(sell * (1 - pct / 100)))
            price = min(price, sell)
            eff_pct = round((sell - price) / sell * 100, 1)
            capped = wanted_pct > r["max_disc"] + 1e-9

            if eff_pct <= 0:
                action = "consider_alternatives"
                note = ("Discount room is zero for this category/floor. Consider a bundle, "
                        "a supplier return, or donating the stock.")
            else:
                action = "markdown"
                note = ("Discount limited by the category cap / price floor."
                        if capped else "Markdown sized to expected unsold units.")
            out.append(MarkdownSuggestion(
                **base, expected_sold=round(expected_sold, 1), at_risk_qty=at_risk,
                suggested_price=price, discount_pct=eff_pct,
                value_at_risk=round(at_risk * r["unit_cost"], 2), action=action, note=note))

    out.sort(key=lambda s: -s.value_at_risk)
    return out