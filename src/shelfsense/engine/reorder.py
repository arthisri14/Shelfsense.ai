"""Reorder suggestions (s,S policy with safety stock) and supplier price-jump alerts."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

from shelfsense.engine.forecast import forecast_demand

_URGENCY_RANK = {"critical": 0, "soon": 1, "routine": 2}


@dataclass(frozen=True)
class ReorderSuggestion:
    product_id: int
    name: str
    brand: str
    pack_size: str
    category: str
    supplier_id: int
    supplier_name: str
    lead_time_days: int
    on_hand: int
    reorder_level: int
    expected_demand: float
    safety_stock: float
    reorder_point: float
    target_stock: float
    order_qty: int
    unit_cost: float
    est_cost: float
    days_of_cover: float | None
    urgency: str
    reason: str


@dataclass(frozen=True)
class CostJump:
    supplier_id: int
    supplier_name: str
    product_id: int
    name: str
    brand: str
    pack_size: str
    old_cost: float
    new_cost: float
    change_pct: float
    effective_date: str
    selling_price: float
    margin_after_pct: float
    floor_price: float


def compute_reorder(conn: sqlite3.Connection, store_id: int, as_of: date,
                    review_days: int = 2, service_z: float = 1.28) -> list[ReorderSuggestion]:
    """Order when on_hand <= reorder point; order up to target = demand over (lead + review) + safety."""
    rows = conn.execute(
        "SELECT p.id AS product_id, p.name, p.brand, p.pack_size, p.category, "
        "       sp.reorder_level, sup.id AS supplier_id, sup.name AS supplier_name, "
        "       sup.lead_time_days, spp.unit_cost, spp.min_order_qty, "
        "       COALESCE(v.qty_on_hand, 0) AS on_hand "
        "FROM store_products sp "
        "JOIN products p ON p.id = sp.product_id "
        "JOIN supplier_products spp ON spp.product_id = p.id "
        "JOIN suppliers sup ON sup.id = spp.supplier_id "
        "LEFT JOIN v_stock v ON v.store_id = sp.store_id AND v.product_id = sp.product_id "
        "WHERE sp.store_id = ? AND sp.active = 1 ORDER BY p.id", (store_id,)).fetchall()

    out: list[ReorderSuggestion] = []
    for r in rows:
        lead = r["lead_time_days"]
        horizon = lead + review_days
        fc = forecast_demand(conn, store_id, r["product_id"], as_of, horizon_days=horizon)
        safety = service_z * fc.total_sd
        reorder_point = fc.total_mean * lead / horizon + safety
        target = fc.total_mean + safety
        on_hand = r["on_hand"]
        if on_hand > reorder_point:
            continue
        need = target - on_hand
        if need <= 1e-9:
            continue
        qty = math.ceil(need)
        moq = r["min_order_qty"]
        if moq > 1:
            qty = math.ceil(qty / moq) * moq

        avg_daily = fc.total_mean / horizon
        cover = (on_hand / avg_daily) if avg_daily > 0 else None
        if on_hand == 0 or (cover is not None and cover < lead):
            urgency = "critical"
        elif cover is not None and cover < horizon:
            urgency = "soon"
        else:
            urgency = "routine"
        out.append(ReorderSuggestion(
            product_id=r["product_id"], name=r["name"], brand=r["brand"],
            pack_size=r["pack_size"], category=r["category"],
            supplier_id=r["supplier_id"], supplier_name=r["supplier_name"],
            lead_time_days=lead, on_hand=on_hand, reorder_level=r["reorder_level"],
            expected_demand=fc.total_mean, safety_stock=round(safety, 2),
            reorder_point=round(reorder_point, 2), target_stock=round(target, 2),
            order_qty=qty, unit_cost=r["unit_cost"],
            est_cost=round(qty * r["unit_cost"], 2),
            days_of_cover=None if cover is None else round(cover, 1),
            urgency=urgency,
            reason=(f"On hand {on_hand} <= reorder point {reorder_point:.1f}; expected demand "
                    f"{fc.total_mean:.1f} over {horizon} days (lead {lead} + review {review_days})")))
    out.sort(key=lambda s: (_URGENCY_RANK[s.urgency],
                            s.days_of_cover if s.days_of_cover is not None else 1e9))
    return out


def detect_cost_jumps(conn: sqlite3.Connection, store_id: int, as_of: date,
                      threshold_pct: float = 7.0, within_days: int = 14) -> list[CostJump]:
    """Recent supplier price rises versus the previous recorded price."""
    rows = conn.execute(
        "WITH ranked AS ("
        "  SELECT supplier_id, product_id, unit_cost, effective_date, "
        "         ROW_NUMBER() OVER (PARTITION BY supplier_id, product_id "
        "                            ORDER BY effective_date DESC) AS rn "
        "  FROM supplier_price_history) "
        "SELECT a.supplier_id, sup.name AS supplier_name, a.product_id, p.name, p.brand, "
        "       p.pack_size, b.unit_cost AS old_cost, a.unit_cost AS new_cost, "
        "       a.effective_date, sp.selling_price, sp.floor_price "
        "FROM ranked a "
        "JOIN ranked b ON b.supplier_id = a.supplier_id AND b.product_id = a.product_id "
        "              AND b.rn = 2 "
        "JOIN suppliers sup ON sup.id = a.supplier_id "
        "JOIN products p ON p.id = a.product_id "
        "JOIN store_products sp ON sp.product_id = a.product_id AND sp.store_id = ? "
        "                       AND sp.active = 1 "
        "WHERE a.rn = 1", (store_id,)).fetchall()

    cutoff = as_of - timedelta(days=within_days)
    jumps = []
    for r in rows:
        if date.fromisoformat(r["effective_date"]) < cutoff:
            continue
        pct = (r["new_cost"] - r["old_cost"]) / r["old_cost"] * 100
        if pct < threshold_pct:
            continue
        margin_after = (r["selling_price"] - r["new_cost"]) / r["selling_price"] * 100
        jumps.append(CostJump(
            supplier_id=r["supplier_id"], supplier_name=r["supplier_name"],
            product_id=r["product_id"], name=r["name"], brand=r["brand"],
            pack_size=r["pack_size"], old_cost=r["old_cost"], new_cost=r["new_cost"],
            change_pct=round(pct, 1), effective_date=r["effective_date"],
            selling_price=r["selling_price"], margin_after_pct=round(margin_after, 1),
            floor_price=r["floor_price"]))
    jumps.sort(key=lambda j: -j.change_pct)
    return jumps