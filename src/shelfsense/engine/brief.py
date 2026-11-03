"""The owner's daily brief: everything the engine knows, in one place."""
from __future__ import annotations

import sqlite3
from datetime import date

from shelfsense.engine.expiry import suggest_markdowns
from shelfsense.engine.insights import unmet_demand
from shelfsense.engine.pricing import check_floor_integrity
from shelfsense.engine.reorder import compute_reorder, detect_cost_jumps


def build_owner_brief(conn: sqlite3.Connection, store_id: int, as_of: date) -> dict:
    reorders = compute_reorder(conn, store_id, as_of)
    expiring = [m for m in suggest_markdowns(conn, store_id, as_of) if m.action != "monitor"]
    return {
        "store_id": store_id,
        "as_of": as_of.isoformat(),
        "reorders": reorders,
        "expiring": expiring,
        "value_at_risk": round(sum(m.value_at_risk for m in expiring), 2),
        "cost_jumps": detect_cost_jumps(conn, store_id, as_of),
        "unmet_demand": unmet_demand(conn, store_id, as_of),
        "floor_problems": check_floor_integrity(conn, store_id),
    }


def format_brief(b: dict) -> str:
    lines = [f"ShelfSense daily brief | store {b['store_id']} | {b['as_of']}", ""]

    crit = [r for r in b["reorders"] if r.urgency == "critical"]
    lines.append(f"STOCK: {len(crit)} critical, {len(b['reorders'])} item(s) to reorder")
    for r in b["reorders"][:6]:
        lines.append(f"  - [{r.urgency}] {r.name} ({r.brand}, {r.pack_size}): "
                     f"{r.on_hand} on hand, order {r.order_qty} from {r.supplier_name} "
                     f"(~Rs {r.est_cost:,.0f})")

    lines += ["", f"EXPIRY: Rs {b['value_at_risk']:,.0f} of stock at risk (at cost)"]
    for m in b["expiring"][:6]:
        extra = (f"-> Rs {m.suggested_price:g} ({m.discount_pct}% off)"
                 if m.action == "markdown" else "")
        lines.append(f"  - {m.name} ({m.brand}): {m.at_risk_qty}/{m.qty} at risk, "
                     f"{m.days_left}d left, {m.action} {extra}".rstrip())

    lines += ["", f"SUPPLIER PRICES: {len(b['cost_jumps'])} recent jump(s)"]
    for j in b["cost_jumps"][:5]:
        lines.append(f"  - {j.name} ({j.brand}): cost {j.old_cost:g} -> {j.new_cost:g} "
                     f"(+{j.change_pct}%), margin now {j.margin_after_pct}%")

    lines += ["", "UNMET DEMAND (last 30 days)"]
    for u in b["unmet_demand"]:
        lines.append(f"  - {u['extracted_item']}: asked {u['asks']}x ({u['signal_type']})")

    if b["floor_problems"]:
        lines += ["", f"WARNING: {len(b['floor_problems'])} product(s) have a floor price "
                      f"below supplier cost"]
    return "\n".join(lines)