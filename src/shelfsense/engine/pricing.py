"""Price rules, the hard price-floor guard, and the bounded bargaining policy.

The floor is enforced HERE, in code. An LLM can phrase offers, but it can never produce a
price below the floor, because every price passes through enforce_floor().
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

from shelfsense.engine.common import ceil_half

MAX_ROUNDS = 3
CONCESSION_PER_ROUND = 0.4   # share of the discount room unlocked per round


class PriceFloorViolation(Exception):
    pass


@dataclass(frozen=True)
class PriceRules:
    store_id: int
    product_id: int
    mrp: float
    selling_price: float
    floor_price: float
    max_discount_pct: float


@dataclass(frozen=True)
class Negotiation:
    decision: str    # 'accept' | 'counter' | 'final'
    price: float
    round_no: int
    code: str        # ACCEPT_LIST_PRICE | ACCEPT_OFFER | COUNTER | FINAL_OFFER


def get_price_rules(conn: sqlite3.Connection, store_id: int, product_id: int) -> PriceRules:
    r = conn.execute(
        "SELECT sp.mrp, sp.selling_price, sp.floor_price, "
        "       COALESCE(dp.max_discount_pct, 0) AS max_disc "
        "FROM store_products sp JOIN products p ON p.id = sp.product_id "
        "LEFT JOIN discount_policy dp ON dp.store_id = sp.store_id AND dp.category = p.category "
        "WHERE sp.store_id = ? AND sp.product_id = ? AND sp.active = 1",
        (store_id, product_id)).fetchone()
    if r is None:
        raise LookupError(f"product {product_id} is not sold by store {store_id}")
    return PriceRules(store_id, product_id, r["mrp"], r["selling_price"],
                      r["floor_price"], r["max_disc"])


def enforce_floor(rules: PriceRules, price: float) -> float:
    """The one gate every outgoing price must pass."""
    if not math.isfinite(price) or price < rules.floor_price - 1e-9:
        raise PriceFloorViolation(
            f"price {price} is below the floor for product {rules.product_id}")
    return round(price, 2)


def negotiate_with_rules(rules: PriceRules, customer_offer: float,
                         round_no: int = 1) -> Negotiation:
    """Bounded bargaining: unlocks 40% more of the discount room each round, never below floor."""
    if not isinstance(round_no, int) or round_no < 1:
        raise ValueError("round_no must be an integer >= 1")
    offer = float(customer_offer)
    if not math.isfinite(offer) or offer <= 0:
        raise ValueError("offer must be a positive number")

    sell, floor = rules.selling_price, rules.floor_price
    if offer >= sell:   # never charge more than the list price
        return Negotiation("accept", enforce_floor(rules, sell), round_no, "ACCEPT_LIST_PRICE")

    room = sell - floor
    share = min(1.0, CONCESSION_PER_ROUND * round_no)
    threshold = min(sell, max(floor, ceil_half(sell - room * share)))
    if offer >= threshold:
        return Negotiation("accept", enforce_floor(rules, offer), round_no, "ACCEPT_OFFER")
    final = round_no >= MAX_ROUNDS
    return Negotiation("final" if final else "counter", enforce_floor(rules, threshold),
                       round_no, "FINAL_OFFER" if final else "COUNTER")


def negotiate(conn: sqlite3.Connection, store_id: int, product_id: int,
              customer_offer: float, round_no: int = 1) -> Negotiation:
    return negotiate_with_rules(get_price_rules(conn, store_id, product_id),
                                customer_offer, round_no)


def check_floor_integrity(conn: sqlite3.Connection, store_id: int) -> list[dict]:
    """Products whose floor price is now below the current supplier cost (selling there loses money)."""
    rows = conn.execute(
        "SELECT p.id AS product_id, p.name, p.brand, sp.floor_price, spp.unit_cost "
        "FROM store_products sp JOIN products p ON p.id = sp.product_id "
        "JOIN supplier_products spp ON spp.product_id = sp.product_id "
        "WHERE sp.store_id = ? AND sp.active = 1 AND sp.floor_price < spp.unit_cost",
        (store_id,))
    return [dict(r) for r in rows]