from datetime import date, timedelta

import pytest

from shelfsense.db.connection import get_connection, init_db
from shelfsense.engine.brief import build_owner_brief, format_brief
from shelfsense.engine.common import ceil_half, data_as_of
from shelfsense.engine.expiry import suggest_markdowns
from shelfsense.engine.forecast import forecast_demand, rolling_backtest
from shelfsense.engine.insights import unmet_demand
from shelfsense.engine.pricing import (
    PriceFloorViolation, PriceRules, check_floor_integrity, enforce_floor,
    get_price_rules, negotiate, negotiate_with_rules,
)
from shelfsense.engine.reorder import compute_reorder, detect_cost_jumps
from shelfsense.engine.stock import (
    InsufficientStock, consume_stock, find_substitutes, get_stock, search_products,
)
from shelfsense.simulator.generate import generate

END = date(2026, 11, 2)


def build(seed: int = 7, days: int = 150):
    c = get_connection(":memory:")
    init_db(c)
    generate(c, end_date=END, days=days, seed=seed)
    return c


@pytest.fixture(scope="module")
def conn():
    c = build()
    yield c
    c.close()


@pytest.fixture()
def fresh():
    c = build()
    yield c
    c.close()


def _multi_batch_product(c, store_id=1):
    row = c.execute("SELECT product_id FROM stock_batches WHERE store_id = ? AND qty > 0 "
                    "GROUP BY product_id HAVING COUNT(*) > 1 ORDER BY product_id LIMIT 1",
                    (store_id,)).fetchone()
    return row[0] if row else None


def _set_single_batch(c, pid, qty, expiry):
    c.execute("DELETE FROM stock_batches WHERE store_id = 1 AND product_id = ?", (pid,))
    c.execute("INSERT INTO stock_batches (store_id, product_id, qty, unit_cost, received_date, "
              "expiry_date) VALUES (1, ?, ?, ?, ?, ?)",
              (pid, qty, 10.0, (END - timedelta(days=5)).isoformat(), expiry.isoformat()))
    c.commit()


# ---------------------------------------------------------------- helpers
def test_ceil_half():
    assert ceil_half(24.4999999999) == 24.5
    assert ceil_half(24.51) == 25.0
    assert ceil_half(24.5) == 24.5


def test_data_as_of(conn):
    assert data_as_of(conn) == END


# ---------------------------------------------------------------- stock
def test_get_stock_is_fefo(conn):
    pid = _multi_batch_product(conn)
    assert pid is not None
    info = get_stock(conn, 1, pid)
    expiries = [b.expiry_date for b in info.batches]
    assert expiries == sorted(expiries)
    assert info.qty_on_hand == sum(b.qty for b in info.batches)


def test_consume_stock_fefo(fresh):
    pid = _multi_batch_product(fresh)
    before = get_stock(fresh, 1, pid)
    first, second = before.batches[0], before.batches[1]
    take = first.qty + 1
    taken = consume_stock(fresh, 1, pid, take)
    assert taken[0] == (first.batch_id, first.qty)
    assert taken[1] == (second.batch_id, 1)
    assert get_stock(fresh, 1, pid).qty_on_hand == before.qty_on_hand - take


def test_consume_stock_all_or_nothing(fresh):
    pid = _multi_batch_product(fresh)
    before = get_stock(fresh, 1, pid).qty_on_hand
    with pytest.raises(InsufficientStock):
        consume_stock(fresh, 1, pid, before + 1)
    assert get_stock(fresh, 1, pid).qty_on_hand == before
    with pytest.raises(ValueError):
        consume_stock(fresh, 1, pid, 0)


def test_search_products(conn):
    res = search_products(conn, 1, "milk")
    assert len(res) >= 2
    assert set(res[0]) == {"product_id", "name", "brand", "pack_size", "selling_price", "in_stock"}
    assert search_products(conn, 1, "%%") == []
    assert search_products(conn, 1, "") == []


def test_substitutes_are_customer_safe(fresh):
    aavin = fresh.execute("SELECT id FROM products WHERE name = 'Milk Packet' "
                          "AND brand = 'Aavin'").fetchone()[0]
    heritage = fresh.execute("SELECT id FROM products WHERE name = 'Milk Packet' "
                             "AND brand = 'Heritage'").fetchone()[0]
    fresh.execute("UPDATE stock_batches SET qty = 0 WHERE store_id = 1 AND product_id = ?", (aavin,))
    _set_single_batch(fresh, heritage, 10, END + timedelta(days=2))
    subs = find_substitutes(fresh, 1, aavin)
    assert [s["product_id"] for s in subs] == [heritage]
    assert subs[0]["match_type"] == "same_group"
    for forbidden in ("floor_price", "unit_cost", "mrp", "qty_on_hand"):
        assert forbidden not in subs[0]


# ---------------------------------------------------------------- forecast
def test_forecast_shape_and_interval(conn):
    fc = forecast_demand(conn, 1, 1, END, horizon_days=7)
    assert len(fc.daily_mean) == 7
    assert fc.total_mean >= 0
    assert fc.low <= fc.total_mean <= fc.high


def test_forecast_with_no_history_is_zero(conn):
    row = conn.execute("SELECT product_id FROM store_products "
                       "WHERE store_id = 2 AND active = 0 LIMIT 1").fetchone()
    if row is None:
        pytest.skip("store 2 sells every product in this seed")
    fc = forecast_demand(conn, 2, row[0], END, horizon_days=5)
    assert fc.total_mean == 0.0 and fc.high == 0.0


@pytest.fixture(scope="module")
def long_conn():
    c = build(days=200)
    yield c
    c.close()


def test_forecast_backtest_is_sane(long_conn):
    """Pooled over several windows, the model must not be meaningfully worse than a plain
    28-day average, and its 80% intervals should cover a reasonable share of outcomes."""
    ev = rolling_backtest(long_conn, 1, END, n_origins=4, step_days=14, holdout_days=14)
    assert ev["wape_model"] < 0.35
    assert ev["wape_model"] <= ev["wape_naive"] * 1.10
    assert ev["interval_coverage"] > 0.5

# ---------------------------------------------------------------- reorder and costs
def test_reorder_covers_stockouts(conn):
    sugg = {s.product_id: s for s in compute_reorder(conn, 1, END)}
    assert all(s.order_qty >= 1 and s.est_cost > 0 for s in sugg.values())
    oos = [r[0] for r in conn.execute(
        "SELECT product_id FROM v_stock WHERE store_id = 1 AND qty_on_hand = 0")]
    for pid in oos:
        if forecast_demand(conn, 1, pid, END, 7).total_mean > 0:
            assert pid in sugg and sugg[pid].urgency == "critical"


def test_cost_jump_detection(fresh):
    sup, cost = fresh.execute("SELECT supplier_id, unit_cost FROM supplier_products "
                              "WHERE product_id = 1").fetchone()
    fresh.execute("INSERT INTO supplier_price_history (supplier_id, product_id, unit_cost, "
                  "effective_date) VALUES (?, 1, ?, ?)",
                  (sup, round(cost * 1.2, 2), (END - timedelta(days=1)).isoformat()))
    fresh.commit()
    jumps = detect_cost_jumps(fresh, 1, END)
    assert any(j.product_id == 1 and j.change_pct >= 15 for j in jumps)


# ---------------------------------------------------------------- pricing
def test_negotiation_never_below_floor(conn):
    pids = [r[0] for r in conn.execute(
        "SELECT product_id FROM store_products WHERE store_id = 1 AND active = 1")]
    for pid in pids:
        rules = get_price_rules(conn, 1, pid)
        offers = [0.01, rules.floor_price * 0.5, rules.floor_price * 0.99, rules.floor_price,
                  (rules.floor_price + rules.selling_price) / 2, rules.selling_price,
                  rules.selling_price * 1.5]
        for offer in offers:
            for rnd in range(1, 6):
                res = negotiate_with_rules(rules, offer, rnd)
                assert rules.floor_price <= res.price <= rules.selling_price


def test_negotiation_concedes_gradually_and_ends(conn):
    rules = get_price_rules(conn, 1, 1)   # product 1 has real discount room
    prices = [negotiate_with_rules(rules, 0.01, r).price for r in (1, 2, 3)]
    assert prices[0] >= prices[1] >= prices[2] >= rules.floor_price
    last = negotiate_with_rules(rules, 0.01, 3)
    assert last.decision == "final" and last.price == rules.floor_price


def test_negotiation_list_price_and_invalid_offers(conn):
    rules = get_price_rules(conn, 1, 1)
    res = negotiate_with_rules(rules, rules.selling_price * 2, 1)
    assert res.price == rules.selling_price and res.code == "ACCEPT_LIST_PRICE"
    for bad in (float("nan"), float("inf"), -5, 0):
        with pytest.raises(ValueError):
            negotiate_with_rules(rules, bad, 1)
    assert negotiate(conn, 1, 1, 0.01, 1).price >= rules.floor_price


def test_enforce_floor():
    rules = PriceRules(1, 1, 100.0, 90.0, 80.0, 5.0)
    with pytest.raises(PriceFloorViolation):
        enforce_floor(rules, 79.99)
    assert enforce_floor(rules, 80.0) == 80.0


def test_floor_integrity_clean(conn):
    assert check_floor_integrity(conn, 1) == []


# ---------------------------------------------------------------- expiry
def test_markdown_for_overstocked_expiring_batch(fresh):
    _set_single_batch(fresh, 1, 500, END + timedelta(days=1))
    sugg = [s for s in suggest_markdowns(fresh, 1, END) if s.product_id == 1]
    assert len(sugg) == 1
    s = sugg[0]
    rules = get_price_rules(fresh, 1, 1)
    assert s.at_risk_qty > 400
    assert rules.floor_price <= s.suggested_price <= rules.selling_price
    assert s.action in ("markdown", "consider_alternatives")


def test_expired_stock_flagged_and_far_expiry_ignored(fresh):
    _set_single_batch(fresh, 2, 5, END - timedelta(days=1))
    _set_single_batch(fresh, 3, 5, END + timedelta(days=60))
    sugg = suggest_markdowns(fresh, 1, END)
    expired = [s for s in sugg if s.product_id == 2]
    assert len(expired) == 1
    assert expired[0].action == "remove_from_shelf" and expired[0].at_risk_qty == 5
    assert not [s for s in sugg if s.product_id == 3]


def test_all_markdowns_respect_floor(conn):
    for s in suggest_markdowns(conn, 1, END):
        rules = get_price_rules(conn, 1, s.product_id)
        assert rules.floor_price <= s.suggested_price <= rules.selling_price


# ---------------------------------------------------------------- insights and brief
def test_unmet_demand_sorted(conn):
    res = unmet_demand(conn, 1, END, days=30, top_n=5)
    assert res
    asks = [r["asks"] for r in res]
    assert asks == sorted(asks, reverse=True)


def test_owner_brief_builds(conn):
    b = build_owner_brief(conn, 1, END)
    for key in ("reorders", "expiring", "value_at_risk", "cost_jumps",
                "unmet_demand", "floor_problems"):
        assert key in b
    assert "ShelfSense daily brief" in format_brief(b)