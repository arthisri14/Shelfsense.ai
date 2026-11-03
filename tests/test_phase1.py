from datetime import date, timedelta

import pytest

from shelfsense.db.connection import get_connection, init_db
from shelfsense.simulator.generate import generate

END = date(2026, 11, 2)
DAYS = 60


def build(seed: int):
    conn = get_connection(":memory:")
    init_db(conn)
    counts = generate(conn, end_date=END, days=DAYS, seed=seed)
    return conn, counts


@pytest.fixture(scope="module")
def conn():
    c, _ = build(7)
    yield c
    c.close()


def test_foreign_keys_are_consistent(conn):
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_floor_price_never_below_cost(conn):
    bad = conn.execute("""
        SELECT COUNT(*) FROM store_products sp
        JOIN supplier_products sc ON sc.product_id = sp.product_id
        WHERE sp.floor_price < sc.unit_cost""").fetchone()[0]
    assert bad == 0


def test_price_ordering(conn):
    bad = conn.execute("""
        SELECT COUNT(*) FROM store_products
        WHERE NOT (floor_price <= selling_price AND selling_price <= mrp)""").fetchone()[0]
    assert bad == 0


def test_no_negative_stock(conn):
    assert conn.execute("SELECT COUNT(*) FROM stock_batches WHERE qty < 0").fetchone()[0] == 0


def test_customer_view_hides_sensitive_columns(conn):
    cols = {r[1] for r in conn.execute("PRAGMA table_info(v_customer_catalog)")}
    for forbidden in ("floor_price", "unit_cost", "reorder_level", "qty_on_hand", "mrp"):
        assert forbidden not in cols
    assert {"selling_price", "in_stock"} <= cols


def test_sales_fall_inside_window(conn):
    lo, hi = conn.execute("SELECT MIN(date(sold_at)), MAX(date(sold_at)) FROM sales").fetchone()
    assert lo >= (END - timedelta(days=DAYS - 1)).isoformat()
    assert hi <= END.isoformat()


def test_phone_numbers_are_hashed(conn):
    lengths = {r[0] for r in conn.execute("SELECT LENGTH(phone_hash) FROM customers")}
    assert lengths == {64}


def test_unmet_demand_and_expiry_data_exist(conn):
    assert conn.execute("SELECT COUNT(*) FROM demand_signals").fetchone()[0] > 0
    soon = conn.execute(
        "SELECT COUNT(*) FROM stock_batches WHERE qty > 0 AND expiry_date <= date(?, '+3 day')",
        (END.isoformat(),)).fetchone()[0]
    assert soon > 0


def test_simulator_is_deterministic():
    _, a = build(123)
    _, b = build(123)
    assert a == b