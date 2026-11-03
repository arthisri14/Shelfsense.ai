"""Synthetic store simulator: fills the database with a realistic year of retail activity.

Deterministic for a given (seed, end_date, days). Uses only synthetic data.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
from datetime import date, timedelta

import numpy as np

from shelfsense.simulator.catalog import (
    ASK_TEMPLATES, CATEGORY_RULES, FESTIVALS, FIRST_NAMES, LAST_INITIALS,
    NOT_IN_CATALOG, PRODUCTS, STORES, SUPPLIERS,
)

WEEKDAY_FACTOR = [0.95, 0.90, 0.90, 0.95, 1.05, 1.25, 1.20]   # Mon..Sun
HOURS = np.arange(7, 22)
HOUR_P = np.array([8, 10, 7, 4, 3, 3, 3, 4, 5, 8, 10, 10, 7, 4, 2], dtype=float)
HOUR_P /= HOUR_P.sum()
PHONE_SALT = "shelfsense-simulator-salt"   # simulator only; real salts live in .env later


def _ceil_half(x: float) -> float:
    return math.ceil(x * 2) / 2


def festival_effect(day: date) -> tuple[float, dict[str, float]]:
    """Overall footfall multiplier and per-category boosts for a given day."""
    mult, cats = 1.0, {}
    for _name, fdate, ramp, peak, boosts in FESTIVALS:
        dist = (fdate - day).days
        if 0 <= dist < ramp:
            strength = 1 - dist / ramp
            mult = max(mult, 1 + (peak - 1) * strength)
            for cat, boost in boosts.items():
                cats[cat] = max(cats.get(cat, 1.0), 1 + (boost - 1) * strength)
    return mult, cats


def generate(conn: sqlite3.Connection, end_date: date | None = None,
             days: int = 365, seed: int = 42) -> dict[str, int]:
    end_date = end_date or date.today()
    start = end_date - timedelta(days=days - 1)
    rng = np.random.default_rng(seed)
    today = end_date

    # ---- stores, products ----------------------------------------------------------------
    for sid, s in enumerate(STORES, 1):
        conn.execute("INSERT INTO stores (id, name) VALUES (?, ?)", (sid, s["name"]))
    for pid, p in enumerate(PRODUCTS, 1):
        conn.execute(
            "INSERT INTO products (id, sku, name, brand, category, pack_size, substitute_group,"
            " shelf_life_days, is_perishable) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, f"SS-{pid:04d}", p.name, p.brand, p.category, p.pack, p.group,
             p.shelf_life, int(p.is_perishable)))

    # ---- pricing: selling price, cost, floor ---------------------------------------------
    price: dict[int, dict] = {}
    for pid, p in enumerate(PRODUCTS, 1):
        margin, max_disc = CATEGORY_RULES[p.category]
        margin = max(2.0, margin + float(rng.uniform(-2, 2)))
        selling = float(round(p.mrp * 0.97)) if p.category in ("Staples", "Produce") else float(p.mrp)
        cost = round(selling * (1 - margin / 100), 2)
        floor = max(_ceil_half(cost * 1.02), _ceil_half(selling * (1 - max_disc / 100)))
        floor = min(floor, selling)
        price[pid] = {"mrp": float(p.mrp), "selling": selling, "cost": cost, "floor": floor}

    for sid in range(1, len(STORES) + 1):
        for cat, (_m, max_disc) in CATEGORY_RULES.items():
            conn.execute("INSERT INTO discount_policy VALUES (?,?,?)", (sid, cat, max_disc))

    # ---- suppliers and cost history --------------------------------------------------------
    cat_supplier, lead_time = {}, {}
    for sup_id, (name, lead, cats) in enumerate(SUPPLIERS, 1):
        conn.execute("INSERT INTO suppliers VALUES (?,?,?)", (sup_id, name, lead))
        lead_time[sup_id] = lead
        for c in cats:
            cat_supplier[c] = sup_id
    for pid, p in enumerate(PRODUCTS, 1):
        sup = cat_supplier[p.category]
        cost = price[pid]["cost"]
        moq = 1 if p.is_perishable else int(rng.choice([6, 12, 24]))
        conn.execute("INSERT INTO supplier_products VALUES (?,?,?,?)", (sup, pid, cost, moq))
        rows, cur = [], cost
        if rng.random() < 0.05:   # a recent supplier price jump (feeds the future alert feature)
            rows.append((sup, pid, cost, (end_date - timedelta(days=5)).isoformat()))
            cur = cost / (1 + float(rng.uniform(0.08, 0.15)))
        for age in (30, 60, 90, 120, 150):
            rows.append((sup, pid, round(cur, 2), (end_date - timedelta(days=age)).isoformat()))
            cur *= 1 + float(rng.normal(0.003, 0.012))
        conn.executemany("INSERT INTO supplier_price_history (supplier_id, product_id, unit_cost,"
                         " effective_date) VALUES (?,?,?,?)", rows)

    # ---- which products each store sells -------------------------------------------------
    active: dict[int, list[int]] = {}
    for sid, s in enumerate(STORES, 1):
        mask = rng.random(len(PRODUCTS)) < s["active_share"]
        active[sid] = [pid for pid, on in enumerate(mask, 1) if on]

    # ---- customers (phone stored only as a salted hash) -----------------------------------
    customers: dict[int, list[int]] = {}
    langs = ["ta-en", "en", "hi-en", "ta"]
    lang_p = [0.5, 0.25, 0.15, 0.10]
    cid = 1
    for sid, s in enumerate(STORES, 1):
        customers[sid] = []
        for _ in range(s["n_customers"]):
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(list(LAST_INITIALS))}."
            phone = f"{int(rng.choice([6, 7, 8, 9]))}{int(rng.integers(0, 10**9)):09d}"
            phone_hash = hashlib.sha256((PHONE_SALT + phone).encode()).hexdigest()
            conn.execute("INSERT INTO customers (id, store_id, display_name, phone_hash, language)"
                         " VALUES (?,?,?,?,?)", (cid, sid, name, phone_hash,
                                                 str(rng.choice(langs, p=lang_p))))
            customers[sid].append(cid)
            cid += 1

    # ---- sales history ----------------------------------------------------------------------
    sale_id, item_id = 1, 1
    demand28: dict[tuple[int, int], int] = {}
    cutoff = end_date - timedelta(days=27)
    for sid, s in enumerate(STORES, 1):
        ids = np.array(active[sid])
        cats = [PRODUCTS[i - 1].category for i in ids]
        base_w = (np.array([PRODUCTS[i - 1].popularity for i in ids])
                  * rng.lognormal(0, 0.25, len(ids)))
        cust = np.array(customers[sid])
        cust_w = 1 / np.arange(1, len(cust) + 1) ** 0.8
        cust_w /= cust_w.sum()
        sales_rows, item_rows = [], []
        for d in range(days):
            day = start + timedelta(days=d)
            fest_mult, fest_cat = festival_effect(day)
            factor = (WEEKDAY_FACTOR[day.weekday()] * (1 + 0.0004 * d)
                      * (1.1 if day.day <= 5 else 1.0) * fest_mult)
            n = int(rng.poisson(s["base_baskets"] * factor))
            if n == 0:
                continue
            w = base_w * np.array([fest_cat.get(c, 1.0) for c in cats])
            w = w / w.sum()
            sizes = 1 + rng.poisson(2.0, n)
            picks = rng.choice(len(ids), size=int(sizes.sum()), p=w)
            qtys = rng.choice([1, 2, 3], size=len(picks), p=[0.78, 0.17, 0.05])
            hours = rng.choice(HOURS, size=n, p=HOUR_P)
            minutes = rng.integers(0, 60, size=n)
            known = rng.random(n) < 0.6
            who = rng.choice(cust, size=n, p=cust_w)
            pos = 0
            for b in range(n):
                sz = int(sizes[b])
                basket: dict[int, int] = {}
                for k in range(pos, pos + sz):
                    pid = int(ids[picks[k]])
                    basket[pid] = basket.get(pid, 0) + int(qtys[k])
                pos += sz
                ts = f"{day.isoformat()} {int(hours[b]):02d}:{int(minutes[b]):02d}:00"
                sales_rows.append((sale_id, sid, int(who[b]) if known[b] else None, ts, "walk_in"))
                for pid, q in basket.items():
                    item_rows.append((item_id, sale_id, pid, q, price[pid]["selling"]))
                    item_id += 1
                    if day >= cutoff:
                        demand28[(sid, pid)] = demand28.get((sid, pid), 0) + q
                sale_id += 1
        conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?)", sales_rows)
        conn.executemany("INSERT INTO sale_items VALUES (?,?,?,?,?)", item_rows)

    # ---- store price list + current stock batches -----------------------------------------
    batch_id = 1
    out_of_stock: dict[int, list[int]] = {}
    for sid in range(1, len(STORES) + 1):
        out_of_stock[sid] = []
        active_set = set(active[sid])
        for pid, p in enumerate(PRODUCTS, 1):
            pr = price[pid]
            daily = max(demand28.get((sid, pid), 0) / 28, 0.1)
            lead = lead_time[cat_supplier[p.category]]
            reorder = max(2, math.ceil(daily * (lead + 1)))
            conn.execute("INSERT INTO store_products VALUES (?,?,?,?,?,?,?)",
                         (sid, pid, pr["mrp"], pr["selling"], pr["floor"], reorder,
                          int(pid in active_set)))
            if pid not in active_set:
                continue

            batches: list[tuple[int, int, str | None]] = []   # qty, age_days, expiry
            if rng.random() < 0.06:          # random stock-out
                out_of_stock[sid].append(pid)
            elif p.is_perishable:
                qty = max(1, math.ceil(daily * float(rng.uniform(0.8, 2.5))))
                age = int(rng.integers(0, max(1, p.shelf_life // 2) + 1))
                batches.append((qty, age, (today - timedelta(days=age)
                                           + timedelta(days=p.shelf_life)).isoformat()))
                if rng.random() < 0.30:      # an older batch about to expire
                    left = int(rng.integers(1, 3))
                    old_age = max(0, p.shelf_life - left)
                    batches.append((max(2, math.ceil(daily * float(rng.uniform(0.3, 1.0)))),
                                    old_age, (today + timedelta(days=left)).isoformat()))
            else:
                cover = float(rng.uniform(0.3, 2.0) if rng.random() < 0.08
                              else rng.uniform(5, 30))
                total = max(1, math.ceil(daily * cover))
                parts = [total] if rng.random() < 0.6 else [math.ceil(total * 0.6),
                                                            max(1, total - math.ceil(total * 0.6))]
                for qty in parts:
                    age = int(rng.integers(1, 46))
                    batches.append((qty, age, (today - timedelta(days=age)
                                               + timedelta(days=p.shelf_life)).isoformat()))
                if rng.random() < 0.07:      # old stock nearing expiry
                    left = int(rng.integers(5, 31))
                    batches.append((int(rng.integers(2, 9)), max(0, p.shelf_life - left),
                                    (today + timedelta(days=left)).isoformat()))
            for qty, age, expiry in batches:
                unit_cost = round(pr["cost"] * float(rng.uniform(0.98, 1.03)), 2)
                conn.execute("INSERT INTO stock_batches VALUES (?,?,?,?,?,?,?)",
                             (batch_id, sid, pid, qty, unit_cost,
                              (today - timedelta(days=age)).isoformat(), expiry))
                batch_id += 1

    # ---- unmet-demand signals (last 90 days) ----------------------------------------------
    signal_rows = []
    sig_id = 1
    nic_w = 1 / np.arange(1, len(NOT_IN_CATALOG) + 1) ** 1.2
    nic_w /= nic_w.sum()
    for sid, s in enumerate(STORES, 1):
        n_nic = int(s["base_baskets"] * 2.5)
        targets = [("not_in_catalog", None, str(rng.choice(NOT_IN_CATALOG, p=nic_w)))
                   for _ in range(n_nic)]
        for pid in out_of_stock[sid]:
            p = PRODUCTS[pid - 1]
            for _ in range(int(rng.integers(2, 9))):
                targets.append(("out_of_stock", pid, f"{p.brand} {p.name.lower()} {p.pack}"))
        for sig_type, pid, item in targets:
            lang, template = ASK_TEMPLATES[int(rng.integers(0, len(ASK_TEMPLATES)))]
            when = end_date - timedelta(days=int(rng.integers(0, 90)))
            ts = f"{when.isoformat()} {int(rng.choice(HOURS)):02d}:{int(rng.integers(0, 60)):02d}:00"
            who = int(rng.choice(customers[sid])) if rng.random() < 0.5 else None
            signal_rows.append((sig_id, sid, who, template.format(item=item), item, pid,
                                sig_type, lang, ts))
            sig_id += 1
    conn.executemany("INSERT INTO demand_signals VALUES (?,?,?,?,?,?,?,?,?)", signal_rows)
    conn.commit()

    counts = {}
    for table in ("stores", "products", "store_products", "suppliers", "customers", "sales",
                  "sale_items", "stock_batches", "demand_signals", "supplier_price_history"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return counts