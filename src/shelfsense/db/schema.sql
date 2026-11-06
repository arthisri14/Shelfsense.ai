-- ShelfSense schema (Phase 1). Safe to run repeatedly (IF NOT EXISTS).
-- Money is in rupees (REAL). Dates are ISO text: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.

CREATE TABLE IF NOT EXISTS stores (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS products (
    id               INTEGER PRIMARY KEY,
    sku              TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    brand            TEXT NOT NULL,
    category         TEXT NOT NULL,
    pack_size        TEXT NOT NULL,
    substitute_group TEXT NOT NULL,          -- same group = valid substitutes
    shelf_life_days  INTEGER CHECK (shelf_life_days IS NULL OR shelf_life_days > 0),
    is_perishable    INTEGER NOT NULL DEFAULT 0 CHECK (is_perishable IN (0, 1))
);

-- Per-store price list. floor_price is the lowest price the bargaining agent may ever offer.
CREATE TABLE IF NOT EXISTS store_products (
    store_id      INTEGER NOT NULL REFERENCES stores(id),
    product_id    INTEGER NOT NULL REFERENCES products(id),
    mrp           REAL NOT NULL CHECK (mrp > 0),
    selling_price REAL NOT NULL CHECK (selling_price > 0),
    floor_price   REAL NOT NULL CHECK (floor_price > 0),
    reorder_level INTEGER NOT NULL DEFAULT 0 CHECK (reorder_level >= 0),
    active        INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    PRIMARY KEY (store_id, product_id),
    CHECK (floor_price <= selling_price AND selling_price <= mrp)
);

CREATE TABLE IF NOT EXISTS discount_policy (
    store_id         INTEGER NOT NULL REFERENCES stores(id),
    category         TEXT NOT NULL,
    max_discount_pct REAL NOT NULL CHECK (max_discount_pct >= 0 AND max_discount_pct <= 50),
    PRIMARY KEY (store_id, category)
);

CREATE TABLE IF NOT EXISTS suppliers (
    id             INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL CHECK (lead_time_days >= 0)
);

CREATE TABLE IF NOT EXISTS supplier_products (
    supplier_id   INTEGER NOT NULL REFERENCES suppliers(id),
    product_id    INTEGER NOT NULL REFERENCES products(id),
    unit_cost     REAL NOT NULL CHECK (unit_cost > 0),
    min_order_qty INTEGER NOT NULL DEFAULT 1 CHECK (min_order_qty >= 1),
    PRIMARY KEY (supplier_id, product_id)
);

CREATE TABLE IF NOT EXISTS supplier_price_history (
    id             INTEGER PRIMARY KEY,
    supplier_id    INTEGER NOT NULL REFERENCES suppliers(id),
    product_id     INTEGER NOT NULL REFERENCES products(id),
    unit_cost      REAL NOT NULL CHECK (unit_cost > 0),
    effective_date TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sph_lookup ON supplier_price_history (supplier_id, product_id, effective_date);

-- Physical stock, tracked per batch so expiry is exact (FEFO: first-expiring, first-out).
CREATE TABLE IF NOT EXISTS stock_batches (
    id            INTEGER PRIMARY KEY,
    store_id      INTEGER NOT NULL REFERENCES stores(id),
    product_id    INTEGER NOT NULL REFERENCES products(id),
    qty           INTEGER NOT NULL CHECK (qty >= 0),
    unit_cost     REAL NOT NULL CHECK (unit_cost > 0),
    received_date TEXT NOT NULL,
    expiry_date   TEXT
);
CREATE INDEX IF NOT EXISTS idx_batches_store_product ON stock_batches (store_id, product_id);
CREATE INDEX IF NOT EXISTS idx_batches_expiry ON stock_batches (expiry_date);

-- Customers: only a salted hash of the phone number is stored (privacy by design).
CREATE TABLE IF NOT EXISTS customers (
    id           INTEGER PRIMARY KEY,
    store_id     INTEGER NOT NULL REFERENCES stores(id),
    display_name TEXT NOT NULL,
    phone_hash   TEXT NOT NULL,
    language     TEXT NOT NULL CHECK (language IN ('en', 'ta', 'ta-en', 'hi-en')),
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sales (
    id          INTEGER PRIMARY KEY,
    store_id    INTEGER NOT NULL REFERENCES stores(id),
    customer_id INTEGER REFERENCES customers(id),
    sold_at     TEXT NOT NULL,
    channel     TEXT NOT NULL DEFAULT 'walk_in' CHECK (channel IN ('walk_in', 'chat'))
);
CREATE INDEX IF NOT EXISTS idx_sales_store_time ON sales (store_id, sold_at);

CREATE TABLE IF NOT EXISTS sale_items (
    id         INTEGER PRIMARY KEY,
    sale_id    INTEGER NOT NULL REFERENCES sales(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    qty        INTEGER NOT NULL CHECK (qty > 0),
    unit_price REAL NOT NULL CHECK (unit_price > 0)
);
CREATE INDEX IF NOT EXISTS idx_items_sale ON sale_items (sale_id);
CREATE INDEX IF NOT EXISTS idx_items_product ON sale_items (product_id);

-- Every "do you have X?" that could not be satisfied. Fuel for the owner's insights.
CREATE TABLE IF NOT EXISTS demand_signals (
    id                 INTEGER PRIMARY KEY,
    store_id           INTEGER NOT NULL REFERENCES stores(id),
    customer_id        INTEGER REFERENCES customers(id),
    raw_text           TEXT NOT NULL,
    extracted_item     TEXT NOT NULL,
    matched_product_id INTEGER REFERENCES products(id),
    signal_type        TEXT NOT NULL CHECK (signal_type IN ('not_in_catalog', 'out_of_stock')),
    language           TEXT NOT NULL,
    created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_store_time ON demand_signals (store_id, created_at);

CREATE TABLE IF NOT EXISTS reservations (
    id           INTEGER PRIMARY KEY,
    store_id     INTEGER NOT NULL REFERENCES stores(id),
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    product_id   INTEGER NOT NULL REFERENCES products(id),
    qty          INTEGER NOT NULL CHECK (qty > 0),
    agreed_price REAL NOT NULL CHECK (agreed_price > 0),
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'confirmed', 'picked_up', 'expired', 'cancelled')),
    created_at   TEXT NOT NULL,
    expires_at   TEXT NOT NULL
);

-- Append-only, hash-chained audit trail (filled from Phase 10 onwards).
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY,
    ts           TEXT NOT NULL,
    store_id     INTEGER,
    actor        TEXT NOT NULL,
    role         TEXT NOT NULL,
    action       TEXT NOT NULL,
    details_json TEXT NOT NULL,
    prev_hash    TEXT NOT NULL,
    hash         TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Views
-- ---------------------------------------------------------------------------

-- Owner/staff view: stock per product, including products with zero stock.
CREATE VIEW IF NOT EXISTS v_stock AS
SELECT sp.store_id,
       sp.product_id,
       COALESCE(SUM(b.qty), 0) AS qty_on_hand,
       MIN(CASE WHEN b.qty > 0 THEN b.expiry_date END) AS earliest_expiry
FROM store_products sp
LEFT JOIN stock_batches b
       ON b.store_id = sp.store_id AND b.product_id = sp.product_id
WHERE sp.active = 1
GROUP BY sp.store_id, sp.product_id;

-- The ONLY thing the customer-facing agent may read. No cost, floor price, margin or exact counts.
CREATE VIEW IF NOT EXISTS v_customer_catalog AS
SELECT sp.store_id,
       p.id AS product_id,
       p.name,
       p.brand,
       p.category,
       p.pack_size,
       p.substitute_group,
       sp.selling_price,
       CASE WHEN s.qty_on_hand > 0 THEN 1 ELSE 0 END AS in_stock
FROM store_products sp
JOIN products p ON p.id = sp.product_id
JOIN v_stock s ON s.store_id = sp.store_id AND s.product_id = sp.product_id
WHERE sp.active = 1;

-- Alternate names (Tamil/Hindi/transliterated) used by search.
CREATE TABLE IF NOT EXISTS product_aliases (
    product_id INTEGER NOT NULL REFERENCES products(id),
    alias      TEXT NOT NULL,
    PRIMARY KEY (product_id, alias)
);