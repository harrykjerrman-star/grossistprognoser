import hashlib
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "grossist.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS suppliers (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT    UNIQUE NOT NULL,
            contact_person   TEXT    NOT NULL DEFAULT '',
            email            TEXT    NOT NULL DEFAULT '',
            phone            TEXT    NOT NULL DEFAULT '',
            lead_time_days   INTEGER NOT NULL DEFAULT 3,
            min_order_qty    INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS products (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT    UNIQUE NOT NULL,
            current_stock     REAL    NOT NULL DEFAULT 0,
            unit              TEXT    NOT NULL DEFAULT 'st',
            stock_initialized INTEGER NOT NULL DEFAULT 0,
            stock_updated_at  TEXT,
            stock_sync_date   TEXT,
            supplier_id       INTEGER REFERENCES suppliers(id),
            price             REAL    NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sales (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(id),
            date       TEXT    NOT NULL,
            quantity   REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sales_product_date
            ON sales(product_id, date);

        CREATE TABLE IF NOT EXISTS expiry_dates (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  INTEGER NOT NULL REFERENCES products(id),
            expiry_date TEXT    NOT NULL,
            quantity    REAL    NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_expiry_product
            ON expiry_dates(product_id, expiry_date);

        CREATE TABLE IF NOT EXISTS waste (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(id),
            date       TEXT    NOT NULL,
            quantity   REAL    NOT NULL,
            unit       TEXT    NOT NULL DEFAULT 'st',
            reason     TEXT    NOT NULL DEFAULT 'övrigt'
        );

        CREATE INDEX IF NOT EXISTS idx_waste_product_date
            ON waste(product_id, date);

        CREATE TABLE IF NOT EXISTS recipes (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    UNIQUE NOT NULL,
            portions INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS recipe_ingredients (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id  INTEGER NOT NULL REFERENCES recipes(id) ON DELETE CASCADE,
            product_id INTEGER NOT NULL REFERENCES products(id),
            quantity   REAL    NOT NULL DEFAULT 0,
            unit       TEXT    NOT NULL DEFAULT 'st'
        );

        CREATE TABLE IF NOT EXISTS bookings (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            date   TEXT    UNIQUE NOT NULL,
            guests INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS dish_sales (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER NOT NULL REFERENCES recipes(id),
            date      TEXT    NOT NULL,
            portions  INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_dish_sales_recipe_date
            ON dish_sales(recipe_id, date);

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS zettle_product_mapping (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            zettle_uuid  TEXT    UNIQUE,
            zettle_name  TEXT    NOT NULL,
            product_id   INTEGER REFERENCES products(id)
        );

        CREATE INDEX IF NOT EXISTS idx_zettle_name
            ON zettle_product_mapping(zettle_name);

        CREATE TABLE IF NOT EXISTS sync_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            source       TEXT    NOT NULL DEFAULT 'zettle',
            status       TEXT    NOT NULL,
            transactions INTEGER NOT NULL DEFAULT 0,
            matched      INTEGER NOT NULL DEFAULT 0,
            unmatched    INTEGER NOT NULL DEFAULT 0,
            message      TEXT
        );

        CREATE TABLE IF NOT EXISTS forecast_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id    INTEGER NOT NULL REFERENCES products(id),
            forecast_date TEXT    NOT NULL,
            target_date   TEXT    NOT NULL,
            predicted     REAL    NOT NULL,
            lower_bound   REAL,
            upper_bound   REAL,
            model         TEXT    NOT NULL DEFAULT 'prophet'
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_forecast_log_unique
            ON forecast_log(product_id, forecast_date, target_date);

        CREATE INDEX IF NOT EXISTS idx_forecast_log_product
            ON forecast_log(product_id, target_date);

        CREATE TABLE IF NOT EXISTS calendar_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT    NOT NULL,
            name       TEXT    NOT NULL,
            event_type TEXT    NOT NULL DEFAULT 'other',
            multiplier REAL    NOT NULL DEFAULT 1.0,
            note       TEXT    NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS anomalies (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id    INTEGER NOT NULL REFERENCES products(id),
            date          TEXT    NOT NULL,
            predicted     REAL    NOT NULL,
            actual        REAL    NOT NULL,
            deviation_pct REAL    NOT NULL,
            reason        TEXT,
            marked_at     TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_anomalies_unique
            ON anomalies(product_id, date);

        CREATE TABLE IF NOT EXISTS ai_chat_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            context    TEXT,
            created_at TEXT NOT NULL
        );
    """)

    # Safe migrations for older databases
    migrations = [
        "ALTER TABLE products ADD COLUMN stock_initialized INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE products ADD COLUMN stock_updated_at  TEXT",
        "ALTER TABLE products ADD COLUMN stock_sync_date   TEXT",
        "ALTER TABLE products ADD COLUMN supplier_id       INTEGER REFERENCES suppliers(id)",
        "ALTER TABLE products ADD COLUMN price             REAL NOT NULL DEFAULT 0",
        "ALTER TABLE waste    ADD COLUMN unit              TEXT NOT NULL DEFAULT 'st'",
    ]
    for sql in migrations:
        try:
            cur.execute(sql)
        except sqlite3.OperationalError:
            pass  # column already exists

    pw_hash = hashlib.sha256("demo123".encode()).hexdigest()
    cur.execute(
        "INSERT OR IGNORE INTO users (email, password_hash) VALUES (?, ?)",
        ("demo@grossist.se", pw_hash),
    )

    conn.commit()
    conn.close()
