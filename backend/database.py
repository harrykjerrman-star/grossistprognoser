import sqlite3
import hashlib
from pathlib import Path

DB_PATH = Path(__file__).parent / "grossist.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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

        CREATE TABLE IF NOT EXISTS products (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    UNIQUE NOT NULL,
            current_stock INTEGER NOT NULL DEFAULT 0,
            unit          TEXT    NOT NULL DEFAULT 'st'
        );

        CREATE TABLE IF NOT EXISTS sales (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL REFERENCES products(id),
            date       TEXT    NOT NULL,
            quantity   INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sales_product_date
            ON sales(product_id, date);
    """)

    pw_hash = hashlib.sha256("demo123".encode()).hexdigest()
    cur.execute(
        "INSERT OR IGNORE INTO users (email, password_hash) VALUES (?, ?)",
        ("demo@grossist.se", pw_hash),
    )

    conn.commit()
    conn.close()
