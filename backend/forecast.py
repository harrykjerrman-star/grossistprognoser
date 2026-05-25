import logging
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from database import get_connection

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

try:
    from prophet import Prophet
    _PROPHET = True
except ImportError:
    _PROPHET = False

SAFETY_MARGIN = 0.10
MIN_PROPHET_ROWS = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_forecast(product_name: str, days: int = 7) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT id, current_stock FROM products WHERE name = ?", (product_name,)
    )
    product = cur.fetchone()
    if not product:
        conn.close()
        return None

    cur.execute(
        """
        SELECT date, SUM(quantity) AS qty
        FROM sales
        WHERE product_id = ?
        GROUP BY date
        ORDER BY date
        """,
        (product["id"],),
    )
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 2:
        return None

    df = pd.DataFrame([{"ds": r["date"], "y": float(r["qty"])} for r in rows])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    current_stock = product["current_stock"]

    method = "prophet"
    if _PROPHET and len(df) >= MIN_PROPHET_ROWS:
        try:
            points = _prophet_forecast(df, days)
        except Exception:
            points = _simple_forecast(df, days)
            method = "fallback"
    else:
        points = _simple_forecast(df, days)
        method = "fallback"

    daily_avg = float(df["y"].mean())
    days_stock = (current_stock / daily_avg) if daily_avg > 0 else 999.0

    total_forecast = sum(p["predicted"] for p in points)
    order_qty = max(0, round(total_forecast * (1 + SAFETY_MARGIN) - current_stock))

    if days_stock < 3:
        status = "urgent"
    elif days_stock < 7:
        status = "soon"
    else:
        status = "plan"

    return {
        "product": product_name,
        "current_stock": current_stock,
        "forecast": points,
        "order_quantity": order_qty,
        "days_stock": round(days_stock, 1),
        "total_forecast_7d": round(total_forecast),
        "status": status,
        "method": method,
    }


# ---------------------------------------------------------------------------
# Forecast backends
# ---------------------------------------------------------------------------

def _prophet_forecast(df: pd.DataFrame, days: int) -> list[dict]:
    model = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=days, freq="D")
    fc = model.predict(future).tail(days)

    return [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted": round(max(0.0, row["yhat"]), 1),
            "lower": round(max(0.0, row["yhat_lower"]), 1),
            "upper": round(max(0.0, row["yhat_upper"]), 1),
        }
        for _, row in fc.iterrows()
    ]


def _simple_forecast(df: pd.DataFrame, days: int) -> list[dict]:
    """Exponentially weighted moving average with weekday seasonality."""
    values = df["y"].values
    weights = np.exp(np.linspace(0, 1, len(values)))
    base = float(np.average(values, weights=weights))
    std = float(values.std()) if len(values) > 1 else base * 0.2

    last_date = df["ds"].max()
    weekday_factors = {0: 0.85, 1: 0.90, 2: 0.93, 3: 0.97, 4: 1.25, 5: 1.35, 6: 1.10}

    points = []
    for i in range(1, days + 1):
        d = last_date + timedelta(days=i)
        factor = weekday_factors.get(d.weekday(), 1.0)
        pred = max(0.0, base * factor)
        points.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "predicted": round(pred, 1),
                "lower": round(max(0.0, pred - std), 1),
                "upper": round(pred + std, 1),
            }
        )
    return points


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def generate_demo_data() -> None:
    import random

    products = {
        "Mjölk 3L":           {"base": 45, "stock": 120},
        "Smör 500g":          {"base": 28, "stock": 40},
        "Ost Präst 1kg":      {"base": 15, "stock": 8},
        "Ägg 12-pack":        {"base": 35, "stock": 90},
        "Bröd Formfranska":   {"base": 52, "stock": 15},
        "Yoghurt Naturell 1L":{"base": 22, "stock": 200},
        "Grädde 1L":          {"base": 18, "stock": 5},
    }

    weekday_factor = {0: 0.80, 1: 0.85, 2: 0.90, 3: 0.95, 4: 1.30, 5: 1.40, 6: 1.10}

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM sales")
    cur.execute("DELETE FROM products")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=89)

    for name, cfg in products.items():
        cur.execute(
            "INSERT INTO products (name, current_stock) VALUES (?, ?)",
            (name, cfg["stock"]),
        )
        pid = cur.lastrowid

        current = start_date
        while current <= end_date:
            wf = weekday_factor.get(current.weekday(), 1.0)
            mf = 1.0 + 0.12 * math.sin(2 * math.pi * current.month / 12)
            base = cfg["base"] * wf * mf
            qty = max(1, int(base + random.gauss(0, base * 0.15)))
            cur.execute(
                "INSERT INTO sales (product_id, date, quantity) VALUES (?, ?, ?)",
                (pid, current.isoformat(), qty),
            )
            current += timedelta(days=1)

    conn.commit()
    conn.close()
