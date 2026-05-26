"""
Data quality module: outlier detection, missing-data analysis, cleaning for forecast.
"""
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from database import get_connection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers_for_product(product_id: int, conn) -> list[dict]:
    """
    Identify sales days whose value exceeds mean + 3*std.
    Returns a list of dicts: date, original_value, interpolated_value, z_score.
    """
    rows = conn.execute(
        "SELECT date, SUM(quantity) AS qty "
        "FROM sales WHERE product_id=? AND quantity > 0 "
        "GROUP BY date ORDER BY date",
        (product_id,)
    ).fetchall()

    if len(rows) < 10:
        return []

    df = pd.DataFrame([{"date": r["date"], "qty": float(r["qty"])} for r in rows])
    df["dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("dt").reset_index(drop=True)

    mean = float(df["qty"].mean())
    std  = float(df["qty"].std())
    if std < 1e-6:
        return []

    threshold = mean + 3 * std
    result    = []
    for i, row in df[df["qty"] > threshold].iterrows():
        # Interpolate from ±3-day neighbours (skip NaN neighbours)
        neighbors = [
            df.iloc[i + k]["qty"]
            for k in [-3, -2, -1, 1, 2, 3]
            if 0 <= i + k < len(df)
        ]
        interp = round(float(np.mean(neighbors)), 1) if neighbors else round(mean, 1)
        result.append({
            "date":               row["date"],
            "original_value":     round(row["qty"], 1),
            "interpolated_value": interp,
            "mean":               round(mean, 1),
            "std":                round(std, 1),
            "z_score":            round((row["qty"] - mean) / std, 1),
        })
    return result


def sync_outliers(product_id: int, conn) -> int:
    """
    Run outlier detection for one product and persist new findings.
    Returns the number of newly inserted rows.
    """
    outliers = detect_outliers_for_product(product_id, conn)
    added = 0
    for ov in outliers:
        conn.execute("""
            INSERT OR IGNORE INTO outliers
                (product_id, date, original_value, interpolated_value, status)
            VALUES (?, ?, ?, ?, 'pending')
        """, (product_id, ov["date"], ov["original_value"], ov["interpolated_value"]))
        added += conn.execute("SELECT changes()").fetchone()[0]
    return added


def scan_all_products() -> dict:
    """
    Run sync_outliers for every product and log to model_train_log.
    Safe to call from a background thread.
    """
    conn = get_connection()
    try:
        products = conn.execute("SELECT id FROM products").fetchall()
        total = 0
        for p in products:
            try:
                total += sync_outliers(p["id"], conn)
            except Exception as exc:
                logger.warning("scan_all_products product %s: %s", p["id"], exc)
        conn.commit()
        conn.execute(
            "INSERT INTO model_train_log (trained_at, method, n_samples, status) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), "quality_scan", len(products), "ok"),
        )
        conn.commit()
        return {"status": "ok", "new_outliers": total, "products": len(products)}
    except Exception as exc:
        logger.error("scan_all_products error: %s", exc)
        return {"status": "error", "message": str(exc)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cleaned series (for forecasting)
# ---------------------------------------------------------------------------

def get_cleaned_series(product_id: int, conn) -> pd.DataFrame:
    """
    Return a cleaned sales DataFrame (ds, y) for forecasting:
    - confirmed outliers replaced with interpolated values
    - closed days removed
    - negative quantities removed
    """
    rows = conn.execute(
        "SELECT date, SUM(quantity) AS qty "
        "FROM sales WHERE product_id=? GROUP BY date ORDER BY date",
        (product_id,)
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    df = pd.DataFrame([{"ds": r["date"], "y": float(r["qty"])} for r in rows])
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)

    # Outlier replacements
    outlier_rows = conn.execute("""
        SELECT date, interpolated_value FROM outliers
        WHERE product_id=? AND status='confirmed_outlier'
          AND interpolated_value IS NOT NULL
    """, (product_id,)).fetchall()
    outlier_map = {r["date"]: float(r["interpolated_value"]) for r in outlier_rows}

    # Closed days
    closed_rows = conn.execute("""
        SELECT date FROM closed_days
        WHERE product_id=? OR product_id IS NULL
    """, (product_id,)).fetchall()
    closed_set = {r["date"] for r in closed_rows}

    for idx, row in df.iterrows():
        ds = row["ds"].strftime("%Y-%m-%d")
        if ds in outlier_map:
            df.at[idx, "y"] = outlier_map[ds]

    # Remove closed days
    df = df[~df["ds"].apply(lambda ts: ts.strftime("%Y-%m-%d") in closed_set)]

    # Remove non-positive values
    df = df[df["y"] > 0]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def get_quality_report(product_id: int, conn) -> dict:
    """
    Compute a quality report for one product.
    """
    rows = conn.execute(
        "SELECT date, SUM(quantity) AS qty "
        "FROM sales WHERE product_id=? AND quantity > 0 "
        "GROUP BY date ORDER BY date",
        (product_id,)
    ).fetchall()

    if not rows:
        return {
            "product_id": product_id, "n_days": 0,
            "quality_score": 1.0, "warnings": [], "recommendations": [],
        }

    n_days     = len(rows)
    first_date = rows[0]["date"]
    last_date  = rows[-1]["date"]

    start      = date.fromisoformat(first_date)
    end        = date.fromisoformat(last_date)
    span       = (end - start).days + 1

    date_set   = {r["date"] for r in rows}

    closed_rows = conn.execute("""
        SELECT date FROM closed_days WHERE product_id=? OR product_id IS NULL
    """, (product_id,)).fetchall()
    closed_set = {r["date"] for r in closed_rows}

    # Count missing (not closed and not present)
    n_missing  = 0
    cur = start
    while cur <= end:
        ds = cur.isoformat()
        if ds not in date_set and ds not in closed_set:
            n_missing += 1
        cur += timedelta(days=1)

    n_pending   = conn.execute(
        "SELECT COUNT(*) FROM outliers WHERE product_id=? AND status='pending'",
        (product_id,)
    ).fetchone()[0]

    n_confirmed = conn.execute(
        "SELECT COUNT(*) FROM outliers WHERE product_id=? AND status='confirmed_outlier'",
        (product_id,)
    ).fetchone()[0]

    last_row = conn.execute("""
        SELECT trained_at FROM model_train_log
        WHERE product_id=? OR product_id IS NULL
        ORDER BY trained_at DESC LIMIT 1
    """, (product_id,)).fetchone()
    last_trained = last_row["trained_at"] if last_row else None

    # Score 1–10
    score        = 10.0
    missing_pct  = n_missing / span * 100 if span else 0

    if n_days < 30:
        score -= 4.0
    elif n_days < 60:
        score -= 2.5
    elif n_days < 90:
        score -= 1.0
    elif n_days < 180:
        score -= 0.5

    if missing_pct > 30:
        score -= 3.0
    elif missing_pct > 20:
        score -= 2.0
    elif missing_pct > 10:
        score -= 1.0

    if n_pending > 5:
        score -= 1.5
    elif n_pending > 0:
        score -= 0.5

    score = max(1.0, min(10.0, score))

    warnings = []
    if n_days < 30:
        warnings.append("Kortare än 30 dagars historik — prognosen är osäker")
    elif n_days < 365:
        warnings.append("Kortare än 365 dagars historik — säsongsmönster kanske inte fångas")
    if missing_pct > 10:
        warnings.append(f"{missing_pct:.0f}% av dagarna saknar data")
    if n_pending > 0:
        warnings.append(f"{n_pending} ogranskade extremvärden")

    recommendations = []
    if n_days < 90:
        recommendations.append("Importera mer historisk data via Excel för bättre prognoser")
    if n_missing > 7:
        recommendations.append("Markera stängda dagar för att förbättra datakvaliteten")
    if n_pending > 0:
        recommendations.append("Granska och bekräfta flaggade extremvärden")
    if n_days >= 90 and missing_pct <= 5 and n_pending == 0:
        recommendations.append("Datakvaliteten är god — inga åtgärder krävs")

    return {
        "product_id":           product_id,
        "n_days":               n_days,
        "first_date":           first_date,
        "last_date":            last_date,
        "total_span_days":      span,
        "n_missing":            n_missing,
        "missing_pct":          round(missing_pct, 1),
        "n_outliers_pending":   n_pending,
        "n_outliers_confirmed": n_confirmed,
        "last_trained":         last_trained,
        "quality_score":        round(score, 1),
        "warnings":             warnings,
        "recommendations":      recommendations,
    }
