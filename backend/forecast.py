import logging
import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from database import get_connection

try:
    from data_quality import get_cleaned_series, get_quality_report
    _HAS_DQ = True
except ImportError:
    _HAS_DQ = False

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("NP.forecaster").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

try:
    from prophet import Prophet
    _PROPHET = True
except ImportError:
    _PROPHET = False

try:
    from neuralprophet import NeuralProphet as _NeuralProphet
    _NEURALPROPHET = True
except ImportError:
    _NEURALPROPHET = False

try:
    from xgboost import XGBRegressor as _XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

SAFETY_MARGIN    = 0.10
MIN_PROPHET_ROWS = 10
MIN_NP_ROWS      = 30
MIN_XGB_ROWS     = 60


# ---------------------------------------------------------------------------
# Swedish holidays
# ---------------------------------------------------------------------------

def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lv = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lv) // 451
    month = (h + lv - 7 * m + 114) // 31
    day   = ((h + lv - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _midsommar_saturday(year: int) -> date:
    jun20 = date(year, 6, 20)
    return jun20 + timedelta(days=(5 - jun20.weekday()) % 7)


def _black_friday(year: int) -> date:
    nov1 = date(year, 11, 1)
    return nov1 + timedelta(days=(4 - nov1.weekday()) % 7 + 21)


def _swedish_holidays(years) -> pd.DataFrame:
    rows = []
    for year in years:
        e   = _easter(year)
        mid = _midsommar_saturday(year)
        bf  = _black_friday(year)
        entries = [
            ("Nyårsdagen",         date(year, 1,  1),  -1, 1),
            ("Trettondagen",       date(year, 1,  6),  -1, 1),
            ("Valborg",            date(year, 4, 30),  -1, 1),
            ("Nationaldagen",      date(year, 6,  6),  -1, 1),
            ("Julafton",           date(year, 12, 24), -2, 1),
            ("Juldag",             date(year, 12, 25), -2, 2),
            ("Annandag jul",       date(year, 12, 26), -1, 1),
            ("Nyårsafton",         date(year, 12, 31), -1, 1),
            ("Långfredag",         e - timedelta(2),   -1, 0),
            ("Påskafton",          e - timedelta(1),   -1, 0),
            ("Påskdagen",          e,                   0, 1),
            ("Annandag påsk",      e + timedelta(1),    0, 1),
            ("Kristi himmelsfärd", e + timedelta(39),   0, 1),
            ("Pingstdagen",        e + timedelta(49),   0, 1),
            ("Midsommarafton",     mid - timedelta(1), -1, 0),
            ("Midsommardagen",     mid,                 0, 1),
            ("Black Friday",       bf,                 -3, 1),
        ]
        for name, d, lw, uw in entries:
            rows.append({"holiday": name, "ds": pd.Timestamp(d),
                         "lower_window": lw, "upper_window": uw})
    return pd.DataFrame(rows)


def _holiday_set(years) -> set:
    holidays_df = _swedish_holidays(years)
    return set(pd.Timestamp(ts).date() for ts in holidays_df["ds"])


# ---------------------------------------------------------------------------
# Forecast backends
# ---------------------------------------------------------------------------

def _neuralprophet_forecast(df: pd.DataFrame, days: int) -> list[dict]:
    n_lags = min(14, max(7, len(df) // 4))
    model  = _NeuralProphet(
        n_forecasts=1,
        n_lags=n_lags,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        epochs=30,
        learning_rate=0.1,
        batch_size=32,
    )
    model.fit(df, freq="D", progress=None)
    future   = model.make_future_dataframe(df, periods=days, n_historic_predictions=False)
    forecast = model.predict(future)
    future_rows = forecast.tail(days)

    std_val = float(df["y"].std()) if len(df) > 1 else float(df["y"].mean()) * 0.15
    results: list[dict] = []
    for _, row in future_rows.iterrows():
        pred = max(0.0, float(row.get("yhat1", 0) or 0))
        results.append({
            "date":      row["ds"].strftime("%Y-%m-%d"),
            "predicted": round(pred, 1),
            "lower":     round(max(0.0, pred - std_val), 1),
            "upper":     round(pred + std_val, 1),
        })
    return results


def _prophet_forecast(df: pd.DataFrame, days: int) -> list[dict]:
    years    = range(df["ds"].dt.year.min() - 1, datetime.now().year + 3)
    holidays = _swedish_holidays(years)
    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=days, freq="D")
    fc     = model.predict(future).tail(days)
    return [
        {
            "date":      row["ds"].strftime("%Y-%m-%d"),
            "predicted": round(max(0.0, row["yhat"]),       1),
            "lower":     round(max(0.0, row["yhat_lower"]), 1),
            "upper":     round(max(0.0, row["yhat_upper"]), 1),
        }
        for _, row in fc.iterrows()
    ]


def _xgboost_forecast(df: pd.DataFrame, days: int) -> list[dict]:
    if not _HAS_XGB:
        return _simple_forecast(df, days)
    try:
        all_years = range(df["ds"].dt.year.min(), datetime.now().year + 2)
        holidays  = _holiday_set(all_years)

        df2 = df.copy()
        df2["dow"]        = df2["ds"].dt.dayofweek
        df2["month"]      = df2["ds"].dt.month
        df2["dom"]        = df2["ds"].dt.day
        df2["is_holiday"] = df2["ds"].apply(lambda ts: int(ts.date() in holidays))
        df2["roll7"]      = df2["y"].rolling(7, min_periods=1).mean()
        df2["lag_7"]      = df2["y"].shift(7)

        train = df2.dropna()
        if len(train) < 15:
            return _simple_forecast(df, days)

        feat_cols = ["dow", "month", "dom", "is_holiday", "roll7", "lag_7"]

        model = _XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", verbosity=0, random_state=42,
        )
        model.fit(train[feat_cols].values, train["y"].values)

        history   = list(df["y"].values)
        last_date = df["ds"].max()
        std_r     = float(np.std(history[-14:]) if len(history) >= 14 else np.std(history))
        preds: list[dict] = []

        for i in range(1, days + 1):
            fd = last_date + timedelta(days=i)

            def _lag(n: int) -> float:
                idx = len(history) - n
                return history[idx] if idx >= 0 else float(np.mean(history[-7:]))

            roll7_val = float(np.mean(history[-7:])) if len(history) >= 7 else float(np.mean(history))
            row_dict  = {
                "dow":        fd.dayofweek,
                "month":      fd.month,
                "dom":        fd.day,
                "is_holiday": int(fd.date() in holidays),
                "roll7":      roll7_val,
                "lag_7":      _lag(7),
            }
            feat = np.array([[row_dict[c] for c in feat_cols]])
            pred = float(max(0.0, model.predict(feat)[0]))

            preds.append({
                "date":      fd.strftime("%Y-%m-%d"),
                "predicted": round(pred, 1),
                "lower":     round(max(0.0, pred - std_r), 1),
                "upper":     round(pred + std_r, 1),
            })
            history.append(pred)

        return preds
    except Exception:
        return _simple_forecast(df, days)


def _simple_forecast(df: pd.DataFrame, days: int) -> list[dict]:
    values  = df["y"].values
    weights = np.exp(np.linspace(0, 1, len(values)))
    base    = float(np.average(values, weights=weights))
    std     = float(values.std()) if len(values) > 1 else base * 0.2
    last_date = df["ds"].max()
    wf = {0: 0.85, 1: 0.90, 2: 0.93, 3: 0.97, 4: 1.25, 5: 1.35, 6: 1.10}
    points = []
    for i in range(1, days + 1):
        d    = last_date + timedelta(days=i)
        pred = max(0.0, base * wf.get(d.weekday(), 1.0))
        points.append({
            "date":      d.strftime("%Y-%m-%d"),
            "predicted": round(pred, 1),
            "lower":     round(max(0.0, pred - std), 1),
            "upper":     round(pred + std, 1),
        })
    return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_forecast(product_name: str, days: int = 7) -> dict | None:
    conn = get_connection()
    cur  = conn.cursor()

    cur.execute(
        "SELECT id, current_stock, stock_initialized, stock_updated_at FROM products WHERE name = ?",
        (product_name,),
    )
    product = cur.fetchone()
    if not product:
        conn.close()
        return None

    cur.execute(
        "SELECT date, SUM(quantity) AS qty FROM sales WHERE product_id=? GROUP BY date ORDER BY date",
        (product["id"],),
    )
    rows = cur.fetchall()

    if len(rows) < 2:
        conn.close()
        return None

    # Use cleaned series (outlier replacements + closed day removal) when available
    if _HAS_DQ:
        df = get_cleaned_series(product["id"], conn)
        if len(df) < 2:
            df = pd.DataFrame([{"ds": r["date"], "y": float(r["qty"])} for r in rows])
            df["ds"] = pd.to_datetime(df["ds"])
            df = df.sort_values("ds").reset_index(drop=True)
    else:
        df = pd.DataFrame([{"ds": r["date"], "y": float(r["qty"])} for r in rows])
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds").reset_index(drop=True)

    # Build history metadata
    history_days = len(df)
    history_warnings = []
    if history_days < 30:
        history_warnings.append("Kortare än 30 dagars historik — prognosen är osäker")
    elif history_days < 365:
        history_warnings.append("Kortare än 365 dagars historik — säsongsmönster kanske inte fångas helt")

    current_stock = product["current_stock"]
    stock_init    = bool(product["stock_initialized"])
    stock_updated = product["stock_updated_at"]

    # ── Choose model ──────────────────────────────────────────────────────────
    method = "fallback"
    points: list[dict] = []

    # 1. Try NeuralProphet (+ optional XGBoost combination at 60 days)
    if _NEURALPROPHET and len(df) >= MIN_NP_ROWS:
        try:
            np_pts = _neuralprophet_forecast(df, days)
            if _HAS_XGB and len(df) >= MIN_XGB_ROWS:
                xgb_pts = _xgboost_forecast(df, days)
                combined = []
                for np_p, xp in zip(np_pts, xgb_pts):
                    pred = round(max(0.0, 0.6 * np_p["predicted"] + 0.4 * xp["predicted"]), 1)
                    combined.append({
                        "date":      np_p["date"],
                        "predicted": pred,
                        "lower":     round(min(np_p["lower"],  xp["lower"]),  1),
                        "upper":     round(max(np_p["upper"],  xp["upper"]),  1),
                    })
                points = combined
                method = "combined_np"
            else:
                points = np_pts
                method = "neuralprophet"
        except Exception:
            pass  # fall through to Prophet

    # 2. Fall back to Prophet (+ XGBoost at 30 days)
    if not points and _PROPHET and len(df) >= MIN_PROPHET_ROWS:
        try:
            prophet_pts = _prophet_forecast(df, days)
            if _HAS_XGB and len(df) >= MIN_XGB_ROWS:
                xgb_pts = _xgboost_forecast(df, days)
                combined = []
                for pp, xp in zip(prophet_pts, xgb_pts):
                    pred = round(max(0.0, 0.6 * pp["predicted"] + 0.4 * xp["predicted"]), 1)
                    combined.append({
                        "date":      pp["date"],
                        "predicted": pred,
                        "lower":     round(min(pp["lower"],  xp["lower"]),  1),
                        "upper":     round(max(pp["upper"],  xp["upper"]),  1),
                    })
                points = combined
                method = "combined"
            else:
                points = prophet_pts
                method = "prophet"
        except Exception:
            pass

    # 3. EWMA fallback
    if not points:
        points = _simple_forecast(df, days)
        method = "fallback"

    # ── Apply calendar event multipliers ─────────────────────────────────────
    today_iso = date.today().isoformat()
    events    = cur.execute(
        "SELECT date, multiplier FROM calendar_events WHERE date >= ?", (today_iso,)
    ).fetchall()
    event_map = {e["date"]: e["multiplier"] for e in events}
    for pt in points:
        if pt["date"] in event_map:
            m = event_map[pt["date"]]
            pt["predicted"] = round(max(0.0, pt["predicted"] * m), 1)
            pt["lower"]     = round(max(0.0, pt["lower"]     * m), 1)
            pt["upper"]     = round(max(0.0, pt["upper"]     * m), 1)
            pt["event"]     = True

    # ── Apply weather multipliers ─────────────────────────────────────────────
    try:
        city_row = cur.execute("SELECT value FROM settings WHERE key='weather_city'").fetchone()
        if city_row and city_row["value"]:
            from weather import get_weather_for_city, weather_multipliers
            wdata  = get_weather_for_city(city_row["value"])
            wmults = weather_multipliers(wdata)
            for pt in points:
                if pt["date"] in wmults:
                    m = wmults[pt["date"]]
                    pt["predicted"] = round(max(0.0, pt["predicted"] * m), 1)
                    pt["lower"]     = round(max(0.0, pt["lower"]     * m), 1)
                    pt["upper"]     = round(max(0.0, pt["upper"]     * m), 1)
                    pt["weather_adjusted"] = True
    except Exception:
        pass

    conn.close()

    daily_avg      = float(df["y"].mean())
    days_stock     = (current_stock / daily_avg) if (daily_avg > 0 and stock_init) else None
    total_forecast = sum(p["predicted"] for p in points)

    if stock_init:
        order_qty = max(0, round(total_forecast * (1 + SAFETY_MARGIN) - current_stock))
    else:
        order_qty = round(total_forecast * (1 + SAFETY_MARGIN))

    if not stock_init:
        status = "unknown"
    elif days_stock is not None and days_stock < 3:
        status = "urgent"
    elif days_stock is not None and days_stock < 7:
        status = "soon"
    else:
        status = "plan"

    return {
        "product":           product_name,
        "current_stock":     current_stock,
        "stock_initialized": stock_init,
        "stock_updated_at":  stock_updated,
        "forecast":          points,
        "forecast_days":     days,
        "order_quantity":    order_qty,
        "days_stock":        round(days_stock, 1) if days_stock is not None else None,
        "total_forecast":    round(total_forecast),
        "status":            status,
        "method":            method,
        "history_days":      history_days,
        "history_warnings":  history_warnings,
    }


# ---------------------------------------------------------------------------
# Demo data generator
# ---------------------------------------------------------------------------

def generate_demo_data() -> None:
    import random

    products = {
        "Mjölk 3L":            {"base": 45, "stock": 120},
        "Smör 500g":           {"base": 28, "stock": 40},
        "Ost Präst 1kg":       {"base": 15, "stock": 8},
        "Ägg 12-pack":         {"base": 35, "stock": 90},
        "Bröd Formfranska":    {"base": 52, "stock": 15},
        "Yoghurt Naturell 1L": {"base": 22, "stock": 200},
        "Grädde 1L":           {"base": 18, "stock": 5},
    }
    wf = {0: 0.80, 1: 0.85, 2: 0.90, 3: 0.95, 4: 1.30, 5: 1.40, 6: 1.10}

    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM sales")
    cur.execute("DELETE FROM products")
    cur.execute("DELETE FROM forecast_log")
    cur.execute("DELETE FROM anomalies")

    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=89)
    now_iso    = datetime.now().isoformat()

    for name, cfg in products.items():
        cur.execute(
            "INSERT INTO products (name, current_stock, stock_initialized, stock_updated_at, stock_sync_date) VALUES (?,?,1,?,?)",
            (name, cfg["stock"], now_iso, end_date.isoformat()),
        )
        pid     = cur.lastrowid
        current = start_date
        while current <= end_date:
            day_f   = wf.get(current.weekday(), 1.0)
            month_f = 1.0 + 0.12 * math.sin(2 * math.pi * current.month / 12)
            qty     = max(1, int(cfg["base"] * day_f * month_f + random.gauss(0, cfg["base"] * 0.15)))
            cur.execute("INSERT INTO sales (product_id, date, quantity) VALUES (?,?,?)",
                        (pid, current.isoformat(), qty))
            current += timedelta(days=1)

    conn.commit()
    conn.close()
