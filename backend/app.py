import hashlib
import json
import os
import secrets
from datetime import date, datetime, timedelta
from functools import wraps
from io import BytesIO

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from database import get_connection, init_db
from forecast import generate_demo_data, get_forecast

app = Flask(__name__)
CORS(app)

_tokens: dict[str, dict] = {}

init_db()


# ─────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────

@app.get("/")
def index():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return send_file(os.path.abspath(path))


# ─────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not token or token not in _tokens:
            return jsonify({"error": "Ej autentiserad"}), 401
        return f(*args, **kwargs)
    return wrapper


def _days_param(default: int = 7) -> int:
    try:
        return max(1, min(90, int(request.args.get("days", default))))
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────
# Column detection helpers
# ─────────────────────────────────────────

def _norm(s: str) -> str:
    """Lowercase, strip accents and non-alphanumeric chars."""
    s = s.lower().strip()
    for a, b in [('å','a'), ('ä','a'), ('ö','o'), ('é','e'), ('ü','u')]:
        s = s.replace(a, b)
    return ''.join(c for c in s if c.isalnum())


# Maps our internal field keys → sets of normalised synonyms
_FIELD_PATTERNS: dict[str, set[str]] = {
    "vara": {
        "vara", "produkt", "product", "item", "artikel", "name",
        "namn", "benaming", "beteckning", "varunamn", "produktnamn",
        "productname", "itemname", "goodsname",
    },
    "datum": {
        "datum", "date", "dag", "tid", "time", "forsaljningsdatum",
        "saledate", "transaktionsdatum", "salesdate", "salesdatum",
        "orderdate", "orderdatum",
    },
    "antal": {
        "antal", "quantity", "qty", "amount", "forsaljning", "sold",
        "salda", "volym", "volume", "stycken", "saldaenheter",
        "quantitysold", "numberofunits", "units", "enheter",
    },
    "lager": {
        "lager", "stock", "saldo", "lagerantal", "inventory",
        "balance", "behallning", "lagersaldo", "stocklevel",
        "currentstock", "stockbalance", "stockqty",
    },
    "utgangsdatum": {
        "utgangsdatum", "expiry", "expirydate", "bestbefore",
        "bastfore", "utgar", "expiration", "expirationdate",
        "bbdate", "bestbeforedate", "bbd", "sellby", "sellbydate",
    },
}


def _suggest_columns(columns: list[str]) -> dict[str, str | None]:
    """Return best-guess column name for each internal field key."""
    normed = {c: _norm(c) for c in columns}
    result: dict[str, str | None] = {}

    for field, patterns in _FIELD_PATTERNS.items():
        best: str | None = None
        # 1) exact match after normalisation
        for col, n in normed.items():
            if n in patterns:
                best = col
                break
        # 2) substring match  (pattern inside column name or vice-versa)
        if best is None:
            for col, n in normed.items():
                if any(p in n or n in p for p in patterns):
                    best = col
                    break
        result[field] = best

    return result


def _resolve_col(df_columns: list[str], col_map: dict, field: str,
                 synonyms: set[str]) -> str | None:
    """Return the actual DataFrame column to use for `field`.
    Prefers explicit col_map, then falls back to synonym scanning."""
    explicit = col_map.get(field)
    if explicit and explicit in df_columns:
        return explicit
    # Legacy auto-detect (keeps backward compatibility)
    return next((c for c in df_columns if c in synonyms), None)


def _expiry_info(product_id: int, conn) -> dict:
    today     = date.today()
    in3_iso   = (today + timedelta(days=3)).isoformat()
    in7_iso   = (today + timedelta(days=7)).isoformat()
    today_iso = today.isoformat()

    rows = conn.execute(
        """SELECT expiry_date, quantity FROM expiry_dates
           WHERE product_id = ? AND expiry_date >= ?
           ORDER BY expiry_date""",
        (product_id, today_iso),
    ).fetchall()

    critical = sum(r["quantity"] for r in rows if r["expiry_date"] < in3_iso)
    warning  = sum(r["quantity"] for r in rows if in3_iso <= r["expiry_date"] < in7_iso)
    return {"critical_qty": critical, "warning_qty": warning}


# ─────────────────────────────────────────
# File preview  (column detection)
# ─────────────────────────────────────────

@app.post("/api/preview")
@require_auth
def preview_file():
    """Read the first rows of an uploaded file and suggest column mappings."""
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400
    f    = request.files["file"]
    name = (f.filename or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f, encoding="utf-8-sig", nrows=10, dtype=str)
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f, nrows=10, dtype=str)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.fillna("", inplace=True)
        columns     = list(df.columns)
        preview     = df.head(5).to_dict(orient="records")
        suggestions = _suggest_columns(columns)

        return jsonify({
            "columns":     columns,
            "preview":     preview,
            "suggestions": suggestions,
            "rows_shown":  len(preview),
        })

    except Exception as exc:
        return jsonify({"error": f"Kunde inte läsa filen: {exc}"}), 500


# ─────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────

@app.post("/api/login")
def login():
    body     = request.get_json(silent=True) or {}
    email    = body.get("email", "").strip()
    password = body.get("password", "")
    if not email or not password:
        return jsonify({"error": "Email och lösenord krävs"}), 400
    pw_hash  = hashlib.sha256(password.encode()).hexdigest()
    conn     = get_connection()
    user     = conn.execute(
        "SELECT id FROM users WHERE email = ? AND password_hash = ?", (email, pw_hash)
    ).fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "Felaktiga inloggningsuppgifter"}), 401
    token = secrets.token_urlsafe(32)
    _tokens[token] = {"email": email, "user_id": user["id"]}
    return jsonify({"token": token, "email": email})


@app.post("/api/logout")
@require_auth
def logout():
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    _tokens.pop(token, None)
    return jsonify({"success": True})


# ─────────────────────────────────────────
# Sales upload  (with auto stock deduction)
# ─────────────────────────────────────────

@app.post("/api/upload")
@require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400
    f    = request.files["file"]
    name = (f.filename or "").lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()  # preserve original case for col_map lookup
        df_lower   = {c: c.lower().strip() for c in df.columns}
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)

        col_map = json.loads(request.form.get("column_map", "{}") or "{}")
        # col_map keys use original casing from the browser; normalise to lower for matching
        col_map = {k: v.lower().strip() for k, v in col_map.items() if v}

        VARA_NAMES  = {"vara", "produkt", "product", "item", "artikel"}
        DATUM_NAMES = {"datum", "date", "dag", "tid"}
        ANTAL_NAMES = {"antal", "quantity", "qty", "amount", "försäljning", "sold"}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",  VARA_NAMES)
        col_datum = _resolve_col(list(df.columns), col_map, "datum", DATUM_NAMES)
        col_antal = _resolve_col(list(df.columns), col_map, "antal", ANTAL_NAMES)

        missing = [n for n, c in [("vara", col_vara), ("datum", col_datum), ("antal", col_antal)] if not c]
        if missing:
            return jsonify({"error": f"Kunde inte hitta kolumn(er): {', '.join(missing)}. Kontrollera mappningen."}), 400

        upload_map: dict[str, list[tuple[str, int]]] = {}
        for _, row in df.iterrows():
            try:
                pname    = str(row[col_vara]).strip()
                date_val = str(row[col_datum]).strip()
                qty      = int(float(str(row[col_antal])))
                if not pname or pname == "nan":
                    continue
                upload_map.setdefault(pname, []).append((date_val, qty))
            except (ValueError, TypeError):
                continue

        conn          = get_connection()
        cur           = conn.cursor()
        inserted      = 0
        auto_deducted = []

        for pname, entries in upload_map.items():
            cur.execute("INSERT OR IGNORE INTO products (name) VALUES (?)", (pname,))
            prod = cur.execute(
                "SELECT id, current_stock, stock_initialized, stock_sync_date FROM products WHERE name = ?",
                (pname,),
            ).fetchone()

            max_upload_date = max(d for d, _ in entries)

            for date_val, qty in entries:
                cur.execute(
                    "INSERT INTO sales (product_id, date, quantity) VALUES (?, ?, ?)",
                    (prod["id"], date_val, qty),
                )
                inserted += 1

            if prod["stock_initialized"]:
                sync_date = prod["stock_sync_date"] or ""
                new_sales = sum(qty for d, qty in entries if d > sync_date)
                if new_sales > 0:
                    new_stock     = max(0, prod["current_stock"] - new_sales)
                    new_sync_date = max(max_upload_date, sync_date)
                    cur.execute(
                        """UPDATE products
                           SET current_stock = ?, stock_sync_date = ?, stock_updated_at = ?
                           WHERE id = ?""",
                        (new_stock, new_sync_date, datetime.now().isoformat(), prod["id"]),
                    )
                    auto_deducted.append(f"{pname} (-{new_sales})")
            else:
                if not prod["stock_sync_date"] or max_upload_date > prod["stock_sync_date"]:
                    cur.execute(
                        "UPDATE products SET stock_sync_date = ? WHERE id = ?",
                        (max_upload_date, prod["id"]),
                    )

        conn.commit()
        conn.close()

        msg = f"Laddade upp {inserted} försäljningsposter"
        if auto_deducted:
            msg += f". Lager automatiskt minskat: {', '.join(auto_deducted)}"
        return jsonify({"success": True, "message": msg, "products": list(upload_map.keys())})

    except Exception as exc:
        return jsonify({"error": f"Fel vid uppladdning: {exc}"}), 500


# ─────────────────────────────────────────
# Products & sales
# ─────────────────────────────────────────

@app.get("/api/products")
@require_auth
def get_products():
    conn = get_connection()
    rows = conn.execute(
        "SELECT name, current_stock, stock_initialized, stock_updated_at FROM products ORDER BY name"
    ).fetchall()
    conn.close()
    return jsonify([
        {
            "name":              r["name"],
            "stock":             r["current_stock"],
            "stock_initialized": bool(r["stock_initialized"]),
            "stock_updated_at":  r["stock_updated_at"],
        }
        for r in rows
    ])


@app.get("/api/sales/<product_name>")
@require_auth
def get_sales(product_name):
    conn = get_connection()
    rows = conn.execute(
        """SELECT s.date, SUM(s.quantity) AS quantity
           FROM   sales s
           JOIN   products p ON s.product_id = p.id
           WHERE  p.name = ?
           GROUP  BY s.date
           ORDER  BY s.date
           LIMIT  90""",
        (product_name,),
    ).fetchall()
    conn.close()
    return jsonify([{"date": r["date"], "quantity": r["quantity"]} for r in rows])


# ─────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────

@app.get("/api/forecast/<product_name>")
@require_auth
def forecast(product_name):
    days   = _days_param(7)
    result = get_forecast(product_name, days=days)
    if result is None:
        return jsonify({"error": "Hittades inte eller för lite data (minst 2 datapunkter)"}), 404
    return jsonify(result)


# ─────────────────────────────────────────
# Recommendations
# ─────────────────────────────────────────

@app.get("/api/recommendations")
@require_auth
def recommendations():
    days  = _days_param(7)
    conn  = get_connection()
    prods = conn.execute("SELECT id, name FROM products ORDER BY name").fetchall()
    conn.close()

    results = []
    for prod in prods:
        data = get_forecast(prod["name"], days=days)
        if data:
            conn2 = get_connection()
            exp   = _expiry_info(prod["id"], conn2)
            conn2.close()
            results.append({
                "product":             prod["name"],
                "status":              data["status"],
                "current_stock":       data["current_stock"],
                "stock_initialized":   data["stock_initialized"],
                "order_quantity":      data["order_quantity"],
                "days_stock":          data["days_stock"],
                "total_forecast":      data["total_forecast"],
                "forecast_days":       data["forecast_days"],
                "expiry_critical_qty": exp["critical_qty"],
                "expiry_warning_qty":  exp["warning_qty"],
            })

    priority = {"urgent": 0, "soon": 1, "unknown": 2, "plan": 3}
    results.sort(key=lambda x: (priority.get(x["status"], 4), -x["order_quantity"]))
    return jsonify(results)


# ─────────────────────────────────────────
# Stock management
# ─────────────────────────────────────────

@app.put("/api/stock/<product_name>")
@require_auth
def update_stock(product_name):
    body = request.get_json(silent=True) or {}
    try:
        stock = int(body.get("stock", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Ogiltigt lagervärde"}), 400

    now_iso = datetime.now().isoformat()
    today   = date.today().isoformat()
    conn    = get_connection()
    conn.execute(
        """UPDATE products
           SET current_stock     = ?,
               stock_initialized = 1,
               stock_updated_at  = ?,
               stock_sync_date   = CASE
                   WHEN stock_sync_date IS NULL OR stock_sync_date < ? THEN ?
                   ELSE stock_sync_date
               END
           WHERE name = ?""",
        (stock, now_iso, today, today, product_name),
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.post("/api/upload-stock")
@require_auth
def upload_stock():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400
    f    = request.files["file"]
    name = (f.filename or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)

        col_map = json.loads(request.form.get("column_map", "{}") or "{}")
        col_map = {k: v.lower().strip() for k, v in col_map.items() if v}

        VARA_NAMES  = {"vara", "produkt", "product", "artikel", "item"}
        LAGER_NAMES = {"lager", "stock", "saldo", "lagerantal", "antal", "quantity"}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",  VARA_NAMES)
        col_lager = _resolve_col(list(df.columns), col_map, "lager", LAGER_NAMES)

        missing = [n for n, c in [("vara", col_vara), ("lager", col_lager)] if not c]
        if missing:
            return jsonify({"error": f"Kunde inte hitta kolumn(er): {', '.join(missing)}"}), 400

        conn             = get_connection()
        updated, skipped = 0, 0
        now_iso          = datetime.now().isoformat()
        today            = date.today().isoformat()

        for _, row in df.iterrows():
            try:
                pname = str(row[col_vara]).strip()
                stock = int(float(str(row[col_lager])))
                if not pname or pname == "nan":
                    continue
            except (ValueError, TypeError):
                skipped += 1
                continue

            result = conn.execute(
                """UPDATE products
                   SET current_stock     = ?,
                       stock_initialized = 1,
                       stock_updated_at  = ?,
                       stock_sync_date   = CASE
                           WHEN stock_sync_date IS NULL OR stock_sync_date < ? THEN ?
                           ELSE stock_sync_date
                       END
                   WHERE name = ?""",
                (stock, now_iso, today, today, pname),
            )
            if result.rowcount > 0:
                updated += 1
            else:
                skipped += 1

        conn.commit()
        conn.close()
        suffix = f", {skipped} hoppades över" if skipped else ""
        return jsonify({
            "success": True,
            "message": f"Uppdaterade lager för {updated} produkt(er){suffix}",
            "updated": updated,
            "skipped": skipped,
        })

    except Exception as exc:
        return jsonify({"error": f"Fel vid uppladdning: {exc}"}), 500


@app.get("/api/stock-template")
@require_auth
def stock_template():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "lager_exempel.xlsx")
    )
    if not os.path.exists(path):
        return jsonify({"error": "Mallfilen hittades inte"}), 404
    return send_file(path, as_attachment=True, download_name="lager_mall.xlsx")


# ─────────────────────────────────────────
# Expiry date management
# ─────────────────────────────────────────

@app.post("/api/upload-expiry")
@require_auth
def upload_expiry():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400
    f    = request.files["file"]
    name = (f.filename or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)

        col_map = json.loads(request.form.get("column_map", "{}") or "{}")
        col_map = {k: v.lower().strip() for k, v in col_map.items() if v}

        VARA_NAMES = {"vara", "produkt", "product", "artikel", "item"}
        DATE_NAMES = {
            "utgångsdatum", "utgangsdatum", "expiry", "expiry_date",
            "datum", "bäst före", "best before", "bast fore",
        }
        ANTAL_NAMES = {"antal", "quantity", "qty", "amount"}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",         VARA_NAMES)
        col_date  = _resolve_col(list(df.columns), col_map, "utgangsdatum", DATE_NAMES)
        col_antal = _resolve_col(list(df.columns), col_map, "antal",        ANTAL_NAMES)

        missing = [n for n, c in [("vara", col_vara), ("utgångsdatum", col_date), ("antal", col_antal)] if not c]
        if missing:
            return jsonify({"error": f"Kunde inte hitta kolumn(er): {', '.join(missing)}"}), 400

        product_entries: dict[str, list[tuple[str, int]]] = {}
        for _, row in df.iterrows():
            try:
                pname    = str(row[col_vara]).strip()
                date_raw = row[col_date]
                qty      = int(float(str(row[col_antal])))
                if not pname or pname == "nan" or qty <= 0:
                    continue
                if isinstance(date_raw, pd.Timestamp):
                    date_str = date_raw.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_raw).strip()[:10]
                product_entries.setdefault(pname, []).append((date_str, qty))
            except (ValueError, TypeError):
                continue

        conn     = get_connection()
        cur      = conn.cursor()
        inserted = 0

        for pname, entries in product_entries.items():
            prod = cur.execute("SELECT id FROM products WHERE name = ?", (pname,)).fetchone()
            if not prod:
                continue
            pid = prod["id"]
            cur.execute("DELETE FROM expiry_dates WHERE product_id = ?", (pid,))
            for date_str, qty in entries:
                cur.execute(
                    "INSERT INTO expiry_dates (product_id, expiry_date, quantity) VALUES (?, ?, ?)",
                    (pid, date_str, qty),
                )
                inserted += 1

        conn.commit()
        conn.close()
        return jsonify({
            "success": True,
            "message": f"Laddade upp {inserted} utgångsdatumrader för {len(product_entries)} produkt(er)",
        })

    except Exception as exc:
        return jsonify({"error": f"Fel vid uppladdning: {exc}"}), 500


@app.get("/api/expiry")
@require_auth
def get_expiry():
    today     = date.today()
    in7_iso   = (today + timedelta(days=7)).isoformat()
    today_iso = today.isoformat()

    conn = get_connection()
    rows = conn.execute(
        """SELECT p.name AS product, e.expiry_date, e.quantity
           FROM   expiry_dates e
           JOIN   products p ON e.product_id = p.id
           WHERE  e.expiry_date >= ? AND e.expiry_date < ?
           ORDER  BY e.expiry_date, p.name""",
        (today_iso, in7_iso),
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        d         = date.fromisoformat(r["expiry_date"])
        days_left = (d - today).days
        urgency   = "critical" if days_left < 3 else "warning"
        result.append({
            "product":     r["product"],
            "expiry_date": r["expiry_date"],
            "quantity":    r["quantity"],
            "days_left":   days_left,
            "urgency":     urgency,
        })
    return jsonify(result)


@app.get("/api/expiry-template")
@require_auth
def expiry_template():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = Workbook()
    ws = wb.active
    ws.title = "Utgångsdatum"

    headers    = ["vara", "utgångsdatum", "antal"]
    col_widths = [28, 16, 10]
    for ci, (h, w) in enumerate(zip(headers, col_widths), 1):
        cell           = ws.cell(row=1, column=ci, value=h)
        cell.font      = Font(bold=True, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor="1E3A8A")
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width = w

    today = date.today()
    examples = [
        ("Mjölk 3L",            (today + timedelta(days=2)).isoformat(),  12),
        ("Grädde 1L",           (today + timedelta(days=5)).isoformat(),   8),
        ("Yoghurt Naturell 1L", (today + timedelta(days=14)).isoformat(), 24),
    ]
    for ri, (vara, datum, antal) in enumerate(examples, 2):
        ws.cell(row=ri, column=1, value=vara)
        ws.cell(row=ri, column=2, value=datum)
        ws.cell(row=ri, column=3, value=antal)

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name="utgangsdatum_mall.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ─────────────────────────────────────────
# Export: order recommendations as Excel
# ─────────────────────────────────────────

@app.get("/api/export")
@require_auth
def export_recommendations():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    days  = _days_param(7)
    conn  = get_connection()
    prods = conn.execute("SELECT id, name FROM products ORDER BY name").fetchall()
    conn.close()

    STATUS_SV   = {"urgent": "Bradskande", "soon": "Snart", "plan": "Planera", "unknown": "Lager okant"}
    STATUS_FILL = {"Bradskande": "FEE2E2", "Snart": "FEF3C7", "Planera": "DCFCE7", "Lager okant": "F1F5F9"}

    rows = []
    for prod in prods:
        data = get_forecast(prod["name"], days=days)
        if not data:
            continue
        conn2 = get_connection()
        exp   = _expiry_info(prod["id"], conn2)
        conn2.close()
        rows.append({
            "Produkt":           prod["name"],
            "Nuv. lager":        data["current_stock"] if data["stock_initialized"] else "Ej angivet",
            f"Prognos {days}d":  data["total_forecast"],
            "Bestall antal":     data["order_quantity"],
            "Status":            STATUS_SV.get(data["status"], data["status"]),
            "Lagerdagar kvar":   data["days_stock"] if data["days_stock"] is not None else "-",
            "Utgar inom 3 dgr":  exp["critical_qty"] if exp["critical_qty"] > 0 else "",
            "Utgar inom 7 dgr":  exp["warning_qty"]  if exp["warning_qty"]  > 0 else "",
        })

    wb = Workbook()
    ws = wb.active
    ws.title = "Bestallningslista"

    headers    = ["Produkt", "Nuv. lager", f"Prognos {days}d", "Bestall antal",
                  "Status", "Lagerdagar kvar", "Utgar inom 3 dgr", "Utgar inom 7 dgr"]
    col_widths = [26, 14, 14, 14, 14, 16, 18, 18]

    for ci, (h, w) in enumerate(zip(headers, col_widths), 1):
        cell           = ws.cell(row=1, column=ci, value=h)
        cell.font      = Font(bold=True, color="FFFFFF")
        cell.fill      = PatternFill("solid", fgColor="1E3A8A")
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width = w

    for ri, row in enumerate(rows, 2):
        vals       = [row[h] for h in headers]
        base_fill  = STATUS_FILL.get(row["Status"])
        if row.get("Utgar inom 3 dgr"):
            base_fill = "FEE2E2"
        elif row.get("Utgar inom 7 dgr"):
            base_fill = "FEF3C7"

        for ci, val in enumerate(vals, 1):
            cell           = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(horizontal="left" if ci == 1 else "center")
            if base_fill:
                cell.fill = PatternFill("solid", fgColor=base_fill)

    ws.append([])
    ws.append(["Exportdatum:", datetime.now().strftime("%Y-%m-%d %H:%M"), f"Horisont: {days} dagar"])

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)

    date_str = datetime.now().strftime("%Y%m%d")
    return send_file(
        buf,
        as_attachment=True,
        download_name=f"bestallningslista_{date_str}_{days}d.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ─────────────────────────────────────────
# Demo data
# ─────────────────────────────────────────

@app.post("/api/demo")
@require_auth
def demo():
    try:
        generate_demo_data()
        return jsonify({"success": True, "message": "Demo-data genererad (7 produkter, 90 dagar)"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        from waitress import serve
        print(f"Startar Waitress (produktion) på port {port} ...")
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        print("Waitress ej installerat — faller tillbaka på Flask dev-server.")
        print("Installera med: pip install waitress")
        app.run(debug=False, host="0.0.0.0", port=port)
