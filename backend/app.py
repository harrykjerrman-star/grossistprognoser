import hashlib
import json
import math
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
    s = s.lower().strip()
    for a, b in [('å','a'), ('ä','a'), ('ö','o'), ('é','e'), ('ü','u')]:
        s = s.replace(a, b)
    return ''.join(c for c in s if c.isalnum())


_FIELD_PATTERNS: dict[str, set[str]] = {
    "vara":         {"vara","produkt","product","item","artikel","name","namn","ingrediens","ingredient"},
    "datum":        {"datum","date","dag","tid","time","forsaljningsdatum","saledate"},
    "antal":        {"antal","quantity","qty","amount","forsaljning","sold","salda","volym","portioner"},
    "lager":        {"lager","stock","saldo","lagerantal","inventory","balance","lagersaldo","forradsaldo","forrad"},
    "utgangsdatum": {"utgangsdatum","expiry","expirydate","bestbefore","utgar","expiration","bbdate"},
    "ratt":         {"ratt","dish","recipe","recept","matratt","retter"},
    "portioner":    {"portioner","portions","servings","antal_portioner","portionsantal"},
}


def _suggest_columns(columns: list[str]) -> dict[str, str | None]:
    normed = {c: _norm(c) for c in columns}
    result: dict[str, str | None] = {}
    for field, patterns in _FIELD_PATTERNS.items():
        best: str | None = None
        for col, n in normed.items():
            if n in patterns:
                best = col; break
        if best is None:
            for col, n in normed.items():
                if any(p in n or n in p for p in patterns):
                    best = col; break
        result[field] = best
    return result


def _resolve_col(df_columns, col_map, field, synonyms):
    explicit = col_map.get(field)
    if explicit and explicit in df_columns:
        return explicit
    return next((c for c in df_columns if c in synonyms), None)


def _expiry_info(product_id: int, conn) -> dict:
    today     = date.today()
    in3_iso   = (today + timedelta(days=3)).isoformat()
    in7_iso   = (today + timedelta(days=7)).isoformat()
    today_iso = today.isoformat()
    rows = conn.execute(
        "SELECT expiry_date, quantity FROM expiry_dates WHERE product_id=? AND expiry_date>=? ORDER BY expiry_date",
        (product_id, today_iso),
    ).fetchall()
    critical = sum(r["quantity"] for r in rows if r["expiry_date"] < in3_iso)
    warning  = sum(r["quantity"] for r in rows if in3_iso <= r["expiry_date"] < in7_iso)
    return {"critical_qty": critical, "warning_qty": warning}


# ─────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────

@app.get("/api/dashboard")
@require_auth
def get_dashboard():
    conn       = get_connection()
    today      = date.today()
    seven_ago  = (today - timedelta(days=7)).isoformat()
    thirty_ago = (today - timedelta(days=30)).isoformat()
    in7_iso    = (today + timedelta(days=7)).isoformat()

    total_products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    urgent_count = conn.execute("""
        SELECT COUNT(*) FROM products p
        WHERE p.stock_initialized = 1
        AND p.current_stock < COALESCE((
            SELECT SUM(s.quantity) / 7.0 * 3
            FROM sales s WHERE s.product_id = p.id AND s.date >= ?
        ), 999999)
    """, (seven_ago,)).fetchone()[0]

    expiring_count = conn.execute(
        "SELECT COUNT(DISTINCT product_id) FROM expiry_dates WHERE expiry_date>=? AND expiry_date<?",
        (today.isoformat(), in7_iso)
    ).fetchone()[0]

    recent_sales = conn.execute(
        "SELECT COALESCE(SUM(quantity),0) FROM sales WHERE date>=?", (seven_ago,)
    ).fetchone()[0]

    daily_rows = conn.execute(
        "SELECT date, SUM(quantity) as total FROM sales WHERE date>=? GROUP BY date ORDER BY date",
        (thirty_ago,)
    ).fetchall()

    # Recipe portions warning count
    recipes = conn.execute("SELECT id FROM recipes").fetchall()
    low_portions = 0
    for rec in recipes:
        ingrs = conn.execute("""
            SELECT ri.quantity, p.current_stock, p.stock_initialized
            FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
            WHERE ri.recipe_id=?
        """, (rec["id"],)).fetchall()
        if not ingrs: continue
        if not all(i["stock_initialized"] for i in ingrs): continue
        max_p = min(
            math.floor(i["current_stock"] / i["quantity"]) if i["quantity"] > 0 else 9999
            for i in ingrs
        )
        if max_p < 10:
            low_portions += 1

    conn.close()
    return jsonify({
        "total_products":      total_products,
        "urgent_count":        urgent_count,
        "expiring_soon_count": expiring_count,
        "recent_sales_7d":     int(recent_sales),
        "daily_sales_30d":     [{"date": r["date"], "total": r["total"]} for r in daily_rows],
        "low_portions_count":  low_portions,
    })


# ─────────────────────────────────────────
# File preview  (column detection)
# ─────────────────────────────────────────

@app.post("/api/preview")
@require_auth
def preview_file():
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
        return jsonify({"columns": columns, "preview": preview, "suggestions": suggestions})
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
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    conn    = get_connection()
    user    = conn.execute(
        "SELECT id FROM users WHERE email=? AND password_hash=?", (email, pw_hash)
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
# Supplier management
# ─────────────────────────────────────────

@app.get("/api/suppliers")
@require_auth
def get_suppliers():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM suppliers ORDER BY name").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.post("/api/suppliers")
@require_auth
def create_supplier():
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return jsonify({"error": "Namn krävs"}), 400
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO suppliers (name,contact_person,email,phone,lead_time_days,min_order_qty) VALUES (?,?,?,?,?,?)",
            (name, body.get("contact_person",""), body.get("email",""), body.get("phone",""),
             int(body.get("lead_time_days", 3)), int(body.get("min_order_qty", 1)))
        )
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return jsonify({"success": True, "id": new_id})
    except Exception as exc:
        conn.close()
        return jsonify({"error": str(exc)}), 500


@app.put("/api/suppliers/<int:supplier_id>")
@require_auth
def update_supplier(supplier_id):
    body = request.get_json(silent=True) or {}
    conn = get_connection()
    conn.execute(
        "UPDATE suppliers SET name=?,contact_person=?,email=?,phone=?,lead_time_days=?,min_order_qty=? WHERE id=?",
        (body.get("name",""), body.get("contact_person",""), body.get("email",""), body.get("phone",""),
         int(body.get("lead_time_days",3)), int(body.get("min_order_qty",1)), supplier_id)
    )
    conn.commit(); conn.close()
    return jsonify({"success": True})


@app.delete("/api/suppliers/<int:supplier_id>")
@require_auth
def delete_supplier(supplier_id):
    conn = get_connection()
    conn.execute("UPDATE products SET supplier_id=NULL WHERE supplier_id=?", (supplier_id,))
    conn.execute("DELETE FROM suppliers WHERE id=?", (supplier_id,))
    conn.commit(); conn.close()
    return jsonify({"success": True})


@app.put("/api/products/<product_name>/supplier")
@require_auth
def set_product_supplier(product_name):
    body        = request.get_json(silent=True) or {}
    supplier_id = body.get("supplier_id")
    conn        = get_connection()
    conn.execute("UPDATE products SET supplier_id=? WHERE name=?", (supplier_id, product_name))
    conn.commit(); conn.close()
    return jsonify({"success": True})


@app.put("/api/products/<product_name>/price")
@require_auth
def set_product_price(product_name):
    body = request.get_json(silent=True) or {}
    try:
        price = float(body.get("price", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Ogiltigt pris"}), 400
    conn = get_connection()
    conn.execute("UPDATE products SET price=? WHERE name=?", (price, product_name))
    conn.commit(); conn.close()
    return jsonify({"success": True})


@app.get("/api/order-email/<int:supplier_id>")
@require_auth
def order_email(supplier_id):
    days = _days_param(14)
    conn = get_connection()
    supplier = conn.execute("SELECT * FROM suppliers WHERE id=?", (supplier_id,)).fetchone()
    if not supplier:
        conn.close()
        return jsonify({"error": "Leverantör hittades inte"}), 404
    prods = conn.execute(
        "SELECT name FROM products WHERE supplier_id=? ORDER BY name", (supplier_id,)
    ).fetchall()
    conn.close()

    order_lines = []
    for p in prods:
        data = get_forecast(p["name"], days=days)
        if data and data["order_quantity"] > 0:
            qty = max(data["order_quantity"], supplier["min_order_qty"])
            order_lines.append({"product": p["name"], "order_quantity": qty})

    contact   = supplier["contact_person"] or supplier["name"]
    today_str = date.today().strftime("%Y-%m-%d")

    if order_lines:
        lines = "\n".join(f"  - {ol['product']}: {ol['order_quantity']}" for ol in order_lines)
    else:
        lines = "  (inga ingredienser behöver beställas just nu)"

    subject  = f"Inköpsorder {today_str} – {supplier['name']}"
    body_txt = (
        f"Hej {contact},\n\n"
        f"Vi önskar göra följande inköpsorder:\n\n"
        f"{lines}\n\n"
        f"Vänligen bekräfta ordern och beräknat leveransdatum.\n\n"
        f"Med vänliga hälsningar"
    )

    return jsonify({
        "supplier":    dict(supplier),
        "to":          supplier["email"] or "",
        "subject":     subject,
        "body":        body_txt,
        "order_lines": order_lines,
        "has_orders":  len(order_lines) > 0,
    })


# ─────────────────────────────────────────
# Recipe management
# ─────────────────────────────────────────

def _recipe_portions_available(recipe_id: int, conn) -> int | None:
    """Calculate max portions from current stock. Returns None if stock not initialized."""
    ingrs = conn.execute("""
        SELECT ri.quantity, p.current_stock, p.stock_initialized, p.name
        FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
        WHERE ri.recipe_id=?
    """, (recipe_id,)).fetchall()
    if not ingrs:
        return 0
    if not all(i["stock_initialized"] for i in ingrs):
        return None
    portions = min(
        math.floor(i["current_stock"] / i["quantity"]) if i["quantity"] > 0 else 99999
        for i in ingrs
    )
    return max(0, portions)


@app.get("/api/recipes")
@require_auth
def get_recipes():
    conn = get_connection()
    recipes = conn.execute("SELECT * FROM recipes ORDER BY name").fetchall()
    result = []
    for r in recipes:
        ingrs = conn.execute("""
            SELECT ri.id, ri.quantity, ri.unit, p.name as ingredient, p.id as product_id,
                   p.current_stock, p.stock_initialized
            FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
            WHERE ri.recipe_id=? ORDER BY p.name
        """, (r["id"],)).fetchall()
        portions_available = _recipe_portions_available(r["id"], conn)
        result.append({
            "id":                 r["id"],
            "name":               r["name"],
            "portions":           r["portions"],
            "portions_available": portions_available,
            "low_stock_warning":  (portions_available is not None and portions_available < 10),
            "ingredients": [{
                "id":          i["id"],
                "ingredient":  i["ingredient"],
                "product_id":  i["product_id"],
                "quantity":    i["quantity"],
                "unit":        i["unit"],
                "stock":       i["current_stock"],
                "initialized": bool(i["stock_initialized"]),
            } for i in ingrs],
        })
    conn.close()
    return jsonify(result)


@app.post("/api/recipes")
@require_auth
def create_recipe():
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    if not name:
        return jsonify({"error": "Namn krävs"}), 400
    portions = int(body.get("portions", 1))
    ingredients = body.get("ingredients", [])  # [{product_id, quantity, unit}]

    conn = get_connection()
    try:
        conn.execute("INSERT INTO recipes (name, portions) VALUES (?,?)", (name, portions))
        recipe_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for ing in ingredients:
            pid = ing.get("product_id")
            qty = float(ing.get("quantity", 0))
            unit = ing.get("unit", "st")
            if pid and qty > 0:
                conn.execute(
                    "INSERT INTO recipe_ingredients (recipe_id,product_id,quantity,unit) VALUES (?,?,?,?)",
                    (recipe_id, pid, qty, unit)
                )
        conn.commit(); conn.close()
        return jsonify({"success": True, "id": recipe_id})
    except Exception as exc:
        conn.close()
        return jsonify({"error": str(exc)}), 500


@app.put("/api/recipes/<int:recipe_id>")
@require_auth
def update_recipe(recipe_id):
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    portions = int(body.get("portions", 1))
    ingredients = body.get("ingredients", [])

    conn = get_connection()
    try:
        conn.execute("UPDATE recipes SET name=?, portions=? WHERE id=?", (name, portions, recipe_id))
        conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (recipe_id,))
        for ing in ingredients:
            pid = ing.get("product_id")
            qty = float(ing.get("quantity", 0))
            unit = ing.get("unit", "st")
            if pid and qty > 0:
                conn.execute(
                    "INSERT INTO recipe_ingredients (recipe_id,product_id,quantity,unit) VALUES (?,?,?,?)",
                    (recipe_id, pid, qty, unit)
                )
        conn.commit(); conn.close()
        return jsonify({"success": True})
    except Exception as exc:
        conn.close()
        return jsonify({"error": str(exc)}), 500


@app.delete("/api/recipes/<int:recipe_id>")
@require_auth
def delete_recipe(recipe_id):
    conn = get_connection()
    conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (recipe_id,))
    conn.execute("DELETE FROM recipes WHERE id=?", (recipe_id,))
    conn.commit(); conn.close()
    return jsonify({"success": True})


@app.get("/api/recipes/<int:recipe_id>/portions")
@require_auth
def recipe_portions(recipe_id):
    conn = get_connection()
    recipe = conn.execute("SELECT * FROM recipes WHERE id=?", (recipe_id,)).fetchone()
    if not recipe:
        conn.close()
        return jsonify({"error": "Recept hittades inte"}), 404
    portions = _recipe_portions_available(recipe_id, conn)
    conn.close()
    return jsonify({"recipe_id": recipe_id, "portions_available": portions})


# ─────────────────────────────────────────
# Dish sales upload (portions → ingredients)
# ─────────────────────────────────────────

@app.post("/api/upload-dish-sales")
@require_auth
def upload_dish_sales():
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
        col_map = {k: v.lower().strip() for k, v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        RATT     = {"rätt","ratt","recept","dish","recipe","matratt"}
        DATUM    = {"datum","date","dag","tid"}
        PORTIONER= {"portioner","portions","antal_portioner","antal","qty","quantity"}

        col_ratt  = _resolve_col(list(df.columns), col_map, "ratt",     RATT)
        col_datum = _resolve_col(list(df.columns), col_map, "datum",    DATUM)
        col_port  = _resolve_col(list(df.columns), col_map, "portioner",PORTIONER)

        missing = [n for n,c in [("rätt",col_ratt),("datum",col_datum),("portioner",col_port)] if not c]
        if missing:
            return jsonify({"error": f"Saknar kolumn(er): {', '.join(missing)}"}), 400

        conn = get_connection()
        inserted_dish = 0
        inserted_ingr = 0
        skipped       = []

        for _, row in df.iterrows():
            try:
                rname = str(row[col_ratt]).strip()
                dval  = str(row[col_datum]).strip()
                port  = int(float(str(row[col_port])))
                if not rname or rname == "nan" or port <= 0: continue
            except (ValueError, TypeError):
                continue

            recipe = conn.execute("SELECT id FROM recipes WHERE name=?", (rname,)).fetchone()
            if not recipe:
                if rname not in skipped: skipped.append(rname)
                continue

            conn.execute(
                "INSERT INTO dish_sales (recipe_id, date, portions) VALUES (?,?,?)",
                (recipe["id"], dval, port)
            )
            inserted_dish += 1

            # Convert to ingredient sales
            ingrs = conn.execute("""
                SELECT ri.product_id, ri.quantity
                FROM recipe_ingredients ri WHERE ri.recipe_id=?
            """, (recipe["id"],)).fetchall()

            for ing in ingrs:
                usage = ing["quantity"] * port
                conn.execute(
                    "INSERT INTO sales (product_id, date, quantity) VALUES (?,?,?)",
                    (ing["product_id"], dval, usage)
                )
                # Update stock if initialized
                prod = conn.execute(
                    "SELECT id, current_stock, stock_initialized, stock_sync_date FROM products WHERE id=?",
                    (ing["product_id"],)
                ).fetchone()
                if prod and prod["stock_initialized"]:
                    sync = prod["stock_sync_date"] or ""
                    if dval > sync:
                        new_stock = max(0, prod["current_stock"] - usage)
                        conn.execute(
                            "UPDATE products SET current_stock=?, stock_sync_date=?, stock_updated_at=? WHERE id=?",
                            (new_stock, dval, datetime.now().isoformat(), prod["id"])
                        )
                inserted_ingr += 1

        conn.commit(); conn.close()
        msg = f"Laddade upp {inserted_dish} rätt-försäljningsposter → {inserted_ingr} ingrediensrader"
        if skipped: msg += f". Okända rätter: {', '.join(skipped)}"
        return jsonify({"success": True, "message": msg})
    except Exception as exc:
        return jsonify({"error": f"Fel: {exc}"}), 500


# ─────────────────────────────────────────
# Bookings
# ─────────────────────────────────────────

@app.get("/api/bookings")
@require_auth
def get_bookings():
    today   = date.today()
    in7_iso = (today + timedelta(days=7)).isoformat()
    conn    = get_connection()
    rows    = conn.execute(
        "SELECT date, guests FROM bookings WHERE date>=? AND date<=? ORDER BY date",
        (today.isoformat(), in7_iso)
    ).fetchall()
    conn.close()
    # Fill in all 7 days even if no booking
    result = []
    for i in range(8):
        d = (today + timedelta(days=i)).isoformat()
        match = next((r for r in rows if r["date"] == d), None)
        result.append({"date": d, "guests": match["guests"] if match else 0})
    return jsonify(result)


@app.post("/api/bookings")
@require_auth
def set_booking():
    body   = request.get_json(silent=True) or {}
    bdate  = body.get("date", "")
    guests = int(body.get("guests", 0))
    if not bdate:
        return jsonify({"error": "Datum krävs"}), 400
    conn = get_connection()
    conn.execute(
        "INSERT INTO bookings (date, guests) VALUES (?,?) ON CONFLICT(date) DO UPDATE SET guests=excluded.guests",
        (bdate, guests)
    )
    conn.commit(); conn.close()
    return jsonify({"success": True})


# ─────────────────────────────────────────
# Daily menu & shopping list
# ─────────────────────────────────────────

@app.post("/api/daily-menu/shopping-list")
@require_auth
def shopping_list():
    """
    Body: {"items": [{"recipe_id": 1, "portions": 20}, ...]}
    Returns ingredient needs vs stock with per-supplier grouping.
    """
    body  = request.get_json(silent=True) or {}
    items = body.get("items", [])  # [{recipe_id, portions}]
    if not items:
        return jsonify({"error": "Inga rätter i menyn"}), 400

    conn = get_connection()
    # Aggregate ingredient needs
    needs: dict[int, dict] = {}  # product_id → {name, need, stock, unit, supplier_id, supplier_name}

    for item in items:
        rid   = item.get("recipe_id")
        port  = float(item.get("portions", 0))
        if not rid or port <= 0: continue

        ingrs = conn.execute("""
            SELECT ri.product_id, ri.quantity, ri.unit,
                   p.name, p.current_stock, p.stock_initialized,
                   p.supplier_id, s.name as supplier_name
            FROM recipe_ingredients ri
            JOIN products p ON ri.product_id=p.id
            LEFT JOIN suppliers s ON p.supplier_id=s.id
            WHERE ri.recipe_id=?
        """, (rid,)).fetchall()

        for ing in ingrs:
            pid = ing["product_id"]
            usage = ing["quantity"] * port
            if pid not in needs:
                needs[pid] = {
                    "product_id":    pid,
                    "ingredient":    ing["name"],
                    "unit":          ing["unit"],
                    "need":          0.0,
                    "stock":         ing["current_stock"] if ing["stock_initialized"] else None,
                    "supplier_id":   ing["supplier_id"],
                    "supplier_name": ing["supplier_name"] or "Okänd",
                }
            needs[pid]["need"] += usage

    result = []
    for pid, row in needs.items():
        stock = row["stock"]
        need  = row["need"]
        to_buy = round(max(0, need - stock), 3) if stock is not None else need
        result.append({
            **row,
            "need":    round(need, 3),
            "to_buy":  to_buy,
            "ok":      (stock is not None and stock >= need),
        })

    # Sort by supplier then ingredient name
    result.sort(key=lambda x: (x["supplier_name"] or "", x["ingredient"]))
    conn.close()
    return jsonify(result)


@app.post("/api/daily-menu/export")
@require_auth
def shopping_list_export():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill

    body  = request.get_json(silent=True) or {}
    items = body.get("items", [])
    menu_date = body.get("date", date.today().isoformat())

    conn = get_connection()
    needs: dict[int, dict] = {}
    for item in items:
        rid  = item.get("recipe_id")
        port = float(item.get("portions", 0))
        if not rid or port <= 0: continue
        ingrs = conn.execute("""
            SELECT ri.product_id, ri.quantity, ri.unit,
                   p.name, p.current_stock, p.stock_initialized,
                   p.supplier_id, s.name as supplier_name
            FROM recipe_ingredients ri
            JOIN products p ON ri.product_id=p.id
            LEFT JOIN suppliers s ON p.supplier_id=s.id
            WHERE ri.recipe_id=?
        """, (rid,)).fetchall()
        for ing in ingrs:
            pid = ing["product_id"]
            if pid not in needs:
                needs[pid] = {"ingredient": ing["name"], "unit": ing["unit"], "need": 0.0,
                              "stock": ing["current_stock"] if ing["stock_initialized"] else None,
                              "supplier_name": ing["supplier_name"] or "Okänd leverantör"}
            needs[pid]["need"] += ing["quantity"] * port

    rows = []
    for row in sorted(needs.values(), key=lambda x: (x["supplier_name"], x["ingredient"])):
        stock = row["stock"]
        need  = row["need"]
        to_buy = round(max(0, need - (stock or 0)), 3)
        rows.append({
            "Leverantör":  row["supplier_name"],
            "Ingrediens":  row["ingredient"],
            "Behövs":      round(need, 3),
            "I förråd":    round(stock, 3) if stock is not None else "Ej angivet",
            "Att köpa":    to_buy,
            "Enhet":       row["unit"],
        })
    conn.close()

    wb = Workbook(); ws = wb.active; ws.title = "Inköpslista"
    ws.append([f"Inköpslista — {menu_date}"])
    ws.cell(1,1).font = Font(bold=True, size=13)
    ws.append([])

    headers = ["Leverantör","Ingrediens","Behövs","I förråd","Att köpa","Enhet"]
    widths  = [22, 24, 12, 12, 12, 10]
    for ci,(h,w) in enumerate(zip(headers,widths),1):
        cell = ws.cell(row=3, column=ci, value=h)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="7C2D12")
        cell.alignment = Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width = w

    for ri, row in enumerate(rows, 4):
        vals = [row[h] for h in headers]
        for ci, val in enumerate(vals, 1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.alignment = Alignment(horizontal="left" if ci <= 2 else "center")
            if row["Att köpa"] and row["Att köpa"] != "Ej angivet":
                try:
                    if float(str(row["Att köpa"])) > 0:
                        cell.fill = PatternFill("solid", fgColor="FEF3C7")
                except (ValueError, TypeError):
                    pass

    buf = BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf, as_attachment=True,
                     download_name=f"inkopslista_{menu_date}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


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

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)
        col_map = {k: v.lower().strip() for k, v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        VARA  = {"vara","produkt","product","item","artikel","ingrediens","ingredient"}
        DATUM = {"datum","date","dag","tid"}
        ANTAL = {"antal","quantity","qty","amount","försäljning","sold"}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",  VARA)
        col_datum = _resolve_col(list(df.columns), col_map, "datum", DATUM)
        col_antal = _resolve_col(list(df.columns), col_map, "antal", ANTAL)

        missing = [n for n,c in [("ingrediens",col_vara),("datum",col_datum),("antal",col_antal)] if not c]
        if missing:
            return jsonify({"error": f"Kunde inte hitta kolumn(er): {', '.join(missing)}"}), 400

        upload_map: dict[str, list] = {}
        for _, row in df.iterrows():
            try:
                pname = str(row[col_vara]).strip()
                dval  = str(row[col_datum]).strip()
                qty   = float(str(row[col_antal]))
                if not pname or pname=="nan": continue
                upload_map.setdefault(pname,[]).append((dval, qty))
            except (ValueError, TypeError):
                continue

        conn = get_connection(); cur = conn.cursor()
        inserted = 0; auto_deducted = []

        for pname, entries in upload_map.items():
            cur.execute("INSERT OR IGNORE INTO products (name) VALUES (?)", (pname,))
            prod = cur.execute(
                "SELECT id,current_stock,stock_initialized,stock_sync_date FROM products WHERE name=?",
                (pname,)
            ).fetchone()
            max_date = max(d for d,_ in entries)
            for dval, qty in entries:
                cur.execute("INSERT INTO sales (product_id,date,quantity) VALUES (?,?,?)",
                            (prod["id"],dval,qty)); inserted += 1
            if prod["stock_initialized"]:
                sync = prod["stock_sync_date"] or ""
                new_sales = sum(q for d,q in entries if d > sync)
                if new_sales > 0:
                    new_stock = max(0, prod["current_stock"] - new_sales)
                    new_sync  = max(max_date, sync)
                    cur.execute(
                        "UPDATE products SET current_stock=?,stock_sync_date=?,stock_updated_at=? WHERE id=?",
                        (new_stock, new_sync, datetime.now().isoformat(), prod["id"])
                    )
                    auto_deducted.append(f"{pname} (-{new_sales})")
            else:
                if not prod["stock_sync_date"] or max_date > prod["stock_sync_date"]:
                    cur.execute("UPDATE products SET stock_sync_date=? WHERE id=?",
                                (max_date, prod["id"]))

        conn.commit(); conn.close()
        msg = f"Laddade upp {inserted} försäljningsposter"
        if auto_deducted: msg += f". Förråd minskat: {', '.join(auto_deducted)}"
        return jsonify({"success":True,"message":msg,"products":list(upload_map.keys())})
    except Exception as exc:
        return jsonify({"error": f"Fel: {exc}"}), 500


# ─────────────────────────────────────────
# Products & sales
# ─────────────────────────────────────────

@app.get("/api/products")
@require_auth
def get_products():
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.name, p.current_stock, p.unit, p.stock_initialized, p.stock_updated_at,
               p.price, p.supplier_id, s.name as supplier_name
        FROM products p LEFT JOIN suppliers s ON p.supplier_id=s.id
        ORDER BY p.name
    """).fetchall()
    conn.close()
    return jsonify([{
        "name":              r["name"],
        "stock":             r["current_stock"],
        "unit":              r["unit"],
        "stock_initialized": bool(r["stock_initialized"]),
        "stock_updated_at":  r["stock_updated_at"],
        "price":             r["price"],
        "supplier_id":       r["supplier_id"],
        "supplier_name":     r["supplier_name"],
    } for r in rows])


@app.get("/api/sales/<product_name>")
@require_auth
def get_sales(product_name):
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.date, SUM(s.quantity) AS quantity
        FROM sales s JOIN products p ON s.product_id=p.id
        WHERE p.name=? GROUP BY s.date ORDER BY s.date LIMIT 90
    """, (product_name,)).fetchall()
    conn.close()
    return jsonify([{"date":r["date"],"quantity":r["quantity"]} for r in rows])


# ─────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────

@app.get("/api/forecast/<product_name>")
@require_auth
def forecast(product_name):
    days   = _days_param(7)
    result = get_forecast(product_name, days=days)
    if result is None:
        return jsonify({"error": "Hittades inte eller för lite data"}), 404
    return jsonify(result)


# ─────────────────────────────────────────
# Recommendations  (supplier-aware status)
# ─────────────────────────────────────────

@app.get("/api/recommendations")
@require_auth
def recommendations():
    days  = _days_param(7)
    conn  = get_connection()
    prods = conn.execute("""
        SELECT p.id, p.name, p.supplier_id,
               s.lead_time_days, s.name as supplier_name,
               s.email as supplier_email, s.phone as supplier_phone
        FROM products p LEFT JOIN suppliers s ON p.supplier_id=s.id
        ORDER BY p.name
    """).fetchall()
    conn.close()

    results = []
    for prod in prods:
        data = get_forecast(prod["name"], days=days)
        if not data: continue

        lead = prod["lead_time_days"] or 3

        if not data["stock_initialized"]:
            status = "unknown"
        elif data["days_stock"] is not None:
            if   data["days_stock"] < lead:      status = "urgent"
            elif data["days_stock"] < lead * 2:  status = "soon"
            else:                                status = "plan"
        else:
            status = data["status"]

        conn2 = get_connection()
        exp   = _expiry_info(prod["id"], conn2)
        conn2.close()

        results.append({
            "product":             prod["name"],
            "status":              status,
            "current_stock":       data["current_stock"],
            "stock_initialized":   data["stock_initialized"],
            "order_quantity":      data["order_quantity"],
            "days_stock":          data["days_stock"],
            "total_forecast":      data["total_forecast"],
            "forecast_days":       data["forecast_days"],
            "expiry_critical_qty": exp["critical_qty"],
            "expiry_warning_qty":  exp["warning_qty"],
            "supplier_id":         prod["supplier_id"],
            "supplier_name":       prod["supplier_name"],
            "supplier_email":      prod["supplier_email"],
            "supplier_phone":      prod["supplier_phone"],
            "lead_time_days":      lead,
        })

    priority = {"urgent":0,"soon":1,"unknown":2,"plan":3}
    results.sort(key=lambda x: (priority.get(x["status"],4), -x["order_quantity"]))
    return jsonify(results)


# ─────────────────────────────────────────
# Stock / förråd management
# ─────────────────────────────────────────

@app.put("/api/stock/<product_name>")
@require_auth
def update_stock(product_name):
    body = request.get_json(silent=True) or {}
    try:
        stock = float(body.get("stock", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Ogiltigt förrådsvalue"}), 400
    now_iso = datetime.now().isoformat()
    today   = date.today().isoformat()
    conn    = get_connection()
    conn.execute("""
        UPDATE products SET current_stock=?, stock_initialized=1, stock_updated_at=?,
        stock_sync_date=CASE WHEN stock_sync_date IS NULL OR stock_sync_date<? THEN ? ELSE stock_sync_date END
        WHERE name=?
    """, (stock, now_iso, today, today, product_name))
    conn.commit(); conn.close()
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
        elif name.endswith((".xlsx",".xls")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)
        col_map = {k: v.lower().strip() for k,v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",
                                 {"vara","produkt","product","artikel","item","ingrediens"})
        col_lager = _resolve_col(list(df.columns), col_map, "lager",
                                 {"lager","stock","saldo","lagerantal","antal","quantity","forrad","forradsaldo"})
        missing = [n for n,c in [("ingrediens",col_vara),("förråd",col_lager)] if not c]
        if missing:
            return jsonify({"error": f"Saknar kolumn(er): {', '.join(missing)}"}), 400

        conn = get_connection()
        updated = skipped = 0
        now_iso = datetime.now().isoformat()
        today   = date.today().isoformat()

        for _, row in df.iterrows():
            try:
                pname = str(row[col_vara]).strip()
                stock = float(str(row[col_lager]))
                if not pname or pname=="nan": continue
            except (ValueError, TypeError):
                skipped += 1; continue
            r = conn.execute("""
                UPDATE products SET current_stock=?,stock_initialized=1,stock_updated_at=?,
                stock_sync_date=CASE WHEN stock_sync_date IS NULL OR stock_sync_date<? THEN ? ELSE stock_sync_date END
                WHERE name=?
            """, (stock, now_iso, today, today, pname))
            updated += r.rowcount if r.rowcount > 0 else 0
            if r.rowcount == 0: skipped += 1

        conn.commit(); conn.close()
        suffix = f", {skipped} hoppades över" if skipped else ""
        return jsonify({"success":True,"message":f"Uppdaterade {updated} ingrediens(er){suffix}","updated":updated})
    except Exception as exc:
        return jsonify({"error": f"Fel: {exc}"}), 500


@app.get("/api/stock-template")
@require_auth
def stock_template():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lager_exempel.xlsx"))
    if not os.path.exists(path):
        return jsonify({"error": "Mallfilen hittades inte"}), 404
    return send_file(path, as_attachment=True, download_name="forrad_mall.xlsx")


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
        elif name.endswith((".xlsx",".xls")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)
        col_map = {k: v.lower().strip() for k,v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        VARA  = {"vara","produkt","product","artikel","item","ingrediens"}
        DATES = {"utgångsdatum","utgangsdatum","expiry","expiry_date","datum","bäst före","best before"}
        ANTAL = {"antal","quantity","qty","amount"}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",         VARA)
        col_date  = _resolve_col(list(df.columns), col_map, "utgangsdatum", DATES)
        col_antal = _resolve_col(list(df.columns), col_map, "antal",        ANTAL)
        missing = [n for n,c in [("ingrediens",col_vara),("utgångsdatum",col_date),("antal",col_antal)] if not c]
        if missing:
            return jsonify({"error": f"Saknar kolumn(er): {', '.join(missing)}"}), 400

        product_entries: dict[str, list] = {}
        for _, row in df.iterrows():
            try:
                pname    = str(row[col_vara]).strip()
                date_raw = row[col_date]
                qty      = float(str(row[col_antal]))
                if not pname or pname=="nan" or qty<=0: continue
                date_str = date_raw.strftime("%Y-%m-%d") if isinstance(date_raw, pd.Timestamp) else str(date_raw).strip()[:10]
                product_entries.setdefault(pname,[]).append((date_str,qty))
            except (ValueError, TypeError):
                continue

        conn = get_connection(); cur = conn.cursor(); inserted = 0
        for pname, entries in product_entries.items():
            prod = cur.execute("SELECT id FROM products WHERE name=?", (pname,)).fetchone()
            if not prod: continue
            cur.execute("DELETE FROM expiry_dates WHERE product_id=?", (prod["id"],))
            for ds, qty in entries:
                cur.execute("INSERT INTO expiry_dates (product_id,expiry_date,quantity) VALUES (?,?,?)",
                            (prod["id"],ds,qty)); inserted += 1
        conn.commit(); conn.close()
        return jsonify({"success":True,"message":f"Laddade upp {inserted} rader för {len(product_entries)} ingrediens(er)"})
    except Exception as exc:
        return jsonify({"error": f"Fel: {exc}"}), 500


@app.get("/api/expiry")
@require_auth
def get_expiry():
    today   = date.today()
    in7_iso = (today + timedelta(days=7)).isoformat()
    conn    = get_connection()
    rows    = conn.execute("""
        SELECT p.name as product, e.expiry_date, e.quantity
        FROM expiry_dates e JOIN products p ON e.product_id=p.id
        WHERE e.expiry_date>=? AND e.expiry_date<? ORDER BY e.expiry_date, p.name
    """, (today.isoformat(), in7_iso)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d         = date.fromisoformat(r["expiry_date"])
        days_left = (d - today).days
        result.append({"product":r["product"],"expiry_date":r["expiry_date"],
                       "quantity":r["quantity"],"days_left":days_left,
                       "urgency":"critical" if days_left < 3 else "warning"})
    return jsonify(result)


@app.get("/api/expiry-template")
@require_auth
def expiry_template():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook(); ws = wb.active; ws.title = "Utgångsdatum"
    for ci,(h,w) in enumerate(zip(["ingrediens","utgångsdatum","antal"],[28,16,10]),1):
        cell=ws.cell(row=1,column=ci,value=h)
        cell.font=Font(bold=True,color="FFFFFF")
        cell.fill=PatternFill("solid",fgColor="7C2D12")
        cell.alignment=Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width=w
    today = date.today()
    for ri,(vara,datum,antal) in enumerate([
        ("Mjölk 3L",(today+timedelta(2)).isoformat(),12),
        ("Grädde 1L",(today+timedelta(5)).isoformat(),8),
        ("Yoghurt Naturell 1L",(today+timedelta(14)).isoformat(),24)],2):
        ws.cell(ri,1,vara); ws.cell(ri,2,datum); ws.cell(ri,3,antal)
    buf=BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf,as_attachment=True,download_name="utgangsdatum_mall.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────
# Waste (svinn)
# ─────────────────────────────────────────

@app.post("/api/waste")
@require_auth
def add_waste():
    body         = request.get_json(silent=True) or {}
    product_name = body.get("product","").strip()
    waste_date   = body.get("date", date.today().isoformat())
    reason       = body.get("reason", "övrigt")
    unit         = body.get("unit", "st")
    try:
        quantity = float(body.get("quantity", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Ogiltigt antal"}), 400
    if not product_name or quantity <= 0:
        return jsonify({"error": "Ingrediens och antal > 0 krävs"}), 400

    conn = get_connection()
    prod = conn.execute("SELECT id FROM products WHERE name=?", (product_name,)).fetchone()
    if not prod:
        conn.close()
        return jsonify({"error": "Ingrediensen finns inte"}), 404
    conn.execute("INSERT INTO waste (product_id,date,quantity,unit,reason) VALUES (?,?,?,?,?)",
                 (prod["id"], waste_date, quantity, unit, reason))
    conn.commit(); conn.close()
    return jsonify({"success":True,"message":f"Registrerade {quantity} {unit} svinn av {product_name}"})


@app.get("/api/waste/report")
@require_auth
def waste_report():
    conn       = get_connection()
    ninety_ago = (date.today() - timedelta(days=90)).isoformat()

    product_rows = conn.execute("""
        SELECT p.name, p.price,
               COALESCE(SUM(w.quantity),0)         AS waste_qty,
               COALESCE(SUM(w.quantity*p.price),0) AS waste_cost
        FROM products p LEFT JOIN waste w ON w.product_id=p.id
        GROUP BY p.id, p.name, p.price
        HAVING waste_qty > 0
        ORDER BY waste_cost DESC
    """).fetchall()

    products = []
    for r in product_rows:
        sales = conn.execute("""
            SELECT COALESCE(SUM(s.quantity),0) FROM sales s
            JOIN products p ON s.product_id=p.id WHERE p.name=? AND s.date>=?
        """, (r["name"], ninety_ago)).fetchone()[0]
        products.append({
            "product":    r["name"],
            "price":      r["price"],
            "waste_qty":  r["waste_qty"],
            "waste_cost": round(r["waste_cost"],2),
            "sales_qty":  float(sales),
            "waste_pct":  round(r["waste_qty"]/sales*100,1) if sales>0 else None,
        })

    monthly = conn.execute("""
        SELECT strftime('%Y-%m',w.date) as month,
               SUM(w.quantity) as qty,
               SUM(w.quantity*p.price) as cost
        FROM waste w JOIN products p ON w.product_id=p.id
        GROUP BY month ORDER BY month
    """).fetchall()

    by_reason = conn.execute("""
        SELECT reason, SUM(w.quantity) as qty, SUM(w.quantity*p.price) as cost
        FROM waste w JOIN products p ON w.product_id=p.id
        GROUP BY reason ORDER BY qty DESC
    """).fetchall()

    # Weekly breakdown (last 8 weeks)
    eight_weeks_ago = (date.today() - timedelta(weeks=8)).isoformat()
    weekly = conn.execute("""
        SELECT strftime('%Y-W%W',w.date) as week,
               SUM(w.quantity) as qty,
               SUM(w.quantity*p.price) as cost
        FROM waste w JOIN products p ON w.product_id=p.id
        WHERE w.date>=?
        GROUP BY week ORDER BY week
    """, (eight_weeks_ago,)).fetchall()

    # Recommendations: top wasters
    recommendations = []
    for p in products[:3]:
        if p["waste_pct"] is not None and p["waste_pct"] > 5:
            recommendations.append(
                f"{p['product']}: {p['waste_pct']}% svinn — överväg att minska inköpskvantiteten"
            )

    conn.close()
    return jsonify({
        "products":        products,
        "monthly":         [{"month":r["month"],"qty":r["qty"],"cost":round(r["cost"] or 0,2)} for r in monthly],
        "weekly":          [{"week":r["week"],"qty":r["qty"],"cost":round(r["cost"] or 0,2)} for r in weekly],
        "by_reason":       [{"reason":r["reason"],"qty":r["qty"],"cost":round(r["cost"] or 0,2)} for r in by_reason],
        "total_cost":      round(sum(p["waste_cost"] for p in products),2),
        "total_qty":       sum(p["waste_qty"] for p in products),
        "recommendations": recommendations,
    })


@app.get("/api/waste/export")
@require_auth
def waste_export():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.name as product, w.date, w.quantity, w.unit, w.reason,
               ROUND(w.quantity*p.price,2) as cost
        FROM waste w JOIN products p ON w.product_id=p.id
        ORDER BY w.date DESC, p.name
    """).fetchall()
    conn.close()

    wb=Workbook(); ws=wb.active; ws.title="Svinnrapport"
    headers=["Ingrediens","Datum","Antal","Enhet","Orsak","Kostnad (kr)"]
    widths=[26,14,10,10,14,14]
    for ci,(h,w) in enumerate(zip(headers,widths),1):
        cell=ws.cell(row=1,column=ci,value=h)
        cell.font=Font(bold=True,color="FFFFFF")
        cell.fill=PatternFill("solid",fgColor="7C2D12")
        cell.alignment=Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width=w
    for ri,r in enumerate(rows,2):
        ws.cell(ri,1,r["product"]); ws.cell(ri,2,r["date"])
        ws.cell(ri,3,r["quantity"]); ws.cell(ri,4,r["unit"])
        ws.cell(ri,5,r["reason"]); ws.cell(ri,6,r["cost"] or 0)
    buf=BytesIO(); wb.save(buf); buf.seek(0)
    date_str=datetime.now().strftime("%Y%m%d")
    return send_file(buf,as_attachment=True,download_name=f"svinnrapport_{date_str}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


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
    prods = conn.execute("""
        SELECT p.id, p.name, p.supplier_id, s.name as supplier_name, s.lead_time_days
        FROM products p LEFT JOIN suppliers s ON p.supplier_id=s.id ORDER BY p.name
    """).fetchall()
    conn.close()

    STATUS_SV   = {"urgent":"Bradskande","soon":"Snart","plan":"Planera","unknown":"Okant"}
    STATUS_FILL = {"Bradskande":"FEE2E2","Snart":"FEF3C7","Planera":"DCFCE7","Okant":"F1F5F9"}

    rows = []
    for prod in prods:
        data = get_forecast(prod["name"], days=days)
        if not data: continue
        lead   = prod["lead_time_days"] or 3
        status = STATUS_SV.get(data["status"], data["status"])
        conn2  = get_connection()
        exp    = _expiry_info(prod["id"], conn2); conn2.close()
        rows.append({
            "Ingrediens":       prod["name"],
            "Nuv. forrad":      data["current_stock"] if data["stock_initialized"] else "Ej angivet",
            f"Prognos {days}d": data["total_forecast"],
            "Bestall antal":    data["order_quantity"],
            "Status":           status,
            "Forradsdagar":     data["days_stock"] if data["days_stock"] is not None else "-",
            "Leverantor":       prod["supplier_name"] or "-",
            "Ledtid (dagar)":   lead,
        })

    wb=Workbook(); ws=wb.active; ws.title="Inkopsrekommendationer"
    headers=["Ingrediens","Nuv. forrad",f"Prognos {days}d","Bestall antal","Status",
             "Forradsdagar","Leverantor","Ledtid (dagar)"]
    widths=[26,14,14,14,14,14,20,14]
    for ci,(h,w) in enumerate(zip(headers,widths),1):
        cell=ws.cell(row=1,column=ci,value=h)
        cell.font=Font(bold=True,color="FFFFFF")
        cell.fill=PatternFill("solid",fgColor="7C2D12")
        cell.alignment=Alignment(horizontal="center")
        ws.column_dimensions[cell.column_letter].width=w
    for ri,row in enumerate(rows,2):
        vals=[row[h] for h in headers]
        base_fill=STATUS_FILL.get(row["Status"])
        for ci,val in enumerate(vals,1):
            cell=ws.cell(row=ri,column=ci,value=val)
            cell.alignment=Alignment(horizontal="left" if ci==1 else "center")
            if base_fill: cell.fill=PatternFill("solid",fgColor=base_fill)
    ws.append([]); ws.append(["Exportdatum:",datetime.now().strftime("%Y-%m-%d %H:%M")])
    buf=BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf,as_attachment=True,
                     download_name=f"inkopsrekommendationer_{datetime.now().strftime('%Y%m%d')}_{days}d.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────
# Demo data
# ─────────────────────────────────────────

@app.post("/api/demo")
@require_auth
def demo():
    try:
        generate_demo_data()
        return jsonify({"success":True,"message":"Demo-data genererad (7 ingredienser, 90 dagar)"})
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
        app.run(debug=False, host="0.0.0.0", port=port)
