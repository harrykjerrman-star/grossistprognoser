import hashlib
import secrets
from functools import wraps

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from database import get_connection, init_db
from forecast import generate_demo_data, get_forecast

app = Flask(__name__)
CORS(app)

_tokens: dict[str, dict] = {}

init_db()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not token or token not in _tokens:
            return jsonify({"error": "Ej autentiserad"}), 401
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/login")
def login():
    body = request.get_json(silent=True) or {}
    email = body.get("email", "").strip()
    password = body.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email och lösenord krävs"}), 400

    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = get_connection()
    user = conn.execute(
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


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
@require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400

    f = request.files["file"]
    name = (f.filename or "").lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(f)
        else:
            return jsonify({"error": "Stödjer bara CSV och Excel (.csv, .xlsx, .xls)"}), 400

        df.columns = df.columns.str.lower().str.strip()

        VARA_NAMES   = {"vara", "produkt", "product", "item", "artikel"}
        DATUM_NAMES  = {"datum", "date", "dag", "tid"}
        ANTAL_NAMES  = {"antal", "quantity", "qty", "amount", "försäljning", "sold"}

        col_vara  = next((c for c in df.columns if c in VARA_NAMES), None)
        col_datum = next((c for c in df.columns if c in DATUM_NAMES), None)
        col_antal = next((c for c in df.columns if c in ANTAL_NAMES), None)

        missing = [n for n, c in [("vara", col_vara), ("datum", col_datum), ("antal", col_antal)] if not c]
        if missing:
            return jsonify({
                "error": f"Saknar kolumn(er): {', '.join(missing)}. "
                         f"Filen har: {', '.join(df.columns.tolist())}"
            }), 400

        conn = get_connection()
        cur = conn.cursor()
        inserted = 0

        for _, row in df.iterrows():
            try:
                product_name = str(row[col_vara]).strip()
                date_val     = str(row[col_datum]).strip()
                quantity     = int(float(str(row[col_antal])))
                if not product_name or product_name == "nan":
                    continue
            except (ValueError, TypeError):
                continue

            cur.execute("INSERT OR IGNORE INTO products (name) VALUES (?)", (product_name,))
            pid = cur.execute(
                "SELECT id FROM products WHERE name = ?", (product_name,)
            ).fetchone()["id"]
            cur.execute(
                "INSERT INTO sales (product_id, date, quantity) VALUES (?, ?, ?)",
                (pid, date_val, quantity),
            )
            inserted += 1

        conn.commit()
        conn.close()

        unique_products = df[col_vara].dropna().astype(str).unique().tolist()
        return jsonify({
            "success": True,
            "message": f"Laddade upp {inserted} poster för {len(unique_products)} produkt(er)",
            "products": unique_products,
        })

    except Exception as exc:
        return jsonify({"error": f"Fel vid uppladdning: {exc}"}), 500


@app.get("/api/products")
@require_auth
def get_products():
    conn = get_connection()
    rows = conn.execute("SELECT name, current_stock FROM products ORDER BY name").fetchall()
    conn.close()
    return jsonify([{"name": r["name"], "stock": r["current_stock"]} for r in rows])


@app.get("/api/sales/<product_name>")
@require_auth
def get_sales(product_name):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT s.date, SUM(s.quantity) AS quantity
        FROM sales s
        JOIN products p ON s.product_id = p.id
        WHERE p.name = ?
        GROUP BY s.date
        ORDER BY s.date
        LIMIT 60
        """,
        (product_name,),
    ).fetchall()
    conn.close()
    return jsonify([{"date": r["date"], "quantity": r["quantity"]} for r in rows])


@app.get("/api/forecast/<product_name>")
@require_auth
def forecast(product_name):
    result = get_forecast(product_name)
    if result is None:
        return jsonify({"error": "Hittades inte eller för lite data (minst 2 datapunkter)"}), 404
    return jsonify(result)


@app.get("/api/recommendations")
@require_auth
def recommendations():
    conn = get_connection()
    names = [r["name"] for r in conn.execute("SELECT name FROM products ORDER BY name").fetchall()]
    conn.close()

    results = []
    for name in names:
        data = get_forecast(name)
        if data:
            results.append({
                "product":         name,
                "status":          data["status"],
                "current_stock":   data["current_stock"],
                "order_quantity":  data["order_quantity"],
                "days_stock":      data["days_stock"],
                "total_forecast_7d": data["total_forecast_7d"],
            })

    priority = {"urgent": 0, "soon": 1, "plan": 2}
    results.sort(key=lambda x: (priority.get(x["status"], 3), -x["order_quantity"]))
    return jsonify(results)


@app.put("/api/stock/<product_name>")
@require_auth
def update_stock(product_name):
    body = request.get_json(silent=True) or {}
    try:
        stock = int(body.get("stock", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Ogiltigt lagervärde"}), 400

    conn = get_connection()
    conn.execute(
        "UPDATE products SET current_stock = ? WHERE name = ?", (stock, product_name)
    )
    conn.commit()
    conn.close()
    return jsonify({"success": True})


@app.post("/api/demo")
@require_auth
def demo():
    try:
        generate_demo_data()
        return jsonify({"success": True, "message": "Demo-data genererad (7 produkter, 90 dagar)"})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



@app.get("/")
def index():
    import os
    from flask import send_file
    path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return send_file(os.path.abspath(path))

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
  
@app.get("/")      return send_file("../frontend/index.html") 
