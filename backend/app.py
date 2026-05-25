import base64
import hashlib
import json
import math
import os
import re
import secrets
import threading
import time
from datetime import date, datetime, timedelta
from difflib import get_close_matches
from functools import wraps
from io import BytesIO

import pandas as pd
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

try:
    import requests as _req
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import pdfplumber as _pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

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
# Encryption helpers (for API key storage)
# ─────────────────────────────────────────

def _encrypt_val(val: str) -> str:
    secret = (os.environ.get("APP_SECRET_KEY", "restaurang_secret_2024xx") * 10)[:32]
    encoded = val.encode("utf-8")
    key_bytes = (secret * (len(encoded) // len(secret) + 1)).encode("utf-8")[:len(encoded)]
    return base64.b64encode(bytes(a ^ b for a, b in zip(encoded, key_bytes))).decode()


def _decrypt_val(encrypted: str) -> str:
    try:
        secret = (os.environ.get("APP_SECRET_KEY", "restaurang_secret_2024xx") * 10)[:32]
        data = base64.b64decode(encrypted)
        key_bytes = (secret * (len(data) // len(secret) + 1)).encode("utf-8")[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_bytes)).decode("utf-8")
    except Exception:
        return ""


# ─────────────────────────────────────────
# Column detection helpers
# ─────────────────────────────────────────

def _norm(s: str) -> str:
    s = s.lower().strip()
    for a, b in [("å","a"),("ä","a"),("ö","o"),("é","e"),("ü","u")]:
        s = s.replace(a, b)
    return "".join(c for c in s if c.isalnum())


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


# ─────────────────────────────────────────
# Delivery note (följesedel) parsing helpers
# ─────────────────────────────────────────

_UNIT_WORDS = {"st","kg","liter","l","dl","g","cl","förp","fp","krt","kartong","pack","pkt","lådor","låda"}
_STOP_WORDS = _UNIT_WORDS | {"kr","sek","moms","inkl","exkl","art","artnr","nr","pris","á","sum","summa",
                              "rabatt","netto","brutto","tot","totalt","belopp","inklmoms","excl","incl"}

def _extract_delivery_text(file_obj, filename: str) -> tuple[str, list[list]]:
    """Return (raw_text, tables) from a PDF or image file."""
    fname = filename.lower()
    raw_text = ""
    tables: list[list] = []

    if fname.endswith(".pdf"):
        if not HAS_PDFPLUMBER:
            return "", []
        file_bytes = file_obj.read()
        with _pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                # Try structured table extraction first
                for tbl in (page.extract_tables() or []):
                    clean = [[str(c).strip() if c else "" for c in row] for row in tbl if any(row)]
                    if clean:
                        tables.append(clean)
                page_text = page.extract_text() or ""
                raw_text += page_text + "\n"
    else:
        # Image: try pytesseract if available, otherwise return empty
        try:
            import pytesseract
            from PIL import Image
            img = Image.open(BytesIO(file_obj.read()))
            raw_text = pytesseract.image_to_string(img, lang="swe+eng") or ""
        except Exception:
            raw_text = ""

    return raw_text, tables


def _parse_delivery_items(raw_text: str, tables: list[list], known_products: list[dict]) -> list[dict]:
    """Parse delivery note text/tables into a list of candidate items."""
    prod_names = [p["name"] for p in known_products]
    results: list[dict] = []
    seen_keys: set = set()

    def _add(raw_line: str, name_guess: str, qty: float, unit: str) -> None:
        key = (name_guess.lower(), round(qty, 3))
        if key in seen_keys:
            return
        seen_keys.add(key)
        matches = get_close_matches(name_guess, prod_names, n=1, cutoff=0.45)
        matched = matches[0] if matches else None
        # also try substring match if no fuzzy hit
        if not matched:
            ng_low = name_guess.lower()
            for pn in prod_names:
                if ng_low in pn.lower() or pn.lower() in ng_low:
                    matched = pn; break
        results.append({
            "raw_text":       raw_line,
            "parsed_name":    name_guess,
            "matched_product": matched,
            "quantity":       qty,
            "unit":           unit,
            "confidence":     "high" if matched else "low",
        })

    # ── 1. Try table rows first (most reliable) ──────────────────────────────
    for table in tables:
        for row in table:
            # Skip obvious header rows
            cell_text = " ".join(row)
            if not cell_text.strip():
                continue
            lower_row = [c.lower() for c in row]
            if any(h in lower_row for h in ["benämning","produkt","artikel","item","description","namn"]):
                continue  # header row

            # Find quantity cell: first cell that parses as a reasonable positive number
            qty_val: float | None = None
            qty_idx: int = -1
            for i, cell in enumerate(row):
                cleaned = cell.strip().replace(" ", "").replace(",", ".").replace("\xa0", "")
                try:
                    v = float(cleaned)
                    if 0 < v <= 100_000:
                        qty_val = v; qty_idx = i; break
                except ValueError:
                    pass

            if qty_val is None:
                continue

            # Find name cell: longest text cell that isn't pure numbers
            name_cell = max(
                (c for i, c in enumerate(row) if i != qty_idx and c.strip() and not re.match(r"^[\d\s.,]+$", c)),
                key=len, default=""
            ).strip()
            if len(name_cell) < 3:
                continue

            # Detect unit from any cell
            unit = "st"
            for cell in row:
                m = re.search(r"\b(st|kg|liter|dl|g|cl|förp|fp|krt|kartong|pack|pkt)\b", cell, re.IGNORECASE)
                if m:
                    unit = m.group(1).lower(); break

            _add(" | ".join(row), name_cell, qty_val, unit)

    # ── 2. Fall back to line-by-line text parsing ─────────────────────────────
    if not results and raw_text.strip():
        for line in raw_text.splitlines():
            line = line.strip()
            if len(line) < 6:
                continue

            # Find numbers in line
            nums = re.findall(r"\b(\d+(?:[.,]\d{1,3})?)\b", line)
            candidates = []
            for n in nums:
                try:
                    v = float(n.replace(",", "."))
                    if 0 < v <= 100_000:
                        candidates.append(v)
                except ValueError:
                    pass
            if not candidates:
                continue

            # Use first reasonable number as quantity (skip article numbers at line start)
            qty_val = candidates[0] if len(candidates) == 1 else (
                candidates[1] if candidates[0] > 99999 else candidates[0]
            )

            # Strip numbers and stop-words to get product name
            name_part = re.sub(r"\b\d+(?:[.,]\d+)?\b", " ", line)
            name_part = re.sub(r"[^\w\s%/-]", " ", name_part)
            words = [w for w in name_part.split() if w.lower() not in _STOP_WORDS and len(w) > 1]
            name_guess = " ".join(words).strip()
            if len(name_guess) < 3:
                continue

            unit_m = re.search(r"\b(st|kg|liter|dl|g|cl|förp|fp|krt|kartong|pack|pkt)\b", line, re.IGNORECASE)
            unit = unit_m.group(1).lower() if unit_m else "st"

            _add(line, name_guess, qty_val, unit)

    return results


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
# Zettle API helpers
# ─────────────────────────────────────────

ZETTLE_TOKEN_URL    = "https://oauth.zettle.com/token"
ZETTLE_PURCHASE_URL = "https://purchase.izettle.com/purchases/v2"
ZETTLE_PRODUCTS_URL = "https://products.izettle.com/organizations/self/products/v2"

_zettle_sync_lock = threading.Lock()


def _zettle_get_token(client_id: str, api_key: str) -> str | None:
    if not HAS_REQUESTS:
        return None
    try:
        r = _req.post(ZETTLE_TOKEN_URL, data={
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "client_id":  client_id,
            "assertion":  api_key,
        }, timeout=15)
        if r.status_code == 200:
            return r.json().get("access_token")
        return None
    except Exception:
        return None


def _zettle_fetch_purchases(token: str, last_hash: str | None = None) -> dict | None:
    if not HAS_REQUESTS:
        return None
    try:
        params: dict = {"limit": 100}
        if last_hash:
            params["lastPurchaseHash"] = last_hash
        r = _req.get(ZETTLE_PURCHASE_URL,
                     headers={"Authorization": f"Bearer {token}"},
                     params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def _zettle_fetch_catalog(token: str) -> list:
    if not HAS_REQUESTS:
        return []
    try:
        r = _req.get(ZETTLE_PRODUCTS_URL,
                     headers={"Authorization": f"Bearer {token}"},
                     timeout=30)
        if r.status_code == 200:
            return r.json()
        return []
    except Exception:
        return []


def _log_sync(status: str, transactions: int, matched: int, unmatched: int, message: str) -> None:
    try:
        conn = get_connection()
        conn.execute(
            "INSERT INTO sync_log (timestamp,source,status,transactions,matched,unmatched,message) VALUES (?,?,?,?,?,?,?)",
            (datetime.now().isoformat(), "zettle", status, transactions, matched, unmatched, message)
        )
        conn.commit(); conn.close()
    except Exception:
        pass


def _run_zettle_sync() -> dict:
    """Execute a full Zettle sync cycle. Returns result dict."""
    if not HAS_REQUESTS:
        return {"status": "error", "message": "Paketet 'requests' saknas. Kör: pip install requests"}

    if not _zettle_sync_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Synkronisering pågår redan"}

    try:
        conn = get_connection()
        rows = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
        conn.close()

        client_id_enc = rows.get("zettle_client_id")
        api_key_enc   = rows.get("zettle_api_key")
        if not client_id_enc or not api_key_enc:
            _log_sync("error", 0, 0, 0, "Zettle-uppgifter saknas")
            return {"status": "error", "message": "Zettle ej konfigurerat. Ange uppgifter under Inställningar → Integrationer."}

        client_id = _decrypt_val(client_id_enc)
        api_key   = _decrypt_val(api_key_enc)
        last_hash = rows.get("zettle_last_hash") or None

        # Authenticate
        token = _zettle_get_token(client_id, api_key)
        if not token:
            msg = "Autentisering mot Zettle misslyckades – kontrollera Client ID och API-nyckel"
            _log_sync("error", 0, 0, 0, msg)
            return {"status": "error", "message": msg}

        # Fetch purchases (paginate up to 5 pages per sync = 500 purchases)
        all_purchases = []
        current_hash  = last_hash
        new_last_hash = last_hash
        for _ in range(5):
            page = _zettle_fetch_purchases(token, current_hash)
            if not page:
                break
            batch = page.get("purchases", [])
            if not batch:
                break
            all_purchases.extend(batch)
            new_last_hash = page.get("lastPurchaseHash", current_hash)
            if len(batch) < 100:
                break  # last page
            current_hash = new_last_hash

        total_transactions = len(all_purchases)
        conn = get_connection(); cur = conn.cursor()
        matched = 0
        unmatched_set: set[str] = set()

        for purchase in all_purchases:
            p_date = (purchase.get("timestamp") or "")[:10]
            if not p_date:
                continue
            for item in purchase.get("products", []):
                p_uuid = (item.get("productUUID") or "").strip()
                p_name = (item.get("name") or "").strip()
                p_qty  = float(item.get("quantity") or 0)
                if p_qty <= 0 or not p_name:
                    continue

                # Look up existing mapping (by UUID first, then by name)
                mapping = cur.execute(
                    "SELECT id, product_id FROM zettle_product_mapping WHERE zettle_uuid=?",
                    (p_uuid,)
                ).fetchone()
                if not mapping:
                    mapping = cur.execute(
                        "SELECT id, product_id FROM zettle_product_mapping WHERE LOWER(zettle_name)=LOWER(?)",
                        (p_name,)
                    ).fetchone()

                if mapping:
                    if not mapping["product_id"]:
                        unmatched_set.add(p_name)
                        continue
                    product_id = mapping["product_id"]
                else:
                    # Attempt automatic name match
                    local = cur.execute(
                        "SELECT id FROM products WHERE LOWER(name)=LOWER(?)", (p_name,)
                    ).fetchone()
                    if local:
                        product_id = local["id"]
                        cur.execute(
                            "INSERT OR IGNORE INTO zettle_product_mapping (zettle_uuid, zettle_name, product_id) VALUES (?,?,?)",
                            (p_uuid or None, p_name, product_id)
                        )
                    else:
                        cur.execute(
                            "INSERT OR IGNORE INTO zettle_product_mapping (zettle_uuid, zettle_name, product_id) VALUES (?,?,NULL)",
                            (p_uuid or None, p_name)
                        )
                        unmatched_set.add(p_name)
                        continue

                # Record sale
                cur.execute(
                    "INSERT INTO sales (product_id, date, quantity) VALUES (?,?,?)",
                    (product_id, p_date, p_qty)
                )

                # Deduct from stock if initialized
                prod = cur.execute(
                    "SELECT current_stock, stock_initialized, stock_sync_date FROM products WHERE id=?",
                    (product_id,)
                ).fetchone()
                if prod and prod["stock_initialized"]:
                    sync_date = prod["stock_sync_date"] or ""
                    if p_date >= sync_date:
                        new_stock = max(0.0, float(prod["current_stock"]) - p_qty)
                        cur.execute(
                            "UPDATE products SET current_stock=?, stock_sync_date=?, stock_updated_at=? WHERE id=?",
                            (new_stock, max(p_date, sync_date), datetime.now().isoformat(), product_id)
                        )
                matched += 1

        now_iso = datetime.now().isoformat()
        # Save last processed hash and sync timestamp
        for k, v in [("zettle_last_hash", new_last_hash or ""), ("zettle_last_sync", now_iso)]:
            if v:
                cur.execute(
                    "INSERT INTO settings (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (k, v)
                )
        conn.commit(); conn.close()

        status  = "ok" if not unmatched_set else "partial"
        message = f"{total_transactions} köp hämtade, {matched} rader matchade"
        if unmatched_set:
            message += f", {len(unmatched_set)} Zettle-produkter utan koppling"
        _log_sync(status, total_transactions, matched, len(unmatched_set), message)

        return {
            "status":       status,
            "transactions": total_transactions,
            "matched":      matched,
            "unmatched":    sorted(unmatched_set),
            "message":      message,
            "last_sync":    now_iso,
        }

    except Exception as exc:
        msg = f"Oväntat fel: {exc}"
        _log_sync("error", 0, 0, 0, msg)
        return {"status": "error", "message": msg}
    finally:
        _zettle_sync_lock.release()


# ─────────────────────────────────────────
# Background sync thread (every 15 min)
# ─────────────────────────────────────────

def _background_sync_loop():
    while True:
        time.sleep(900)   # 15 minutes
        try:
            _run_zettle_sync()
        except Exception:
            pass


_bg_sync_thread = threading.Thread(target=_background_sync_loop, daemon=True, name="zettle-bg-sync")
_bg_sync_thread.start()


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

    recipes = conn.execute("SELECT id FROM recipes").fetchall()
    low_portions = 0
    for rec in recipes:
        ingrs = conn.execute("""
            SELECT ri.quantity, p.current_stock, p.stock_initialized
            FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
            WHERE ri.recipe_id=?
        """, (rec["id"],)).fetchall()
        if not ingrs or not all(i["stock_initialized"] for i in ingrs):
            continue
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
# File preview
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
# Settings – Zettle integration
# ─────────────────────────────────────────

@app.get("/api/settings/zettle")
@require_auth
def get_zettle_settings():
    conn = get_connection()
    rows  = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM settings").fetchall()}
    unmapped = conn.execute(
        "SELECT COUNT(*) FROM zettle_product_mapping WHERE product_id IS NULL"
    ).fetchone()[0]
    total_mapped = conn.execute(
        "SELECT COUNT(*) FROM zettle_product_mapping WHERE product_id IS NOT NULL"
    ).fetchone()[0]
    conn.close()

    has_creds = bool(rows.get("zettle_client_id"))
    cid_masked = None
    if has_creds:
        cid_plain  = _decrypt_val(rows["zettle_client_id"])
        cid_masked = cid_plain[:6] + "…" if len(cid_plain) > 6 else cid_plain

    return jsonify({
        "configured":     has_creds,
        "client_id_hint": cid_masked,
        "last_sync":      rows.get("zettle_last_sync"),
        "unmapped_count": unmapped,
        "mapped_count":   total_mapped,
        "has_requests":   HAS_REQUESTS,
    })


@app.put("/api/settings/zettle")
@require_auth
def save_zettle_settings():
    body      = request.get_json(silent=True) or {}
    client_id = body.get("client_id", "").strip()
    api_key   = body.get("api_key", "").strip()
    if not client_id or not api_key:
        return jsonify({"error": "Client ID och API-nyckel krävs"}), 400
    conn = get_connection()
    for k, v in [("zettle_client_id", _encrypt_val(client_id)),
                 ("zettle_api_key",   _encrypt_val(api_key))]:
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, v)
        )
    conn.commit(); conn.close()
    return jsonify({"success": True, "message": "Zettle-uppgifter sparade"})


@app.delete("/api/settings/zettle")
@require_auth
def delete_zettle_settings():
    conn = get_connection()
    conn.execute(
        "DELETE FROM settings WHERE key IN ('zettle_client_id','zettle_api_key','zettle_last_hash','zettle_last_sync')"
    )
    conn.commit(); conn.close()
    return jsonify({"success": True})


# ─────────────────────────────────────────
# Zettle sync endpoints
# ─────────────────────────────────────────

@app.post("/api/zettle/sync")
@require_auth
def manual_zettle_sync():
    result = _run_zettle_sync()
    return jsonify(result), (200 if result["status"] != "error" else 400)


@app.get("/api/zettle/products")
@require_auth
def get_zettle_products():
    conn     = get_connection()
    mappings = conn.execute("""
        SELECT zm.id, zm.zettle_uuid, zm.zettle_name, zm.product_id,
               p.name as product_name
        FROM zettle_product_mapping zm
        LEFT JOIN products p ON zm.product_id=p.id
        ORDER BY (zm.product_id IS NULL) DESC, LOWER(zm.zettle_name)
    """).fetchall()
    products = conn.execute("SELECT id, name FROM products ORDER BY name").fetchall()
    conn.close()
    return jsonify({
        "mappings": [dict(r) for r in mappings],
        "products": [dict(p) for p in products],
    })


@app.post("/api/zettle/mapping")
@require_auth
def save_zettle_mapping():
    body       = request.get_json(silent=True) or {}
    mapping_id = body.get("mapping_id")
    product_id = body.get("product_id")   # None → unmap
    if not mapping_id:
        return jsonify({"error": "mapping_id krävs"}), 400
    conn = get_connection()
    conn.execute(
        "UPDATE zettle_product_mapping SET product_id=? WHERE id=?",
        (product_id, mapping_id)
    )
    conn.commit(); conn.close()
    return jsonify({"success": True})


@app.get("/api/zettle/log")
@require_auth
def get_sync_log():
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sync_log ORDER BY timestamp DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


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
    lines     = "\n".join(f"  - {ol['product']}: {ol['order_quantity']}" for ol in order_lines) \
                if order_lines else "  (inga ingredienser behöver beställas just nu)"

    subject  = f"Inköpsorder {today_str} – {supplier['name']}"
    body_txt = (f"Hej {contact},\n\nVi önskar göra följande inköpsorder:\n\n{lines}\n\n"
                f"Vänligen bekräfta ordern och beräknat leveransdatum.\n\nMed vänliga hälsningar")

    return jsonify({"supplier": dict(supplier), "to": supplier["email"] or "",
                    "subject": subject, "body": body_txt, "order_lines": order_lines,
                    "has_orders": len(order_lines) > 0})


# ─────────────────────────────────────────
# Recipe management
# ─────────────────────────────────────────

def _recipe_portions_available(recipe_id: int, conn) -> int | None:
    ingrs = conn.execute("""
        SELECT ri.quantity, p.current_stock, p.stock_initialized
        FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
        WHERE ri.recipe_id=?
    """, (recipe_id,)).fetchall()
    if not ingrs:
        return 0
    if not all(i["stock_initialized"] for i in ingrs):
        return None
    return max(0, min(
        math.floor(i["current_stock"] / i["quantity"]) if i["quantity"] > 0 else 99999
        for i in ingrs
    ))


@app.get("/api/recipes")
@require_auth
def get_recipes():
    conn    = get_connection()
    recipes = conn.execute("SELECT * FROM recipes ORDER BY name").fetchall()
    result  = []
    for r in recipes:
        ingrs = conn.execute("""
            SELECT ri.id, ri.quantity, ri.unit, p.name as ingredient,
                   p.id as product_id, p.current_stock, p.stock_initialized
            FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
            WHERE ri.recipe_id=? ORDER BY p.name
        """, (r["id"],)).fetchall()
        pa = _recipe_portions_available(r["id"], conn)
        result.append({
            "id": r["id"], "name": r["name"], "portions": r["portions"],
            "portions_available": pa,
            "low_stock_warning":  (pa is not None and pa < 10),
            "ingredients": [{"id": i["id"], "ingredient": i["ingredient"],
                              "product_id": i["product_id"], "quantity": i["quantity"],
                              "unit": i["unit"], "stock": i["current_stock"],
                              "initialized": bool(i["stock_initialized"])} for i in ingrs],
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
    conn = get_connection()
    try:
        conn.execute("INSERT INTO recipes (name, portions) VALUES (?,?)", (name, int(body.get("portions",1))))
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        for ing in body.get("ingredients", []):
            pid = ing.get("product_id"); qty = float(ing.get("quantity", 0))
            if pid and qty > 0:
                conn.execute("INSERT INTO recipe_ingredients (recipe_id,product_id,quantity,unit) VALUES (?,?,?,?)",
                             (rid, pid, qty, ing.get("unit","st")))
        conn.commit(); conn.close()
        return jsonify({"success": True, "id": rid})
    except Exception as exc:
        conn.close(); return jsonify({"error": str(exc)}), 500


@app.put("/api/recipes/<int:recipe_id>")
@require_auth
def update_recipe(recipe_id):
    body = request.get_json(silent=True) or {}
    conn = get_connection()
    try:
        conn.execute("UPDATE recipes SET name=?, portions=? WHERE id=?",
                     (body.get("name",""), int(body.get("portions",1)), recipe_id))
        conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (recipe_id,))
        for ing in body.get("ingredients", []):
            pid = ing.get("product_id"); qty = float(ing.get("quantity", 0))
            if pid and qty > 0:
                conn.execute("INSERT INTO recipe_ingredients (recipe_id,product_id,quantity,unit) VALUES (?,?,?,?)",
                             (recipe_id, pid, qty, ing.get("unit","st")))
        conn.commit(); conn.close()
        return jsonify({"success": True})
    except Exception as exc:
        conn.close(); return jsonify({"error": str(exc)}), 500


@app.delete("/api/recipes/<int:recipe_id>")
@require_auth
def delete_recipe(recipe_id):
    conn = get_connection()
    conn.execute("DELETE FROM recipe_ingredients WHERE recipe_id=?", (recipe_id,))
    conn.execute("DELETE FROM recipes WHERE id=?", (recipe_id,))
    conn.commit(); conn.close()
    return jsonify({"success": True})


# ─────────────────────────────────────────
# Dish sales upload
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

        col_ratt  = _resolve_col(list(df.columns), col_map, "ratt",     {"rätt","ratt","recept","dish","recipe","matratt"})
        col_datum = _resolve_col(list(df.columns), col_map, "datum",    {"datum","date","dag","tid"})
        col_port  = _resolve_col(list(df.columns), col_map, "portioner",{"portioner","portions","antal_portioner","antal","qty","quantity"})

        missing = [n for n,c in [("rätt",col_ratt),("datum",col_datum),("portioner",col_port)] if not c]
        if missing:
            return jsonify({"error": f"Saknar kolumn(er): {', '.join(missing)}"}), 400

        conn = get_connection(); inserted_dish = 0; inserted_ingr = 0; skipped = []

        for _, row in df.iterrows():
            try:
                rname = str(row[col_ratt]).strip(); dval = str(row[col_datum]).strip()
                port  = int(float(str(row[col_port])))
                if not rname or rname=="nan" or port<=0: continue
            except (ValueError, TypeError): continue

            recipe = conn.execute("SELECT id FROM recipes WHERE name=?", (rname,)).fetchone()
            if not recipe:
                if rname not in skipped: skipped.append(rname)
                continue

            conn.execute("INSERT INTO dish_sales (recipe_id, date, portions) VALUES (?,?,?)",
                         (recipe["id"], dval, port))
            inserted_dish += 1

            for ing in conn.execute(
                "SELECT product_id, quantity FROM recipe_ingredients WHERE recipe_id=?", (recipe["id"],)
            ).fetchall():
                usage = ing["quantity"] * port
                conn.execute("INSERT INTO sales (product_id, date, quantity) VALUES (?,?,?)",
                             (ing["product_id"], dval, usage))
                prod = conn.execute(
                    "SELECT id, current_stock, stock_initialized, stock_sync_date FROM products WHERE id=?",
                    (ing["product_id"],)
                ).fetchone()
                if prod and prod["stock_initialized"] and dval > (prod["stock_sync_date"] or ""):
                    conn.execute(
                        "UPDATE products SET current_stock=?, stock_sync_date=?, stock_updated_at=? WHERE id=?",
                        (max(0, prod["current_stock"]-usage), dval, datetime.now().isoformat(), prod["id"])
                    )
                inserted_ingr += 1

        conn.commit(); conn.close()
        msg = f"Laddade upp {inserted_dish} rätt-poster → {inserted_ingr} ingrediensrader"
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
# Daily menu / shopping list
# ─────────────────────────────────────────

@app.post("/api/daily-menu/shopping-list")
@require_auth
def shopping_list():
    body  = request.get_json(silent=True) or {}
    items = body.get("items", [])
    if not items:
        return jsonify({"error": "Inga rätter i menyn"}), 400

    conn = get_connection()
    needs: dict[int, dict] = {}

    for item in items:
        rid  = item.get("recipe_id"); port = float(item.get("portions", 0))
        if not rid or port <= 0: continue
        for ing in conn.execute("""
            SELECT ri.product_id, ri.quantity, ri.unit, p.name, p.current_stock,
                   p.stock_initialized, p.supplier_id, s.name as supplier_name
            FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
            LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE ri.recipe_id=?
        """, (rid,)).fetchall():
            pid = ing["product_id"]
            if pid not in needs:
                needs[pid] = {"product_id": pid, "ingredient": ing["name"], "unit": ing["unit"],
                              "need": 0.0, "stock": ing["current_stock"] if ing["stock_initialized"] else None,
                              "supplier_id": ing["supplier_id"], "supplier_name": ing["supplier_name"] or "Okänd"}
            needs[pid]["need"] += ing["quantity"] * port

    result = []
    for row in needs.values():
        stock = row["stock"]; need = row["need"]
        row.update({"need": round(need, 3), "to_buy": round(max(0, need-(stock or 0)), 3),
                    "ok": (stock is not None and stock >= need)})
        result.append(row)
    result.sort(key=lambda x: (x["supplier_name"] or "", x["ingredient"]))
    conn.close()
    return jsonify(result)


@app.post("/api/daily-menu/export")
@require_auth
def shopping_list_export():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    body      = request.get_json(silent=True) or {}
    items     = body.get("items", [])
    menu_date = body.get("date", date.today().isoformat())
    conn      = get_connection()
    needs: dict[int, dict] = {}
    for item in items:
        rid = item.get("recipe_id"); port = float(item.get("portions",0))
        if not rid or port<=0: continue
        for ing in conn.execute("""
            SELECT ri.product_id, ri.quantity, ri.unit, p.name, p.current_stock,
                   p.stock_initialized, s.name as supplier_name
            FROM recipe_ingredients ri JOIN products p ON ri.product_id=p.id
            LEFT JOIN suppliers s ON p.supplier_id=s.id WHERE ri.recipe_id=?
        """, (rid,)).fetchall():
            pid = ing["product_id"]
            if pid not in needs:
                needs[pid] = {"ingredient": ing["name"], "unit": ing["unit"], "need": 0.0,
                              "stock": ing["current_stock"] if ing["stock_initialized"] else None,
                              "supplier_name": ing["supplier_name"] or "Okänd leverantör"}
            needs[pid]["need"] += ing["quantity"] * port
    conn.close()

    wb = Workbook(); ws = wb.active; ws.title = "Inköpslista"
    ws.append([f"Inköpslista — {menu_date}"]); ws.cell(1,1).font = Font(bold=True, size=13); ws.append([])
    headers = ["Leverantör","Ingrediens","Behövs","I förråd","Att köpa","Enhet"]; widths = [22,24,12,12,12,10]
    for ci,(h,w) in enumerate(zip(headers,widths),1):
        cell = ws.cell(row=3,column=ci,value=h)
        cell.font = Font(bold=True,color="FFFFFF"); cell.fill = PatternFill("solid",fgColor="7C2D12")
        cell.alignment = Alignment(horizontal="center"); ws.column_dimensions[cell.column_letter].width=w
    for ri,row in enumerate(sorted(needs.values(),key=lambda x:(x["supplier_name"],x["ingredient"])),4):
        stock = row["stock"]; need = row["need"]; to_buy = round(max(0,need-(stock or 0)),3)
        for ci,val in enumerate([row["supplier_name"],row["ingredient"],round(need,3),
                                  round(stock,3) if stock is not None else "Ej angivet",to_buy,row["unit"]],1):
            cell = ws.cell(row=ri,column=ci,value=val)
            cell.alignment = Alignment(horizontal="left" if ci<=2 else "center")
            if to_buy>0: cell.fill = PatternFill("solid",fgColor="FEF3C7")
    buf = BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f"inkopslista_{menu_date}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────
# Sales upload
# ─────────────────────────────────────────

@app.post("/api/upload")
@require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400
    f    = request.files["file"]
    name = (f.filename or "").lower()
    try:
        if name.endswith(".csv"): df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx",".xls")): df = pd.read_excel(f)
        else: return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)
        col_map = {k: v.lower().strip() for k,v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",  {"vara","produkt","product","item","artikel","ingrediens","ingredient"})
        col_datum = _resolve_col(list(df.columns), col_map, "datum", {"datum","date","dag","tid"})
        col_antal = _resolve_col(list(df.columns), col_map, "antal", {"antal","quantity","qty","amount","försäljning","sold"})

        missing = [n for n,c in [("ingrediens",col_vara),("datum",col_datum),("antal",col_antal)] if not c]
        if missing:
            return jsonify({"error": f"Kunde inte hitta kolumn(er): {', '.join(missing)}"}), 400

        upload_map: dict[str, list] = {}
        for _, row in df.iterrows():
            try:
                pname = str(row[col_vara]).strip(); dval = str(row[col_datum]).strip()
                qty   = float(str(row[col_antal]))
                if not pname or pname=="nan": continue
                upload_map.setdefault(pname,[]).append((dval, qty))
            except (ValueError, TypeError): continue

        conn = get_connection(); cur = conn.cursor(); inserted = 0; auto_deducted = []
        for pname, entries in upload_map.items():
            cur.execute("INSERT OR IGNORE INTO products (name) VALUES (?)", (pname,))
            prod = cur.execute(
                "SELECT id,current_stock,stock_initialized,stock_sync_date FROM products WHERE name=?", (pname,)
            ).fetchone()
            max_date = max(d for d,_ in entries)
            for dval, qty in entries:
                cur.execute("INSERT INTO sales (product_id,date,quantity) VALUES (?,?,?)",
                            (prod["id"],dval,qty)); inserted += 1
            if prod["stock_initialized"]:
                sync = prod["stock_sync_date"] or ""
                new_sales = sum(q for d,q in entries if d > sync)
                if new_sales > 0:
                    cur.execute(
                        "UPDATE products SET current_stock=?,stock_sync_date=?,stock_updated_at=? WHERE id=?",
                        (max(0, prod["current_stock"]-new_sales), max(max_date,sync),
                         datetime.now().isoformat(), prod["id"])
                    )
                    auto_deducted.append(f"{pname} (-{new_sales})")
            else:
                if not prod["stock_sync_date"] or max_date > prod["stock_sync_date"]:
                    cur.execute("UPDATE products SET stock_sync_date=? WHERE id=?", (max_date, prod["id"]))
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
        FROM products p LEFT JOIN suppliers s ON p.supplier_id=s.id ORDER BY p.name
    """).fetchall()
    conn.close()
    return jsonify([{"name":r["name"],"stock":r["current_stock"],"unit":r["unit"],
                     "stock_initialized":bool(r["stock_initialized"]),"stock_updated_at":r["stock_updated_at"],
                     "price":r["price"],"supplier_id":r["supplier_id"],"supplier_name":r["supplier_name"]} for r in rows])


@app.get("/api/sales/<product_name>")
@require_auth
def get_sales(product_name):
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.date, SUM(s.quantity) AS quantity FROM sales s
        JOIN products p ON s.product_id=p.id WHERE p.name=?
        GROUP BY s.date ORDER BY s.date LIMIT 90
    """, (product_name,)).fetchall()
    conn.close()
    return jsonify([{"date":r["date"],"quantity":r["quantity"]} for r in rows])


# ─────────────────────────────────────────
# Forecast & recommendations
# ─────────────────────────────────────────

@app.get("/api/forecast/<product_name>")
@require_auth
def forecast(product_name):
    days   = _days_param(7)
    result = get_forecast(product_name, days=days)
    if result is None:
        return jsonify({"error": "Hittades inte eller för lite data"}), 404
    return jsonify(result)


@app.get("/api/recommendations")
@require_auth
def recommendations():
    days  = _days_param(7)
    conn  = get_connection()
    prods = conn.execute("""
        SELECT p.id, p.name, p.supplier_id,
               s.lead_time_days, s.name as supplier_name,
               s.email as supplier_email, s.phone as supplier_phone
        FROM products p LEFT JOIN suppliers s ON p.supplier_id=s.id ORDER BY p.name
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

        conn2 = get_connection(); exp = _expiry_info(prod["id"], conn2); conn2.close()
        results.append({"product": prod["name"], "status": status,
                        "current_stock": data["current_stock"], "stock_initialized": data["stock_initialized"],
                        "order_quantity": data["order_quantity"], "days_stock": data["days_stock"],
                        "total_forecast": data["total_forecast"], "forecast_days": data["forecast_days"],
                        "expiry_critical_qty": exp["critical_qty"], "expiry_warning_qty": exp["warning_qty"],
                        "supplier_id": prod["supplier_id"], "supplier_name": prod["supplier_name"],
                        "supplier_email": prod["supplier_email"], "supplier_phone": prod["supplier_phone"],
                        "lead_time_days": lead})

    results.sort(key=lambda x: ({"urgent":0,"soon":1,"unknown":2,"plan":3}.get(x["status"],4), -x["order_quantity"]))
    return jsonify(results)


# ─────────────────────────────────────────
# Stock / förråd
# ─────────────────────────────────────────

@app.put("/api/stock/<product_name>")
@require_auth
def update_stock(product_name):
    body = request.get_json(silent=True) or {}
    try: stock = float(body.get("stock", 0))
    except (ValueError, TypeError): return jsonify({"error": "Ogiltigt förrådsvärde"}), 400
    now_iso = datetime.now().isoformat(); today = date.today().isoformat()
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
        if name.endswith(".csv"): df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx",".xls")): df = pd.read_excel(f)
        else: return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)
        col_map = {k: v.lower().strip() for k,v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",  {"vara","produkt","product","artikel","item","ingrediens"})
        col_lager = _resolve_col(list(df.columns), col_map, "lager", {"lager","stock","saldo","lagerantal","antal","quantity","forrad","forradsaldo"})
        missing = [n for n,c in [("ingrediens",col_vara),("förråd",col_lager)] if not c]
        if missing:
            return jsonify({"error": f"Saknar kolumn(er): {', '.join(missing)}"}), 400

        conn = get_connection(); updated = skipped = 0
        now_iso = datetime.now().isoformat(); today = date.today().isoformat()
        for _, row in df.iterrows():
            try:
                pname = str(row[col_vara]).strip(); stock = float(str(row[col_lager]))
                if not pname or pname=="nan": continue
            except (ValueError, TypeError): skipped += 1; continue
            r = conn.execute("""
                UPDATE products SET current_stock=?,stock_initialized=1,stock_updated_at=?,
                stock_sync_date=CASE WHEN stock_sync_date IS NULL OR stock_sync_date<? THEN ? ELSE stock_sync_date END
                WHERE name=?
            """, (stock, now_iso, today, today, pname))
            updated += r.rowcount if r.rowcount>0 else 0
            if r.rowcount==0: skipped += 1
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
# Expiry
# ─────────────────────────────────────────

@app.post("/api/upload-expiry")
@require_auth
def upload_expiry():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400
    f    = request.files["file"]
    name = (f.filename or "").lower()
    try:
        if name.endswith(".csv"): df = pd.read_csv(f, encoding="utf-8-sig")
        elif name.endswith((".xlsx",".xls")): df = pd.read_excel(f)
        else: return jsonify({"error": "Stödjer bara CSV och Excel"}), 400

        df.columns = df.columns.str.strip()
        df.rename(columns=lambda c: c.lower().strip(), inplace=True)
        col_map = {k: v.lower().strip() for k,v in
                   json.loads(request.form.get("column_map","{}") or "{}").items() if v}

        col_vara  = _resolve_col(list(df.columns), col_map, "vara",         {"vara","produkt","product","artikel","item","ingrediens"})
        col_date  = _resolve_col(list(df.columns), col_map, "utgangsdatum", {"utgångsdatum","utgangsdatum","expiry","expiry_date","datum","bäst före","best before"})
        col_antal = _resolve_col(list(df.columns), col_map, "antal",        {"antal","quantity","qty","amount"})
        missing = [n for n,c in [("ingrediens",col_vara),("utgångsdatum",col_date),("antal",col_antal)] if not c]
        if missing:
            return jsonify({"error": f"Saknar kolumn(er): {', '.join(missing)}"}), 400

        product_entries: dict[str, list] = {}
        for _, row in df.iterrows():
            try:
                pname = str(row[col_vara]).strip(); date_raw = row[col_date]
                qty   = float(str(row[col_antal]))
                if not pname or pname=="nan" or qty<=0: continue
                date_str = date_raw.strftime("%Y-%m-%d") if isinstance(date_raw, pd.Timestamp) else str(date_raw).strip()[:10]
                product_entries.setdefault(pname,[]).append((date_str,qty))
            except (ValueError, TypeError): continue

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
        d = date.fromisoformat(r["expiry_date"]); days_left = (d - today).days
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
        cell = ws.cell(row=1,column=ci,value=h)
        cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor="7C2D12")
        cell.alignment=Alignment(horizontal="center"); ws.column_dimensions[cell.column_letter].width=w
    today = date.today()
    for ri,(vara,datum,antal) in enumerate([("Mjölk 3L",(today+timedelta(2)).isoformat(),12),
                                             ("Grädde 1L",(today+timedelta(5)).isoformat(),8),
                                             ("Yoghurt 1L",(today+timedelta(14)).isoformat(),24)],2):
        ws.cell(ri,1,vara); ws.cell(ri,2,datum); ws.cell(ri,3,antal)
    buf = BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="utgangsdatum_mall.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────
# Waste (svinn)
# ─────────────────────────────────────────

@app.post("/api/waste")
@require_auth
def add_waste():
    body     = request.get_json(silent=True) or {}
    pname    = body.get("product","").strip()
    w_date   = body.get("date", date.today().isoformat())
    reason   = body.get("reason", "övrigt")
    unit     = body.get("unit", "st")
    try: quantity = float(body.get("quantity", 0))
    except (ValueError, TypeError): return jsonify({"error": "Ogiltigt antal"}), 400
    if not pname or quantity <= 0:
        return jsonify({"error": "Ingrediens och antal > 0 krävs"}), 400
    conn = get_connection()
    prod = conn.execute("SELECT id FROM products WHERE name=?", (pname,)).fetchone()
    if not prod:
        conn.close(); return jsonify({"error": "Ingrediensen finns inte"}), 404
    conn.execute("INSERT INTO waste (product_id,date,quantity,unit,reason) VALUES (?,?,?,?,?)",
                 (prod["id"], w_date, quantity, unit, reason))
    conn.commit(); conn.close()
    return jsonify({"success":True,"message":f"Registrerade {quantity} {unit} svinn av {pname}"})


@app.get("/api/waste/report")
@require_auth
def waste_report():
    conn       = get_connection()
    ninety_ago = (date.today() - timedelta(days=90)).isoformat()
    product_rows = conn.execute("""
        SELECT p.name, p.price, COALESCE(SUM(w.quantity),0) AS waste_qty,
               COALESCE(SUM(w.quantity*p.price),0) AS waste_cost
        FROM products p LEFT JOIN waste w ON w.product_id=p.id
        GROUP BY p.id, p.name, p.price HAVING waste_qty > 0 ORDER BY waste_cost DESC
    """).fetchall()
    products = []
    for r in product_rows:
        sales = conn.execute("""
            SELECT COALESCE(SUM(s.quantity),0) FROM sales s
            JOIN products p ON s.product_id=p.id WHERE p.name=? AND s.date>=?
        """, (r["name"], ninety_ago)).fetchone()[0]
        products.append({"product":r["name"],"price":r["price"],"waste_qty":r["waste_qty"],
                         "waste_cost":round(r["waste_cost"],2),"sales_qty":float(sales),
                         "waste_pct":round(r["waste_qty"]/sales*100,1) if sales>0 else None})
    monthly = conn.execute("""
        SELECT strftime('%Y-%m',w.date) as month, SUM(w.quantity) as qty, SUM(w.quantity*p.price) as cost
        FROM waste w JOIN products p ON w.product_id=p.id GROUP BY month ORDER BY month
    """).fetchall()
    weekly = conn.execute("""
        SELECT strftime('%Y-W%W',w.date) as week, SUM(w.quantity) as qty, SUM(w.quantity*p.price) as cost
        FROM waste w JOIN products p ON w.product_id=p.id
        WHERE w.date>=? GROUP BY week ORDER BY week
    """, ((date.today()-timedelta(weeks=8)).isoformat(),)).fetchall()
    by_reason = conn.execute("""
        SELECT reason, SUM(w.quantity) as qty, SUM(w.quantity*p.price) as cost
        FROM waste w JOIN products p ON w.product_id=p.id GROUP BY reason ORDER BY qty DESC
    """).fetchall()
    conn.close()
    reco = [f"{p['product']}: {p['waste_pct']}% svinn — minska inköpen" for p in products[:3]
            if p["waste_pct"] is not None and p["waste_pct"] > 5]
    return jsonify({"products":products,
                    "monthly":[{"month":r["month"],"qty":r["qty"],"cost":round(r["cost"] or 0,2)} for r in monthly],
                    "weekly": [{"week":r["week"],"qty":r["qty"],"cost":round(r["cost"] or 0,2)} for r in weekly],
                    "by_reason":[{"reason":r["reason"],"qty":r["qty"],"cost":round(r["cost"] or 0,2)} for r in by_reason],
                    "total_cost":round(sum(p["waste_cost"] for p in products),2),
                    "total_qty":sum(p["waste_qty"] for p in products),
                    "recommendations":reco})


@app.get("/api/waste/export")
@require_auth
def waste_export():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.name as product, w.date, w.quantity, w.unit, w.reason,
               ROUND(w.quantity*p.price,2) as cost
        FROM waste w JOIN products p ON w.product_id=p.id ORDER BY w.date DESC, p.name
    """).fetchall()
    conn.close()
    wb=Workbook(); ws=wb.active; ws.title="Svinnrapport"
    headers=["Ingrediens","Datum","Antal","Enhet","Orsak","Kostnad (kr)"]; widths=[26,14,10,10,14,14]
    for ci,(h,w) in enumerate(zip(headers,widths),1):
        cell=ws.cell(row=1,column=ci,value=h)
        cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor="7C2D12")
        cell.alignment=Alignment(horizontal="center"); ws.column_dimensions[cell.column_letter].width=w
    for ri,r in enumerate(rows,2):
        ws.cell(ri,1,r["product"]); ws.cell(ri,2,r["date"]); ws.cell(ri,3,r["quantity"])
        ws.cell(ri,4,r["unit"]); ws.cell(ri,5,r["reason"]); ws.cell(ri,6,r["cost"] or 0)
    buf=BytesIO(); wb.save(buf); buf.seek(0)
    return send_file(buf,as_attachment=True,download_name=f"svinnrapport_{datetime.now().strftime('%Y%m%d')}.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ─────────────────────────────────────────
# Export: recommendations
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
        status = STATUS_SV.get(data["status"], data["status"])
        conn2  = get_connection(); exp = _expiry_info(prod["id"], conn2); conn2.close()
        rows.append({"Ingrediens":prod["name"],"Nuv. forrad":data["current_stock"] if data["stock_initialized"] else "Ej angivet",
                     f"Prognos {days}d":data["total_forecast"],"Bestall antal":data["order_quantity"],
                     "Status":status,"Forradsdagar":data["days_stock"] if data["days_stock"] is not None else "-",
                     "Leverantor":prod["supplier_name"] or "-","Ledtid":prod["lead_time_days"] or 3})
    wb=Workbook(); ws=wb.active; ws.title="Inkopsrekommendationer"
    headers=["Ingrediens","Nuv. forrad",f"Prognos {days}d","Bestall antal","Status","Forradsdagar","Leverantor","Ledtid"]
    widths=[26,14,14,14,14,14,20,10]
    for ci,(h,w) in enumerate(zip(headers,widths),1):
        cell=ws.cell(row=1,column=ci,value=h)
        cell.font=Font(bold=True,color="FFFFFF"); cell.fill=PatternFill("solid",fgColor="7C2D12")
        cell.alignment=Alignment(horizontal="center"); ws.column_dimensions[cell.column_letter].width=w
    for ri,row in enumerate(rows,2):
        for ci,val in enumerate([row[h] for h in headers],1):
            cell=ws.cell(row=ri,column=ci,value=val); cell.alignment=Alignment(horizontal="left" if ci==1 else "center")
            bf=STATUS_FILL.get(row["Status"])
            if bf: cell.fill=PatternFill("solid",fgColor=bf)
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


# ─────────────────────────────────────────
# Delivery note (följesedel)
# ─────────────────────────────────────────

@app.post("/api/upload-delivery-note")
@require_auth
def upload_delivery_note():
    if "file" not in request.files:
        return jsonify({"error": "Ingen fil uppladdad"}), 400

    f       = request.files["file"]
    fname   = (f.filename or "").lower()
    allowed = (".pdf", ".jpg", ".jpeg", ".png")
    if not any(fname.endswith(ext) for ext in allowed):
        return jsonify({"error": "Stödda format: PDF, JPG, PNG"}), 400

    # Check capabilities
    is_image = fname.endswith((".jpg", ".jpeg", ".png"))
    if is_image:
        has_ocr = False
        try:
            import pytesseract; has_ocr = True  # noqa: F401
        except ImportError:
            pass
        if not has_ocr:
            return jsonify({
                "error": "OCR (pytesseract) saknas. Installera Tesseract + pytesseract, "
                         "eller ladda upp PDF-versionen av följesedeln."
            }), 400
    elif not HAS_PDFPLUMBER:
        return jsonify({
            "error": "pdfplumber saknas. Kör: pip install pdfplumber och starta om servern."
        }), 400

    try:
        raw_text, tables = _extract_delivery_text(f, f.filename or "")
    except Exception as exc:
        return jsonify({"error": f"Kunde inte läsa filen: {exc}"}), 500

    if not raw_text.strip() and not tables:
        return jsonify({"error": "Ingen text kunde extraheras ur filen. "
                                  "Prova en annan fil eller ett digitalt skannat PDF-dokument."}), 400

    conn    = get_connection()
    products = [dict(r) for r in conn.execute(
        "SELECT id, name FROM products ORDER BY name"
    ).fetchall()]
    conn.close()

    items = _parse_delivery_items(raw_text, tables, products)

    return jsonify({
        "items":      items,
        "total":      len(items),
        "matched":    sum(1 for i in items if i["matched_product"]),
        "unmatched":  sum(1 for i in items if not i["matched_product"]),
        "has_text":   bool(raw_text.strip()),
        "has_tables": bool(tables),
    })


@app.post("/api/delivery-note/apply")
@require_auth
def apply_delivery_note():
    """Apply confirmed delivery note items: add quantities to current stock."""
    body  = request.get_json(force=True) or {}
    items = body.get("items", [])       # [{product_name, quantity, unit}]
    note  = body.get("note", "")[:200]  # optional free-text reference

    if not items:
        return jsonify({"error": "Inga rader att spara"}), 400

    conn    = get_connection()
    cur     = conn.cursor()
    updated = 0
    created = 0
    today   = date.today().isoformat()

    for item in items:
        pname = str(item.get("product_name", "")).strip()
        qty   = float(item.get("quantity", 0))
        if not pname or qty <= 0:
            continue

        # Ensure product exists
        cur.execute("INSERT OR IGNORE INTO products (name) VALUES (?)", (pname,))
        prod = cur.execute(
            "SELECT id, current_stock, stock_initialized FROM products WHERE name=?", (pname,)
        ).fetchone()
        if not prod:
            continue

        new_stock = (prod["current_stock"] if prod["stock_initialized"] else 0) + qty
        cur.execute(
            "UPDATE products SET current_stock=?, stock_initialized=1, stock_updated_at=? WHERE id=?",
            (new_stock, datetime.now().isoformat(), prod["id"])
        )
        # Log as a sales entry with negative quantity (inleverans = negative consumption)
        cur.execute(
            "INSERT INTO sales (product_id, date, quantity) VALUES (?, ?, ?)",
            (prod["id"], today, -qty)   # negative = inleverans (delivery in)
        )
        if prod["stock_initialized"]:
            updated += 1
        else:
            created += 1

    conn.commit()
    conn.close()

    total = updated + created
    msg = f"Följesedel inlagd: {total} rad{'er' if total != 1 else ''} uppdaterade i förrådet"
    if note:
        msg += f" ({note})"
    return jsonify({"success": True, "message": msg, "updated": updated, "created": created})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        from waitress import serve
        print(f"Startar Waitress (produktion) på port {port} ...")
        serve(app, host="0.0.0.0", port=port, threads=4)
    except ImportError:
        print("Waitress ej installerat — faller tillbaka på Flask dev-server.")
        app.run(debug=False, host="0.0.0.0", port=port)
