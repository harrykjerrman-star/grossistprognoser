import time
from datetime import datetime, timedelta

try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

SMHI_FORECAST_URL = (
    "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/"
    "version/2/geotype/point/lon/{lon}/lat/{lat}/data.json"
)

SWEDISH_CITIES: dict[str, tuple[float, float]] = {
    "stockholm":    (59.331, 18.065),
    "göteborg":     (57.707, 11.967),
    "malmö":        (55.605, 13.003),
    "uppsala":      (59.858, 17.645),
    "västerås":     (59.611, 16.545),
    "örebro":       (59.274, 15.213),
    "linköping":    (58.411, 15.621),
    "helsingborg":  (56.046, 12.694),
    "jönköping":    (57.782, 14.161),
    "norrköping":   (58.594, 16.188),
    "lund":         (55.704, 13.191),
    "umeå":         (63.826, 20.266),
    "gävle":        (60.675, 17.142),
    "borås":        (57.721, 12.940),
    "sundsvall":    (62.391, 17.307),
    "eskilstuna":   (59.371, 16.510),
    "karlstad":     (59.374, 13.504),
    "växjö":        (56.877, 14.809),
    "halmstad":     (56.674, 12.858),
    "luleå":        (65.584, 22.157),
    "kalmar":       (56.661, 16.366),
    "falun":        (60.605, 15.632),
    "östersund":    (63.176, 14.636),
    "trollhättan":  (58.284, 12.288),
    "skövde":       (58.388, 13.845),
    "nyköping":     (58.753, 17.009),
}

_weather_cache: dict[str, tuple[float, list]] = {}
_CACHE_TTL = 3600  # 1 hour


def get_city_coordinates(city: str) -> tuple[float, float] | None:
    key = city.lower().strip()
    if key in SWEDISH_CITIES:
        return SWEDISH_CITIES[key]
    for name, coords in SWEDISH_CITIES.items():
        if key in name or name in key:
            return coords
    return None


def _wsymb_to_emoji(wsymb: int) -> str:
    mapping = {
        1: "☀️", 2: "🌤", 3: "⛅", 4: "🌥", 5: "☁️", 6: "☁️",
        7: "🌫", 8: "🌦", 9: "🌧", 10: "🌧", 11: "⛈",
        12: "🌨", 13: "🌨", 14: "🌨", 15: "🌨", 16: "❄️", 17: "❄️",
        18: "🌧", 19: "🌧", 20: "🌧", 21: "⛈",
        22: "🌨", 23: "🌨", 24: "🌨", 25: "❄️", 26: "❄️", 27: "❄️",
    }
    return mapping.get(int(wsymb), "🌡")


def fetch_smhi_forecast(lat: float, lon: float) -> list[dict]:
    """Fetch 7-day daily forecast from SMHI. Cached for 1 hour."""
    cache_key = f"{lat:.2f},{lon:.2f}"
    now = time.time()
    if cache_key in _weather_cache:
        cached_at, data = _weather_cache[cache_key]
        if now - cached_at < _CACHE_TTL:
            return data

    if not HAS_REQUESTS:
        return []

    try:
        url = SMHI_FORECAST_URL.format(lat=f"{lat:.4f}", lon=f"{lon:.4f}")
        resp = _requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []

        raw = resp.json()
        hourly: list[dict] = []

        for ts in raw.get("timeSeries", []):
            dt_str = ts.get("validTime", "")
            if not dt_str:
                continue
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            params = {p["name"]: p["values"][0] for p in ts.get("parameters", []) if p.get("values")}
            temp   = params.get("t")
            precip = float(params.get("pmean", 0) or 0)
            wsymb  = params.get("Wsymb2", 1) or 1
            if temp is None:
                continue
            hourly.append({"dt": dt, "temp": float(temp), "precip": precip, "wsymb": wsymb})

        from collections import defaultdict
        days_map: dict = defaultdict(list)
        for h in hourly:
            days_map[h["dt"].strftime("%Y-%m-%d")].append(h)

        result = []
        today = datetime.now().date()
        for i in range(7):
            day = today + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            if day_str in days_map:
                hrs    = days_map[day_str]
                temps  = [h["temp"] for h in hrs]
                precips = [h["precip"] for h in hrs]
                wsymbs  = [h["wsymb"] for h in hrs]
                noon_symb = wsymbs[len(wsymbs) // 2] if wsymbs else 1
                result.append({
                    "date":          day_str,
                    "temp_max":      round(max(temps), 1),
                    "temp_min":      round(min(temps), 1),
                    "temp_avg":      round(sum(temps) / len(temps), 1),
                    "precipitation": round(sum(precips), 1),
                    "symbol":        int(noon_symb),
                    "emoji":         _wsymb_to_emoji(noon_symb),
                })

        _weather_cache[cache_key] = (now, result)
        return result
    except Exception:
        return []


def get_weather_for_city(city: str) -> list[dict]:
    """Get 7-day weather forecast for a named Swedish city."""
    if not city:
        return []
    coords = get_city_coordinates(city)
    if not coords:
        return []
    return fetch_smhi_forecast(coords[0], coords[1])


def weather_multipliers(weather_days: list[dict]) -> dict[str, float]:
    """
    Return {date_str: multiplier} based on weather conditions.
    Heavy rain lowers expected footfall; extreme temps nudge it slightly.
    """
    result: dict[str, float] = {}
    for day in weather_days:
        mult = 1.0
        if day["precipitation"] > 10:
            mult *= 0.93
        elif day["precipitation"] > 5:
            mult *= 0.97
        if day["temp_avg"] < -10:
            mult *= 1.08   # very cold → people stock up
        elif day["temp_avg"] > 28:
            mult *= 1.04   # heat wave → cold-item boost
        if round(mult, 3) != 1.0:
            result[day["date"]] = round(mult, 3)
    return result
