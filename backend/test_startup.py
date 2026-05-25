"""Quick smoke test – run from backend/ directory."""
import sys

print("Testing imports...")
import database
import forecast
import app
print("  imports OK")

print("Testing DB migration...")
from database import init_db
init_db()
print("  init_db OK")

print("Testing Swedish holidays...")
from forecast import _easter, _midsommar_saturday, _black_friday, _swedish_holidays
e = _easter(2025)
m = _midsommar_saturday(2025)
b = _black_friday(2025)
print(f"  Easter 2025:      {e}")
print(f"  Midsommar 2025:   {m}")
print(f"  Black Friday 2025:{b}")
h = _swedish_holidays(range(2024, 2027))
print(f"  Holiday rows:     {len(h)}")

print("Generating demo data...")
from forecast import generate_demo_data, get_forecast
generate_demo_data()
print("  demo data OK")

print("Testing forecast (14 days)...")
fc = get_forecast("Mjölk 3L", days=14)
assert fc is not None, "Forecast returned None"
assert fc["forecast_days"] == 14
assert fc["stock_initialized"] is True
assert len(fc["forecast"]) == 14
print(f"  order_qty={fc['order_quantity']}  status={fc['status']}  method={fc['method']}")

print("Testing forecast (30 days)...")
fc30 = get_forecast("Grädde 1L", days=30)
assert fc30 is not None
assert fc30["forecast_days"] == 30
print(f"  order_qty={fc30['order_quantity']}  status={fc30['status']}")

print("\nALL TESTS PASSED")
