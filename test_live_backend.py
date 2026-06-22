"""
Live endpoint test against production Gold Scalper backend.
"""

import urllib.request
import urllib.error
import json
import sys
import ssl

BASE_URL = "https://gold-scalper-qyhg.onrender.com"
TEST_DEVICE_ID = "test-live-scan-001"
HEADERS = {"X-Device-Id": TEST_DEVICE_ID}

results = []


def req(method, path, body=None, headers=None):
    url = BASE_URL + path
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)

    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, method=method, headers=hdrs)

    try:
        ctx = ssl.create_default_context()
        resp = urllib.request.urlopen(r, timeout=20, context=ctx)
        status = resp.status
        resp_body = resp.read().decode()
        try:
            resp_json = json.loads(resp_body)
        except Exception:
            resp_json = resp_body[:300]
        return status, resp_json, None
    except urllib.error.HTTPError as e:
        status = e.code
        try:
            resp_json = json.loads(e.read().decode())
        except Exception:
            resp_json = str(e)[:300]
        return status, resp_json, None
    except Exception as e:
        return None, None, str(e)


def test(name, method, path, body=None, headers=None, expect_status=None):
    status, resp, err = req(method, path, body, headers)
    print(f"\n{'='*60}")
    print(f"{method} {path}")
    print(f"{'='*60}")
    if err:
        print(f"  ERROR: {err}")
        results.append((name, "ERROR", err))
        return
    print(f"  HTTP {status}")
    if isinstance(resp, dict):
        print(f"  Response: {json.dumps(resp, indent=4)}")
    else:
        print(f"  Response: {resp}")

    if status < 500:
        results.append((name, "PASS", f"HTTP {status}"))
    else:
        results.append((name, "FAIL", f"HTTP {status}"))


# ── 1. Health & Status ──
test("Health Check", "GET", "/health")
test("Account Info", "GET", "/api/account")
test("Bot State", "GET", "/api/state")
test("Positions", "GET", "/api/positions")

# ── 2. Admin Bot Control ──
test("Start Bot", "POST", "/api/bot/start")
test("Stop Bot", "POST", "/api/bot/stop")
test("Close All", "POST", "/api/trades/close_all")
test("Update Settings", "POST", "/api/bot/settings",
     {"max_daily_loss": "15.00", "lot_multiplier": "3"})
test("Login MT5", "POST", "/api/bot/login",
     {"server": "Test", "account": "12345", "password": "pw"})

# ── 3. MT5 Accounts ──
test("List MT5 Accounts", "GET", "/api/accounts")
test("Add MT5 Account", "POST", "/api/accounts",
     {"label": "Test", "server": "Srv", "account": "111", "password": "pw"})
test("Delete MT5 Account", "DELETE", "/api/accounts/test123")

# ── 4. Device Accounts ──
test("List Device Accounts", "GET", "/api/device/accounts", headers=HEADERS)
test("Add Device Account", "POST", "/api/device/accounts",
     {"api_key": "test_key", "identifier": "live-scan@test.com",
      "password": "pass123", "demo": True},
     headers=HEADERS)
test("Delete Device Account", "DELETE", "/api/device/accounts/live-scan@test.com",
     headers=HEADERS)

# ── 5. Device Bot Control ──
test("Device Start Bot", "POST", "/api/device/bot/start", headers=HEADERS)
test("Device Stop Bot", "POST", "/api/device/bot/stop", headers=HEADERS)
test("Device Bot State", "GET", "/api/device/bot/state", headers=HEADERS)
test("Device Bot Logs", "GET", "/api/device/bot/logs", headers=HEADERS)

# ── 6. Device Config ──
test("Get Config", "GET", "/api/device/bot/config", headers=HEADERS)
test("Update Config", "POST", "/api/device/bot/config",
     {"LOT_MULTIPLIER": "3", "MAX_DAILY_LOSS_USD": "20.00"},
     headers=HEADERS)

# ── 7. Trades & Performance ──
test("Bot Trades", "GET", "/api/device/bot/trades", headers=HEADERS)
test("Performance", "GET", "/api/device/bot/performance", headers=HEADERS)
test("Equity Curve", "GET", "/api/device/bot/equity_curve?period=all",
     headers=HEADERS)
test("Equity Curve Yearly", "GET", "/api/device/bot/equity_curve?period=yearly",
     headers=HEADERS)
test("Equity Curve Monthly", "GET", "/api/device/bot/equity_curve?period=monthly",
     headers=HEADERS)

# ── 8. Device Close All ──
test("Device Close All", "POST", "/api/device/trades/close_all", headers=HEADERS)

# ── 9. Subscription ──
test("Get Subscription", "GET", "/api/device/subscription", headers=HEADERS)
test("Check Subscription", "POST", "/api/device/subscription/check", headers=HEADERS)

# ── 10. Payments ──
test("Init Payment", "POST", "/api/payment/initialize",
     {"email": "live-scan@test.com"}, headers=HEADERS)
test("Verify Payment", "POST", "/api/payment/verify",
     {"reference": "test_ref_live"})
test("MaxelPay Init", "POST", "/api/payment/maxelpay/init",
     {"amount": 50.0}, headers=HEADERS)
test("MaxelPay Callback", "POST", "/api/payment/maxelpay/callback",
     {"event": "payment.completed",
      "data": {"orderId": "maxel_live_001", "status": "paid", "totalPaidUsd": 50.0}})

# ── SUMMARY ──
print("\n\n" + "="*70)
print("LIVE BACKEND TEST SUMMARY")
print("="*70)
passed = sum(1 for r in results if r[1] == "PASS")
failed = sum(1 for r in results if r[1] != "PASS")
for name, status, detail in results:
    icon = "OK" if status == "PASS" else "FAIL"
    print(f"  [{icon}] {name} -> {status} ({detail})")
print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed}")
print("="*70)
