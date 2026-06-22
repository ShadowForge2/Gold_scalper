"""
Test trade opening and closing endpoints on the live Gold Scalper backend.
"""
import urllib.request, urllib.error, json, ssl, time, sys

BASE = "https://gold-scalper-qyhg.onrender.com"
DEVICE_ID = "trade-test-device-001"

ssl_ctx = ssl.create_default_context()

def req(method, path, body=None, headers=None):
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(BASE + path, data=data, method=method, headers=hdrs)
    try:
        resp = urllib.request.urlopen(r, timeout=25, context=ssl_ctx)
        return resp.status, json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode()), None
        except:
            return e.code, str(e), None
    except Exception as e:
        return None, None, str(e)

def test(name, method, path, body=None, headers=None):
    status, resp, err = req(method, path, body, headers)
    status_str = f"HTTP {status}" if status else "ERROR"
    ok = status is not None and status < 500
    icon = "OK" if ok else "FAIL"
    print(f"\n[{icon}] {method} {path} ({status_str})")
    if err:
        print(f"       ERROR: {err}")
    elif isinstance(resp, dict):
        # Print relevant fields, skip verbose logs
        r = {k:v for k,v in resp.items() if k != "last_logs"}
        print(f"       {json.dumps(r, indent=2)[:500]}")
    else:
        print(f"       {resp}")
    return status, resp, err

# ── Step 1: Check current state (is market open? connected?) ──
print("="*70)
print("STEP 1: CHECK CURRENT STATE & HEALTH")
print("="*70)

s, health, _ = test("Health", "GET", "/health")
print(f"\n  Market open? {health.get('state') != 'MARKET_CLOSED' if health else 'unknown'}")
print(f"  Connected: {health.get('connected') if health else 'unknown'}")

s, state, _ = test("Bot State", "GET", "/api/state")
if state and "positions" in state:
    pos = state.get("positions", {})
    print(f"  Open positions: {pos.get('open_count', '?')}")
    print(f"  Daily PnL: {pos.get('daily_pnl', '?')}")

s, acct, _ = test("Account Info", "GET", "/api/account")
if acct:
    print(f"  Balance: ${acct.get('balance', '?')}")
    print(f"  Equity: ${acct.get('equity', '?')}")

# ── Step 2: Start the bot ──
print("\n" + "="*70)
print("STEP 2: START BOT (admin)")
print("="*70)
test("Start Bot", "POST", "/api/bot/start")
time.sleep(2)

# Check state after start
s, state2, _ = test("State after start", "GET", "/api/state")
if state2:
    print(f"  Bot state: {state2.get('state', '?')}")
    risk = state2.get("risk", {})
    if risk:
        print(f"  Cooldown: {risk.get('cooldown_active', '?')}")
        print(f"  Session trades: {risk.get('session_trades', '?')}")
        print(f"  Daily PnL: {risk.get('daily_pnl', '?')}")

# ── Step 3: Check positions ──
print("\n" + "="*70)
print("STEP 3: CHECK POSITIONS")
print("="*70)
test("Get Positions", "GET", "/api/positions")

# ── Step 4: Close all trades ──
print("\n" + "="*70)
print("STEP 4: CLOSE ALL TRADES (admin)")
print("="*70)
test("Close All Admin", "POST", "/api/trades/close_all")

# Verify after close
time.sleep(1)
test("Positions after close", "GET", "/api/positions")
test("State after close", "GET", "/api/state")

# ── Step 5: Stop the bot ──
print("\n" + "="*70)
print("STEP 5: STOP BOT")
print("="*70)
test("Stop Bot", "POST", "/api/bot/stop")

# ── Step 6: Device-based trade endpoints ──
print("\n" + "="*70)
print("STEP 6: DEVICE TRADE ENDPOINTS")
print("="*70)

# First register device with the real Capital.com credentials from env
CAPITAL_API_KEY = "fSMMyfGbqGcwh6OX"
CAPITAL_IDENTIFIER = "bashirabdulganiyy9@gmail.com"
CAPITAL_PASSWORD = "Thunder@g2"

print("\n--- Adding Capital.com account to device ---")
s, add_resp, _ = test("Add Device Account", "POST", "/api/device/accounts",
    {"api_key": CAPITAL_API_KEY, "identifier": CAPITAL_IDENTIFIER,
     "password": CAPITAL_PASSWORD, "demo": True},
    headers={"X-Device-Id": DEVICE_ID})

if s == 200:
    print("\n--- Starting device bot ---")
    s2, start_resp, _ = test("Device Start Bot", "POST", "/api/device/bot/start",
        headers={"X-Device-Id": DEVICE_ID})
    time.sleep(3)

    print("\n--- Device bot state ---")
    test("Device Bot State", "GET", "/api/device/bot/state",
        headers={"X-Device-Id": DEVICE_ID})

    print("\n--- Device bot trades ---")
    test("Device Bot Trades", "GET", "/api/device/bot/trades",
        headers={"X-Device-Id": DEVICE_ID})

    print("\n--- Device bot performance ---")
    test("Device Bot Performance", "GET", "/api/device/bot/performance",
        headers={"X-Device-Id": DEVICE_ID})

    print("\n--- Device close all trades ---")
    test("Device Close All", "POST", "/api/device/trades/close_all",
        headers={"X-Device-Id": DEVICE_ID})

    print("\n--- State after device close ---")
    test("Device Bot State After", "GET", "/api/device/bot/state",
        headers={"X-Device-Id": DEVICE_ID})

    print("\n--- Stopping device bot ---")
    test("Device Stop Bot", "POST", "/api/device/bot/stop",
        headers={"X-Device-Id": DEVICE_ID})

    print("\n--- Cleanup: remove device account ---")
    test("Delete Device Account", "DELETE", f"/api/device/accounts/{CAPITAL_IDENTIFIER}",
        headers={"X-Device-Id": DEVICE_ID})
else:
    print("\n  Skipping device trade tests - could not register account")

print("\n" + "="*70)
print("TRADE ENDPOINT TESTS COMPLETE")
print("="*70)
