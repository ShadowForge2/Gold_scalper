"""
Comprehensive endpoint test for Gold Scalper API.
Tests ALL endpoints and reports pass/fail status.
"""

import os
import sys
import json
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch config and broker-dependent imports BEFORE importing app
import config as cfg

class MockClient:
    def __init__(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def connect(self):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    def get_account_info(self):
        return {
            "balance": 1000.0,
            "equity": 1000.0,
            "margin": 0.0,
            "free_margin": 1000.0,
            "leverage": 100,
            "server": "MockServer-Demo",
            "account": "12345",
            "name": "Mock Account",
            "currency": "USD",
        }

    def get_positions(self):
        return []

    def get_symbol_info(self, symbol):
        return {
            "symbol": symbol,
            "bid": 1950.50,
            "ask": 1950.60,
            "spread": 10,
            "digits": 2,
            "point": 0.01,
            "tick_size": 0.01,
            "tick_value": 0.1,
        }

    def get_rates(self, symbol, timeframe, count):
        return []

    def order_send(self, request):
        return {"retcode": 10009, "order": 12345}

    def order_calc_margin(self, action, symbol, volume, price):
        return 100.0

    def order_calc_profit(self, action, symbol, volume, price_open, price_close):
        return 50.0

    def login(self, account, password, server):
        return True

    def shutdown(self):
        pass


class MockBot:
    def __init__(self):
        self.state = "IDLE"
        self.client = MockClient()
        self.symbol = cfg.SYMBOL
        self.logger = MagicMock()
        self.position_manager = MagicMock()
        self.position_manager.summary.return_value = {
            "positions": [],
            "total_profit": 0.0,
            "total_volume": 0.0,
            "count": 0,
        }
        self.position_manager.refresh = MagicMock()
        self.signal_engine = MagicMock()

    def get_state_summary(self):
        return {
            "state": self.state,
            "connected": self.client.is_connected(),
            "symbol": self.symbol,
            "broker": cfg.BROKER,
            "balance": 1000.0,
            "equity": 1000.0,
            "bias": "BULLISH",
            "bias_strength": 0.65,
            "signal": {"direction": "BUY", "score": 0.55, "reason": "Test"},
            "positions": [],
            "closed_trades": [],
            "risk": {
                "daily_loss": 0.0,
                "max_daily_loss": cfg.MAX_DAILY_LOSS_USD,
                "consecutive_losses": 0,
                "max_consecutive_losses": cfg.MAX_CONSECUTIVE_LOSSES,
            },
            "scaler": {
                "starting_balance": 1000.0,
                "current_balance": 1000.0,
                "lot_multiplier": cfg.LOT_MULTIPLIER,
                "trades_today": 0,
                "max_trades_per_session": cfg.MAX_TRADES_PER_SESSION,
            },
        }

    def start(self):
        self.state = "RUNNING"

    def stop(self):
        self.state = "STOPPED"

    def update_settings(self, settings):
        pass

    def login(self, server, account, password):
        return {"success": True, "account": {"server": server, "account": account}}

    def list_accounts(self):
        return [
            {"label": "Demo1", "server": "Server-Demo", "account": "12345"},
            {"label": "Live1", "server": "Server-Live", "account": "67890"},
        ]

    def add_account(self, label, server, account, password):
        return {"success": True, "message": "Account added"}

    def remove_account(self, account_id):
        return {"success": True, "message": "Account removed"}

    async def emergency_close(self):
        return 0

    async def initialize(self):
        self.client.connect()

    async def shutdown(self):
        self.client.disconnect()

    async def run(self):
        pass


class MockBotPool:
    def __init__(self):
        self._bots = {}
        self._logs = {}
        self._states = {}
        self._running = {}

    def start(self, identifier, api_key, password, demo=True):
        self._running[identifier] = True
        self._bots[identifier] = True
        return {"success": True}

    def stop(self, identifier):
        self._running.pop(identifier, None)
        return {"success": True}

    def is_running(self, identifier):
        return self._running.get(identifier, False)

    def get_state(self, identifier):
        return {
            "running": self.is_running(identifier),
            "account": {
                "balance": 1000.0,
                "equity": 1000.0,
                "margin": 0.0,
                "free_margin": 1000.0,
            },
            "bot": {
                "state": "RUNNING" if self.is_running(identifier) else "STOPPED",
                "positions": {"positions": [], "total_profit": 0, "count": 0},
                "closed_trades": [],
                "scaler": {"starting_balance": 1000.0, "current_balance": 1000.0},
            },
        }

    def get_logs(self, identifier):
        return self._logs.get(identifier, [])

    def add_log(self, identifier, message, level="INFO"):
        if identifier not in self._logs:
            self._logs[identifier] = []
        self._logs[identifier].append(
            {"time": datetime.utcnow().isoformat(), "message": message, "level": level}
        )

    def add_log_once(self, identifier, message, level="INFO"):
        self.add_log(identifier, message, level)

    def update_settings(self, identifier, settings):
        pass

    async def emergency_close(self, identifier):
        return 0

    def stop_all(self):
        self._running.clear()


# Mock for subscription module
import app.subscription as sub_mod
sub_mod.ensure_device = AsyncMock()
sub_mod.ensure_device.return_value = {
    "device_id": "test-device-123",
    "accounts": [
        {
            "api_key": "test_key",
            "identifier": "test@example.com",
            "password": "test_pass",
            "demo": True,
        }
    ],
}
sub_mod.get_device = AsyncMock()
sub_mod.get_device.return_value = {
    "device_id": "test-device-123",
    "accounts": [
        {
            "api_key": "test_key",
            "identifier": "test@example.com",
            "password": "test_pass",
            "demo": True,
        }
    ],
}
sub_mod.add_account = AsyncMock()
sub_mod.add_account.return_value = True
sub_mod.remove_account = AsyncMock()
sub_mod.remove_account.return_value = True
sub_mod.restore_device_by_capital_id = AsyncMock()
sub_mod.start_trial = AsyncMock()
sub_mod.get_subscription = AsyncMock()
sub_mod.get_subscription.return_value = {
    "demo": True,
    "trial_active": True,
    "days_remaining": 25,
    "can_trade": True,
    "subscribed": False,
    "due_amount": 0.0,
    "unpaid_fees": 0.0,
    "is_new": False,
    "current_month_profit": 0.0,
    "current_month_fee": 0.0,
    "trial_end": (datetime.utcnow() + timedelta(days=25)).isoformat(),
}
sub_mod.can_start_live = AsyncMock()
sub_mod.can_start_live.return_value = True
sub_mod.initialize_payment = MagicMock()
sub_mod.initialize_payment.return_value = {
    "authorization_url": "https://paystack.com/ref/test123",
    "reference": "test_ref_123",
    "access_code": "test_code_123",
}
sub_mod.verify_payment = AsyncMock()
sub_mod.verify_payment.return_value = {
    "identifier": "test@example.com",
    "amount": 5000.0,
    "status": "success",
}
sub_mod.create_maxelpay_payment = MagicMock()
sub_mod.create_maxelpay_payment.return_value = {
    "data": {"checkoutUrl": "https://maxelpay.com/pay/test123"},
    "orderId": "maxel_test_123",
}
sub_mod._maxelpay_register_order = MagicMock()
sub_mod._maxelpay_get_identifier = MagicMock()
sub_mod._maxelpay_get_identifier.return_value = "test@example.com"
sub_mod._get_sub_record = AsyncMock()
sub_mod._get_sub_record.return_value = {
    "paid_amount": 0.0,
    "subscribed": False,
    "monthly_periods": [],
}
sub_mod._save_sub_record = AsyncMock()

# Now import after mocks are in place
from fastapi.testclient import TestClient
from app.api import create_app

# Create app with mock objects
mock_bot = MockBot()
mock_pool = MockBotPool()
mock_db_check = lambda: True
app = create_app(mock_bot, bot_pool=mock_pool, db_check=mock_db_check)
client = TestClient(app)

TEST_DEVICE_ID = "test-device-abc-123"
TEST_HEADERS = {"X-Device-Id": TEST_DEVICE_ID}


def test_endpoint(name, method, url, **kwargs):
    """Test a single endpoint and return result."""
    headers = kwargs.pop("headers", {})
    if "X-Device-Id" in str(kwargs.get("json", {})) or any(
        k for k in headers if "device" in k.lower()
    ):
        pass  # headers already specified

    result = {"name": name, "method": method, "url": url, "status": None, "error": None}

    body = kwargs.pop("json", None)

    try:
        if method == "GET":
            resp = client.get(url, headers=headers, **kwargs)
        elif method == "POST":
            resp = client.post(url, headers=headers, json=body, **kwargs)
        elif method == "DELETE":
            resp = client.delete(url, headers=headers, **kwargs)
        else:
            result["error"] = f"Unknown method: {method}"
            result["status"] = "FAIL"
            return result

        result["http_status"] = resp.status_code
        try:
            result["response"] = resp.json()
        except Exception:
            result["response"] = resp.text[:200]

        if resp.status_code < 500:
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["error"] = f"HTTP {resp.status_code}"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)

    return result


def print_results(results):
    """Print formatted test results."""
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] != "PASS"]

    print("=" * 80)
    print(f"{'ENDPOINT TEST RESULTS':^80}")
    print("=" * 80)

    for r in results:
        status_icon = "[OK]" if r["status"] == "PASS" else "[FAIL]"
        print(f"\n{status_icon} [{r['status']}] {r['method']} {r['url']}")
        if r.get("http_status"):
            print(f"   HTTP {r['http_status']}")
        if r.get("error"):
            print(f"   ERROR: {r['error']}")
        if r.get("response"):
            resp_str = json.dumps(r["response"], indent=2)
            if len(resp_str) > 300:
                resp_str = resp_str[:300] + "..."
            print(f"   Response: {resp_str}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {len(passed)} passed, {len(failed)} failed out of {len(results)}")
    print("=" * 80)


def run_all_tests():
    results = []

    # ── Health & Status ──
    results.append(test_endpoint("Health Check", "GET", "/health"))
    results.append(test_endpoint("Get Account Info", "GET", "/api/account"))
    results.append(test_endpoint("Get Bot State", "GET", "/api/state"))
    results.append(test_endpoint("Get Positions", "GET", "/api/positions"))

    # ── Admin Bot Control ──
    results.append(test_endpoint("Start Bot", "POST", "/api/bot/start"))
    results.append(test_endpoint("Stop Bot", "POST", "/api/bot/stop"))
    results.append(test_endpoint("Close All Trades", "POST", "/api/trades/close_all"))
    results.append(
        test_endpoint(
            "Update Settings",
            "POST",
            "/api/bot/settings",
            json={"max_daily_loss": "15.00", "lot_multiplier": "3"},
        )
    )
    results.append(
        test_endpoint(
            "Login MT5",
            "POST",
            "/api/bot/login",
            json={"server": "TestServer", "account": "12345", "password": "secret"},
        )
    )

    # ── MT5 Account Management ──
    results.append(test_endpoint("List MT5 Accounts", "GET", "/api/accounts"))
    results.append(
        test_endpoint(
            "Add MT5 Account",
            "POST",
            "/api/accounts",
            json={"label": "Test", "server": "Srv", "account": "111", "password": "pw"},
        )
    )
    results.append(test_endpoint("Delete MT5 Account", "DELETE", "/api/accounts/test123"))

    # ── Device Account Management ──
    results.append(
        test_endpoint("List Device Accounts", "GET", "/api/device/accounts", headers=TEST_HEADERS)
    )
    results.append(
        test_endpoint(
            "Add Device Account",
            "POST",
            "/api/device/accounts",
            headers=TEST_HEADERS,
            json={
                "api_key": "new_key",
                "identifier": "new@test.com",
                "password": "new_pass",
                "demo": True,
            },
        )
    )
    results.append(
        test_endpoint(
            "Delete Device Account",
            "DELETE",
            "/api/device/accounts/new@test.com",
            headers=TEST_HEADERS,
        )
    )

    # ── Device Bot Control ──
    results.append(
        test_endpoint("Device Start Bot", "POST", "/api/device/bot/start", headers=TEST_HEADERS)
    )
    results.append(
        test_endpoint("Device Stop Bot", "POST", "/api/device/bot/stop", headers=TEST_HEADERS)
    )
    results.append(
        test_endpoint("Device Bot State", "GET", "/api/device/bot/state", headers=TEST_HEADERS)
    )
    results.append(
        test_endpoint("Device Bot Logs", "GET", "/api/device/bot/logs", headers=TEST_HEADERS)
    )

    # ── Device Bot Config ──
    results.append(
        test_endpoint("Get Device Config", "GET", "/api/device/bot/config", headers=TEST_HEADERS)
    )
    results.append(
        test_endpoint(
            "Update Device Config",
            "POST",
            "/api/device/bot/config",
            headers=TEST_HEADERS,
            json={"LOT_MULTIPLIER": "3", "MAX_DAILY_LOSS_USD": "20.00"},
        )
    )

    # ── Device Bot Trades & Performance ──
    results.append(
        test_endpoint("Device Bot Trades", "GET", "/api/device/bot/trades", headers=TEST_HEADERS)
    )
    results.append(
        test_endpoint(
            "Device Bot Performance", "GET", "/api/device/bot/performance", headers=TEST_HEADERS
        )
    )
    results.append(
        test_endpoint(
            "Device Bot Equity Curve", "GET", "/api/device/bot/equity_curve?period=all", headers=TEST_HEADERS
        )
    )
    results.append(
        test_endpoint(
            "Device Bot Equity Curve Yearly",
            "GET",
            "/api/device/bot/equity_curve?period=yearly",
            headers=TEST_HEADERS,
        )
    )
    results.append(
        test_endpoint(
            "Device Bot Equity Curve Monthly",
            "GET",
            "/api/device/bot/equity_curve?period=monthly",
            headers=TEST_HEADERS,
        )
    )

    # ── Device Close All ──
    results.append(
        test_endpoint(
            "Device Close All", "POST", "/api/device/trades/close_all", headers=TEST_HEADERS
        )
    )

    # ── Subscription ──
    results.append(
        test_endpoint(
            "Get Subscription", "GET", "/api/device/subscription", headers=TEST_HEADERS
        )
    )
    results.append(
        test_endpoint(
            "Check Subscription",
            "POST",
            "/api/device/subscription/check",
            headers=TEST_HEADERS,
        )
    )

    # ── Payment ──
    results.append(
        test_endpoint(
            "Initialize Payment",
            "POST",
            "/api/payment/initialize",
            headers=TEST_HEADERS,
            json={"email": "test@example.com"},
        )
    )
    results.append(
        test_endpoint(
            "Verify Payment",
            "POST",
            "/api/payment/verify",
            json={"reference": "test_ref_123"},
        )
    )
    results.append(
        test_endpoint(
            "MaxelPay Init",
            "POST",
            "/api/payment/maxelpay/init",
            headers=TEST_HEADERS,
            json={"amount": 50.0},
        )
    )
    results.append(
        test_endpoint(
            "MaxelPay Callback",
            "POST",
            "/api/payment/maxelpay/callback",
            json={
                "event": "payment.completed",
                "data": {
                    "orderId": "maxel_test_123",
                    "status": "paid",
                    "totalPaidUsd": 50.0,
                },
            },
        )
    )

    print_results(results)
    return results


if __name__ == "__main__":
    run_all_tests()
