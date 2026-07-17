import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import httpx

from app import database as db_mod

logger = logging.getLogger("GoldScalper")

PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET_KEY", "")
MAXELPAY_API_KEY = os.getenv("MAXELPAY_API_KEY", "")
MAXELPAY_BASE = "https://api.maxelpay.com/api/v1"

_firebase_app = None

def _get_firebase_app():
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app
    try:
        import firebase_admin
        from firebase_admin import credentials
        cred_json_str = os.getenv("FIREBASE_CREDENTIALS_JSON", "")
        cred_path = os.getenv("FIREBASE_CREDENTIALS", "")
        if cred_json_str:
            import tempfile
            cred_data = json.loads(cred_json_str)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(cred_data, f)
                f.flush()
                cred = credentials.Certificate(f.name)
                _firebase_app = firebase_admin.initialize_app(cred)
            os.unlink(f.name)
            logger.info("Firebase Admin initialized from env var")
        elif cred_path and os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            _firebase_app = firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin initialized from %s", cred_path)
        elif firebase_admin._DEFAULT_APP_NAME not in firebase_admin._apps:
            _firebase_app = firebase_admin.initialize_app()
            logger.info("Firebase Admin initialized with default credentials")
        else:
            _firebase_app = firebase_admin.get_app()
    except Exception as e:
        logger.warning("Firebase Admin init failed: %s", e)
        _firebase_app = None
    return _firebase_app


async def _send_fcm_push(identifier: str, title: str, body: str, data: Optional[Dict] = None):
    try:
        app = _get_firebase_app()
        if app is None:
            return
        from firebase_admin import messaging as fcm
        rows = await db_mod.database.fetch_all(
            """SELECT f.fcm_token FROM fcm_tokens f
               JOIN accounts a ON a.device_id = f.device_id
               WHERE a.identifier = :ident""",
            {"ident": identifier},
        )
        if not rows:
            return
        tokens = [r["fcm_token"] for r in rows if r.get("fcm_token")]
        if not tokens:
            return
        message = fcm.MulticastMessage(
            notification=fcm.Notification(title=title, body=body),
            data={k: str(v) for k, v in (data or {}).items()},
            tokens=tokens,
        )
        response = fcm.send_each(message)
        logger.info("FCM push sent: %d success, %d failed out of %d",
                     response.success_count, response.failure_count, len(tokens))
    except Exception as e:
        logger.debug("FCM push failed: %s", e)


# ── Device tracking (async DB) ──────────────────────────────────

_device_cache: Dict[str, Dict] = {}
_device_cache_ts: Dict[str, float] = {}
_DEVICE_CACHE_TTL = 5.0


async def _cached_device(device_id: str, force: bool = False) -> Optional[Dict]:
    now = time.time()
    if not force:
        cached = _device_cache.get(device_id)
        ts = _device_cache_ts.get(device_id, 0)
        if cached is not None and now - ts < _DEVICE_CACHE_TTL:
            return cached
    row = await db_mod.database.fetch_one(
        "SELECT * FROM devices WHERE device_id = :did", {"did": device_id}
    )
    if row is None:
        _device_cache[device_id] = None
        _device_cache_ts[device_id] = now
        return None
    rows = await db_mod.database.fetch_all(
        "SELECT * FROM accounts WHERE device_id = :did", {"did": device_id}
    )
    data = dict(row) | {"accounts": [dict(a) for a in rows]}
    _device_cache[device_id] = data
    _device_cache_ts[device_id] = now
    return data


async def ensure_device(device_id: str) -> Dict:
    row = await db_mod.database.fetch_one(
        "SELECT * FROM devices WHERE device_id = :did", {"did": device_id}
    )
    if row:
        account_rows = await db_mod.database.fetch_all(
            "SELECT * FROM accounts WHERE device_id = :did", {"did": device_id}
        )
        return dict(row) | {"accounts": [dict(a) for a in account_rows]}

    now = datetime.utcnow().isoformat()
    await db_mod.database.execute(
        "INSERT INTO devices (device_id, first_seen) VALUES (:did, :fs)",
        {"did": device_id, "fs": now},
    )
    return {"device_id": device_id, "first_seen": now, "accounts": []}


async def get_device(device_id: str, force: bool = False) -> Optional[Dict]:
    return await _cached_device(device_id, force=force)


async def get_accounts(device_id: str) -> List[Dict]:
    rows = await db_mod.database.fetch_all(
        "SELECT * FROM accounts WHERE device_id = :did", {"did": device_id}
    )
    return [dict(r) for r in rows]


async def add_account(device_id: str, api_key: str, identifier: str, password: str, demo: bool = True) -> bool:
    _device_cache.pop(device_id, None)
    await ensure_device(device_id)
    await db_mod.database.execute(
        """INSERT INTO accounts (device_id, api_key, identifier, password, demo)
           VALUES (:did, :ak, :id, :pw, :dm)
           ON CONFLICT (device_id, identifier)
           DO UPDATE SET api_key = :ak2, password = :pw2, demo = :dm2""",
        {"did": device_id, "ak": api_key, "id": identifier, "pw": password, "dm": int(demo),
         "ak2": api_key, "pw2": password, "dm2": int(demo)},
    )
    return True


async def remove_account(device_id: str, identifier: str) -> bool:
    _device_cache.pop(device_id, None)
    result = await db_mod.database.execute(
        "DELETE FROM accounts WHERE device_id = :did AND identifier = :id",
        {"did": device_id, "id": identifier},
    )
    # asyncpg returns status string "DELETE N", aiosqlite returns int or None
    if isinstance(result, int):
        return result > 0
    # asyncpg: parse "DELETE N"
    if isinstance(result, str) and result.startswith("DELETE"):
        parts = result.split()
        return len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0
    return bool(result)


async def set_account_active(identifier: str, active: bool):
    await db_mod.database.execute(
        "UPDATE accounts SET active = :act WHERE identifier = :id",
        {"act": int(active), "id": identifier},
    )


async def get_account_by_identifier(identifier: str) -> Optional[Dict]:
    row = await db_mod.database.fetch_one(
        "SELECT api_key, identifier, password, demo, active FROM accounts WHERE identifier = :id LIMIT 1",
        {"id": identifier},
    )
    return dict(row) if row else None


async def get_active_accounts() -> List[Dict]:
    rows = await db_mod.database.fetch_all(
        """SELECT DISTINCT identifier, api_key, password, demo
           FROM accounts WHERE active = 1"""
    )
    return [dict(r) for r in rows]


async def restore_device_by_capital_id(identifier: str, new_device_id: str) -> Optional[Dict]:
    _device_cache.pop(new_device_id, None)
    old_device = await db_mod.database.fetch_one(
        """SELECT DISTINCT a.device_id FROM accounts a
           WHERE a.identifier = :id AND a.device_id != :ndid""",
        {"id": identifier, "ndid": new_device_id},
    )
    if old_device is None:
        return None

    old_did = old_device["device_id"]
    _device_cache.pop(old_did, None)

    existing = await db_mod.database.fetch_one(
        "SELECT 1 FROM devices WHERE device_id = :did", {"did": new_device_id}
    )

    if existing is None:
        now = datetime.utcnow().isoformat()
        await db_mod.database.execute(
            "UPDATE devices SET device_id = :ndid, restored_from = :old WHERE device_id = :odid",
            {"ndid": new_device_id, "old": old_did, "odid": old_did},
        )
        await db_mod.database.execute(
            "UPDATE accounts SET device_id = :ndid WHERE device_id = :odid",
            {"ndid": new_device_id, "odid": old_did},
        )
    else:
        await db_mod.database.execute(
            """INSERT INTO accounts (device_id, api_key, identifier, password, demo, active)
               SELECT :ndid, api_key, identifier, password, demo, active FROM accounts
               WHERE device_id = :odid AND identifier NOT IN (
                   SELECT identifier FROM accounts WHERE device_id = :ndid2
               )""",
            {"ndid": new_device_id, "odid": old_did, "ndid2": new_device_id},
        )
        await db_mod.database.execute("DELETE FROM devices WHERE device_id = :odid", {"odid": old_did})
        await db_mod.database.execute("DELETE FROM accounts WHERE device_id = :odid", {"odid": old_did})

    return await get_device(new_device_id)


# ── Subscription / Trial / 30-Day Profit Tracking ─────────────────

PERIOD_DAYS = 30


async def _get_sub_record(identifier: str) -> Optional[Dict]:
    row = await db_mod.database.fetch_one(
        "SELECT * FROM subscriptions WHERE identifier = :id", {"id": identifier}
    )
    if row is None:
        return None
    record = dict(row)
    period_rows = await db_mod.database.fetch_all(
        "SELECT * FROM monthly_periods WHERE identifier = :id ORDER BY period_start",
        {"id": identifier},
    )
    record["monthly_periods"] = [dict(p) for p in period_rows]
    return record


async def _save_sub_record(record: Dict):
    ident = record["identifier"]
    await db_mod.database.execute(
        """INSERT INTO subscriptions (identifier, first_connected_at, trial_end, subscribed, subscription_end, paid_amount)
           VALUES (:id, :fca, :te, :sub, :se, :pa)
           ON CONFLICT (identifier)
           DO UPDATE SET first_connected_at = :fca2, trial_end = :te2, subscribed = :sub2,
                         subscription_end = :se2, paid_amount = :pa2""",
        {"id": ident, "fca": record.get("first_connected_at"), "te": record.get("trial_end"),
         "sub": int(record.get("subscribed", False)), "se": record.get("subscription_end"),
         "pa": record.get("paid_amount", 0.0),
         "fca2": record.get("first_connected_at"), "te2": record.get("trial_end"),
         "sub2": int(record.get("subscribed", False)), "se2": record.get("subscription_end"),
         "pa2": record.get("paid_amount", 0.0)},
    )

    for p in record.get("monthly_periods", []):
        await db_mod.database.execute(
            """INSERT INTO monthly_periods (identifier, period_start, period_end, starting_balance,
               ending_balance, cumulative_profit, fee_15pct, fee_paid, paid_at)
               VALUES (:id, :ps, :pe, :sb, :eb, :cp, :fp, :fpd, :pa)
               ON CONFLICT (identifier, period_start)
               DO UPDATE SET period_end = :pe2, ending_balance = :eb2, cumulative_profit = :cp2,
                             fee_15pct = :fp2, fee_paid = :fpd2, paid_at = :pa2""",
            {"id": ident, "ps": p["period_start"], "pe": p.get("period_end"),
             "sb": p.get("starting_balance"), "eb": p.get("ending_balance"),
             "cp": p.get("cumulative_profit", 0.0), "fp": p.get("fee_15pct", 0.0),
             "fpd": int(p.get("fee_paid", False)), "pa": p.get("paid_at"),
             "pe2": p.get("period_end"), "eb2": p.get("ending_balance"),
             "cp2": p.get("cumulative_profit", 0.0), "fp2": p.get("fee_15pct", 0.0),
             "fpd2": int(p.get("fee_paid", False)), "pa2": p.get("paid_at")},
        )


def _ensure_current_period(record: Dict, current_balance: float):
    now = datetime.utcnow()
    periods = record.setdefault("monthly_periods", [])

    if not periods:
        fc = record.get("first_connected_at")
        period_start = datetime.fromisoformat(fc) if fc else now
        periods.append({
            "period_start": period_start.isoformat(),
            "period_end": None,
            "starting_balance": current_balance,
            "ending_balance": None,
            "cumulative_profit": 0.0,
            "fee_15pct": 0.0,
            "fee_paid": False,
            "paid_at": None,
        })
        return

    current = periods[-1]
    period_start = datetime.fromisoformat(current["period_start"])

    if now >= period_start + timedelta(days=PERIOD_DAYS):
        current["period_end"] = (period_start + timedelta(days=PERIOD_DAYS)).isoformat()
        current["ending_balance"] = current_balance
        profit = current_balance - current["starting_balance"]
        current["cumulative_profit"] = round(profit, 2)
        current["fee_15pct"] = round(profit * 0.15, 2) if profit > 0 else 0.0

        periods.append({
            "period_start": current["period_end"],
            "period_end": None,
            "starting_balance": current_balance,
            "ending_balance": None,
            "cumulative_profit": 0.0,
            "fee_15pct": 0.0,
            "fee_paid": False,
            "paid_at": None,
        })
    else:
        profit = current_balance - current["starting_balance"]
        current["cumulative_profit"] = round(profit, 2)
        current["fee_15pct"] = round(profit * 0.15, 2) if profit > 0 else 0.0


async def start_trial(identifier: str, balance: float):
    existing = await _get_sub_record(identifier)
    if existing:
        return
    now = datetime.utcnow()
    record = {
        "identifier": identifier,
        "first_connected_at": now.isoformat(),
        "trial_end": (now + timedelta(days=30)).isoformat(),
        "subscribed": False,
        "subscription_end": None,
        "paid_amount": 0.0,
        "monthly_periods": [
            {
                "period_start": now.isoformat(),
                "period_end": None,
                "starting_balance": balance,
                "ending_balance": None,
                "cumulative_profit": 0.0,
                "fee_15pct": 0.0,
                "fee_paid": False,
                "paid_at": None,
            }
        ],
    }
    await _save_sub_record(record)


async def get_subscription(identifier: str, current_balance: float = 0.0) -> Dict:
    record = await _get_sub_record(identifier)
    now = datetime.utcnow()

    empty = {
        "trial_active": False,
        "trial_end": None,
        "subscribed": False,
        "subscription_end": None,
        "due_amount": 0.0,
        "unpaid_fees": 0.0,
        "days_remaining": 0,
        "can_trade": True,
        "is_new": True,
        "monthly_periods": [],
        "current_month_profit": 0.0,
        "current_month_fee": 0.0,
        "demo": False,
    }

    if record is None:
        return empty

    if current_balance > 0:
        _ensure_current_period(record, current_balance)
        await _save_sub_record(record)

    periods = record.get("monthly_periods", [])
    current = periods[-1] if periods else {}

    due = 0.0
    if record.get("trial_end"):
        trial_end_dt = datetime.fromisoformat(record["trial_end"])
        if now >= trial_end_dt:
            for p in periods:
                if p.get("period_end") and not p.get("fee_paid") and p.get("fee_15pct", 0) > 0:
                    due += p["fee_15pct"]
            due = round(due, 2)

    if record.get("subscribed") and record.get("subscription_end"):
        sub_end_dt = datetime.fromisoformat(record["subscription_end"])
        if now < sub_end_dt:
            dr = (sub_end_dt - now).days
            return {
                "trial_active": False,
                "trial_end": record.get("trial_end"),
                "subscribed": True,
                "subscription_end": record["subscription_end"],
                "due_amount": due,
                "unpaid_fees": due,
                "days_remaining": dr,
                "can_trade": True,
                "is_new": False,
                "monthly_periods": periods,
                "current_month_profit": current.get("cumulative_profit", 0.0),
                "current_month_fee": current.get("fee_15pct", 0.0),
                "demo": False,
            }

    trial_end_str = record.get("trial_end")
    if trial_end_str:
        trial_end_dt = datetime.fromisoformat(trial_end_str)
        if now < trial_end_dt:
            dr = (trial_end_dt - now).days
            return {
                "trial_active": True,
                "trial_end": trial_end_str,
                "subscribed": False,
                "subscription_end": None,
                "due_amount": 0.0,
                "unpaid_fees": 0.0,
                "days_remaining": dr,
                "can_trade": True,
                "is_new": False,
                "monthly_periods": periods,
                "current_month_profit": current.get("cumulative_profit", 0.0),
                "current_month_fee": current.get("fee_15pct", 0.0),
                "demo": False,
            }

    return {
        "trial_active": False,
        "trial_end": record.get("trial_end"),
        "subscribed": record.get("subscribed", False),
        "subscription_end": record.get("subscription_end"),
        "due_amount": due,
        "unpaid_fees": due,
        "days_remaining": 0,
        "can_trade": due == 0.0,
        "is_new": False,
        "monthly_periods": periods,
        "current_month_profit": current.get("cumulative_profit", 0.0),
        "current_month_fee": current.get("fee_15pct", 0.0),
        "demo": False,
    }


async def can_start_live(identifier: str, current_balance: float = 0.0) -> bool:
    sub = await get_subscription(identifier, current_balance)
    return sub["can_trade"]


# ── Paystack ──────────────────────────────────────────────────────

PAYSTACK_BASE = "https://api.paystack.co"


def _paystack_headers() -> Dict:
    return {
        "Authorization": f"Bearer {PAYSTACK_SECRET}",
        "Content-Type": "application/json",
    }


def verify_paystack_webhook(body: bytes, signature: str) -> bool:
    if not PAYSTACK_SECRET or not signature:
        return False
    expected = hmac.new(PAYSTACK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def initialize_payment(email: str, amount_kobo: int, metadata: Dict = None, channels: List[str] = None, currency: str = "NGN") -> Optional[Dict]:
    if not PAYSTACK_SECRET:
        return None
    try:
        body = {"email": email, "amount": amount_kobo, "currency": currency}
        if metadata:
            body["metadata"] = metadata
        if channels:
            body["channels"] = channels
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(f"{PAYSTACK_BASE}/transaction/initialize", headers=_paystack_headers(), json=body)
        if r.is_success:
            return r.json().get("data")
    except Exception:
        pass
    return None


async def verify_payment(reference: str) -> Optional[Dict]:
    if not PAYSTACK_SECRET:
        return None
    if await _is_payment_processed(reference):
        return {"error": "already_processed"}
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{PAYSTACK_BASE}/transaction/verify/{reference}", headers=_paystack_headers())
        if not r.is_success:
            return None
        data = r.json().get("data", {})
        if data.get("status") != "success":
            return None
        meta = data.get("metadata", {}) or {}
        identifier = meta.get("identifier", "")
        amount_paid = float(data.get("amount", 0)) / 100

        record = await _get_sub_record(identifier)
        if record:
            record["paid_amount"] = record.get("paid_amount", 0) + amount_paid
            record["subscribed"] = True
            existing_end = record.get("subscription_end")
            if existing_end and datetime.fromisoformat(existing_end) > datetime.utcnow():
                record["subscription_end"] = (datetime.fromisoformat(existing_end) + timedelta(days=30)).isoformat()
            else:
                record["subscription_end"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
            remaining = amount_paid
            for p in record.get("monthly_periods", []):
                if not p.get("fee_paid") and p.get("fee_15pct", 0) > 0:
                    if remaining >= p["fee_15pct"]:
                        p["fee_paid"] = True
                        p["paid_at"] = datetime.utcnow().isoformat()
                        remaining -= p["fee_15pct"]
                    else:
                        break
            await _save_sub_record(record)
            await _mark_payment_processed(reference, identifier, "paystack", amount_paid)

        return {
            "identifier": identifier,
            "amount": amount_paid,
            "subscription_end": record["subscription_end"] if record else None,
        }
    except Exception:
        pass
    return None


async def process_paystack_webhook(event: str, data: dict) -> bool:
    if event != "charge.success":
        return False
    reference = data.get("reference", "")
    if not reference or await _is_payment_processed(reference):
        return False
    result = await verify_payment(reference)
    return result is not None and "error" not in result


# ── MaxelPay ─────────────────────────────────────────────────────

_maxelpay_orders: dict = {}


async def _maxelpay_register_order(order_id: str, identifier: str):
    _maxelpay_orders[order_id] = identifier
    try:
        await db_mod.database.execute(
            "INSERT INTO pending_orders (order_id, identifier, gateway, created_at) "
            "VALUES (:oid, :ident, 'maxelpay', :ca) "
            "ON CONFLICT (order_id) DO NOTHING",
            {"oid": order_id, "ident": identifier, "ca": datetime.utcnow().isoformat()},
        )
    except Exception:
        pass


async def _maxelpay_get_identifier(order_id: str) -> str:
    ident = _maxelpay_orders.get(order_id, "")
    if not ident:
        try:
            row = await db_mod.database.fetch_one(
                "SELECT identifier FROM pending_orders WHERE order_id = :oid",
                {"oid": order_id},
            )
            if row:
                ident = row["identifier"]
        except Exception:
            pass
    return ident


async def _is_payment_processed(ref_key: str) -> bool:
    try:
        row = await db_mod.database.fetch_one(
            "SELECT 1 FROM processed_payments WHERE ref_key = :rk", {"rk": ref_key}
        )
        return row is not None
    except Exception:
        return False


async def _mark_payment_processed(ref_key: str, identifier: str, gateway: str, amount: float):
    try:
        await db_mod.database.execute(
            "INSERT INTO processed_payments (ref_key, identifier, gateway, amount, processed_at) "
            "VALUES (:rk, :ident, :gw, :amt, :pa) "
            "ON CONFLICT (ref_key) DO NOTHING",
            {"rk": ref_key, "ident": identifier, "gw": gateway, "amt": amount, "pa": datetime.utcnow().isoformat()},
        )
    except Exception:
        pass


def _maxelpay_headers() -> Dict:
    return {
        "X-API-KEY": MAXELPAY_API_KEY,
        "Content-Type": "application/json",
    }


def verify_maxelpay_webhook(body: bytes, signature: str) -> bool:
    if not MAXELPAY_API_KEY or not signature:
        return False
    key = MAXELPAY_API_KEY.encode()
    # Try compact-normalized (matches Node.js JSON.stringify())
    try:
        normalized = json.dumps(json.loads(body), separators=(',', ':'), ensure_ascii=True)
        expected = hmac.new(key, normalized.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True
    except Exception:
        pass
    # Fallback: raw body in case MaxelPay signs the exact HTTP body
    try:
        expected = hmac.new(key, body, hashlib.sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return True
    except Exception:
        pass
    return False


async def create_maxelpay_payment(amount_usd: float, order_id: str, description: str = "",
                                   success_url: str = "", cancel_url: str = "",
                                   callback_url: str = "") -> Optional[Dict]:
    if not MAXELPAY_API_KEY:
        return None
    try:
        payload = {
            "orderId": order_id,
            "amount": amount_usd,
            "currency": "USD",
            "description": description or f"Gold Scalper subscription payment",
            "successUrl": success_url or "https://gold-scalper-qyhg.onrender.com/api/payment/maxelpay/success",
            "cancelUrl": cancel_url or "https://gold-scalper-qyhg.onrender.com/api/payment/maxelpay/cancel",
            "callbackUrl": callback_url or "https://gold-scalper-qyhg.onrender.com/api/payment/maxelpay/callback",
        }
        headers = _maxelpay_headers()
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(f"{MAXELPAY_BASE}/payments/sessions", headers=headers, json=payload)
        if r.is_success:
            return r.json()
    except Exception:
        pass
    return None


async def process_maxelpay_callback(order_id: str, status: str, amount: float) -> bool:
    ref_key = f"maxelpay_{order_id}"
    if await _is_payment_processed(ref_key):
        return False
    ident = await _maxelpay_get_identifier(order_id)
    if not ident:
        return False
    if status != "paid":
        return False
    record = await _get_sub_record(ident)
    if not record:
        logger.warning("MaxelPay callback: no subscription record for %s (order %s), payment $%.2f dropped", ident, order_id, amount)
        return True
    record["paid_amount"] = record.get("paid_amount", 0) + amount
    record["subscribed"] = True
    existing_end = record.get("subscription_end")
    if existing_end and datetime.fromisoformat(existing_end) > datetime.utcnow():
        record["subscription_end"] = (datetime.fromisoformat(existing_end) + timedelta(days=30)).isoformat()
    else:
        record["subscription_end"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
    remaining = amount
    for p in record.get("monthly_periods", []):
        if not p.get("fee_paid") and p.get("fee_15pct", 0) > 0:
            if remaining >= p["fee_15pct"]:
                p["fee_paid"] = True
                p["paid_at"] = datetime.utcnow().isoformat()
                remaining -= p["fee_15pct"]
            else:
                break
    await _save_sub_record(record)
    await _mark_payment_processed(ref_key, ident, "maxelpay", amount)
    return True


# ── Notifications ────────────────────────────────────────────────

_main_loop = None

def set_main_loop(loop):
    global _main_loop
    _main_loop = loop


def _schedule_on_main(coro):
    """Schedule a coroutine on the main event loop (for cross-thread DB access)."""
    import asyncio
    loop = _main_loop
    if loop is None or loop.is_closed():
        return
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is loop:
        return None
    return asyncio.run_coroutine_threadsafe(coro, loop)


async def create_notification(identifier: str, ntype: str, title: str, message: str, data: Optional[Dict] = None):
    import asyncio
    nid = uuid.uuid4().hex
    values = {
        "nid": nid,
        "ident": identifier,
        "type": ntype,
        "title": title,
        "msg": message,
        "data": json.dumps(data) if data else None,
        "ca": datetime.utcnow().isoformat(),
    }
    sql = """INSERT INTO notifications (id, identifier, type, title, message, data, created_at)
             VALUES (:nid, :ident, :type, :title, :msg, :data, :ca)"""
    try:
        future = _schedule_on_main(db_mod.database.execute(sql, values))
        if future is not None:
            await asyncio.wrap_future(future)
        else:
            await db_mod.database.execute(sql, values)
    except Exception as e:
        logger.warning("DB notification insert failed: %s", e)
    try:
        future = _schedule_on_main(_send_fcm_push(identifier, title, message, data))
        if future is None:
            asyncio.create_task(_send_fcm_push(identifier, title, message, data))
    except Exception:
        pass


async def get_notifications(identifier: str, limit: int = 50) -> List[Dict]:
    rows = await db_mod.database.fetch_all(
        """SELECT id, type, title, message, data, is_read, created_at
           FROM notifications WHERE identifier = :ident
           ORDER BY created_at DESC LIMIT :lim""",
        {"ident": identifier, "lim": limit},
    )
    return [dict(r) for r in rows]


async def mark_notification_read(notification_id: str):
    await db_mod.database.execute(
        "UPDATE notifications SET is_read = 1 WHERE id = :id",
        {"id": notification_id},
    )


async def mark_all_notifications_read(identifier: str):
    await db_mod.database.execute(
        "UPDATE notifications SET is_read = 1 WHERE identifier = :ident AND is_read = 0",
        {"ident": identifier},
    )


async def get_unread_notification_count(identifier: str) -> int:
    row = await db_mod.database.fetch_one(
        "SELECT COUNT(*) as cnt FROM notifications WHERE identifier = :ident AND is_read = 0",
        {"ident": identifier},
    )
    return row["cnt"] if row else 0
