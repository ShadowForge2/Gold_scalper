import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import requests as http_requests

from app import database as db_mod

PAYSTACK_SECRET = os.getenv("PAYSTACK_SECRET_KEY", "")


# ── Device tracking (async DB) ──────────────────────────────────

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


async def get_device(device_id: str) -> Optional[Dict]:
    row = await db_mod.database.fetch_one(
        "SELECT * FROM devices WHERE device_id = :did", {"did": device_id}
    )
    if row is None:
        return None
    rows = await db_mod.database.fetch_all(
        "SELECT * FROM accounts WHERE device_id = :did", {"did": device_id}
    )
    return dict(row) | {"accounts": [dict(a) for a in rows]}


async def get_accounts(device_id: str) -> List[Dict]:
    rows = await db_mod.database.fetch_all(
        "SELECT * FROM accounts WHERE device_id = :did", {"did": device_id}
    )
    return [dict(r) for r in rows]


async def add_account(device_id: str, api_key: str, identifier: str, password: str, demo: bool = True) -> bool:
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
    result = await db_mod.database.execute(
        "DELETE FROM accounts WHERE device_id = :did AND identifier = :id",
        {"did": device_id, "id": identifier},
    )
    return result > 0


async def restore_device_by_capital_id(identifier: str, new_device_id: str) -> Optional[Dict]:
    old_device = await db_mod.database.fetch_one(
        """SELECT DISTINCT a.device_id FROM accounts a
           WHERE a.identifier = :id AND a.device_id != :ndid""",
        {"id": identifier, "ndid": new_device_id},
    )
    if old_device is None:
        return None

    old_did = old_device["device_id"]

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
            """INSERT INTO accounts (device_id, api_key, identifier, password, demo)
               SELECT :ndid, api_key, identifier, password, demo FROM accounts
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


def initialize_payment(email: str, amount_kobo: int, metadata: Dict = None) -> Optional[Dict]:
    if not PAYSTACK_SECRET:
        return None
    try:
        body = {"email": email, "amount": amount_kobo}
        if metadata:
            body["metadata"] = metadata
        r = http_requests.post(f"{PAYSTACK_BASE}/transaction/initialize", headers=_paystack_headers(), json=body, timeout=15)
        if r.ok:
            return r.json().get("data")
    except Exception:
        pass
    return None


async def verify_payment(reference: str) -> Optional[Dict]:
    if not PAYSTACK_SECRET:
        return None
    try:
        r = http_requests.get(f"{PAYSTACK_BASE}/transaction/verify/{reference}", headers=_paystack_headers(), timeout=15)
        if not r.ok:
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
            record["subscription_end"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
            for p in record.get("monthly_periods", []):
                if not p.get("fee_paid") and p.get("fee_15pct", 0) > 0:
                    p["fee_paid"] = True
                    p["paid_at"] = datetime.utcnow().isoformat()
            await _save_sub_record(record)

        return {
            "identifier": identifier,
            "amount": amount_paid,
            "subscription_end": record["subscription_end"] if record else None,
        }
    except Exception:
        pass
    return None
