import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from app.bot import Bot
from app.bot_pool import BotPool
from app.subscription import (
    ensure_device, get_device,
    add_account, remove_account, restore_device_by_capital_id,
    start_trial, get_subscription,
    can_start_live, initialize_payment, verify_payment,
    create_cryptomus_payment, get_cryptomus_payment,
    _get_sub_record, _save_sub_record,
    _cryptomus_register_order, _cryptomus_get_identifier,
)
from app.capital_client import CapitalClient
import config as cfg



class AddAccountRequest(BaseModel):
    api_key: str
    identifier: str
    password: str
    demo: bool = True

class VerifyCredentialsRequest(BaseModel):
    api_key: str
    identifier: str
    password: str
    demo: bool = True

class PaystackInitRequest(BaseModel):
    email: str
    channels: Optional[List[str]] = None


def sanitize_account(acct: Dict) -> Dict:
    pw = acct.get("password", "")
    masked = pw[:2] + "****" + pw[-1:] if len(pw) > 5 else "****"
    return {k: v if k != "password" else masked for k, v in acct.items()}


def create_app(bot: Bot, bot_pool: Optional[BotPool] = None, db_check=None) -> FastAPI:
    app = FastAPI(title="Gold Scalper", version="2.0.0")

    origins = [
        "http://localhost:8080", "http://127.0.0.1:8080",
        "http://localhost:8081", "http://127.0.0.1:8081",
        "http://localhost:8082", "http://127.0.0.1:8082",
        "http://localhost:9090", "http://127.0.0.1:9090",
        "http://localhost:9091", "http://127.0.0.1:9091",
        "http://localhost:9092", "http://127.0.0.1:9092",
        "https://gold-scalper-qyhg.onrender.com", "https://gold-scalper.onrender.com",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _db_ok() -> bool:
        return db_check() if db_check else True

    @app.get("/health")
    async def health():
        connected = bot.client is not None and bot.client.is_connected()
        db_connected = _db_ok()
        db_type = "disconnected"
        if db_connected:
            try:
                from app import database as _db_mod
                url = str(getattr(_db_mod.database, "url", ""))
                db_type = "postgresql" if "postgres" in url else "sqlite"
            except Exception:
                db_type = "unknown"
        return {
            "status": "healthy" if connected else "degraded",
            "state": bot.state,
            "connected": connected,
            "db_connected": db_connected,
            "db_type": db_type,
            "broker": cfg.BROKER,
            "symbol": bot.symbol,
        }

    @app.get("/api/account")
    async def get_account():
        info = bot.client.get_account_info()
        if info is None:
            return JSONResponse(status_code=503, content={"error": "MT5 not connected"})
        return info

    @app.get("/api/state")
    async def get_state():
        return bot.get_state_summary()

    @app.get("/api/positions")
    async def get_positions():
        bot.position_manager.refresh()
        return bot.position_manager.summary()

    @app.post("/api/bot/start")
    async def start_bot():
        bot.start()
        return {"message": "Bot started", "state": bot.state}

    @app.post("/api/bot/stop")
    async def stop_bot():
        bot.stop()
        return {"message": "Bot stopped", "state": bot.state}

    @app.post("/api/trades/close_all")
    async def close_all():
        count = await bot.emergency_close()
        return {"message": f"Closed {count} position(s)", "closed_count": count}

    @app.post("/api/bot/settings")
    async def update_settings(settings: dict):
        bot.update_settings(settings)
        return {"message": "Settings updated", "settings": settings}

    @app.post("/api/bot/login")
    async def bot_login(data: dict):
        server = data.get("server", "")
        account = data.get("account", "")
        password = data.get("password", "")
        result = bot.login(server, account, password)
        if result["success"]:
            return {"message": "Login successful", "account": result["account"]}
        return JSONResponse(status_code=401, content={"error": result.get("error", "Login failed")})

    @app.get("/api/accounts")
    async def list_accounts():
        return {"accounts": bot.list_accounts()}

    @app.post("/api/accounts")
    async def mt_add_account(data: dict):
        label = data.get("label", "")
        server = data.get("server", "")
        account = data.get("account", "")
        password = data.get("password", "")
        result = bot.add_account(label, server, account, password)
        if result["success"]:
            return {"message": result["message"], "accounts": bot.list_accounts()}
        return JSONResponse(status_code=400, content=result)

    @app.delete("/api/accounts/{account_id}")
    async def mt_remove_account(account_id: str):
        result = bot.remove_account(account_id)
        if result["success"]:
            return {"message": result["message"], "accounts": bot.list_accounts()}
        return JSONResponse(status_code=404, content=result)

    def _no_db():
        return JSONResponse(status_code=503, content={"error": "Database not connected"})

    # ── Device-based Account Management ──────────────────────────
    @app.get("/api/device/accounts")
    async def device_list_accounts(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        dev = await ensure_device(device_id or "unknown")
        return {"accounts": [sanitize_account(a) for a in dev.get("accounts", [])]}

    @app.post("/api/device/accounts")
    async def device_add_account(data: AddAccountRequest, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        await restore_device_by_capital_id(data.identifier, did)
        await ensure_device(did)
        await add_account(did, data.api_key, data.identifier, data.password, data.demo)
        dev = await get_device(did)
        return {"success": True, "accounts": [sanitize_account(a) for a in (dev.get("accounts") or [])]}

    @app.delete("/api/device/accounts/{identifier}")
    async def device_remove_account(identifier: str, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        ok = await remove_account(did, identifier)
        if not ok:
            return JSONResponse(status_code=404, content={"error": "Account not found"})
        dev = await get_device(did)
        return {"success": True, "accounts": [sanitize_account(a) for a in (dev.get("accounts") or [])]}

    # ── Device Bot Control (keyed by Capital.com identifier) ────
    @app.post("/api/device/bot/start")
    async def device_start_bot(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return JSONResponse(status_code=503, content={"error": "Bot pool not available"})
        dev = await get_device(did)
        if dev is None:
            return JSONResponse(status_code=400, content={"error": "Device not registered"})
        accounts = dev.get("accounts", [])
        if not accounts:
            return JSONResponse(status_code=400, content={"error": "No accounts saved"})
        acct = accounts[0]
        ident = acct["identifier"]
        demo = acct.get("demo", True)

        if bot_pool.is_running(ident):
            return JSONResponse(status_code=400, content={"error": "Bot already running for this account"})

        if not demo:
            state_data = bot_pool.get_state(ident)
            bal = 0.0
            if state_data and state_data.get("account") and not state_data["account"].get("error"):
                bal = state_data["account"].get("balance", 0)
            if not await can_start_live(ident, bal):
                sub = await get_subscription(ident, bal)
                return JSONResponse(status_code=402, content={
                    "error": "Trial expired. Unpaid fees must be settled.",
                    "subscription": sub,
                })

        result = bot_pool.start(
            identifier=ident,
            api_key=acct["api_key"],
            password=acct["password"],
            demo=demo,
        )
        if result["success"]:
            if not demo:
                state_data = bot_pool.get_state(ident)
                bal = 0.0
                if state_data and state_data.get("account") and not state_data["account"].get("error"):
                    bal = state_data["account"].get("balance", 0)
                await start_trial(ident, bal)
                sub = await get_subscription(ident, bal)
                dr = sub.get("days_remaining", 30)
                bot_pool.add_log(ident, f"Live bot started. Trial active: {dr} day(s) remaining.", "INFO")
                if dr <= 3:
                    bot_pool.add_log(ident, f"Trial ending soon ({dr} day(s)). Subscribe to keep trading.", "WARNING")
                current_profit = sub.get("current_month_profit", 0)
                current_fee = sub.get("current_month_fee", 0)
                if current_fee > 0:
                    bot_pool.add_log(ident, f"Monthly profit: ${current_profit:.2f}. Fee due: ${current_fee:.2f}.", "INFO")
            else:
                bot_pool.add_log(ident, "Demo bot started. Unlimited free usage.", "INFO")
            bot_pool.add_log(ident, "Credentials connected successfully.", "INFO")
            return {"message": "Bot started"}
        return JSONResponse(status_code=400, content=result)

    @app.post("/api/device/bot/stop")
    async def device_stop_bot(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return JSONResponse(status_code=503, content={"error": "Bot pool not available"})
        dev = await get_device(did)
        if not dev:
            return JSONResponse(status_code=400, content={"error": "Device not found"})
        accounts = dev.get("accounts", [])
        if not accounts:
            return {"message": "No accounts to stop"}
        ident = accounts[0]["identifier"]
        result = bot_pool.stop(ident)
        if result["success"]:
            bot_pool.add_log(ident, "Bot stopped.", "WARNING")
        return result

    @app.get("/api/device/bot/state")
    async def device_bot_state(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return JSONResponse(status_code=503, content={"error": "Bot pool not available"})
        dev = await get_device(did)
        if not dev:
            return {"running": False, "state": None}
        accounts = dev.get("accounts", [])
        if not accounts:
            return {"running": False, "state": None}
        ident = accounts[0]["identifier"]
        state = bot_pool.get_state(ident)
        if state is None:
            return {"running": False, "state": None}
        running = bot_pool.is_running(ident) or False
        return {**state, "running": running}

    @app.get("/api/device/bot/logs")
    async def device_bot_logs(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return JSONResponse(status_code=503, content={"error": "Bot pool not available"})
        dev = await get_device(did)
        if not dev:
            return {"logs": []}
        accounts = dev.get("accounts", [])
        if not accounts:
            return {"logs": []}
        ident = accounts[0]["identifier"]
        logs = bot_pool.get_logs(ident)
        return {"logs": logs}

    # ── Device Bot Config ───────────────────────────────────────
    @app.get("/api/device/bot/config")
    async def device_bot_config(device_id: str = Header(None, alias="X-Device-Id")):
        return {
            "LOT_MULTIPLIER": str(cfg.LOT_MULTIPLIER),
            "MAX_DAILY_LOSS_USD": str(cfg.MAX_DAILY_LOSS_USD),
            "MAX_EVENT_LOSS_USD": str(cfg.MAX_EVENT_LOSS_USD),
            "SIGNAL_ENTRY_THRESHOLD": str(cfg.SIGNAL_ENTRY_THRESHOLD),
            "EXIT_THRESHOLD_TIGHT": str(cfg.EXIT_THRESHOLD_TIGHT),
            "MAX_SPREAD_PIPS": str(cfg.MAX_SPREAD_PIPS),
            "MAX_TRADES_PER_EVENT": str(cfg.MAX_TRADES_PER_EVENT),
            "MAX_TRADES_PER_SESSION": str(cfg.MAX_TRADES_PER_SESSION),
            "MAX_CONSECUTIVE_LOSSES": str(cfg.MAX_CONSECUTIVE_LOSSES),
            "RE_ENTRY_COOLDOWN_SEC": str(cfg.RE_ENTRY_COOLDOWN_SEC),
            "BIAS_UPDATE_INTERVAL_SEC": str(cfg.BIAS_UPDATE_INTERVAL_SEC),
            "ALLOWED_SESSIONS": cfg.ALLOWED_SESSIONS,
            "MIN_BALANCE": str(cfg.MIN_BALANCE),
            "MAX_LOT": str(cfg.MAX_LOT),
            "LOT_SIZE": str(cfg.LOT_SIZE),
        }

    @app.post("/api/device/bot/config")
    async def device_bot_config_update(data: dict, device_id: str = Header(None, alias="X-Device-Id")):
        if bot_pool is None:
            return JSONResponse(status_code=503, content={"error": "Bot pool not available"})
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return JSONResponse(status_code=400, content={"error": "No accounts"})
        ident = dev["accounts"][0]["identifier"]
        # Normalize env-var style keys to the format update_settings expects
        KEY_MAP = {
            "MAX_DAILY_LOSS_USD": "max_daily_loss",
            "MAX_EVENT_LOSS_USD": "max_event_loss",
            "MAX_TRADES_PER_EVENT": "max_trades_per_event",
            "MAX_TRADES_PER_SESSION": "max_trades_per_session",
            "RE_ENTRY_COOLDOWN_SEC": "cooldown_seconds",
            "MAX_CONSECUTIVE_LOSSES": "consecutive_loss_limit",
            "LOT_MULTIPLIER": "lot_multiplier",
            "SIGNAL_ENTRY_THRESHOLD": "signal_entry_threshold",
            "EXIT_THRESHOLD_TIGHT": "exit_threshold_tight",
            "MAX_SPREAD_PIPS": "max_spread_pips",
            "BIAS_UPDATE_INTERVAL_SEC": "bias_update_interval_sec",
            "ALLOWED_SESSIONS": "allowed_sessions",
        }
        normalized = {}
        for k, v in data.items():
            target = KEY_MAP.get(k, k)
            try:
                normalized[target] = float(v)
            except (ValueError, TypeError):
                normalized[target] = v
        if bot_pool.is_running(ident):
            bot_pool.update_settings(ident, normalized)
        else:
            bot_pool.add_log(ident, "Config saved (bot not running, settings will apply on next start)", "INFO")
        return {"success": True, "message": "Config updated"}

    # ── Device Bot Trades ───────────────────────────────────────
    @app.get("/api/device/bot/trades")
    async def device_bot_trades(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return {"trades": []}
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"trades": []}
        ident = dev["accounts"][0]["identifier"]
        state = bot_pool.get_state(ident)
        trades = []
        if state and state.get("bot"):
            bot_data = state["bot"]
            pm = bot_data.get("positions", {})
            positions = pm.get("positions", []) if isinstance(pm, dict) else []
            for pos in positions:
                time_val = pos.get("time", "")
                if hasattr(time_val, "isoformat"):
                    time_val = time_val.isoformat()
                trades.append({
                    "entry_time": time_val,
                    "direction": pos.get("type", "BUY"),
                    "lot": pos.get("volume", 0),
                    "entry_price": pos.get("price_open", 0),
                    "current_price": pos.get("price_current", 0),
                    "pnl": pos.get("profit", 0),
                    "ticket": pos.get("ticket", 0),
                })
        return {"trades": trades}

    # ── Device Bot Performance ──────────────────────────────────
    @app.get("/api/device/bot/performance")
    async def device_bot_performance(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return {"trades": 0}
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"trades": 0}
        ident = dev["accounts"][0]["identifier"]
        state = bot_pool.get_state(ident)
        if not state or not state.get("bot"):
            return {"trades": 0}
        bot_data = state["bot"]
        scaler = bot_data.get("scaler") or {}
        account = state.get("account") or {}
        balance = account.get("balance", 0) or 0
        starting = scaler.get("starting_balance", 0) or 0

        closed = bot_data.get("closed_trades", []) or []
        wins = sum(1 for t in closed if (t.get("profit") or 0) > 0)
        losses = sum(1 for t in closed if (t.get("profit") or 0) <= 0)
        total = wins + losses
        gross_profit = sum(t.get("profit", 0) for t in closed if (t.get("profit") or 0) > 0)
        gross_loss = abs(sum(t.get("profit", 0) for t in closed if (t.get("profit") or 0) < 0))
        net_pnl = gross_profit - gross_loss
        win_rate = round(wins / total * 100, 1) if total > 0 else 0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (round(gross_profit, 2) if gross_profit > 0 else 0)
        avg_win = round(gross_profit / wins, 2) if wins > 0 else 0
        avg_loss = round(gross_loss / losses, 2) if losses > 0 else 0

        max_dd = 0.0
        if closed:
            sorted_closed = sorted(closed, key=lambda t: t.get("closed_at", ""))
            running = starting
            peak = starting
            for t in sorted_closed:
                running += t.get("profit", 0)
                if running > peak:
                    peak = running
                dd = peak - running
                if dd > max_dd:
                    max_dd = dd

        return {
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(net_pnl, 2),
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_dd": round(max_dd, 2),
            "starting_balance": round(starting, 2),
            "ending_balance": round(balance, 2),
            "return_pct": round((net_pnl / starting * 100) if starting > 0 else 0, 2),
            "monthly": [],
            "daily": [],
        }

    # ── Device Bot Equity Curve ─────────────────────────────────
    @app.get("/api/device/bot/equity_curve")
    async def device_equity_curve(period: str = "all", device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return {"points": []}
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"points": []}
        ident = dev["accounts"][0]["identifier"]
        state = bot_pool.get_state(ident)
        if not state or not state.get("bot"):
            return {"points": []}
        bot_data = state["bot"]
        scaler = bot_data.get("scaler") or {}
        starting = scaler.get("starting_balance", 0) or 0
        closed = bot_data.get("closed_trades", []) or []

        sorted_closed = sorted(closed, key=lambda t: t.get("closed_at", ""))
        points = []
        running = starting
        now = datetime.utcnow()
        account = state.get("account") or {}
        balance = account.get("balance", 0) or 0

        if not closed or starting <= 0:
            if balance > 0:
                return {"points": [{"time": now.isoformat(), "balance": round(balance, 2)}]}
            return {"points": []}

        if period == "yearly":
            year_start = datetime(now.year, 1, 1)
            running = starting
            points.append({"time": year_start.isoformat(), "balance": round(starting, 2)})
            monthly = {}
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                running += t.get("profit", 0)
                month_key = closed_at[:7]
                monthly[month_key] = round(running, 2)
            for m in sorted(monthly.keys()):
                dt = datetime.fromisoformat(m + "-01")
                if dt >= year_start:
                    points.append({"time": dt.isoformat(), "balance": monthly[m]})
            account = state.get("account") or {}
            balance = account.get("balance", 0) or 0
            if balance > 0:
                points.append({"time": now.isoformat(), "balance": round(balance, 2)})

        elif period == "monthly":
            month_start = datetime(now.year, now.month, 1)
            running = starting
            daily = {}
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                running += t.get("profit", 0)
                day_key = closed_at[:10]
                if day_key >= month_start.strftime("%Y-%m-%d"):
                    daily[day_key] = round(running, 2)
            if daily:
                first_day = min(daily.keys())
                if first_day > month_start.strftime("%Y-%m-%d"):
                    points.append({"time": month_start.isoformat(), "balance": round(starting, 2)})
            for d in sorted(daily.keys()):
                points.append({"time": d + "T00:00:00", "balance": daily[d]})
            account = state.get("account") or {}
            balance = account.get("balance", 0) or 0
            if balance > 0:
                points.append({"time": now.isoformat(), "balance": round(balance, 2)})

        else:
            points.append({"time": sorted_closed[0].get("closed_at", ""), "balance": round(starting, 2)})
            for t in sorted_closed:
                running += t.get("profit", 0)
                points.append({
                    "time": t.get("closed_at", ""),
                    "balance": round(running, 2),
                })
            account = state.get("account") or {}
            balance = account.get("balance", 0) or 0
            if balance > 0:
                points.append({"time": now.isoformat(), "balance": round(balance, 2)})

        return {"points": points}

    # ── Subscription ─────────────────────────────────────────────
    @app.post("/api/device/trades/close_all")
    async def device_close_all(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        if bot_pool is None:
            return JSONResponse(status_code=503, content={"error": "Bot pool not available"})
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return JSONResponse(status_code=400, content={"error": "No accounts"})
        ident = dev["accounts"][0]["identifier"]
        count = await bot_pool.emergency_close(ident)
        return {"message": f"Closed {count} position(s)", "closed_count": count}

    @app.get("/api/device/subscription")
    async def device_subscription(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"error": "No accounts", "trial_active": False, "can_trade": True, "demo": True, "is_new": True}
        acct = dev["accounts"][0]
        ident = acct["identifier"]
        demo = acct.get("demo", True)
        bal = 0.0
        if bot_pool:
            state = bot_pool.get_state(ident)
            if state and state.get("account") and not state["account"].get("error"):
                bal = state["account"].get("balance", 0)
        sub = await get_subscription(ident, bal)
        sub["demo"] = bool(demo)
        if bot_pool:
            if not sub.get("can_trade") and sub.get("trial_end"):
                bot_pool.add_log_once(ident, "Trial ended. Subscription required to continue trading.", "WARNING")
            elif sub.get("trial_active"):
                dr = sub.get("days_remaining", 0)
                if dr == 1:
                    bot_pool.add_log_once(ident, "Trial ends tomorrow! Subscribe to continue.", "WARNING")
                elif 2 <= dr <= 3:
                    bot_pool.add_log_once(ident, f"Trial ending in {dr} day(s). Please subscribe.", "WARNING")
        return sub

    @app.post("/api/device/subscription/check")
    async def device_subscription_check(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"error": "No accounts", "trial_active": False, "can_trade": True, "demo": True, "is_new": True}
        acct = dev["accounts"][0]
        ident = acct["identifier"]
        demo = acct.get("demo", True)
        bal = 0.0
        if bot_pool:
            state = bot_pool.get_state(ident)
            if state and state.get("account") and not state["account"].get("error"):
                bal = state["account"].get("balance", 0)
        sub = await get_subscription(ident, bal)
        sub["demo"] = bool(demo)
        if bot_pool:
            if not sub.get("can_trade") and sub.get("trial_end"):
                bot_pool.add_log_once(ident, "Trial ended. Subscription required to continue trading.", "WARNING")
            elif sub.get("trial_active"):
                dr = sub.get("days_remaining", 0)
                if dr == 1:
                    bot_pool.add_log_once(ident, "Trial ends tomorrow! Subscribe to continue.", "WARNING")
                elif 2 <= dr <= 3:
                    bot_pool.add_log_once(ident, f"Trial ending in {dr} day(s). Please subscribe.", "WARNING")
        return sub

    # ── Paystack Payment ─────────────────────────────────────────
    @app.post("/api/payment/initialize")
    async def payment_initialize(data: PaystackInitRequest, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return JSONResponse(status_code=400, content={"error": "No accounts found"})
        ident = dev["accounts"][0]["identifier"]
        bal = 0.0
        if bot_pool:
            state = bot_pool.get_state(ident)
            if state and state.get("account") and not state["account"].get("error"):
                bal = state["account"].get("balance", 0)
        sub = await get_subscription(ident, bal)
        due = sub.get("unpaid_fees", sub.get("due_amount", 0))
        due_kobo = int(due * 100)
        if due_kobo < 100:
            due_kobo = 5000  # Min 50 NGN
        if bot_pool:
            bot_pool.add_log(ident, f"Initializing payment of ${due_kobo/100:.2f}...", "INFO")
        result = initialize_payment(data.email, due_kobo, metadata={"identifier": ident}, channels=data.channels)
        if result is None:
            if bot_pool:
                bot_pool.add_log(ident, "Payment gateway error", "ERROR")
            return JSONResponse(status_code=500, content={"error": "Payment gateway error"})
        if bot_pool:
            bot_pool.add_log(ident, "Payment link generated", "INFO")
        return {
            "authorization_url": result.get("authorization_url"),
            "reference": result.get("reference"),
            "access_code": result.get("access_code"),
        }

    @app.post("/api/payment/verify")
    async def payment_verify(data: dict):
        ref = data.get("reference", "")
        result = await verify_payment(ref)
        if result is None:
            return JSONResponse(status_code=400, content={"error": "Payment verification failed"})
        ident = result.get("identifier", "")
        amount = result.get("amount", 0)
        if bot_pool:
            bot_pool.add_log(ident, f"Payment of ${amount:.2f} verified. Subscription active for 30 more days.", "INFO")
        return {"message": "Payment verified", "data": result}

    # ── Cryptomus ──────────────────────────────────────────────────
    @app.post("/api/payment/cryptomus/init")
    async def cryptomus_init(data: dict, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return JSONResponse(status_code=400, content={"error": "No accounts found"})
        ident = dev["accounts"][0]["identifier"]
        bal = 0.0
        if bot_pool:
            state = bot_pool.get_state(ident)
            if state and state.get("account") and not state["account"].get("error"):
                bal = state["account"].get("balance", 0)
        sub = await get_subscription(ident, bal)
        due = sub.get("unpaid_fees", sub.get("due_amount", 0))
        amount = float(data.get("amount", due))
        if amount <= 0:
            amount = due
        if amount <= 0:
            amount = 1.0
        email = data.get("email", "")
        order_id = f"crypt_{uuid.uuid4().hex[:16]}"
        _cryptomus_register_order(order_id, ident)
        result = create_cryptomus_payment(
            amount_usd=amount, order_id=order_id, email=email,
            url_return=data.get("url_return", ""),
            url_callback=data.get("url_callback", ""),
        )
        if result is None:
            return JSONResponse(status_code=500, content={"error": "Cryptomus payment gateway error"})
        if bot_pool:
            bot_pool.add_log(ident, f"Cryptomus payment of ${amount:.2f} created", "INFO")
        return {
            "order_id": order_id,
            "amount": amount,
            "payment_url": result.get("url"),
            "uuid": result.get("uuid"),
        }

    @app.post("/api/payment/cryptomus/status")
    async def cryptomus_status(data: dict):
        order_id = data.get("order_id", "")
        if not order_id:
            return JSONResponse(status_code=400, content={"error": "order_id required"})
        result = get_cryptomus_payment(order_id)
        if result is None:
            return JSONResponse(status_code=404, content={"error": "Payment not found"})
        return result

    @app.post("/api/payment/cryptomus/callback")
    async def cryptomus_callback(data: dict):
        order_id = data.get("order_id", "")
        status = data.get("status", "")
        if not order_id or not status:
            return {"ok": False}
        ident = _cryptomus_get_identifier(order_id)
        if status in ("paid", "paid_over"):
            amount = float(data.get("amount", 0))
            record = await _get_sub_record(ident) if ident else None
            if record:
                record["paid_amount"] = record.get("paid_amount", 0) + amount
                record["subscribed"] = True
                record["subscription_end"] = (datetime.utcnow() + timedelta(days=30)).isoformat()
                for p in record.get("monthly_periods", []):
                    if not p.get("fee_paid") and p.get("fee_15pct", 0) > 0:
                        p["fee_paid"] = True
                        p["paid_at"] = datetime.utcnow().isoformat()
                await _save_sub_record(record)
                if bot_pool:
                    bot_pool.add_log(ident, f"Cryptomus payment of ${amount:.2f} verified. Subscription active.", "INFO")
        return {"ok": True}

    return app
