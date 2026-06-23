import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, Request
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
    verify_paystack_webhook, process_paystack_webhook,
    create_maxelpay_payment, process_maxelpay_callback,
    verify_maxelpay_webhook,
    _maxelpay_register_order, _maxelpay_get_identifier,
)
from app.capital_client import CapitalClient
import config as cfg



import logging
logger = logging.getLogger("GoldScalper")

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
    clean = dict(acct)
    clean["api_key"] = "****"
    clean["password"] = "****"
    return clean


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
        ident = data.identifier
        dev = await get_device(did)
        existing = None
        if dev:
            existing = next((a for a in dev.get("accounts", []) if a["identifier"] == ident), None)
            if existing:
                type_changed = bool(existing.get("demo", True)) != data.demo
                creds_changed = existing.get("api_key") != data.api_key or existing.get("password") != data.password
                if bot_pool and (type_changed or creds_changed) and bot_pool.is_running(ident):
                    return JSONResponse(status_code=409, content={
                        "error": "Stop the bot before changing account type or credentials.",
                        "action_required": "stop_bot",
                    })

        temp = CapitalClient()
        ok = temp.initialize(api_key=data.api_key, identifier=ident, password=data.password, demo=data.demo)
        err_msg = temp.last_error()[1] if not ok else ""
        temp.shutdown()
        if not ok:
            logger.warning("Capital.com auth failed for %s with demo=%s: %s", ident, data.demo, err_msg)
            # Diagnostic: try the opposite mode
            temp2 = CapitalClient()
            ok2 = temp2.initialize(api_key=data.api_key, identifier=ident, password=data.password, demo=not data.demo)
            err2 = temp2.last_error()[1] if not ok2 else ""
            temp2.shutdown()
            if ok2:
                logger.warning("DIAG: %s works with demo=%s (opposite of what user selected)", ident, not data.demo)
            else:
                logger.warning("DIAG: %s also fails with demo=%s: %s", ident, not data.demo, err2)
            return JSONResponse(status_code=401, content={
                "error": f"Broker authentication failed: {err_msg}",
                "action_required": "check_credentials",
            })

        await restore_device_by_capital_id(ident, did)
        await ensure_device(did)
        await add_account(did, data.api_key, ident, data.password, data.demo)
        if bot_pool and existing:
            prev_type = "demo" if existing.get("demo", True) else "live"
            new_type = "demo" if data.demo else "live"
            if prev_type != new_type:
                bot_pool.add_log(ident, f"Account switched from {prev_type} to {new_type}.", "INFO")
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
                bal = 0.0
                temp_client = CapitalClient()
                if temp_client.initialize(
                    api_key=acct["api_key"],
                    identifier=acct["identifier"],
                    password=acct["password"],
                    demo=demo,
                ):
                    info = temp_client.get_account_info()
                    if info:
                        bal = info.get("balance", 0.0)
                    temp_client.shutdown()
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
        open_positions = bot_pool.open_count(ident)
        if open_positions > 0:
            return JSONResponse(status_code=409, content={
                "error": f"Close all {open_positions} open position(s) before stopping the bot.",
                "open_count": open_positions,
                "action_required": "close_all",
            })
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
                    "exit_time": None,
                    "exit_price": None,
                    "pnl": pos.get("profit", 0),
                    "ticket": pos.get("ticket", 0),
                })
            closed = bot_data.get("closed_trades", []) or []
            for t in closed:
                closed_at = t.get("closed_at", "")
                trades.append({
                    "entry_time": t.get("entry_time", ""),
                    "direction": t.get("type", "BUY"),
                    "lot": t.get("volume", 0),
                    "entry_price": t.get("entry_price", 0),
                    "current_price": None,
                    "exit_time": closed_at,
                    "exit_price": t.get("exit_price", 0),
                    "pnl": t.get("profit", 0),
                    "ticket": t.get("ticket", ""),
                })
        return {"trades": trades}

    # ── helpers ──

    def _monthly_breakdown(closed_trades, start_bal):
        if not closed_trades:
            return []
        from collections import OrderedDict
        monthly = OrderedDict()
        running = start_bal
        for t in closed_trades:
            closed_at = t.get("closed_at", "")
            month_key = closed_at[:7] if len(closed_at) >= 7 else ""
            if not month_key:
                continue
            if month_key not in monthly:
                monthly[month_key] = {"pnl": 0.0, "wins": 0, "total": 0, "start_bal": running}
            pnl = t.get("profit", 0) or 0
            monthly[month_key]["pnl"] += pnl
            monthly[month_key]["total"] += 1
            if pnl > 0:
                monthly[month_key]["wins"] += 1
            running += pnl
        result = []
        for mk, v in monthly.items():
            wr = round(v["wins"] / v["total"] * 100, 1) if v["total"] > 0 else 0
            result.append({"month": mk, "trades": v["total"], "pnl": round(v["pnl"], 2), "wr": wr})
        return result

    def _daily_breakdown(closed_trades):
        if not closed_trades:
            return []
        from collections import OrderedDict
        daily = OrderedDict()
        for t in closed_trades:
            closed_at = t.get("closed_at", "")
            day_key = closed_at[:10] if len(closed_at) >= 10 else ""
            if not day_key:
                continue
            if day_key not in daily:
                daily[day_key] = {"pnl": 0.0, "wins": 0, "total": 0}
            pnl = t.get("profit", 0) or 0
            daily[day_key]["pnl"] += pnl
            daily[day_key]["total"] += 1
            if pnl > 0:
                daily[day_key]["wins"] += 1
        result = []
        for dk, v in daily.items():
            wr = round(v["wins"] / v["total"] * 100, 1) if v["total"] > 0 else 0
            result.append({"date": dk, "trades": v["total"], "pnl": round(v["pnl"], 2), "wr": wr})
        return result

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
        sorted_closed = sorted(closed, key=lambda t: t.get("closed_at", "")) if closed else []
        if closed:
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
            "monthly": _monthly_breakdown(sorted_closed, starting),
            "daily": _daily_breakdown(sorted_closed),
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
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                running += t.get("profit", 0)
            balance_at_start = running
            running = starting
            monthly = {}
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                running += t.get("profit", 0)
                month_key = closed_at[:7]
                monthly[month_key] = round(running, 2)
            points.append({"time": year_start.isoformat(), "balance": round(balance_at_start, 2)})
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
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                running += t.get("profit", 0)
            balance_at_start = running
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
                    points.append({"time": month_start.isoformat(), "balance": round(balance_at_start, 2)})
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
        due_kobo = int(due * cfg.USD_TO_NGN_RATE * 100)
        if due_kobo < 5000:
            due_kobo = 5000  # Min 50 NGN
        if bot_pool:
            bot_pool.add_log(ident, f"Initializing payment of ₦{due_kobo/100:.2f}...", "INFO")
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
        if result.get("error") == "already_processed":
            if bot_pool:
                bot_pool.add_log(ref[:12], "Duplicate verify call ignored (already processed).", "INFO")
            return {"message": "Already verified", "data": result}
        ident = result.get("identifier", "")
        amount = result.get("amount", 0)
        if bot_pool:
            bot_pool.add_log(ident, f"Payment of ₦{amount:.2f} verified. Subscription active for 30 more days.", "INFO")
        return {"message": "Payment verified", "data": result}

    @app.post("/api/payment/paystack/webhook")
    async def paystack_webhook(request: Request):
        body = await request.body()
        sig = request.headers.get("x-paystack-signature", "")
        if not verify_paystack_webhook(body, sig):
            return JSONResponse(status_code=401, content={"error": "Invalid signature"})
        payload = json.loads(body)
        event = payload.get("event", "")
        data = payload.get("data", {})
        ok = await process_paystack_webhook(event, data)
        return {"ok": ok}

    # ── MaxelPay ──────────────────────────────────────────────────
    @app.post("/api/payment/maxelpay/init")
    async def maxelpay_init(data: dict, device_id: str = Header(None, alias="X-Device-Id")):
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
        if amount < 1.0:
            amount = 1.0
        order_id = f"maxel_{uuid.uuid4().hex[:16]}"
        await _maxelpay_register_order(order_id, ident)
        result = create_maxelpay_payment(
            amount_usd=amount, order_id=order_id,
            description=f"Gold Scalper subscription payment - {ident}",
        )
        if result is None:
            return JSONResponse(status_code=500, content={"error": "MaxelPay payment gateway error"})
        if bot_pool:
            bot_pool.add_log(ident, f"MaxelPay payment of ${amount:.2f} created", "INFO")
        checkout_url = (
            result.get("data", {}).get("checkoutUrl") or
            result.get("checkoutUrl") or
            result.get("url") or
            ""
        )
        return {
            "order_id": order_id,
            "amount": amount,
            "payment_url": checkout_url,
        }

    @app.post("/api/payment/maxelpay/callback")
    async def maxelpay_callback(request: Request):
        body = await request.body()
        signature = request.headers.get("X-MaxelPay-Signature", "")
        if not verify_maxelpay_webhook(body, signature):
            return JSONResponse(status_code=401, content={"ok": False, "error": "invalid signature"})
        data = json.loads(body)
        event = data.get("event", "")
        payload = data.get("data", {})
        order_id = payload.get("orderId", "")
        status = payload.get("status", "")
        if not order_id or not event:
            return {"ok": False}
        if event == "payment.completed":
            amount = float(payload.get("totalPaidUsd", payload.get("amount", 0)))
            ok = await process_maxelpay_callback(order_id, status, amount)
            if ok and bot_pool:
                ident = await _maxelpay_get_identifier(order_id)
                if ident:
                    bot_pool.add_log(ident, f"MaxelPay payment of ${amount:.2f} verified. Subscription active.", "INFO")
        return {"ok": True}

    return app
