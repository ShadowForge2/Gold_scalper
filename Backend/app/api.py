from collections import deque
import asyncio
import json
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
import uvicorn

from app.bot import Bot
from app.bot_pool import BotPool
from app.subscription import (
    ensure_device, get_device,
    add_account, remove_account, restore_device_by_capital_id,
    set_account_active, get_active_accounts, get_account_by_identifier,
    start_trial, get_subscription,
    can_start_live, initialize_payment, verify_payment,
    verify_paystack_webhook, process_paystack_webhook,
    create_maxelpay_payment, process_maxelpay_callback,
    verify_maxelpay_webhook,
    _maxelpay_register_order, _maxelpay_get_identifier,
    get_notifications, get_unread_notification_count,
    mark_notification_read, mark_all_notifications_read,
    create_notification,
)
from app.capital_client import CapitalClient
from app import email_service
import config as cfg



import logging
logger = logging.getLogger("GoldScalper")

class AddAccountRequest(BaseModel):
    api_key: str
    identifier: str
    password: str
    demo: bool = True

    @field_validator("api_key", "identifier", "password")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        return (v or "").strip()

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("API key is too short — it looks incomplete.")
        if any(ch.isspace() for ch in v):
            raise ValueError("API key must not contain spaces.")
        return v

    @field_validator("identifier")
    @classmethod
    def _validate_identifier(cls, v: str) -> str:
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Identifier must be a valid email address.")
        return v.lower()

    @field_validator("password")
    @classmethod
    def _validate_password(cls, v: str) -> str:
        if len(v) < 4:
            raise ValueError("Password is too short.")
        return v

class VerifyCredentialsRequest(BaseModel):
    api_key: str
    identifier: str
    password: str
    demo: bool = True

class PaystackInitRequest(BaseModel):
    email: str
    channels: Optional[List[str]] = None


class EmailPrefsRequest(BaseModel):
    allow_email: bool = False
    allow_push: bool = True
    allow_marketing: bool = False
    send_welcome: bool = False


def sanitize_account(acct: Dict) -> Dict:
    clean = dict(acct)
    clean["api_key"] = "****"
    clean["password"] = "****"
    return clean


# ── Credential validation guard ──────────────────────────────────────
# In-memory failed-attempt tracking keyed by "device_id|ip". Protects the
# /api/device/accounts endpoint from credential stuffing / brute forcing.
_AUTH_MAX_ATTEMPTS = int(os.environ.get("AUTH_MAX_ATTEMPTS", "5"))
_AUTH_LOCKOUT_SECONDS = int(os.environ.get("AUTH_LOCKOUT_SECONDS", "300"))
_AUTH_ATTEMPTS: Dict[str, Dict] = {}
_AUTH_LOCK = asyncio.Lock()


def _auth_key(request: Request, device_id: str) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"{device_id or 'unknown'}|{ip}"


async def _auth_locked(key: str) -> Optional[int]:
    """Return seconds remaining on lockout, or None if not locked."""
    async with _AUTH_LOCK:
        entry = _AUTH_ATTEMPTS.get(key)
        if not entry:
            return None
        if entry.get("locked_until", 0) > time.time():
            return int(entry["locked_until"] - time.time()) + 1
        return None


async def _auth_record_failure(key: str) -> Optional[int]:
    """Record a failed attempt; returns lockout seconds if now locked out."""
    async with _AUTH_LOCK:
        now = time.time()
        _prune_auth_entries(now)
        entry = _AUTH_ATTEMPTS.setdefault(key, {"count": 0, "first": now, "locked_until": 0})
        if entry["locked_until"] > now:
            return int(entry["locked_until"] - now) + 1
        if now - entry["first"] > 3600:
            entry["count"] = 0
            entry["first"] = now
        entry["count"] += 1
        if entry["count"] >= _AUTH_MAX_ATTEMPTS:
            entry["locked_until"] = now + _AUTH_LOCKOUT_SECONDS
            entry["count"] = 0
            return _AUTH_LOCKOUT_SECONDS
        return None


def _prune_auth_entries(now: float) -> None:
    """Drop tracking entries idle for over an hour to bound memory."""
    stale = [k for k, v in _AUTH_ATTEMPTS.items()
             if now - max(v.get("first", 0), v.get("locked_until", 0)) > 3600]
    for k in stale:
        _AUTH_ATTEMPTS.pop(k, None)


async def _auth_clear(key: str) -> None:
    async with _AUTH_LOCK:
        _AUTH_ATTEMPTS.pop(key, None)


# ── Polling rate limiter ──────────────────────────────────────────────
# Per-device sliding window limiter for high-frequency polling endpoints.
# Returns the last cached response silently when exceeded — never shows
# an error to the user.
_POLL_WINDOW_SECONDS = int(os.environ.get("POLL_WINDOW_SECONDS", "10"))
_POLL_MAX_REQUESTS = int(os.environ.get("POLL_MAX_REQUESTS", "15"))
_poll_windows: Dict[str, deque] = {}
_poll_cache: Dict[str, dict] = {}
_POLL_LOCK = asyncio.Lock()


def _poll_rate_key(request: Request, device_id: str) -> str:
    ip = request.client.host if request.client else "unknown"
    return f"{device_id or 'unknown'}|{ip}"


_poll_counter: int = 0


async def _poll_check(request: Request, device_id: str) -> Optional[dict]:
    """Return cached response if rate-limited, or None if request is allowed."""
    global _poll_counter
    key = _poll_rate_key(request, device_id)
    now = time.time()
    async with _POLL_LOCK:
        _poll_counter += 1
        if _poll_counter % 500 == 0:
            _poll_prune(now)
        window = _poll_windows.get(key)
        if window is None:
            window = deque()
            _poll_windows[key] = window
        # drop timestamps outside window
        while window and window[0] < now - _POLL_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= _POLL_MAX_REQUESTS:
            return _poll_cache.get(key)
        window.append(now)
        return None


def _poll_store(key: str, response: dict) -> None:
    """Cache a response for silent rate-limit delivery."""
    _poll_cache[key] = response


def _poll_prune(now: float = 0.0) -> None:
    """Drop stale entries to bound memory (called opportunistically)."""
    now = now or time.time()
    stale = [k for k, v in _poll_windows.items() if not v or v[-1] < now - 300]
    for k in stale:
        _poll_windows.pop(k, None)
        _poll_cache.pop(k, None)


def _fire_async(coro):
    """Fire a background coroutine on the running loop (best-effort)."""
    try:
        asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        pass


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

    _static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.isdir(_static_dir):
        app.mount("/static", StaticFiles(directory=_static_dir), name="static")

    def _db_ok() -> bool:
        return db_check() if db_check else True

    @app.get("/")
    async def root():
        return {"service": "Gold Scalper", "status": "running"}

    @app.get("/api/config")
    async def get_config():
        """Read-only dump of the env-resolved values the running process loaded.
        Lets us confirm no stale Render dashboard override beats the repo
        defaults for pull params / blacklist / broker mode. No secrets."""
        symbols = list(getattr(cfg, "SYMBOLS", []))
        pull_params = {}
        for sym in symbols:
            pp = (getattr(cfg, "SYMBOL_PULL_PARAMS", {}) or {}).get(sym, {}) or {}
            pull_params[sym] = {
                "pull_r": pp.get("pull_r"),
                "trail_r": pp.get("trail_r"),
                "max_hold": pp.get("max_hold"),
                "round_trip": pp.get("round_trip"),
            }
        return {
            "broker": getattr(cfg, "BROKER", None),
            "demo": bool(getattr(cfg, "CAPITAL_DEMO", False)),
            "symbols": symbols,
            "blacklist": sorted(
                str(s).strip().upper() for s in (getattr(cfg, "BLACKLIST_SYMBOLS", []) or []) if str(s).strip()
            ),
            "pull_enabled": {
                sym: bool((getattr(cfg, "PULL_ENGINE_ENABLED", {}) or {}).get(sym, False))
                for sym in symbols
            },
            "pull_params": pull_params,
            "pull_auto_tune_enabled": bool(getattr(cfg, "PULL_AUTO_TUNE_ENABLED", True)),
        }

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
        failover = getattr(bot, "_failover", None)
        failover_info = None
        if failover is not None:
            primary_alive = True
            try:
                primary_alive = await failover.check_primary_alive()
            except Exception:
                pass
            failover_info = {**failover.status(), "primary_alive": primary_alive}
        return {
            "status": "healthy" if connected else "degraded",
            "state": bot.state,
            "connected": connected,
            "db_connected": db_connected,
            "db_type": db_type,
            "broker": cfg.BROKER,
            "symbol": bot.symbol,
            "failover": failover_info,
        }

    @app.get("/api/account")
    async def get_account():
        if bot.client is None:
            return JSONResponse(status_code=503, content={"error": "Client not initialized"})
        info = bot.client.get_account_info()
        if info is None:
            return JSONResponse(status_code=503, content={"error": "Account not connected"})
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

    async def _restore_bot_after_payment(identifier: str):
        # Only auto-restart a bot that was stopped because its subscription
        # lapsed (account still marked active in the DB). A bot the user
        # stopped manually was deactivated (active=0) and must stay stopped.
        if not bot_pool:
            return
        if bot_pool.is_running(identifier):
            return
        acct = await get_account_by_identifier(identifier)
        if not acct or not acct.get("active"):
            return
        await bot_pool.start(
            identifier=acct["identifier"],
            api_key=acct["api_key"],
            password=acct["password"],
            demo=bool(acct.get("demo", True)),
        )
        await set_account_active(identifier, True)

    # ── Device-based Account Management ──────────────────────────
    @app.get("/api/device/accounts")
    async def device_list_accounts(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        dev = await ensure_device(device_id or "unknown")
        return {"accounts": [sanitize_account(a) for a in dev.get("accounts", [])]}

    @app.post("/api/device/accounts")
    async def device_add_account(data: AddAccountRequest, request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        ident = data.identifier
        key = _auth_key(request, did)

        remaining = await _auth_locked(key)
        if remaining:
            logger.warning("Auth attempts throttled for %s (%ss remaining)", did, remaining)
            return JSONResponse(status_code=429, content={
                "error": f"Too many failed attempts. Try again in {remaining} seconds.",
                "retry_after": remaining,
                "action_required": "wait",
            })

        dev = await get_device(did)
        existing = None
        if dev:
            existing = next((a for a in dev.get("accounts", [])
                             if a["identifier"].strip().lower() == ident), None)
            if existing:
                type_changed = bool(existing.get("demo", True)) != data.demo
                creds_changed = existing.get("api_key") != data.api_key or existing.get("password") != data.password
                if bot_pool and (type_changed or creds_changed) and bot_pool.is_running(ident):
                    return JSONResponse(status_code=409, content={
                        "error": "Stop the bot before changing account type or credentials.",
                        "action_required": "stop_bot",
                    })

        def _validate_creds():
            c = CapitalClient()
            c._timeout = 10
            ok = c.initialize(api_key=data.api_key, identifier=ident, password=data.password, demo=data.demo)
            hint = c.last_error_hint()
            last_err = c.last_error()
            err_msg = str(last_err[1]) if last_err and len(last_err) > 1 else str(last_err or "")
            c.shutdown()
            return ok, hint, err_msg

        ok, hint, err_msg = await asyncio.to_thread(_validate_creds)
        if not ok:
            logger.warning("Capital.com auth failed for %s with demo=%s: %s", ident, data.demo, err_msg)

            lockout = await _auth_record_failure(key)
            content = {
                "error": hint or f"Broker authentication failed: {err_msg}",
                "action_required": "check_credentials",
            }
            if lockout:
                content["error"] = f"Too many failed attempts. Try again in {lockout} seconds."
                content["retry_after"] = lockout
                content["action_required"] = "wait"
            return JSONResponse(status_code=401, content=content)

        await _auth_clear(key)
        await restore_device_by_capital_id(ident, did)
        await ensure_device(did)
        await add_account(did, data.api_key, ident, data.password, data.demo)
        if bot_pool and existing:
            prev_type = "demo" if existing.get("demo", True) else "live"
            new_type = "demo" if data.demo else "live"
            if prev_type != new_type:
                bot_pool.add_log(ident, f"Account switched from {prev_type} to {new_type}.", "INFO")
        dev = await get_device(did)
        # auto-start if previously active
        for acct in dev.get("accounts", []):
            if acct["identifier"].strip().lower() == ident and acct.get("active") and bot_pool and not bot_pool.is_running(ident):
                bot_pool.add_log(ident, "Account was previously active — auto-starting...", "INFO")
                await asyncio.to_thread(
                    bot_pool.start,
                    identifier=ident, api_key=data.api_key,
                    password=data.password, demo=data.demo,
                    skip_validation=True,
                )
                await set_account_active(ident, True)
                if not data.demo:
                    bal = 0.0
                    def _auto_balance():
                        tc = CapitalClient()
                        tc._timeout = 10
                        if tc.initialize(api_key=data.api_key, identifier=ident, password=data.password, demo=False):
                            bi = tc.get_account_info()
                            tc.shutdown()
                            return bi.get("balance", 0.0) if bi else 0.0
                        tc.shutdown()
                        return 0.0
                    bal = await asyncio.to_thread(_auto_balance)
                    await start_trial(ident, bal)
                break
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

        if await asyncio.to_thread(bot_pool.is_running, ident):
            return JSONResponse(status_code=400, content={"error": "Bot already running for this account"})

        if not demo:
            state_data = await asyncio.to_thread(bot_pool.get_state, ident)
            bal = 0.0
            if state_data and state_data.get("account") and not state_data["account"].get("error"):
                bal = state_data["account"].get("balance", 0)
            if not await can_start_live(ident, bal):
                sub = await get_subscription(ident, bal)
                return JSONResponse(status_code=402, content={
                    "error": "Trial expired. Unpaid fees must be settled.",
                    "subscription": sub,
                })

        result = await asyncio.to_thread(
            bot_pool.start,
            identifier=ident,
            api_key=acct["api_key"],
            password=acct["password"],
            demo=demo,
            skip_validation=True,
        )
        if result["success"]:
            await set_account_active(ident, True)
            if not demo:
                async def _post_start_live():
                    def _fetch_balance():
                        tc = CapitalClient()
                        tc._timeout = 10
                        if tc.initialize(
                            api_key=acct["api_key"],
                            identifier=acct["identifier"],
                            password=acct["password"],
                            demo=demo,
                        ):
                            info = tc.get_account_info()
                            tc.shutdown()
                            return info.get("balance", 0.0) if info else 0.0
                        tc.shutdown()
                        return 0.0
                    bal = await asyncio.to_thread(_fetch_balance)
                    trial_created = await start_trial(ident, bal)
                    sub = await get_subscription(ident, bal)
                    if trial_created and sub.get("trial_end"):
                        _fire_async(email_service.email_billing_welcome(ident, sub.get("trial_end") or ""))
                    dr = sub.get("days_remaining", 30)
                    bot_pool.add_log(ident, f"Live bot started. Trial active: {dr} day(s) remaining.", "INFO")
                    if dr <= 3:
                        bot_pool.add_log(ident, f"Trial ending soon ({dr} day(s)). Subscribe to keep trading.", "WARNING")
                    current_profit = sub.get("current_month_profit", 0)
                    current_fee = sub.get("current_month_fee", 0)
                    if current_fee > 0:
                        bot_pool.add_log(ident, f"Monthly profit: ${current_profit:.2f}. Fee due: ${current_fee:.2f}.", "INFO")
                _fire_async(_post_start_live())
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
        open_positions = await asyncio.to_thread(bot_pool.open_count, ident)
        if open_positions > 0:
            return JSONResponse(status_code=409, content={
                "error": f"Close all {open_positions} open position(s) before stopping the bot.",
                "open_count": open_positions,
                "action_required": "close_all",
            })
        result = await asyncio.to_thread(bot_pool.stop, ident)
        if result["success"]:
            await set_account_active(ident, False)
            bot_pool.add_log(ident, "Bot stopped.", "WARNING")
        return result

    @app.get("/api/device/bot/state")
    async def device_bot_state(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
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
        state = await asyncio.to_thread(bot_pool.get_state, ident)
        if state is None:
            return {"running": False, "state": None}
        running = await asyncio.to_thread(bot_pool.is_running, ident) or False
        result = {**state, "running": running}
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

    @app.get("/api/device/bot/logs")
    async def device_bot_logs(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
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
        result = {"logs": logs}
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

    @app.get("/api/device/bot/connection-status")
    async def device_bot_connection_status(device_id: str = Header(None, alias="X-Device-Id")):
        """Lightweight connection check — no Capital.com calls, reads bot state only."""
        did = device_id or "unknown"
        if bot_pool is None:
            return {"connected": False, "connecting": False, "error": "Bot pool not available", "broker": "Capital.com"}
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"connected": False, "connecting": False, "error": "No account configured", "broker": "Capital.com"}
        ident = dev["accounts"][0]["identifier"]
        running = await asyncio.to_thread(bot_pool.is_running, ident)
        if not running:
            return {"connected": False, "connecting": False, "error": None, "broker": "Capital.com"}
        state = await asyncio.to_thread(bot_pool.get_state, ident)
        if state is None:
            return {"connected": False, "connecting": True, "error": None, "broker": "Capital.com"}
        account = state.get("account") or {}
        has_error = account.get("error") is not None
        bot_data = state.get("bot") or {}
        bot_state = bot_data.get("state", "")
        if has_error:
            return {"connected": False, "connecting": False, "error": account.get("error", "Connection lost"), "broker": "Capital.com"}
        if bot_state in ("STOPPED",):
            return {"connected": False, "connecting": False, "error": "Bot stopped", "broker": "Capital.com"}
        return {"connected": True, "connecting": False, "error": None, "broker": "Capital.com"}

    # ── Device Bot Config ───────────────────────────────────────
    @app.get("/api/device/bot/config")
    async def device_bot_config(device_id: str = Header(None, alias="X-Device-Id")):
        did = device_id or "unknown"
        dev = await get_device(did)
        ident = dev["accounts"][0]["identifier"] if dev and dev.get("accounts") else None
        if ident and bot_pool and bot_pool.is_running(ident):
            bot_config = bot_pool.get_bot_config(ident)
            if bot_config:
                return bot_config
        return {
            "LOT_MULTIPLIER": str(cfg.LOT_MULTIPLIER),
            "MAX_SPREAD_PIPS": str(cfg.MAX_SPREAD_PIPS),
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
            "LOT_MULTIPLIER": "lot_multiplier",
            "MAX_SPREAD_PIPS": "max_spread_pips",
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
    async def device_bot_trades(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
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
            acct_data = state.get("account", {})
            balance = acct_data.get("balance", 0) if isinstance(acct_data, dict) else 0
            signal = bot_data.get("signal", {}) or {}
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
                    "symbol": pos.get("symbol", ""),
                    "score": signal.get("score", 0),
                    "exit_reason": "",
                    "balance": balance,
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
                    "symbol": t.get("symbol", ""),
                    "score": t.get("score", 0),
                    "exit_reason": t.get("exit_reason", ""),
                    "balance": t.get("balance", 0),
                })
        result = {"trades": trades}
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

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
    async def device_bot_performance(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
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
        account = state.get("account") or {}
        balance = account.get("balance", 0) or 0
        starting = bot_data.get("starting_balance", 0) or 0

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
        sorted_closed = sorted(closed, key=lambda t: t.get("closed_at") or "") if closed else []
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

        result = {
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
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

    # ── Device Bot Equity Curve ─────────────────────────────────
    @app.get("/api/device/bot/equity_curve")
    async def device_equity_curve(request: Request, period: str = "all", device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
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
        starting = bot_data.get("starting_balance", 0) or 0
        closed = bot_data.get("closed_trades", []) or []

        sorted_closed = sorted(closed, key=lambda t: t.get("closed_at") or "")
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
            year_start_str = year_start.strftime("%Y-%m-%d")
            running = starting
            balance_at_start = running
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                if closed_at[:10] < year_start_str:
                    balance_at_start += t.get("profit", 0)
                running += t.get("profit", 0)
            monthly = {}
            running = starting
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
            month_start_str = month_start.strftime("%Y-%m-%d")
            running = starting
            balance_at_start = running
            for t in sorted_closed:
                closed_at = t.get("closed_at", "")
                if not closed_at:
                    continue
                if closed_at[:10] < month_start_str:
                    balance_at_start += t.get("profit", 0)
                running += t.get("profit", 0)
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

        result = {"points": points}
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

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
    async def device_subscription(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
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
        _poll_store(_poll_rate_key(request, device_id), sub)
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
        result = await initialize_payment(data.email, due_kobo, metadata={"identifier": ident}, channels=data.channels)
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
        await _restore_bot_after_payment(ident)
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
        if ok:
            meta = (data.get("metadata") or {}) if isinstance(data, dict) else {}
            ident = meta.get("identifier", "")
            if ident:
                await _restore_bot_after_payment(ident)
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
        try:
            amount = float(data.get("amount", due))
        except (ValueError, TypeError):
            amount = due
        amount = max(amount, due, 1.0)
        order_id = f"maxel_{uuid.uuid4().hex[:16]}"
        await _maxelpay_register_order(order_id, ident)
        result = await create_maxelpay_payment(
            amount_usd=amount, order_id=order_id,
            description=f"Gold Scalper subscription payment - {ident}",
        )
        if result is None:
            return JSONResponse(status_code=500, content={"error": "MaxelPay payment gateway error"})
        if bot_pool:
            bot_pool.add_log(ident, f"MaxelPay payment of ${amount:.2f} created", "INFO")
        checkout_url = (
            result.get("data", {}).get("paymentUrl") or
            result.get("paymentUrl") or
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
            try:
                amount = float(payload.get("totalPaidUsd", payload.get("amount", 0)))
            except (ValueError, TypeError):
                amount = 0
            ok = await process_maxelpay_callback(order_id, status, amount)
            if ok and bot_pool:
                ident = await _maxelpay_get_identifier(order_id)
                if ident:
                    bot_pool.add_log(ident, f"MaxelPay payment of ${amount:.2f} verified. Subscription active.", "INFO")
                    await _restore_bot_after_payment(ident)
        return {"ok": True}

    # ── Email / Notification Preferences ────────────────────────────
    @app.get("/api/device/email/status")
    async def device_email_status(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            result = {
                "configured": False,
                "email": None,
                "allow_email": False,
                "allow_push": True,
                "allow_marketing": False,
                "is_new": True,
            }
            _poll_store(_poll_rate_key(request, device_id), result)
            return result
        ident = dev["accounts"][0]["identifier"]
        prefs = await email_service.get_email_prefs(ident)
        result = {
            "configured": email_service.email_configured(),
            "email": ident,
            "allow_email": prefs["allow_email"],
            "allow_push": prefs["allow_push"],
            "allow_marketing": prefs["allow_marketing"],
            "is_new": prefs.get("updated_at") is None,
        }
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

    @app.post("/api/device/email/prefs")
    async def device_email_prefs(data: EmailPrefsRequest, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return JSONResponse(status_code=400, content={"error": "No accounts found"})
        ident = dev["accounts"][0]["identifier"]
        await email_service.set_email_prefs(
            ident,
            allow_email=data.allow_email,
            allow_push=data.allow_push,
            allow_marketing=data.allow_marketing,
        )
        if data.send_welcome and data.allow_email:
            try:
                sub = await get_subscription(ident)
                if sub.get("trial_active") or sub.get("trial_end"):
                    _fire_async(email_service.email_billing_welcome(ident, sub.get("trial_end") or ""))
            except Exception as e:
                logger.debug("Welcome email on consent failed: %s", e)
        return {
            "success": True,
            "email": ident,
            "allow_email": data.allow_email,
            "allow_push": data.allow_push,
            "allow_marketing": data.allow_marketing,
        }

    # ── Notifications ──────────────────────────────────────────────
    @app.get("/api/device/notifications")
    async def device_notifications(request: Request, device_id: str = Header(None, alias="X-Device-Id")):
        cached = await _poll_check(request, device_id)
        if cached is not None:
            return cached
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            result = {"notifications": [], "unread_count": 0}
            _poll_store(_poll_rate_key(request, device_id), result)
            return result
        all_notifs = []
        total_unread = 0
        seen = set()
        for acct in dev["accounts"]:
            ident = acct["identifier"]
            for n in await get_notifications(ident):
                if n["id"] not in seen:
                    seen.add(n["id"])
                    all_notifs.append(n)
            total_unread += await get_unread_notification_count(ident)
        all_notifs.sort(key=lambda n: n.get("created_at") or "", reverse=True)
        result = {"notifications": all_notifs[:50], "unread_count": total_unread}
        _poll_store(_poll_rate_key(request, device_id), result)
        return result

    @app.post("/api/device/notifications/mark-read")
    async def device_notifications_mark_read(data: dict, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"success": False}
        nid = data.get("id")
        if nid:
            await mark_notification_read(nid)
        else:
            for acct in dev["accounts"]:
                await mark_all_notifications_read(acct["identifier"])
        return {"success": True}

    @app.post("/api/device/fcm-token")
    async def device_fcm_token(data: dict, device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        token = data.get("fcm_token", "")
        if not token:
            return {"success": False}
        from datetime import datetime
        from app import database as db_mod
        await db_mod.execute(
            """INSERT INTO fcm_tokens (device_id, fcm_token, updated_at)
               VALUES (:did, :token, :ua)
               ON CONFLICT (device_id) DO UPDATE SET fcm_token = :token, updated_at = :ua""",
            {"did": did, "token": token, "ua": datetime.utcnow().isoformat()},
        )
        return {"success": True}

    # ── Deploy safety check ─────────────────────────────────────
    @app.get("/api/deploy/check")
    async def deploy_check():
        """Check if it's safe to deploy/restart the server.

        Returns open trade counts per bot. Deploy is safe when all bots
        have zero open positions (no risk of orphaned trades during restart).
        """
        if bot_pool is None:
            return {"safe": True, "bots": [], "message": "No bot pool — nothing to protect"}
        open_bots = []
        total_open = 0
        for ident in list(bot_pool._bots.keys()):
            try:
                state = await asyncio.to_thread(bot_pool.get_state, ident)
                if state is None:
                    continue
                bot_data = state.get("bot") or {}
                positions = bot_data.get("positions") or {}
                open_count = positions.get("open_count", 0)
                total_open += open_count
                if open_count > 0:
                    open_bots.append({
                        "identifier": ident,
                        "open_positions": open_count,
                        "state": bot_data.get("state", "UNKNOWN"),
                        "symbol": bot_data.get("symbol", ""),
                    })
            except Exception:
                pass
        safe = total_open == 0
        return {
            "safe": safe,
            "total_open_positions": total_open,
            "bots_with_positions": open_bots,
            "message": "Deploy safe — no open trades" if safe else f"BLOCKED — {total_open} open position(s) across {len(open_bots)} bot(s)",
        }

    return app
