import os
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
)
import config as cfg



class AddAccountRequest(BaseModel):
    api_key: str
    identifier: str
    password: str
    demo: bool = True

class PaystackInitRequest(BaseModel):
    email: str
    channels: Optional[List[str]] = None


def _device_id(x_device_id: Optional[str] = Header(None)) -> str:
    if not x_device_id:
        return "unknown"
    return x_device_id


def sanitize_account(acct: Dict) -> Dict:
    pw = acct.get("password", "")
    masked = pw[:2] + "****" + pw[-1:] if len(pw) > 5 else "****"
    return {k: v if k != "password" else masked for k, v in acct.items()}


def create_app(bot: Bot, bot_pool: Optional[BotPool] = None, db_check=None) -> FastAPI:
    app = FastAPI(title="Gold Scalper", version="2.0.0")

    cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in cors_origins],
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
    async def add_account(data: dict):
        label = data.get("label", "")
        server = data.get("server", "")
        account = data.get("account", "")
        password = data.get("password", "")
        result = bot.add_account(label, server, account, password)
        if result["success"]:
            return {"message": result["message"], "accounts": bot.list_accounts()}
        return JSONResponse(status_code=400, content=result)

    @app.delete("/api/accounts/{account_id}")
    async def remove_account(account_id: str):
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
        state["running"] = True
        return state

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

    # ── Subscription ─────────────────────────────────────────────
    @app.get("/api/device/subscription")
    async def device_subscription(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"error": "No accounts", "trial_active": False, "can_trade": False}
        ident = dev["accounts"][0]["identifier"]
        bal = 0.0
        if bot_pool:
            state = bot_pool.get_state(ident)
            if state and state.get("account") and not state["account"].get("error"):
                bal = state["account"].get("balance", 0)
        sub = await get_subscription(ident, bal)
        return sub

    @app.post("/api/device/subscription/check")
    async def device_subscription_check(device_id: str = Header(None, alias="X-Device-Id")):
        if not _db_ok():
            return _no_db()
        did = device_id or "unknown"
        dev = await get_device(did)
        if not dev or not dev.get("accounts"):
            return {"error": "No accounts", "trial_active": False, "can_trade": False}
        ident = dev["accounts"][0]["identifier"]
        bal = 0.0
        if bot_pool:
            state = bot_pool.get_state(ident)
            if state and state.get("account") and not state["account"].get("error"):
                bal = state["account"].get("balance", 0)
        sub = await get_subscription(ident, bal)
        if not sub.get("can_trade") and sub.get("trial_end"):
            if bot_pool:
                bot_pool.add_log(ident, "Trial ended. Subscription required to continue trading.", "WARNING")
        elif sub.get("trial_active"):
            dr = sub.get("days_remaining", 0)
            if dr == 1:
                if bot_pool:
                    bot_pool.add_log(ident, "Trial ends tomorrow! Subscribe to continue.", "WARNING")
            elif 2 <= dr <= 3:
                if bot_pool:
                    bot_pool.add_log(ident, f"Trial ending in {dr} day(s). Please subscribe.", "WARNING")
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
        return {"authorization_url": result.get("authorization_url"), "reference": result.get("reference")}

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

    return app
