import asyncio
import os
from datetime import datetime
import uvicorn
from app.api import create_app
from app.bot import Bot
from app.bot_pool import BotPool
from app import database as db_mod
from app.database import init_db
from app.subscription import get_active_accounts, can_start_live, start_trial, get_subscription
from app.failover import FailoverManager, HEARTBEAT_INTERVAL, OP_TIMEOUT
from app import email_service
import config as cfg

bot = Bot()
bot_pool = BotPool()
failover = FailoverManager()
_db_connected = False

_background_tasks: set = set()


def _fire_task(coro, name: str = "task"):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)

    def _done(t):
        _background_tasks.discard(t)
        exc = t.exception()
        if exc and not isinstance(exc, asyncio.CancelledError):
            bot.logger.error(f"{name} failed: {exc}")

    task.add_done_callback(_done)
    return task


async def _try_start_user_bot(ident: str, api_key: str, password: str, demo: bool):
    if bot_pool.is_running(ident):
        return
    if not demo:
        try:
            if not await can_start_live(ident):
                bot.logger.warning(f"Skipping live account {ident}: subscription not active")
                return
            await start_trial(ident, 0.0)
        except Exception as e:
            bot.logger.warning(f"Sub check failed for {ident}: {e}. Will still attempt.")
    result = bot_pool.start(identifier=ident, api_key=api_key, password=password, demo=demo)
    if result["success"]:
        bot.logger.info(f"Restored user bot: {ident}")
    else:
        bot.logger.warning(f"Failed to restore user bot {ident}: {result.get('error', 'unknown')}")


async def startup_db():
    global _db_connected
    try:
        await init_db()
        _db_connected = True
        bot.logger.info("Database connected")
    except Exception as e:
        _db_connected = False
        bot.logger.warning(f"Database unavailable ({e}). Running without DB.")


async def shutdown_db():
    if _db_connected:
        try:
            await asyncio.wait_for(db_mod.database.disconnect(), timeout=15)
        except Exception as e:
            bot.logger.warning(f"Database disconnect failed: {e}")


def is_db_connected() -> bool:
    return _db_connected


def _failover_ready() -> bool:
    client = getattr(bot, "client", None)
    if client is None:
        return False
    try:
        return bool(client.is_connected())
    except Exception:
        return False


async def _failover_step():
    """One full heartbeat cycle. Kept separate so it can be time-boxed."""
    await failover.send_heartbeat()
    if await failover.should_takeover():
        bot.logger.warning("FAILOVER: acquired leader lease — primary mode activated")
        if bot.state == bot.STATES["STOPPED"]:
            await bot.initialize()
            bot._account_id = cfg.CAPITAL_IDENTIFIER
            _fire_task(bot.run(), name="bot.run_recovered")
        await _restore_user_bots()
    elif failover.should_step_down():
        bot.logger.warning(
            "FAILOVER: lost leader lease — backup mode activated, trading paused"
        )
        bot_pool.stop_all(close_positions=False)
        bot.logger.warning("FAILOVER: stopped user bots — backup is heartbeat-only")


async def failover_heartbeat_loop():
    while True:
        try:
            age = failover.heartbeat_age()
            if age > HEARTBEAT_INTERVAL * 3:
                bot.logger.warning(
                    f"FAILOVER: heartbeat stale ({age:.0f}s) — self-healing, forcing beat"
                )
            try:
                await asyncio.wait_for(_failover_step(), timeout=OP_TIMEOUT * 3)
            except asyncio.TimeoutError:
                bot.logger.error("FAILOVER: heartbeat step timed out — skipping this cycle")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bot.logger.error(f"Failover heartbeat error: {e}")
        except asyncio.CancelledError:
            bot.logger.info("FAILOVER: heartbeat loop cancelled — shutting down")
            break
        except Exception as e:
            bot.logger.error(f"Failover heartbeat loop failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _restore_user_bots():
    if not _db_connected:
        return
    try:
        accounts = await get_active_accounts()
        for acct in accounts:
            _fire_task(
                _try_start_user_bot(
                    ident=acct["identifier"],
                    api_key=acct["api_key"],
                    password=acct["password"],
                    demo=bool(acct.get("demo", True)),
                ),
                name=f"user_bot_{acct.get('identifier', '?')}",
            )
        if accounts:
            bot.logger.info(f"Scheduled {len(accounts)} user bot(s) for restoration")
    except Exception as e:
        bot.logger.warning(f"Failed to restore user bots: {e}")


# ── Email scheduler (trial reminders / fee due / promos / daily PnL) ──

_EMAIL_SCHEDULE_INTERVAL_SEC = int(os.getenv("EMAIL_SCHEDULE_INTERVAL_SEC", "3600"))


def _closed_today_pnl(state) -> float:
    """Sum today's closed-trade PnL from a bot_pool state dict."""
    try:
        closed = (state.get("bot") or {}).get("closed_trades", []) or []
    except Exception:
        return 0.0
    today = datetime.utcnow().strftime("%Y-%m-%d")
    total = 0.0
    for t in closed:
        ca = (t.get("closed_at") or "") if isinstance(t, dict) else ""
        if ca[:10] == today:
            try:
                total += float(t.get("profit", 0) or 0)
            except (TypeError, ValueError):
                pass
    return round(total, 2)


async def _email_scheduler_one(ident: str, state=None):
    """Handle a single account's scheduled emails (best-effort, never raises)."""
    try:
        sub = await get_subscription(ident)
    except Exception as e:
        bot.logger.debug(f"Email scheduler sub lookup failed for {ident}: {e}")
        return

    now = datetime.utcnow()

    # 1) Trial reminders — milestone-based so we never spam: exactly at
    #    14 days and 7 days before the trial ends. The final "trial ended"
    #    notice is sent below when the trial actually expires.
    if sub.get("trial_active"):
        dr = sub.get("days_remaining", 0)
        if dr in (14, 7):
            await email_service.email_billing_trial_ending(ident, dr)
        return

    # 2) Expired / not subscribed — trial-expired, fee-due, and promo nudges.
    if not sub.get("subscribed"):
        trial_end = sub.get("trial_end")
        expired = False
        if trial_end:
            try:
                expired = datetime.fromisoformat(trial_end) <= now
            except (ValueError, TypeError):
                expired = False
        if expired:
            await email_service.email_billing_trial_expired(ident)
            due = sub.get("due_amount", 0) or 0
            if due > 0:
                await email_service.email_billing_fee_due(ident, float(due))
            await email_service.email_promo(ident)

    # 3) Daily PnL recap — only for running live bots with trades today.
    if state is not None:
        pnl = _closed_today_pnl(state)
        if pnl != 0.0:
            day_label = now.strftime("%A, %b %d")
            account = state.get("account") or {}
            balance = float(account.get("balance", 0) or 0)
            trades = 0
            try:
                trades = len((state.get("bot") or {}).get("closed_trades", []) or [])
            except Exception:
                pass
            await email_service.email_daily_pnl(ident, pnl, day_label, trades, balance)


async def email_scheduler_step():
    if not _db_connected or not email_service.email_configured():
        return
    # With two instances (primary + backup) sharing one DB, only the active
    # leader runs the scheduled sends — otherwise both would send the same
    # trial reminder / promo / daily PnL recap. Transactional emails fired
    # from API requests (payment receipts, trade alerts) are unaffected.
    if failover.enabled:
        try:
            if not await failover.can_trade():
                return
        except Exception as e:
            bot.logger.warning(f"Email scheduler leader check failed: {e}")
    try:
        rows = await db_mod.fetch_all("SELECT DISTINCT identifier FROM accounts")
    except Exception as e:
        bot.logger.warning(f"Email scheduler: account scan failed: {e}")
        return
    if not rows:
        return
    for row in rows:
        ident = row["identifier"]
        state = None
        try:
            if bot_pool.is_running(ident):
                state = await asyncio.to_thread(bot_pool.get_state, ident)
        except Exception:
            state = None
        try:
            await _email_scheduler_one(ident, state)
        except Exception as e:
            bot.logger.debug(f"Email scheduler step failed for {ident}: {e}")


async def email_scheduler_loop():
    while True:
        try:
            await email_scheduler_step()
        except asyncio.CancelledError:
            bot.logger.info("Email scheduler cancelled — shutting down")
            raise
        except Exception as e:
            bot.logger.error(f"Email scheduler loop failed: {e}")
        await asyncio.sleep(_EMAIL_SCHEDULE_INTERVAL_SEC)


app = create_app(bot, bot_pool=bot_pool, db_check=is_db_connected)


@app.on_event("startup")
async def startup():
    import asyncio as _aio
    from app.subscription import set_main_loop
    set_main_loop(_aio.get_running_loop())
    await startup_db()
    if _db_connected:
        await failover.init_db(
            db_url=str(db_mod.database.url),
            database=db_mod.database,
        )
    await bot.initialize()
    bot._account_id = cfg.CAPITAL_IDENTIFIER
    bot._failover = failover
    failover.set_readiness_fn(_failover_ready)
    _fire_task(bot.run(), name="bot.run")
    if failover.enabled:
        _fire_task(failover_heartbeat_loop(), name="failover.heartbeat")
    if _db_connected:
        _fire_task(email_scheduler_loop(), name="email.scheduler")
        if failover.enabled:
            bot.logger.info("FAILOVER enabled — user bots start only on lease leadership")
        else:
            await _restore_user_bots()


@app.on_event("shutdown")
async def shutdown():
    try:
        await failover.release_lease()
    except Exception as e:
        bot.logger.warning(f"Failover lease release on shutdown failed: {e}")
    for t in list(_background_tasks):
        t.cancel()
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)
    await bot.shutdown()
    bot_pool.stop_all()
    await shutdown_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.API_PORT))
    uvicorn.run(app, host=cfg.API_HOST, port=port)
