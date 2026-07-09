import asyncio
import os
import uvicorn
from app.api import create_app
from app.bot import Bot
from app.bot_pool import BotPool
from app import database as db_mod
from app.database import init_db
from app.subscription import get_active_accounts, can_start_live, start_trial
import config as cfg

bot = Bot()
bot_pool = BotPool()
_db_connected = False


def _fire_task(coro, name: str = "task"):
    task = asyncio.create_task(coro)
    task.add_done_callback(
        lambda t: bot.logger.error(
            f"{name} failed: {t.exception()}"
        ) if t.exception() else None
    )
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
        await db_mod.database.disconnect()


def is_db_connected() -> bool:
    return _db_connected


app = create_app(bot, bot_pool=bot_pool, db_check=is_db_connected)


@app.on_event("startup")
async def startup():
    await startup_db()
    await bot.initialize()
    _fire_task(bot.run(), name="bot.run")
    if _db_connected:
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


@app.on_event("shutdown")
async def shutdown():
    await bot.shutdown()
    bot_pool.stop_all()
    await shutdown_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.API_PORT))
    uvicorn.run(app, host=cfg.API_HOST, port=port)
