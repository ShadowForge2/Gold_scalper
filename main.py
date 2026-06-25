import asyncio
import os
import uvicorn
from app.api import create_app
from app.bot import Bot
from app.bot_pool import BotPool
from app import database as db_mod
from app.database import init_db
from app.subscription import get_active_accounts, can_start_live
from app.capital_client import CapitalClient
import config as cfg

bot = Bot()
bot_pool = BotPool()
_db_connected = False


async def _try_start_user_bot(ident: str, api_key: str, password: str, demo: bool):
    if bot_pool.is_running(ident):
        return
    if not demo:
        try:
            if not await can_start_live(ident):
                bot.logger.warning(f"Skipping live account {ident}: subscription not active")
                return
        except Exception as e:
            bot.logger.warning(f"Sub check failed for {ident}: {e}. Will still attempt.")
    temp = CapitalClient()
    ok = temp.initialize(api_key=api_key, identifier=ident, password=password, demo=demo)
    temp.shutdown()
    if not ok:
        bot.logger.warning(f"Skipping account {ident}: credentials no longer valid")
        return
    result = bot_pool.start(identifier=ident, api_key=api_key, password=password, demo=demo)
    if result["success"]:
        bot.logger.info(f"Restored user bot: {ident}")


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
    asyncio.create_task(bot.run())
    if _db_connected:
        try:
            accounts = await get_active_accounts()
            for acct in accounts:
                asyncio.create_task(
                    _try_start_user_bot(
                        ident=acct["identifier"],
                        api_key=acct["api_key"],
                        password=acct["password"],
                        demo=bool(acct.get("demo", True)),
                    )
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
