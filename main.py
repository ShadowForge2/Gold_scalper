import asyncio
import logging
import os
import uvicorn
from app.api import create_app
from app.bot import Bot
from app.bot_pool import BotPool
from app.database import database, init_db
import config as cfg

bot = Bot()
bot_pool = BotPool()

_db_connected = False


async def startup_db():
    global _db_connected
    try:
        await database.connect()
        await init_db()
        _db_connected = True
        bot.logger.info("Database connected")
    except Exception as e:
        _db_connected = False
        bot.logger.warning(f"Database unavailable ({e}). Running without DB.")


async def shutdown_db():
    if _db_connected:
        await database.disconnect()


def is_db_connected() -> bool:
    return _db_connected


app = create_app(bot, bot_pool=bot_pool, db_check=is_db_connected)


@app.on_event("startup")
async def startup():
    await startup_db()
    await bot.initialize()
    asyncio.create_task(bot.run())


@app.on_event("shutdown")
async def shutdown():
    await bot.shutdown()
    bot_pool.stop_all()
    await shutdown_db()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.API_PORT))
    uvicorn.run(app, host=cfg.API_HOST, port=port)
