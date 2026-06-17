import asyncio
import os
import uvicorn
from app.api import create_app
from app.bot import Bot
from app.bot_pool import BotPool
from app.database import database, init_db
import config as cfg

bot = Bot()
bot_pool = BotPool()
app = create_app(bot, bot_pool=bot_pool)


@app.on_event("startup")
async def startup():
    await database.connect()
    await init_db()
    await bot.initialize()
    asyncio.create_task(bot.run())


@app.on_event("shutdown")
async def shutdown():
    await bot.shutdown()
    bot_pool.stop_all()
    await database.disconnect()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.API_PORT))
    uvicorn.run(app, host=cfg.API_HOST, port=port)
