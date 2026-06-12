import asyncio
import os
import uvicorn
from app.api import create_app
from app.bot import Bot
import config as cfg

bot = Bot()
app = create_app(bot)


@app.on_event("startup")
async def startup():
    await bot.initialize()
    asyncio.create_task(bot.run())


@app.on_event("shutdown")
async def shutdown():
    await bot.shutdown()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", cfg.API_PORT))
    uvicorn.run(app, host=cfg.API_HOST, port=port)
