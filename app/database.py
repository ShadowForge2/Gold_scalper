import asyncio
import os
from databases import Database

SQLITE_URL = "sqlite+aiosqlite:///./gold_scalper.db"

_env_url = os.getenv("DATABASE_URL", "").strip()
if _env_url.startswith("postgresql://") and "+" not in _env_url:
    _env_url = _env_url.replace("postgresql://", "postgresql+asyncpg://", 1)

database = Database(_env_url if _env_url else SQLITE_URL)

CREATE_DEVICES = """
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    first_seen TEXT NOT NULL,
    restored_from TEXT
)
"""

CREATE_ACCOUNTS = """
CREATE TABLE IF NOT EXISTS accounts (
    device_id TEXT NOT NULL,
    api_key TEXT NOT NULL,
    identifier TEXT NOT NULL,
    password TEXT NOT NULL,
    demo INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (device_id, identifier)
)
"""

CREATE_SUBSCRIPTIONS = """
CREATE TABLE IF NOT EXISTS subscriptions (
    identifier TEXT PRIMARY KEY,
    first_connected_at TEXT,
    trial_end TEXT,
    subscribed INTEGER DEFAULT 0,
    subscription_end TEXT,
    paid_amount REAL DEFAULT 0.0
)
"""

CREATE_PERIODS = """
CREATE TABLE IF NOT EXISTS monthly_periods (
    identifier TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT,
    starting_balance REAL,
    ending_balance REAL,
    cumulative_profit REAL DEFAULT 0.0,
    fee_15pct REAL DEFAULT 0.0,
    fee_paid INTEGER DEFAULT 0,
    paid_at TEXT,
    PRIMARY KEY (identifier, period_start)
)
"""


async def _try_pg() -> bool:
    try:
        async with asyncio.timeout(5):
            await database.connect()
            await database.execute("SELECT 1")
            return True
    except Exception:
        return False


async def _try_sqlite() -> bool:
    try:
        await database.connect()
        for sql in [CREATE_DEVICES, CREATE_ACCOUNTS, CREATE_SUBSCRIPTIONS, CREATE_PERIODS]:
            await database.execute(sql)
        return True
    except Exception:
        return False


async def init_db():
    global database, _env_url
    if _env_url:
        ok = await _try_pg()
        if ok:
            for sql in [CREATE_DEVICES, CREATE_ACCOUNTS, CREATE_SUBSCRIPTIONS, CREATE_PERIODS]:
                await database.execute(sql)
            return
    database = Database(SQLITE_URL)
    await _try_sqlite()
