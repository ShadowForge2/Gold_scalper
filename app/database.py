import asyncio
import os
from databases import Database

PG_URL = os.getenv("DATABASE_URL", "")
SQLITE_URL = "sqlite+aiosqlite:///./gold_scalper.db"

database = Database(SQLITE_URL)
_using_pg = False

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


async def init_db():
    global database, _using_pg
    if PG_URL:
        pg_url = PG_URL
        if pg_url.startswith("postgresql://") and "+" not in pg_url:
            pg_url = pg_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        try:
            db = Database(pg_url, min_size=1, max_size=1)
            await asyncio.wait_for(db.connect(), timeout=5.0)
            await db.execute("SELECT 1")
            database = db
            _using_pg = True
        except Exception:
            pass
    if not _using_pg:
        await database.connect()
    for sql in [CREATE_DEVICES, CREATE_ACCOUNTS, CREATE_SUBSCRIPTIONS, CREATE_PERIODS]:
        await database.execute(sql)
