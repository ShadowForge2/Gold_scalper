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
    active INTEGER NOT NULL DEFAULT 0,
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

CREATE_PENDING_ORDERS = """
CREATE TABLE IF NOT EXISTS pending_orders (
    order_id TEXT PRIMARY KEY,
    identifier TEXT NOT NULL,
    gateway TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

CREATE_PROCESSED_PAYMENTS = """
CREATE TABLE IF NOT EXISTS processed_payments (
    ref_key TEXT PRIMARY KEY,
    identifier TEXT NOT NULL,
    gateway TEXT NOT NULL,
    amount REAL NOT NULL,
    processed_at TEXT NOT NULL
)
"""

CREATE_NOTIFICATIONS_PG = """
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    identifier TEXT NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,
    is_read INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
)
"""

CREATE_NOTIFICATIONS_SQLITE = """
CREATE TABLE IF NOT EXISTS notifications (
    id TEXT PRIMARY KEY,
    identifier TEXT NOT NULL,
    type TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,
    is_read INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
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
        for sql in ALL_TABLES_SQLITE:
            await database.execute(sql)
        return True
    except Exception:
        return False


ALL_TABLES = [CREATE_DEVICES, CREATE_ACCOUNTS, CREATE_SUBSCRIPTIONS, CREATE_PERIODS, CREATE_PENDING_ORDERS, CREATE_PROCESSED_PAYMENTS]
ALL_TABLES_SQLITE = [*ALL_TABLES, CREATE_NOTIFICATIONS_SQLITE]
ALL_TABLES_PG = [*ALL_TABLES, CREATE_NOTIFICATIONS_PG]


async def init_db():
    global database, _env_url
    if _env_url:
        ok = await _try_pg()
        if ok:
            for sql in ALL_TABLES_PG:
                await database.execute(sql)
            return
    database = Database(SQLITE_URL)
    await _try_sqlite()
    # migration: add active column if missing
    for col in ["active"]:
        try:
            await database.execute(f"ALTER TABLE accounts ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
