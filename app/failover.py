"""
Failover integration for Gold Scalper.

Environment variables:
  FAILOVER_ROLE=primary|backup  — which role this instance plays
  FAILOVER_ENABLED=true|false   — enable/disable failover (default: false)
  DATABASE_URL=postgresql://...  — shared DB (REQUIRED for failover)
"""
import asyncio
import json
import os
import time
import logging

import config as cfg

logger = logging.getLogger("failover")

HEARTBEAT_INTERVAL = 30
HEARTBEAT_TIMEOUT = 120


class FailoverManager:
    def __init__(self):
        self.enabled = cfg._env_bool("FAILOVER_ENABLED", False)
        self.role = cfg._env_str("FAILOVER_ROLE", "primary")
        self.identifier = cfg.CAPITAL_IDENTIFIER
        self.is_leader = self.role == "primary"
        self._db = None

    async def init_db(self, database):
        if not self.enabled:
            return
        self._db = database
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS failover_heartbeats (
                identifier TEXT PRIMARY KEY,
                role TEXT,
                last_beat FLOAT
            )
        """)
        logger.info(f"Failover initialized as {self.role}")

    async def send_heartbeat(self):
        if not self.enabled or not self._db:
            return
        try:
            await self._db.execute("""
                INSERT INTO failover_heartbeats (identifier, role, last_beat)
                VALUES ($1, $2, $3)
                ON CONFLICT (identifier) DO UPDATE SET
                    role = $2, last_beat = $3
            """, [self.identifier, self.role, time.time()])
        except Exception as e:
            logger.error(f"Heartbeat send failed: {e}")

    async def check_primary_alive(self) -> bool:
        if not self._db:
            return True
        try:
            row = await self._db.fetch_one("""
                SELECT last_beat FROM failover_heartbeats
                WHERE role = 'primary' AND identifier != $1
            """, self.identifier)
            if not row:
                return False
            return (time.time() - row["last_beat"]) < HEARTBEAT_TIMEOUT
        except Exception:
            return True

    async def should_takeover(self) -> bool:
        if not self.enabled or self.role != "backup":
            return False
        alive = await self.check_primary_alive()
        if not alive and not self.is_leader:
            logger.warning("PRIMARY DOWN — taking over as leader!")
            self.is_leader = True
            return True
        if alive and self.is_leader:
            logger.info("Primary recovered — stepping back to backup")
            self.is_leader = False
        return False

    def can_trade(self) -> bool:
        if not self.enabled:
            return True
        return self.is_leader
