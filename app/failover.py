"""
Failover integration for Gold Scalper.

Environment variables:
  FAILOVER_ROLE=primary|backup  — which role this instance plays
  FAILOVER_ENABLED=true|false   — enable/disable failover (default: false)
  DATABASE_URL=postgresql://...  — shared DB (REQUIRED for failover)

Each instance identifies itself by a unique instance id (RENDER_INSTANCE_ID,
then HOSTNAME, then hostname) rather than CAPITAL_IDENTIFIER, so multiple
instances can share the heartbeat table without overwriting each other's row.
"""
import asyncio
import json
import os
import socket
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
        self.instance_id = (
            os.environ.get("RENDER_INSTANCE_ID")
            or os.environ.get("HOSTNAME")
            or socket.gethostname()
            or f"{self.role}-{os.getpid()}"
        )
        self.is_leader = self.role == "primary"
        self._db = None
        self._last_beat = 0.0

    async def init_db(self, database):
        if not self.enabled:
            return
        self._db = database
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS failover_heartbeats (
                identifier TEXT PRIMARY KEY,
                role TEXT,
                last_beat FLOAT,
                is_leader INTEGER DEFAULT 0
            )
        """)
        try:
            await self._db.execute(
                "ALTER TABLE failover_heartbeats ADD COLUMN is_leader INTEGER DEFAULT 0"
            )
        except Exception:
            pass  # column already exists
        logger.info(f"Failover initialized as {self.role} (instance={self.instance_id})")

    async def send_heartbeat(self):
        if not self.enabled or not self._db:
            return
        try:
            self._last_beat = time.time()
            await self._db.execute("""
                INSERT INTO failover_heartbeats (identifier, role, last_beat, is_leader)
                VALUES (:identifier, :role, :last_beat, :is_leader)
                ON CONFLICT (identifier) DO UPDATE SET
                    role = :role, last_beat = :last_beat, is_leader = :is_leader
            """, {
                "identifier": self.instance_id,
                "role": self.role,
                "last_beat": self._last_beat,
                "is_leader": 1 if self.is_leader else 0,
            })
        except Exception as e:
            logger.error(f"Heartbeat send failed: {e}")

    async def check_primary_alive(self) -> bool:
        if not self._db:
            return True
        try:
            row = await self._db.fetch_one("""
                SELECT last_beat FROM failover_heartbeats
                WHERE role = 'primary' AND identifier != :identifier
                ORDER BY last_beat DESC LIMIT 1
            """, {"identifier": self.instance_id})
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

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "role": self.role,
            "instance_id": self.instance_id,
            "is_leader": self.is_leader,
            "last_beat": self._last_beat,
            "last_beat_age": round(time.time() - self._last_beat, 1) if self._last_beat else None,
        }
