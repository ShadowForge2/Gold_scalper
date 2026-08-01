"""
Failover integration for Gold Scalper.

Environment variables:
  FAILOVER_ROLE=primary|backup  — which role this instance plays
  FAILOVER_ENABLED=true|false   — enable/disable failover (default: false)
  DATABASE_URL=postgresql://...  — shared DB (REQUIRED for failover)

Leadership model:
  A single "leader lease" row (identifier '__leader__') in the shared DB acts
  as the lock. Every instance heartbeats its own row for observability, and
  the leader renews the lease every HEARTBEAT_INTERVAL. A ready instance may
  acquire the lease only when the previous owner's lease has expired, so at
  most one instance trades at a time (no split-brain), and a recovered old
  primary automatically steps down because it can no longer renew a lease it
  doesn't own.

Readiness:
  A readiness callback (set via set_readiness_fn) decides whether the instance
  is actually able to trade (account reachable, main loop healthy). A leader
  that is not ready stops renewing the lease, so a ready fallback can take
  over once the lease expires.
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
HEARTBEAT_TIMEOUT = 120          # lease length; failover latency if the leader stops renewing
OP_TIMEOUT = 15.0                # max time a single DB call may take before it's aborted
LEADER_LOCK = "__leader__"


class FailoverManager:
    def __init__(self, readiness_fn=None):
        self.enabled = cfg._env_bool("FAILOVER_ENABLED", False)
        self.role = cfg._env_str("FAILOVER_ROLE", "primary")
        self.instance_id = (
            os.environ.get("RENDER_INSTANCE_ID")
            or os.environ.get("HOSTNAME")
            or socket.gethostname()
            or f"{self.role}-{os.getpid()}"
        )
        self.is_leader = False            # leadership decided by the lease, not the role
        self._db = None
        self._readiness_fn = readiness_fn
        self._last_beat = 0.0
        self._lease_expires_at = 0.0
        self._last_lease_check = 0.0
        self._mode = "unknown"
        self._became_leader = False
        self._lost_leader = False

    def set_readiness_fn(self, fn):
        self._readiness_fn = fn

    def _ready(self) -> bool:
        if not self._readiness_fn:
            return True
        try:
            return bool(self._readiness_fn())
        except Exception as e:
            logger.error(f"Readiness check failed ({e}); treating as not ready")
            return False

    async def _exec(self, sql, values=None):
        """DB execute with a hard timeout so a hung connection can never
        stall the heartbeat loop."""
        return await asyncio.wait_for(
            self._db.execute(sql, values or {}), timeout=OP_TIMEOUT
        )

    async def _fetch(self, sql, values=None):
        """DB fetch_one with a hard timeout (see _exec)."""
        return await asyncio.wait_for(
            self._db.fetch_one(sql, values or {}), timeout=OP_TIMEOUT
        )

    async def init_db(self, database):
        if not self.enabled:
            return
        self._db = database
        try:
            await self._exec("""
                CREATE TABLE IF NOT EXISTS failover_heartbeats (
                    identifier TEXT PRIMARY KEY,
                    role TEXT,
                    last_beat FLOAT,
                    is_leader INTEGER DEFAULT 0,
                    owner TEXT,
                    expires_at FLOAT
                )
            """)
            for col, ddl in (("is_leader", "INTEGER DEFAULT 0"),
                             ("owner", "TEXT"),
                             ("expires_at", "FLOAT")):
                try:
                    await self._exec(
                        f"ALTER TABLE failover_heartbeats ADD COLUMN {col} {ddl}"
                    )
                except Exception:
                    pass  # column already exists
            await self._exec("""
                INSERT INTO failover_heartbeats (identifier, role, last_beat, is_leader, owner, expires_at)
                VALUES (:id, '', 0.0, 0, NULL, 0.0)
                ON CONFLICT (identifier) DO NOTHING
            """, {"id": LEADER_LOCK})
        except Exception as e:
            self._db = None
            self.enabled = False
            logger.error(f"Failover init failed ({e}); failover disabled — running without a lease")
            return
        logger.info(f"Failover initialized as {self.role} (instance={self.instance_id})")

    async def _acquire_or_renew_lease(self) -> bool:
        """Atomically take or renew the leader lease. Returns True if this
        instance now holds the lease."""
        now = time.time()
        expires = now + HEARTBEAT_TIMEOUT
        row = await self._fetch("""
            INSERT INTO failover_heartbeats (identifier, role, last_beat, is_leader, owner, expires_at)
            VALUES (:id, :role, :now, 1, :owner, :expires)
            ON CONFLICT (identifier) DO UPDATE SET
                role = :role,
                last_beat = :now,
                is_leader = 1,
                owner = :owner,
                expires_at = :expires
            WHERE failover_heartbeats.owner = :owner
               OR failover_heartbeats.expires_at < :now
            RETURNING identifier
        """, {
            "id": LEADER_LOCK,
            "role": self.role,
            "now": now,
            "owner": self.instance_id,
            "expires": expires,
        })
        return row is not None

    async def _write_instance_beat(self):
        """Record this instance's own heartbeat row for observability."""
        try:
            await self._exec("""
                INSERT INTO failover_heartbeats (identifier, role, last_beat, is_leader, owner, expires_at)
                VALUES (:id, :role, :beat, :is_leader, NULL, 0.0)
                ON CONFLICT (identifier) DO UPDATE SET
                    role = :role, last_beat = :beat, is_leader = :is_leader
            """, {
                "id": self.instance_id,
                "role": self.role,
                "beat": self._last_beat,
                "is_leader": 1 if self.is_leader else 0,
            })
        except Exception as e:
            logger.error(f"Instance heartbeat write failed: {e}")

    async def send_heartbeat(self):
        if not self.enabled or not self._db:
            return
        self._became_leader = False
        self._lost_leader = False
        try:
            self._last_beat = time.time()
            was_leader = self.is_leader
            ready = self._ready()

            if not ready:
                # Do not renew — the lease will expire and a ready fallback
                # can take over. Keep last-known leadership locally.
                logger.warning(
                    f"NOT READY — pausing heartbeat renewal (lease expires in "
                    f"{max(0.0, self._lease_expires_at - time.time()):.0f}s). "
                    f"{'PRIMARY' if was_leader else 'BACKUP'} heartbeat not sent."
                )
                await self._write_instance_beat()
                return

            acquired = await self._acquire_or_renew_lease()
            self.is_leader = acquired
            if acquired:
                self._lease_expires_at = time.time() + HEARTBEAT_TIMEOUT
            else:
                self._lease_expires_at = 0.0

            await self._write_instance_beat()

            if acquired and not was_leader:
                self._became_leader = True
                self._mode = "primary"
                logger.warning(f"PRIMARY MODE ACTIVATED — acquired leader lease (instance={self.instance_id})")
            elif not acquired and was_leader:
                self._lost_leader = True
                self._mode = "backup"
                logger.warning(f"BACKUP MODE ACTIVATED — lost leader lease to another instance")
            elif acquired:
                self._mode = "primary"
                left = max(0.0, self._lease_expires_at - time.time())
                logger.info(f"PRIMARY HEARTBEAT sent — lease ok, expires in {left:.0f}s")
            else:
                self._mode = "backup"
                alive = await self.check_primary_alive()
                age = f"<1s"
                try:
                    row = await self._fetch(
                        "SELECT expires_at FROM failover_heartbeats WHERE identifier = :id",
                        {"id": LEADER_LOCK},
                    )
                    if row and row["expires_at"]:
                        age = f"{max(0.0, row['expires_at'] - time.time()):.0f}s"
                except Exception:
                    pass
                logger.info(
                    f"BACKUP MODE — checking primary heartbeat "
                    f"(primary_alive={alive}, lease_left={age})"
                )
        except Exception as e:
            logger.error(f"Heartbeat send failed: {e}")

    async def release_lease(self):
        """Gracefully give up the leader lease (shutdown / not-ready)."""
        if not self.enabled or not self._db:
            return
        try:
            await self._exec("""
                UPDATE failover_heartbeats
                SET is_leader = 0, owner = NULL, expires_at = 0.0
                WHERE identifier = :id AND owner = :owner
            """, {"id": LEADER_LOCK, "owner": self.instance_id})
            self.is_leader = False
            self._lease_expires_at = 0.0
            logger.warning("Leader lease released (backup mode now)")
        except Exception as e:
            logger.error(f"Lease release failed: {e}")

    async def check_primary_alive(self) -> bool:
        """True if someone currently holds a live leader lease."""
        if not self._db:
            return True
        try:
            row = await self._fetch(
                "SELECT expires_at FROM failover_heartbeats WHERE identifier = :id",
                {"id": LEADER_LOCK},
            )
            if not row or not row["expires_at"]:
                return False
            return row["expires_at"] > time.time()
        except Exception:
            return True

    async def should_takeover(self) -> bool:
        return self._became_leader

    def should_step_down(self) -> bool:
        return self._lost_leader

    def heartbeat_age(self) -> float:
        """Seconds since this instance last beat (sentinel if never)."""
        if not self._last_beat:
            return HEARTBEAT_INTERVAL * 100
        return time.time() - self._last_beat

    async def can_trade(self) -> bool:
        if not self.enabled:
            return True
        now = time.time()
        if self._last_lease_check and now - self._last_lease_check < 5.0:
            return self.is_leader and now < self._lease_expires_at
        self._last_lease_check = now
        try:
            row = await self._fetch(
                "SELECT owner, expires_at FROM failover_heartbeats WHERE identifier = :id",
                {"id": LEADER_LOCK},
            )
            if not row:
                self.is_leader = False
                self._lease_expires_at = 0.0
                return False
            mine = row["owner"] == self.instance_id
            fresh = bool(row["expires_at"] and row["expires_at"] > now)
            self.is_leader = mine and fresh
            if not mine:
                self._lease_expires_at = 0.0
            return self.is_leader
        except Exception:
            return self.is_leader and now < self._lease_expires_at

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self._mode,
            "role": self.role,
            "instance_id": self.instance_id,
            "is_leader": self.is_leader,
            "last_beat": self._last_beat,
            "last_beat_age": round(time.time() - self._last_beat, 1) if self._last_beat else None,
            "lease_expires_in": round(max(0.0, self._lease_expires_at - time.time()), 1)
                                if self._lease_expires_at else None,
            "lease_holder": self.instance_id if self.is_leader else None,
        }
