"""
Failover integration for Gold Scalper — robust version.

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

Robustness changes vs the original implementation:

  1. Dedicated DB layer.  When DATABASE_URL points at Postgres, the failover
     manager opens its OWN small asyncpg pool (min_size=1, max_size=2) instead
     of sharing the app-wide `databases` object.  The `databases` library
     wraps every asyncio task in a per-task connection wrapper whose acquire
     assertion ("Connection is already acquired") is fragile under
     cancellation/concurrency; sharing one instance across the bot, the API,
     subscription writes and the heartbeat repeatedly tripped it, which froze
     the bot out of its own account.  A dedicated pool sidesteps that layer
     entirely and is concurrency-safe by design.  When DATABASE_URL is not a
     Postgres URL (e.g. local SQLite), failover falls back to the shared
     `databases` object, serialized through a lock so SQLite never sees
     concurrent writers.

  2. Serialized + retried operations.  Every failover DB call runs under a
     single asyncio.Lock and is retried with backoff on transient failures
     ("already acquired", "database is locked", timeouts, connection errors).
     A single hiccup can no longer kill a heartbeat cycle.

  3. Resilient leadership.  A failed heartbeat preserves the last known lease
     instead of dropping it: the bot keeps trading (and keeps managing its
     positions) for as long as the lease it already holds is still valid.
     `can_trade()` also self-heals — if the lease is free (expired) it
     re-acquires it inline, so a dead/restarted heartbeat loop cannot leave
     the bot permanently frozen.

  4. Management never blocks on leadership.  `can_manage()` returns True
     unless another instance demonstrably holds a fresh lease, so position
     management, safety exits and reconnect logic always run even during a
     failover transition or a DB outage.
"""
import asyncio
import os
import re
import socket
import time
import logging

import config as cfg

logger = logging.getLogger("failover")

HEARTBEAT_INTERVAL = 30
HEARTBEAT_TIMEOUT = 120          # lease length; failover latency if the leader stops renewing
OP_TIMEOUT = 15.0                # max time a single DB call may take before it's aborted
LEADER_LOCK = "__leader__"

RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 0.5

# Substrings that mark a failure as transient / retryable.
TRANSIENT_MARKERS = (
    "already acquired",
    "not acquired",
    "database is locked",
    "database is busy",
    "busy",
    "deadlock",
    "connection refused",
    "connection reset",
    "cannot connect",
    "timed out",
    "timeout",
    "pool exhausted",
    "server closed the connection",
    "ssl error",
)

_NAMED_PARAM = re.compile(r":([A-Za-z_][A-Za-z0-9_]*)")


def _to_positional(sql, values):
    """Convert SQLAlchemy-style :name bind params to asyncpg $1..$n."""
    if not values:
        return sql, []
    keys = []

    def _repl(m):
        keys.append(m.group(1))
        return f"${len(keys)}"

    sql = _NAMED_PARAM.sub(_repl, sql)
    args = [values[k] for k in keys]
    return sql, args


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
        self._db = None                   # fallback: shared `databases` Database (sqlite)
        self._pool = None                 # primary: dedicated asyncpg pool (postgres)
        self._db_url = ""
        self._readiness_fn = readiness_fn
        self._last_beat = 0.0
        self._lease_expires_at = 0.0
        self._last_lease_check = 0.0
        self._last_success = 0.0
        self._mode = "unknown"
        self._became_leader = False
        self._lost_leader = False
        self._op_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # DB layer
    # ------------------------------------------------------------------
    async def _raw_run(self, sql, values, kind):
        if self._pool is not None:
            pg_sql, args = _to_positional(sql, values or {})
            async with self._pool.acquire() as conn:
                if kind == "execute":
                    return await conn.execute(pg_sql, *args)
                return await conn.fetchrow(pg_sql, *args)
        if self._db is None:
            raise RuntimeError("Failover DB not initialized")
        # Shared `databases` object (sqlite fallback): go through the app-wide
        # serialized layer so failover can never wedge it and vice-versa.
        from app import database as db_mod
        if kind == "execute":
            return await db_mod.execute(sql, values or {})
        return await db_mod.fetch_one(sql, values or {})

    @staticmethod
    def _is_transient(exc) -> bool:
        msg = str(exc).lower()
        return any(m in msg for m in TRANSIENT_MARKERS)

    async def _retry(self, op_factory):
        last_exc = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return await asyncio.wait_for(op_factory(), timeout=OP_TIMEOUT)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                if not self._is_transient(e):
                    raise
                logger.warning(
                    f"failover DB transient failure "
                    f"(attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}"
                )
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
        raise last_exc

    async def _exec(self, sql, values=None):
        """Serialized, retried execute with a hard timeout."""
        async with self._op_lock:
            return await self._retry(lambda: self._raw_run(sql, values or {}, "execute"))

    async def _fetchrow(self, sql, values=None):
        """Serialized, retried fetch_one with a hard timeout."""
        async with self._op_lock:
            return await self._retry(lambda: self._raw_run(sql, values or {}, "fetchrow"))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    async def init_db(self, db_url=None, database=None):
        if not self.enabled:
            return
        self._db = database
        url = str(db_url or "")
        if not url and database is not None:
            try:
                url = str(database.url)
            except Exception:
                url = ""
        self._db_url = url

        if "postgres" in url:
            pg_url = url.replace("postgresql+asyncpg://", "postgresql://", 1)
            try:
                import asyncpg
                self._pool = await asyncio.wait_for(
                    asyncpg.create_pool(
                        pg_url, min_size=1, max_size=2, command_timeout=OP_TIMEOUT
                    ),
                    timeout=OP_TIMEOUT,
                )
                logger.info(
                    f"Failover using dedicated asyncpg pool (instance={self.instance_id})"
                )
            except Exception as e:
                self._pool = None
                logger.error(
                    f"Failover could not open dedicated asyncpg pool ({e}); "
                    f"falling back to shared database"
                )
        else:
            logger.info(
                f"Failover using shared database ({url or 'unknown backend'}; "
                f"instance={self.instance_id})"
            )

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
            if self._pool is not None:
                try:
                    await self._pool.close()
                except Exception:
                    pass
                self._pool = None
            self._db = None
            self.enabled = False
            logger.error(f"Failover init failed ({e}); failover disabled — running without a lease")
            return
        logger.info(f"Failover initialized as {self.role} (instance={self.instance_id})")

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

    # ------------------------------------------------------------------
    # Lease
    # ------------------------------------------------------------------
    async def _acquire_or_renew_lease(self) -> bool:
        """Atomically take or renew the leader lease. Returns True if this
        instance now holds the lease."""
        now = time.time()
        expires = now + HEARTBEAT_TIMEOUT
        row = await self._fetchrow("""
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
        if not self.enabled or (self._pool is None and self._db is None):
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
                self._last_success = time.time()
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
                age = "<1s"
                try:
                    row = await self._fetchrow(
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
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Transient failure: preserve the last-known lease instead of
            # dropping leadership, so the bot is never frozen out of its own
            # account by a single DB hiccup.
            logger.error(
                f"Heartbeat send failed: {e} — keeping last-known lease "
                f"(expires in {max(0.0, self._lease_expires_at - time.time()):.0f}s)"
            )

    async def release_lease(self):
        """Gracefully give up the leader lease (shutdown / not-ready)."""
        if not self.enabled or (self._pool is None and self._db is None):
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
        if self._pool is None and self._db is None:
            return True
        try:
            row = await self._fetchrow(
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
        """Whether this instance may open NEW positions.

        Never returns False just because of a transient DB error: on failure it
        preserves the last-known lease until it expires. If the lease is free
        (expired) it re-acquires it inline, so a dead or restarted heartbeat
        loop cannot leave the bot permanently frozen.
        """
        if not self.enabled:
            return True
        now = time.time()
        if self._last_lease_check and now - self._last_lease_check < 5.0:
            return self.is_leader and now < self._lease_expires_at
        self._last_lease_check = now
        try:
            row = await self._fetchrow(
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
            if mine and fresh:
                self._lease_expires_at = row["expires_at"]
            elif not mine:
                self._lease_expires_at = 0.0
            if not fresh and not (self._last_beat and now - self._last_beat <= HEARTBEAT_INTERVAL):
                # Lease is free and the heartbeat has not just run — acquire
                # it here so a single instance never gets stuck.
                self._last_lease_check = 0.0
                acquired = await self._acquire_or_renew_lease()
                if acquired:
                    self.is_leader = True
                    self._lease_expires_at = time.time() + HEARTBEAT_TIMEOUT
                    self._last_success = time.time()
                    logger.warning(
                        f"can_trade: lease was free — acquired leadership inline "
                        f"(instance={self.instance_id})"
                    )
            return self.is_leader
        except asyncio.CancelledError:
            raise
        except Exception:
            # Transient DB failure: keep last-known state until the lease
            # expires, so management and trading continue through a hiccup.
            return self.is_leader and now < self._lease_expires_at

    async def can_manage(self) -> bool:
        """Whether this instance may manage (and exit) open positions.

        Intentionally permissive: exits/management must never freeze on a
        failover transition or DB outage. Only a demonstrably fresh lease held
        by another instance makes this return False.
        """
        if not self.enabled:
            return True
        now = time.time()
        if self.is_leader and now < self._lease_expires_at:
            return True
        # We have been heartbeating recently (single-instance self-heal) or
        # hold leadership up to its lease — keep managing.
        if self._last_success and now - self._last_success <= HEARTBEAT_TIMEOUT:
            return True
        try:
            row = await self._fetchrow(
                "SELECT owner, expires_at FROM failover_heartbeats WHERE identifier = :id",
                {"id": LEADER_LOCK},
            )
            if row and row["expires_at"] and row["expires_at"] > now:
                return row["owner"] == self.instance_id   # fresh lease: only its owner manages
            return True        # nobody leads — manage (can_trade will acquire)
        except Exception:
            return True        # DB down: never freeze the account

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self._mode,
            "role": self.role,
            "instance_id": self.instance_id,
            "backend": "asyncpg" if self._pool is not None else "shared",
            "is_leader": self.is_leader,
            "last_beat": self._last_beat,
            "last_beat_age": round(time.time() - self._last_beat, 1) if self._last_beat else None,
            "lease_expires_in": round(max(0.0, self._lease_expires_at - time.time()), 1)
                                if self._lease_expires_at else None,
            "lease_holder": self.instance_id if self.is_leader else None,
        }
