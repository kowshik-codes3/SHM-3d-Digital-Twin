"""SQLite DatabaseManager for the SHM Digital Twin persistence layer.

Uses Python's built-in sqlite3 with an asyncio lock for thread-safe access.
The DB file is stored at backend/data/shm.db (created on first run).
"""

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger("SHM.Database")

# Default DB path relative to repo root — created automatically
_DEFAULT_DB_PATH = Path(__file__).parents[3] / "data" / "shm.db"


class DatabaseManager:
    """Async-safe SQLite wrapper.

    Parameters
    ----------
    db_path : Path or str
        Filesystem path to the SQLite database file.
        Parent directories are created automatically.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Open connection and create schema (idempotent)."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")   # concurrent reads
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._create_schema()
        logger.info(f"SQLite database initialised at {self.db_path}")

    def close(self) -> None:
        """Close the connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS alerts (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                structure_id         TEXT    NOT NULL,
                structure_name       TEXT    NOT NULL,
                timestamp            REAL    NOT NULL,
                system_status        TEXT    NOT NULL,   -- WARNING | CRITICAL
                localized_defect_node TEXT,
                nodes_json           TEXT    NOT NULL,   -- JSON blob: per-node snapshot
                acknowledged         INTEGER NOT NULL DEFAULT 0,
                created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
            );

            CREATE INDEX IF NOT EXISTS idx_alerts_structure
                ON alerts(structure_id, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_alerts_status
                ON alerts(system_status, acknowledged);
        """)
        self._conn.commit()

    # ── Low-level helpers ──────────────────────────────────────────────────────

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("DatabaseManager.init() must be called before use.")
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def commit(self) -> None:
        self.conn.commit()
