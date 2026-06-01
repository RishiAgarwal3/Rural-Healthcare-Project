from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class SQLiteCheckpoint:
    """
    Simple checkpoint DB for stage progress.
    Keeps one row per key (e.g., call_id) per stage table.
    """

    def __init__(self, db_path: Path, table: str):
        self.db_path = Path(db_path)
        self.table = table
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    bytes INTEGER,
                    path TEXT,
                    url TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table}_status ON {self.table}(status)")
            conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {self.table} WHERE key = ?",
                (key,),
            ).fetchone()
            return dict(row) if row else None

    def is_done(self, key: str) -> bool:
        row = self.get(key)
        return bool(row) and row.get("status") == "done"

    def upsert(
        self,
        key: str,
        status: str,
        http_status: int | None = None,
        bytes_: int | None = None,
        path: str | None = None,
        url: str | None = None,
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.table}
                    (key, status, http_status, bytes, path, url, error, updated_at)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    status=excluded.status,
                    http_status=excluded.http_status,
                    bytes=excluded.bytes,
                    path=excluded.path,
                    url=excluded.url,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (key, status, http_status, bytes_, path, url, error, now),
            )
            conn.commit()
