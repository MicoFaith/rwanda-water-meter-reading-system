#!/usr/bin/env python3
"""
database.py

Lightweight SQLite persistence layer for the Water Meter Reading System.

Stores every successful or uncertain prediction made via the /predict API.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "wmrs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meter_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    image_name TEXT,
    reading TEXT,
    confidence REAL,
    status TEXT NOT NULL,
    warning TEXT
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    """Create the database file and table if they don't already exist."""
    with _connect() as conn:
        conn.execute(_SCHEMA)


def save_reading(
    image_name: str,
    reading: str | None,
    confidence: float | None,
    status: str,
    warning: str | None = None,
) -> int | None:
    """Insert one reading record. Returns the new row id, or None on failure."""
    try:
        with _connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO meter_readings
                    (image_name, reading, confidence, status, warning)
                VALUES (?, ?, ?, ?, ?)
                """,
                (image_name, reading, confidence, status, warning),
            )
            return cursor.lastrowid
    except sqlite3.Error as exc:
        print(f"[database] Failed to save reading: {exc}")
        return None


def get_recent_readings(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent readings, newest first."""
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM meter_readings ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        print(f"[database] Failed to fetch recent readings: {exc}")
        return []


def get_reading(reading_id: int) -> dict[str, Any] | None:
    """Return a single reading by id, or None if not found or on error."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM meter_readings WHERE id = ?",
                (reading_id,),
            ).fetchone()
            return dict(row) if row else None
    except sqlite3.Error as exc:
        print(f"[database] Failed to fetch reading {reading_id}: {exc}")
        return None
