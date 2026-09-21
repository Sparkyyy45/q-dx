"""
Thread-safe SQLite connection management and automated schema initialization.
Enforces Write-Ahead Logging (WAL) mode and foreign key constraints.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config.settings import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    abha_id TEXT UNIQUE,
    name TEXT NOT NULL,
    age_years REAL NOT NULL,
    gender INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS screenings (
    id TEXT PRIMARY KEY,
    patient_id TEXT,
    abha_id TEXT,
    model_name TEXT NOT NULL,
    age_years REAL NOT NULL,
    gender INTEGER NOT NULL,
    height REAL NOT NULL,
    weight REAL NOT NULL,
    ap_hi REAL NOT NULL,
    ap_lo REAL NOT NULL,
    cholesterol INTEGER NOT NULL,
    gluc INTEGER NOT NULL,
    smoke INTEGER NOT NULL,
    alco INTEGER NOT NULL,
    active INTEGER NOT NULL,
    bmi REAL NOT NULL,
    probability REAL NOT NULL,
    prediction INTEGER NOT NULL,
    risk_tier TEXT NOT NULL,
    applied_threshold REAL NOT NULL,
    explanation_method TEXT,
    top_factors_json TEXT,
    clinical_summary TEXT,
    fhir_bundle_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    user_role TEXT DEFAULT 'clinician',
    details_json TEXT,
    client_ip TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT DEFAULT 'Clinician',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_screenings_created ON screenings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_screenings_abha ON screenings(abha_id);
CREATE INDEX IF NOT EXISTS idx_patients_abha ON patients(abha_id);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(token);
"""


def init_db(db_path: Optional[Union[Path, str, sqlite3.Connection]] = None) -> None:
    """Initialize SQLite database with required tables and indexes."""
    if isinstance(db_path, sqlite3.Connection):
        db_path.execute("PRAGMA journal_mode=WAL;")
        db_path.execute("PRAGMA foreign_keys=ON;")
        db_path.executescript(SCHEMA_SQL)
        db_path.commit()
        return

    if db_path is None:
        target_path = Path(settings.DATABASE_PATH)
    else:
        target_path = Path(db_path)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db_connection(db_path: Optional[Union[Path, str]] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional database connection context."""
    if db_path is None:
        target_path = Path(settings.DATABASE_PATH)
    else:
        target_path = Path(db_path)

    if not target_path.exists():
        init_db(target_path)

    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

