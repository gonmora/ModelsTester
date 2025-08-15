# -*- coding: utf-8 -*-
from __future__ import annotations
import sqlite3, json, uuid, datetime as dt
from typing import Optional, Dict, Any

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  run_key TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL,
  started_at TEXT,
  finished_at TEXT,
  selection_json TEXT NOT NULL,
  metrics_json TEXT,
  artifacts_json TEXT,
  error_msg TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);
"""

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, timeout=30, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init_db(con: sqlite3.Connection) -> None:
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            con.execute(s + ";")

def exists(con: sqlite3.Connection, run_key: str) -> bool:
    cur = con.execute("SELECT 1 FROM runs WHERE run_key = ? LIMIT 1;", (run_key,))
    return cur.fetchone() is not None

def create_run(con: sqlite3.Connection, run_key: str, selection: Dict[str, Any]) -> str:
    run_id = f"{dt.datetime.utcnow().isoformat()}_{uuid.uuid4().hex[:8]}"
    now = dt.datetime.utcnow().isoformat()
    con.execute(
        "INSERT INTO runs (run_id, run_key, status, started_at, selection_json) VALUES (?,?,?,?,?)",
        (run_id, run_key, "PENDING", now, json.dumps(selection, sort_keys=True))
    )
    # opcional: marcar RUNNING de inmediato
    con.execute("UPDATE runs SET status='RUNNING' WHERE run_id=?", (run_id,))
    return run_id

def finish_run(
    con: sqlite3.Connection,
    run_id: str,
    status: str,  # SUCCESS | FAILED | CANCELLED
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    err: Optional[str] = None,
) -> None:
    now = dt.datetime.utcnow().isoformat()
    con.execute(
        """UPDATE runs
           SET status=?, finished_at=?, metrics_json=?, artifacts_json=?, error_msg=?
           WHERE run_id=?""",
        (status, now,
         json.dumps(metrics, sort_keys=True) if metrics else None,
         json.dumps(artifacts, sort_keys=True) if artifacts else None,
         err, run_id)
    )
