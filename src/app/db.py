from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY, ts TEXT NOT NULL, features TEXT NOT NULL,
    score REAL NOT NULL, label INTEGER NOT NULL, uncertainty REAL NOT NULL,
    model_version TEXT NOT NULL, fallback_applied INTEGER NOT NULL,
    ood_log_density REAL, segment INTEGER
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY REFERENCES predictions(request_id),
    y_true INTEGER NOT NULL, ts TEXT NOT NULL
);
"""


def db_path() -> str:
    return os.environ.get("FEEDBACK_DB", str(ROOT / "feedback.db"))


def init_db() -> None:
    with sqlite3.connect(db_path()) as con:
        con.executescript(SCHEMA)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def log_prediction(request_id: str, features: Dict, score: float, label: int, uncertainty: float,
                   model_version: str, fallback_applied: bool, ood_log_density: Optional[float],
                   segment: Optional[int]) -> None:
    with sqlite3.connect(db_path()) as con:
        con.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (request_id, _now(), json.dumps(features, ensure_ascii=False), float(score), int(label),
                     float(uncertainty), model_version, int(fallback_applied), ood_log_density, segment))


def log_feedback(request_id: str, y_true: int) -> bool:
    with sqlite3.connect(db_path()) as con:
        if con.execute("SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)).fetchone() is None:
            return False
        con.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?)", (request_id, int(y_true), _now()))
    return True


def fetch_prediction(request_id: str) -> Optional[Dict]:
    with sqlite3.connect(db_path()) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM predictions WHERE request_id = ?", (request_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["features"] = json.loads(out["features"])
    return out


def fetch_predictions(with_feedback: Optional[bool] = None) -> List[Dict]:
    q = "SELECT p.*, f.y_true FROM predictions p LEFT JOIN feedback f USING (request_id)"
    if with_feedback is True:
        q += " WHERE f.y_true IS NOT NULL"
    elif with_feedback is False:
        q += " WHERE f.y_true IS NULL"
    with sqlite3.connect(db_path()) as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(q + " ORDER BY p.ts")]
    for r in rows:
        r["features"] = json.loads(r["features"])
    return rows
