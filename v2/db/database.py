"""
CycloVision V2 - Database Manager
Handles SQLite persistence for storms, observations, model runs, audit reviews, and web push subscriptions.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

DB_PATH = os.getenv("CYCLOVISION_DB_PATH", os.path.join(os.path.dirname(__file__), "cyclovision_v2.db"))

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    target_path = db_path or DB_PATH
    conn = sqlite3.connect(target_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path: Optional[str] = None) -> None:
    """Initializes tables for audit trail and push subscriptions if not existing."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_reviews (
                audit_id TEXT PRIMARY KEY,
                decision TEXT NOT NULL,
                forecaster_notes TEXT,
                timestamp TEXT NOT NULL
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS push_subscriptions (
                endpoint TEXT PRIMARY KEY,
                p256dh TEXT,
                auth TEXT,
                created_at TEXT NOT NULL
            );
        """)

        cursor.execute("SELECT COUNT(*) FROM audit_reviews;")
        if cursor.fetchone()[0] == 0:
            initial_records = [
                ("AUD-1049", "APPROVED", "SST alignment confirmed via ERSSTv6", "2026-09-17 18:00 UTC"),
                ("AUD-1050", "APPROVED", "Eye-wall microwave signature verified", "2026-09-17 19:30 UTC"),
                ("AUD-1051", "PENDING", "Awaiting final radar pass", "2026-09-17 21:00 UTC")
            ]
            cursor.executemany(
                "INSERT INTO audit_reviews (audit_id, decision, forecaster_notes, timestamp) VALUES (?, ?, ?, ?)",
                initial_records
            )
        conn.commit()

def save_audit_review(audit_id: str, decision: str, forecaster_notes: str) -> Dict[str, Any]:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO audit_reviews (audit_id, decision, forecaster_notes, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(audit_id) DO UPDATE SET
                decision = excluded.decision,
                forecaster_notes = excluded.forecaster_notes,
                timestamp = excluded.timestamp;
        """, (audit_id, decision, forecaster_notes, now_str))
        conn.commit()
    return {
        "audit_id": audit_id,
        "decision": decision,
        "forecaster_notes": forecaster_notes,
        "timestamp": now_str
    }

def get_audit_trail() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT audit_id, decision, forecaster_notes, timestamp FROM audit_reviews ORDER BY rowid ASC;")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

def save_push_subscriber(sub: Dict[str, Any]) -> bool:
    endpoint = sub.get("endpoint", "")
    if not endpoint:
        return False
    keys = sub.get("keys", {})
    p256dh = keys.get("p256dh", "")
    auth = keys.get("auth", "")
    now_str = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO push_subscriptions (endpoint, p256dh, auth, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(endpoint) DO UPDATE SET
                p256dh = excluded.p256dh,
                auth = excluded.auth,
                created_at = excluded.created_at;
        """, (endpoint, p256dh, auth, now_str))
        conn.commit()
    return True

def get_push_subscribers() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT endpoint, p256dh, auth, created_at FROM push_subscriptions;")
        rows = cursor.fetchall()
        subscribers = []
        for r in rows:
            subscribers.append({
                "endpoint": r["endpoint"],
                "keys": {
                    "p256dh": r["p256dh"],
                    "auth": r["auth"]
                },
                "created_at": r["created_at"]
            })
        return subscribers

def get_storms() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT storm_id, name, basin, season, category, peak_wind_kts, min_pressure_hpa, source FROM storms;")
        return [dict(r) for r in cursor.fetchall()]

def get_storm_observations(storm_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT obs_id, storm_id, valid_time, lat, lon, wind_kts, pressure_hpa, stage FROM observations WHERE storm_id = ? ORDER BY obs_id ASC;", (storm_id,))
        return [dict(r) for r in cursor.fetchall()]

init_db()
