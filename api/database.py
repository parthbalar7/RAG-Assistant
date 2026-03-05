"""
SQLite database for users, chat sessions, and analytics.
"""

import json
import sqlite3
import time
import uuid
import logging
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.database_path


def _get_conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT DEFAULT 'New Chat',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            metadata TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS analytics (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            query TEXT NOT NULL,
            category TEXT,
            retrieval_count INTEGER,
            latency_ms REAL,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            search_type TEXT,
            created_at REAL NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS integrity_scans (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            health_score INTEGER NOT NULL,
            health_band TEXT,
            counts TEXT,
            recommendations TEXT,
            fingerprints TEXT,
            sampled_chunks INTEGER,
            total_chunks INTEGER,
            duration_ms REAL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS integrity_issues (
            id TEXT PRIMARY KEY,
            scan_id TEXT NOT NULL,
            type TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            evidence TEXT,
            recommendation TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES integrity_scans(id)
        );
        CREATE TABLE IF NOT EXISTS eval_runs (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            total_cases INTEGER NOT NULL,
            hit_rate REAL,
            avg_mrr REAL,
            avg_faithfulness REAL,
            avg_relevance REAL,
            details TEXT,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_eval_runs_user ON eval_runs(user_id, created_at);
        CREATE TABLE IF NOT EXISTS compliance_scans (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            framework TEXT NOT NULL,
            framework_full_name TEXT,
            summary TEXT,
            risk_score INTEGER NOT NULL,
            issues TEXT,
            compliant_areas TEXT,
            sampled_chunks INTEGER,
            total_chunks INTEGER,
            duration_ms REAL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_compliance_scans_user ON compliance_scans(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_integrity_scans_created ON integrity_scans(created_at);
        CREATE INDEX IF NOT EXISTS idx_integrity_issues_scan ON integrity_issues(scan_id);

        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_analytics_created ON analytics(created_at);
    """)
    conn.close()
    logger.info("Database initialized")


def create_user(username, password_hash, display_name=""):
    conn = _get_conn()
    user_id = str(uuid.uuid4())[:8]
    now = time.time()
    conn.execute("INSERT INTO users (id, username, password_hash, display_name, created_at) VALUES (?,?,?,?,?)",
                 (user_id, username, password_hash, display_name or username, now))
    conn.commit()
    conn.close()
    return {"id": user_id, "username": username, "display_name": display_name or username}


def get_user_by_username(username):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def create_session(user_id, title="New Chat"):
    conn = _get_conn()
    session_id = str(uuid.uuid4())[:12]
    now = time.time()
    conn.execute("INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                 (session_id, user_id, title, now, now))
    conn.commit()
    conn.close()
    return {"id": session_id, "user_id": user_id, "title": title, "created_at": now}


def get_user_sessions(user_id):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_title(session_id, title):
    conn = _get_conn()
    conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, time.time(), session_id))
    conn.commit()
    conn.close()


def delete_session(session_id):
    conn = _get_conn()
    conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def add_message(session_id, role, content, sources=None, metadata=None):
    conn = _get_conn()
    msg_id = str(uuid.uuid4())[:12]
    now = time.time()
    conn.execute("INSERT INTO messages (id, session_id, role, content, sources, metadata, created_at) VALUES (?,?,?,?,?,?,?)",
                 (msg_id, session_id, role, content, json.dumps(sources) if sources else None,
                  json.dumps(metadata) if metadata else None, now))
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    conn.close()
    return {"id": msg_id, "role": role, "content": content, "sources": sources, "created_at": now}


def get_session_messages(session_id):
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,)).fetchall()
    conn.close()
    results = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"]) if d["sources"] else None
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else None
        results.append(d)
    return results


def log_query(user_id, query, category, retrieval_count, latency_ms, model, input_tokens, output_tokens, search_type="hybrid"):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO analytics (id, user_id, query, category, retrieval_count, latency_ms, model, input_tokens, output_tokens, search_type, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4())[:12], user_id, query, category, retrieval_count, latency_ms, model, input_tokens, output_tokens, search_type, time.time()))
    conn.commit()
    conn.close()


def get_analytics(days=7):
    conn = _get_conn()
    cutoff = time.time() - (days * 86400)
    total = conn.execute("SELECT COUNT(*) as c FROM analytics WHERE created_at > ?", (cutoff,)).fetchone()["c"]
    avg_latency = conn.execute("SELECT AVG(latency_ms) as a FROM analytics WHERE created_at > ?", (cutoff,)).fetchone()["a"]
    total_tokens = conn.execute("SELECT SUM(input_tokens + output_tokens) as t FROM analytics WHERE created_at > ?", (cutoff,)).fetchone()["t"]
    categories = conn.execute("SELECT category, COUNT(*) as c FROM analytics WHERE created_at > ? GROUP BY category ORDER BY c DESC", (cutoff,)).fetchall()
    daily = conn.execute("SELECT DATE(created_at, 'unixepoch') as day, COUNT(*) as c, AVG(latency_ms) as avg_lat FROM analytics WHERE created_at > ? GROUP BY day ORDER BY day", (cutoff,)).fetchall()
    search_types = conn.execute("SELECT search_type, COUNT(*) as c FROM analytics WHERE created_at > ? GROUP BY search_type", (cutoff,)).fetchall()
    recent = conn.execute("SELECT query, category, latency_ms, created_at FROM analytics ORDER BY created_at DESC LIMIT 20").fetchall()
    conn.close()
    return {
        "total_queries": total,
        "avg_latency_ms": round(avg_latency or 0, 1),
        "total_tokens": total_tokens or 0,
        "categories": [{"name": r["category"] or "unknown", "count": r["c"]} for r in categories],
        "daily": [{"date": r["day"], "queries": r["c"], "avg_latency": round(r["avg_lat"] or 0, 1)} for r in daily],
        "search_types": [{"type": r["search_type"] or "unknown", "count": r["c"]} for r in search_types],
        "recent_queries": [{"query": r["query"], "category": r["category"], "latency_ms": round(r["latency_ms"] or 0, 1), "time": r["created_at"]} for r in recent],
    }


# ── Knowledge Integrity & Risk Radar ──

def get_latest_integrity_fingerprints():
    """Return most recent per-file fingerprints dict (or empty)."""
    conn = _get_conn()
    row = conn.execute("SELECT fingerprints FROM integrity_scans ORDER BY created_at DESC LIMIT 1").fetchone()
    conn.close()
    if not row or not row["fingerprints"]:
        return {}
    try:
        return json.loads(row["fingerprints"]) or {}
    except Exception:
        return {}


def save_integrity_scan(user_id, scan_result: dict):
    """
    Persist an integrity scan summary + issues.
    """
    conn = _get_conn()
    scan_id = str(uuid.uuid4())[:12]
    now = time.time()

    health = scan_result.get("health", {}) or {}
    counts = health.get("counts", {}) or {}
    recs = scan_result.get("recommendations", []) or []
    fingerprints = scan_result.get("fingerprints", {}) or {}

    conn.execute(
        "INSERT INTO integrity_scans (id, user_id, health_score, health_band, counts, recommendations, fingerprints, sampled_chunks, total_chunks, duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            scan_id,
            user_id,
            int(health.get("score", 0)),
            health.get("band", ""),
            json.dumps(counts),
            json.dumps(recs),
            json.dumps(fingerprints),
            int(scan_result.get("sampled_chunks", 0)),
            int(scan_result.get("total_chunks", 0)),
            float(scan_result.get("duration_ms", 0)),
            now,
        ),
    )

    issues = scan_result.get("issues", []) or []
    for iss in issues:
        conn.execute(
            "INSERT INTO integrity_issues (id, scan_id, type, severity, title, description, evidence, recommendation, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                str(uuid.uuid4())[:12],
                scan_id,
                iss.get("type", ""),
                iss.get("severity", "medium"),
                iss.get("title", ""),
                iss.get("description", ""),
                json.dumps(iss.get("evidence", []) or []),
                iss.get("recommendation", ""),
                now,
            ),
        )

    conn.commit()
    conn.close()
    return scan_id


def get_integrity_history(days: int = 30, limit: int = 30):
    conn = _get_conn()
    cutoff = time.time() - (days * 86400)
    scans = conn.execute(
        "SELECT id, health_score, health_band, counts, sampled_chunks, total_chunks, duration_ms, created_at FROM integrity_scans WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
        (cutoff, limit),
    ).fetchall()

    out = []
    for s in scans:
        d = dict(s)
        d["counts"] = json.loads(d["counts"]) if d.get("counts") else {}
        out.append(d)
    conn.close()
    return {"scans": out}


def save_eval_run(user_id, metrics: dict):
    conn = _get_conn()
    run_id = str(uuid.uuid4())[:12]
    now = time.time()
    conn.execute(
        "INSERT INTO eval_runs (id, user_id, total_cases, hit_rate, avg_mrr, avg_faithfulness, avg_relevance, details, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, user_id, metrics.get("total_cases", 0), metrics.get("retrieval_hit_rate"),
         metrics.get("avg_mrr"), metrics.get("avg_faithfulness"), metrics.get("avg_relevance"),
         json.dumps(metrics.get("cases", [])), now),
    )
    conn.commit()
    conn.close()
    return run_id


def get_eval_history(user_id=None, limit=20):
    conn = _get_conn()
    if user_id:
        rows = conn.execute(
            "SELECT id, total_cases, hit_rate, avg_mrr, avg_faithfulness, avg_relevance, created_at FROM eval_runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, total_cases, hit_rate, avg_mrr, avg_faithfulness, avg_relevance, created_at FROM eval_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_compliance_scan(user_id: str, result: dict) -> str:
    conn = _get_conn()
    scan_id = str(uuid.uuid4())[:12]
    now = time.time()
    conn.execute(
        "INSERT INTO compliance_scans (id, user_id, framework, framework_full_name, summary, risk_score, issues, compliant_areas, sampled_chunks, total_chunks, duration_ms, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (scan_id, user_id, result.get("framework", ""), result.get("framework_full_name", ""),
         result.get("summary", ""), int(result.get("risk_score", 0)),
         json.dumps(result.get("issues", [])), json.dumps(result.get("compliant_areas", [])),
         int(result.get("sampled_chunks", 0)), int(result.get("total_chunks", 0)),
         float(result.get("duration_ms", 0)), now),
    )
    conn.commit()
    conn.close()
    return scan_id


def get_compliance_history(user_id: str, limit: int = 20) -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, framework, framework_full_name, risk_score, sampled_chunks, total_chunks, duration_ms, created_at FROM compliance_scans WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_integrity_scan(scan_id: str):
    conn = _get_conn()
    scan = conn.execute("SELECT * FROM integrity_scans WHERE id = ?", (scan_id,)).fetchone()
    if not scan:
        conn.close()
        return None
    issues = conn.execute("SELECT * FROM integrity_issues WHERE scan_id = ? ORDER BY created_at ASC", (scan_id,)).fetchall()
    conn.close()
    s = dict(scan)
    s["counts"] = json.loads(s["counts"]) if s.get("counts") else {}
    s["recommendations"] = json.loads(s["recommendations"]) if s.get("recommendations") else []
    s["fingerprints"] = json.loads(s["fingerprints"]) if s.get("fingerprints") else {}
    iss_out = []
    for r in issues:
        d = dict(r)
        d["evidence"] = json.loads(d["evidence"]) if d.get("evidence") else []
        iss_out.append(d)
    return {"scan": s, "issues": iss_out}
