"""
Long-term trainee memory (SQLite, stdlib only).

What the coach agent remembers across sessions:
- sessions: every scored roleplay (script, product line, score, per-dimension
  scores, what to improve, which challenge points were missed)
- notes:    things the coach decided to remember ("目标：下周考核账单类")

profile() turns raw rows into what a coach actually needs: average score,
trend, weakest dimensions, recurring misses, product lines never practiced.

Storage: MEMORY_DB_PATH (default data/memory.db). On Cloud Run the local disk
is ephemeral — see README "Known limitations" for the persistence options.
"""

import json
import os
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trainee_id TEXT NOT NULL,
    ts REAL NOT NULL,
    script_title TEXT,
    business_type TEXT,
    overall_score INTEGER,
    dimensions TEXT,      -- JSON [{name, score}]
    improvements TEXT,    -- JSON [str]
    missed TEXT           -- JSON [str]
);
CREATE INDEX IF NOT EXISTS idx_sessions_trainee ON sessions (trainee_id, ts);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trainee_id TEXT NOT NULL,
    ts REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_trainee ON notes (trainee_id, ts);
"""

MAX_NOTES_PER_TRAINEE = 50


class Memory:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.environ.get(
            "MEMORY_DB_PATH", str(Path(__file__).parent.parent / "data" / "memory.db"))
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One connection shared across FastAPI's threadpool, guarded by a lock.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # -- writes ---------------------------------------------------------------

    def record_session(self, trainee_id: str, script: dict, evaluation: dict,
                       business_type: str = "") -> int:
        dims = [{"name": d.get("name", ""), "score": d.get("score")}
                for d in evaluation.get("dimensions", [])]
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO sessions (trainee_id, ts, script_title, business_type, overall_score,"
                " dimensions, improvements, missed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (trainee_id, time.time(), script.get("title", ""), business_type,
                 evaluation.get("overall_score"),
                 json.dumps(dims, ensure_ascii=False),
                 json.dumps(evaluation.get("improvements", [])[:5], ensure_ascii=False),
                 json.dumps(evaluation.get("missed_challenge_points", [])[:5], ensure_ascii=False)))
            return cur.lastrowid

    def add_note(self, trainee_id: str, text: str) -> int:
        text = text.strip()[:500]
        if not text:
            raise ValueError("empty note")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO notes (trainee_id, ts, text) VALUES (?, ?, ?)",
                (trainee_id, time.time(), text))
            # Bounded memory: keep only the newest notes.
            self._conn.execute(
                "DELETE FROM notes WHERE trainee_id = ? AND id NOT IN "
                "(SELECT id FROM notes WHERE trainee_id = ? ORDER BY ts DESC, id DESC LIMIT ?)",
                (trainee_id, trainee_id, MAX_NOTES_PER_TRAINEE))
            return cur.lastrowid

    def forget(self, trainee_id: str) -> None:
        """Delete everything stored about a trainee (user-facing 'clear my data')."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM sessions WHERE trainee_id = ?", (trainee_id,))
            self._conn.execute("DELETE FROM notes WHERE trainee_id = ?", (trainee_id,))

    # -- reads ----------------------------------------------------------------

    def _rows(self, sql: str, args: tuple) -> list:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def sessions(self, trainee_id: str, limit: int = 50) -> list:
        rows = self._rows("SELECT * FROM sessions WHERE trainee_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
                          (trainee_id, limit))
        for r in rows:
            for k in ("dimensions", "improvements", "missed"):
                r[k] = json.loads(r[k] or "[]")
        return rows

    def notes(self, trainee_id: str, limit: int = 10) -> list:
        return self._rows("SELECT id, ts, text FROM notes WHERE trainee_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
                          (trainee_id, limit))

    def profile(self, trainee_id: str, all_business_types: Optional[list] = None) -> dict:
        """Coach-ready summary of a trainee's history."""
        sessions = self.sessions(trainee_id)
        scores = [s["overall_score"] for s in sessions if isinstance(s["overall_score"], int)]
        dim_scores = defaultdict(list)
        for s in sessions:
            for d in s["dimensions"]:
                if isinstance(d.get("score"), (int, float)):
                    dim_scores[d["name"]].append(d["score"])
        weakest = sorted(((n, sum(v) / len(v), len(v)) for n, v in dim_scores.items()),
                         key=lambda x: x[1])[:3]
        missed = Counter(m for s in sessions for m in s["missed"])
        practiced = Counter(s["business_type"] for s in sessions if s["business_type"])

        trend = None
        if len(scores) >= 4:
            # scores are newest-first: compare the recent half with the older half
            half = len(scores) // 2
            trend = round(sum(scores[:half]) / half - sum(scores[half:]) / (len(scores) - half), 1)

        return {
            "trainee_id": trainee_id,
            "sessions_count": len(sessions),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
            "trend_vs_earlier": trend,
            "recent_sessions": [
                {"title": s["script_title"], "business_type": s["business_type"],
                 "score": s["overall_score"],
                 "when": time.strftime("%Y-%m-%d", time.localtime(s["ts"]))}
                for s in sessions[:5]
            ],
            "weakest_dimensions": [{"name": n, "avg": round(a, 1), "n": c} for n, a, c in weakest],
            "recurring_missed_points": [m for m, c in missed.most_common(5) if c >= 1],
            "practiced_business_types": dict(practiced),
            "never_practiced": [b for b in (all_business_types or []) if b not in practiced],
            "notes": [n["text"] for n in self.notes(trainee_id)],
        }


_MEMORY: Optional[Memory] = None
_mem_lock = threading.Lock()


def get_memory() -> Memory:
    global _MEMORY
    with _mem_lock:
        if _MEMORY is None:
            _MEMORY = Memory()
        return _MEMORY
