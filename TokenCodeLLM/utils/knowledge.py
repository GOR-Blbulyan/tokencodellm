"""Knowledge base storage and retrieval utilities."""

from __future__ import annotations

import json
import sqlite3
import time
import urllib.parse
import urllib.request
from typing import List, Tuple


class CorpusDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()

    def _create_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS texts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                quality REAL NOT NULL,
                created_at INTEGER NOT NULL
            );
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mode TEXT NOT NULL,
                epochs INTEGER NOT NULL,
                steps INTEGER NOT NULL,
                avg_loss REAL,
                created_at INTEGER NOT NULL
            );
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
        """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            );
        """
        )
        self.conn.commit()

    def insert_texts(self, texts: List[str]):
        now = int(time.time())
        self.conn.executemany("INSERT INTO texts (text, created_at) VALUES (?, ?);", [(t, now) for t in texts])
        self.conn.commit()

    def save_generation(self, prompt: str, response: str, quality: float):
        now = int(time.time())
        self.conn.execute(
            "INSERT INTO generations (prompt, response, quality, created_at) VALUES (?, ?, ?, ?)",
            (prompt, response, quality, now),
        )
        self.conn.commit()

    def save_turn(self, role: str, content: str):
        now = int(time.time())
        self.conn.execute(
            "INSERT INTO conversations (role, content, created_at) VALUES (?, ?, ?)",
            (role, content, now),
        )
        self.conn.execute("INSERT INTO texts (text, created_at) VALUES (?, ?)", (f"{role}: {content}", now))
        self.conn.commit()

    def recent_conversations(self, limit: int = 20) -> List[Tuple[str, str]]:
        cur = self.conn.execute(
            "SELECT role, content FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = list(reversed(cur.fetchall()))
        return [(str(r[0]), str(r[1])) for r in rows]

    def count_conversations(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM conversations;").fetchone()[0]

    def set_memory(self, key: str, value: str):
        now = int(time.time())
        self.conn.execute(
            "INSERT OR REPLACE INTO user_memory (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, now),
        )
        self.conn.commit()

    def get_memory(self) -> dict[str, str]:
        cur = self.conn.execute("SELECT key, value FROM user_memory ORDER BY key")
        return {str(row[0]): str(row[1]) for row in cur.fetchall()}

    def clear_all(self):
        self.conn.execute("DELETE FROM texts")
        self.conn.execute("DELETE FROM generations")
        self.conn.execute("DELETE FROM training_runs")
        self.conn.execute("DELETE FROM conversations")
        self.conn.execute("DELETE FROM user_memory")
        self.conn.commit()

    def log_training_run(self, mode: str, epochs: int, steps: int, avg_loss: float | None):
        now = int(time.time())
        self.conn.execute(
            "INSERT INTO training_runs (mode, epochs, steps, avg_loss, created_at) VALUES (?, ?, ?, ?, ?)",
            (mode, epochs, steps, avg_loss, now),
        )
        self.conn.commit()

    def count_texts(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM texts;").fetchone()[0]

    def sample_texts(self, limit: int = 100) -> List[str]:
        cur = self.conn.execute("SELECT text FROM texts ORDER BY RANDOM() LIMIT ?", (limit,))
        return [r[0] for r in cur.fetchall()]

    def search(self, query: str, limit: int = 10) -> List[Tuple[int, str]]:
        like = f"%{query}%"
        cur = self.conn.execute("SELECT id, text FROM texts WHERE text LIKE ? LIMIT ?", (like, limit))
        return cur.fetchall()

    def stats(self) -> dict:
        total_generations = self.conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
        avg_quality = self.conn.execute("SELECT COALESCE(AVG(quality), 0.0) FROM generations").fetchone()[0]
        training_runs = self.conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
        last_loss = self.conn.execute(
            "SELECT avg_loss FROM training_runs WHERE avg_loss IS NOT NULL ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return {
            "texts": self.count_texts(),
            "generations": total_generations,
            "avg_quality": float(avg_quality),
            "training_runs": training_runs,
            "last_loss": float(last_loss[0]) if last_loss else None,
            "conversations": self.count_conversations(),
            "memory_keys": len(self.get_memory()),
        }


def external_search(query: str) -> str:
    """Tiny Wikipedia summary fetch used as an optional external knowledge source."""
    safe_query = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_query}"
    req = urllib.request.Request(url, headers={"User-Agent": "TokenCodeAI/4.0"})
    with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310 (trusted endpoint)
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("extract", "No external summary found.")


def _flatten_for_text(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        out: list[str] = []
        for _, v in value.items():
            out.extend(_flatten_for_text(v))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_for_text(item))
        return out
    return [str(value)]


def fetch_hf_rows_as_texts(url: str, timeout_s: int = 12, max_rows: int = 100) -> List[str]:
    req = urllib.request.Request(url, headers={"User-Agent": "TokenCodeAI/6.1"})
    with urllib.request.urlopen(req, timeout=timeout_s) as response:  # noqa: S310 (trusted endpoint)
        payload = json.loads(response.read().decode("utf-8"))

    rows = payload.get("rows", [])
    out: list[str] = []
    for row_obj in rows[:max_rows]:
        row = row_obj.get("row", row_obj)
        chunks = _flatten_for_text(row)
        merged = " ".join([c for c in chunks if c])
        merged = " ".join(merged.split())
        if merged:
            out.append(merged)
    return out
