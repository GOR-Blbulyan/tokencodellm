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
        }


def external_search(query: str) -> str:
    """Tiny Wikipedia summary fetch used as an optional external knowledge source."""
    safe_query = urllib.parse.quote(query)
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_query}"
    req = urllib.request.Request(url, headers={"User-Agent": "TokenCodeAI/4.0"})
    with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310 (trusted endpoint)
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("extract", "No external summary found.")
