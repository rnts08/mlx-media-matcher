"""SQLite storage for the image index.

One database file per media tree (default: <root>/.media-match.db). Full-text
search uses FTS5 with the trigram tokenizer (substring matching), falling back
to LIKE queries when the tokenizer or query is unsuitable.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass

DEFAULT_DB_NAME = ".media-match.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha1 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    phash TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '',
    scanned_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_files_sha1 ON files(sha1);
CREATE INDEX IF NOT EXISTS idx_files_phash ON files(phash);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class FileRecord:
    id: int
    path: str
    size: int
    mtime: float
    sha1: str
    width: int
    height: int
    phash: str
    description: str
    keywords: str


class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._fts_available = self._setup_fts()
        self.conn.commit()

    def _setup_fts(self) -> bool:
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                    description, keywords, tokenize='trigram'
                )
                """
            )
            return True
        except sqlite3.OperationalError:
            return False

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    # -- files --------------------------------------------------------------

    def get_file(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()

    def upsert_file(
        self,
        *,
        path: str,
        size: int,
        mtime: float,
        sha1: str,
        width: int,
        height: int,
        phash: str,
        description: str,
        keywords: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO files(path, size, mtime, sha1, width, height, phash,
                              description, keywords)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size, mtime = excluded.mtime,
                sha1 = excluded.sha1, width = excluded.width,
                height = excluded.height, phash = excluded.phash,
                description = excluded.description,
                keywords = excluded.keywords,
                scanned_at = datetime('now')
            """,
            (path, size, mtime, sha1, width, height, phash, description, keywords),
        )
        if self._fts_available:
            row = self.conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
            self.conn.execute("DELETE FROM files_fts WHERE rowid = ?", (row["id"],))
            self.conn.execute(
                "INSERT INTO files_fts(rowid, description, keywords) VALUES(?, ?, ?)",
                (row["id"], description, keywords),
            )
        self.conn.commit()

    def all_files(self) -> list[FileRecord]:
        rows = self.conn.execute("SELECT * FROM files ORDER BY path").fetchall()
        return [
            FileRecord(
                id=r["id"], path=r["path"], size=r["size"], mtime=r["mtime"],
                sha1=r["sha1"], width=r["width"], height=r["height"],
                phash=r["phash"], description=r["description"],
                keywords=r["keywords"],
            )
            for r in rows
        ]

    def count_files(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]

    # -- search -------------------------------------------------------------

    def search(self, terms: list[str], limit: int = 50) -> list[sqlite3.Row]:
        """AND-search over description + keywords."""
        terms = [t.strip() for t in terms if t.strip()]
        if not terms:
            return []
        if self._fts_available and all(len(t) >= 3 for t in terms):
            match = " AND ".join(f'"{t}"' for t in terms)
            try:
                return self.conn.execute(
                    """
                    SELECT f.* FROM files f
                    JOIN files_fts ON files_fts.rowid = f.id
                    WHERE files_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                pass  # fall through to LIKE
        clauses, params = [], []
        for t in terms:
            clauses.append("(description LIKE ? OR keywords LIKE ?)")
            params.extend([f"%{t}%", f"%{t}%"])
        params.append(limit)
        return self.conn.execute(
            f"SELECT * FROM files WHERE {' AND '.join(clauses)} LIMIT ?",
            params,
        ).fetchall()

    def keyword_counts(self, limit: int = 40) -> list[tuple[str, int]]:
        rows = self.conn.execute("SELECT keywords FROM files").fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            for kw in r["keywords"].split(","):
                kw = kw.strip().lower()
                if kw:
                    counts[kw] = counts.get(kw, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def close(self) -> None:
        self.conn.close()


def db_path_for(root: str, override: str | None = None) -> str:
    if override:
        return override
    return os.path.join(os.path.abspath(root), DEFAULT_DB_NAME)
