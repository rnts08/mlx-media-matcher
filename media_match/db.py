"""SQLite storage for the media index.

One database file per media tree (default: <root>/.media-match.db). Full-text
search uses FTS5 with the trigram tokenizer (substring matching), falling back
to LIKE queries when the tokenizer or query is unsuitable.

Two kinds of media live side by side: images (one row in `files`) and videos
(one row in `videos` plus one row per sampled keyframe in `video_frames`). All
the image code kept working unchanged; videos are additive.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field

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
    embedding BLOB,
    scanned_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_files_sha1 ON files(sha1);
CREATE INDEX IF NOT EXISTS idx_files_phash ON files(phash);

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    sha1 TEXT NOT NULL,
    duration REAL NOT NULL DEFAULT 0,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps REAL NOT NULL DEFAULT 0,
    codec TEXT NOT NULL DEFAULT '',
    bitrate INTEGER NOT NULL DEFAULT 0,
    frame_count INTEGER NOT NULL DEFAULT 0,
    signature TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    keywords TEXT NOT NULL DEFAULT '',
    scanned_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_videos_sha1 ON videos(sha1);

CREATE TABLE IF NOT EXISTS video_frames (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    t REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    phash TEXT NOT NULL,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_vf_video ON video_frames(video_id);
CREATE INDEX IF NOT EXISTS idx_vf_phash ON video_frames(phash);

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


@dataclass
class VideoRecord:
    """A video row, shaped to duck-type FileRecord for groups/cleanup/tui.

    `phash` maps to the video's cheap min-frame signature and `width`/`height`
    to the video resolution, so the image grouping and TUI code can consume
    VideoRecords without changes. `phashes` carries the per-keyframe hashes the
    video-specific matcher actually uses.
    """

    id: int
    path: str
    size: int
    mtime: float
    sha1: str
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    frame_count: int
    signature: str
    description: str
    keywords: str
    phashes: list[str] = field(default_factory=list)

    @property
    def phash(self) -> str:
        return self.signature


class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self._fts_available = self._setup_fts()
        self.conn.commit()

    def _migrate(self) -> None:
        columns = {r["name"] for r in self.conn.execute("PRAGMA table_info(files)")}
        if "embedding" not in columns:
            self.conn.execute("ALTER TABLE files ADD COLUMN embedding BLOB")

    def _setup_fts(self) -> bool:
        try:
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                    description, keywords, tokenize='trigram'
                )
                """
            )
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS videos_fts USING fts5(
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
        embedding: bytes | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO files(path, size, mtime, sha1, width, height, phash,
                              description, keywords, embedding)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size, mtime = excluded.mtime,
                sha1 = excluded.sha1, width = excluded.width,
                height = excluded.height, phash = excluded.phash,
                description = excluded.description,
                keywords = excluded.keywords,
                embedding = excluded.embedding,
                scanned_at = datetime('now')
            """,
            (path, size, mtime, sha1, width, height, phash, description,
             keywords, embedding),
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

    # -- videos -------------------------------------------------------------

    def get_video(self, path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM videos WHERE path = ?", (path,)).fetchone()

    def upsert_video(
        self,
        *,
        path: str,
        size: int,
        mtime: float,
        sha1: str,
        duration: float,
        width: int,
        height: int,
        fps: float,
        codec: str,
        bitrate: int,
        signature: str,
        description: str = "",
        keywords: str = "",
    ) -> int:
        self.conn.execute(
            """
            INSERT INTO videos(path, size, mtime, sha1, duration, width, height,
                               fps, codec, bitrate, signature, description,
                               keywords)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size, mtime = excluded.mtime,
                sha1 = excluded.sha1, duration = excluded.duration,
                width = excluded.width, height = excluded.height,
                fps = excluded.fps, codec = excluded.codec,
                bitrate = excluded.bitrate, signature = excluded.signature,
                description = excluded.description,
                keywords = excluded.keywords,
                scanned_at = datetime('now')
            """,
            (path, size, mtime, sha1, duration, width, height, fps, codec,
             bitrate, signature, description, keywords),
        )
        vid = self.conn.execute(
            "SELECT id FROM videos WHERE path = ?", (path,)
        ).fetchone()["id"]
        if self._fts_available:
            self.conn.execute("DELETE FROM videos_fts WHERE rowid = ?", (vid,))
            self.conn.execute(
                "INSERT INTO videos_fts(rowid, description, keywords) VALUES(?, ?, ?)",
                (vid, description, keywords),
            )
        self.conn.commit()
        return vid

    def set_video_frames(
        self, video_id: int, frames: list[tuple[int, float, int, int, str, bytes | None]]
    ) -> None:
        """(idx, timestamp, width, height, phash, embedding) per keyframe."""
        self.conn.execute("DELETE FROM video_frames WHERE video_id = ?", (video_id,))
        self.conn.executemany(
            "INSERT INTO video_frames(video_id, idx, t, width, height, phash, embedding) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            [
                (video_id, idx, t, w, h, ph, emb)
                for idx, t, w, h, ph, emb in frames
            ],
        )
        self.conn.execute(
            "UPDATE videos SET frame_count = ? WHERE id = ?", (len(frames), video_id)
        )
        self.conn.commit()

    def video_frames(self, video_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM video_frames WHERE video_id = ? ORDER BY idx", (video_id,)
        ).fetchall()

    def _video_record(self, r: sqlite3.Row) -> VideoRecord:
        return VideoRecord(
            id=r["id"], path=r["path"], size=r["size"], mtime=r["mtime"],
            sha1=r["sha1"], duration=r["duration"], width=r["width"],
            height=r["height"], fps=r["fps"], codec=r["codec"],
            bitrate=r["bitrate"], frame_count=r["frame_count"],
            signature=r["signature"], description=r["description"],
            keywords=r["keywords"],
        )

    def all_videos(self) -> list[VideoRecord]:
        rows = self.conn.execute("SELECT * FROM videos ORDER BY path").fetchall()
        return [self._video_record(r) for r in rows]

    def all_videos_with_frames(self) -> list[VideoRecord]:
        """VideoRecords with their per-keyframe pHashes attached (for grouping)."""
        records = {r.id: r for r in self.all_videos()}
        rows = self.conn.execute(
            "SELECT video_id, phash FROM video_frames ORDER BY video_id, idx"
        ).fetchall()
        for r in rows:
            record = records.get(r["video_id"])
            if record is not None:
                record.phashes.append(r["phash"])
        return list(records.values())

    def video_embeddings(self) -> dict[int, list[tuple[float, bytes]]]:
        """video_id -> [(timestamp, float32 blob), ...] for embedded keyframes."""
        out: dict[int, list[tuple[float, bytes]]] = {}
        rows = self.conn.execute(
            "SELECT video_id, t, embedding FROM video_frames "
            "WHERE embedding IS NOT NULL ORDER BY video_id, idx"
        ).fetchall()
        for r in rows:
            out.setdefault(r["video_id"], []).append((r["t"], r["embedding"]))
        return out

    def set_video_description(self, video_id: int, description: str, keywords: str) -> None:
        self.conn.execute(
            "UPDATE videos SET description = ?, keywords = ? WHERE id = ?",
            (description, keywords, video_id),
        )
        if self._fts_available:
            self.conn.execute("DELETE FROM videos_fts WHERE rowid = ?", (video_id,))
            self.conn.execute(
                "INSERT INTO videos_fts(rowid, description, keywords) VALUES(?, ?, ?)",
                (video_id, description, keywords),
            )
        self.conn.commit()

    def delete_videos(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM videos WHERE id IN ({marks})", ids)
        self.conn.execute(f"DELETE FROM video_frames WHERE video_id IN ({marks})", ids)
        if self._fts_available:
            self.conn.execute(f"DELETE FROM videos_fts WHERE rowid IN ({marks})", ids)
        self.conn.commit()

    def count_videos(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM videos").fetchone()["n"]

    def delete_media(self, records: list[object]) -> None:
        """Delete mixed image/video records, routing each to the right table.

        Image and video ids come from separate tables, so the caller must not
        assume a single ``delete_files`` call is safe for both.
        """
        file_ids = [r.id for r in records if not isinstance(r, VideoRecord)]  # type: ignore[attr-defined]
        video_ids = [r.id for r in records if isinstance(r, VideoRecord)]  # type: ignore[attr-defined]
        self.delete_files(file_ids)
        self.delete_videos(video_ids)

    # -- CLIP embeddings ------------------------------------------------------

    def set_description(self, file_id: int, description: str, keywords: str) -> None:
        self.conn.execute(
            "UPDATE files SET description = ?, keywords = ? WHERE id = ?",
            (description, keywords, file_id),
        )
        if self._fts_available:
            self.conn.execute("DELETE FROM files_fts WHERE rowid = ?", (file_id,))
            self.conn.execute(
                "INSERT INTO files_fts(rowid, description, keywords) VALUES(?, ?, ?)",
                (file_id, description, keywords),
            )
        self.conn.commit()

    def files_missing_embedding(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path FROM files WHERE embedding IS NULL ORDER BY path"
        ).fetchall()

    def set_embeddings(self, vectors: list[tuple[int, bytes]]) -> None:
        self.conn.executemany(
            "UPDATE files SET embedding = ? WHERE id = ?",
            [(blob, fid) for fid, blob in vectors],
        )
        self.conn.commit()

    def all_embeddings(self) -> tuple[list[int], list[bytes]]:
        """(file ids, raw float32 embedding blobs) for indexed files that have one."""
        rows = self.conn.execute(
            "SELECT id, embedding FROM files WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        return [r["id"] for r in rows], [r["embedding"] for r in rows]

    def embedding_for_path(self, path: str) -> bytes | None:
        row = self.conn.execute(
            "SELECT embedding FROM files WHERE path = ?", (path,)
        ).fetchone()
        return row["embedding"] if row else None

    def delete_files(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM files WHERE id IN ({marks})", ids)
        if self._fts_available:
            self.conn.execute(
                f"DELETE FROM files_fts WHERE rowid IN ({marks})", ids
            )
        self.conn.commit()

    # -- search -------------------------------------------------------------

    def search(self, terms: list[str], limit: int = 50) -> list[sqlite3.Row]:
        """AND-search over image description + keywords."""
        return self._fts_like_search("files", "files_fts", terms, limit)

    def search_videos(self, terms: list[str], limit: int = 50) -> list[sqlite3.Row]:
        """AND-search over video description + keywords."""
        return self._fts_like_search("videos", "videos_fts", terms, limit)

    def _fts_like_search(
        self, table: str, fts_table: str, terms: list[str], limit: int
    ) -> list[sqlite3.Row]:
        terms = [t.strip() for t in terms if t.strip()]
        if not terms:
            return []
        if self._fts_available and all(len(t) >= 3 for t in terms):
            match = " AND ".join(f'"{t}"' for t in terms)
            try:
                return self.conn.execute(
                    f"""
                    SELECT f.* FROM {table} f
                    JOIN {fts_table} ON {fts_table}.rowid = f.id
                    WHERE {fts_table} MATCH ?
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
            f"SELECT * FROM {table} WHERE {' AND '.join(clauses)} LIMIT ?",
            params,
        ).fetchall()

    def keyword_counts(self, limit: int = 40, include_videos: bool = False) -> list[tuple[str, int]]:
        tables = ["files"] + (["videos"] if include_videos else [])
        counts: dict[str, int] = {}
        for table in tables:
            rows = self.conn.execute(f"SELECT keywords FROM {table}").fetchall()
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


def open_db(root: str, override: str | None = None, create: bool = True) -> "Database | None":
    """Open the index for a media tree.

    Read-only commands should pass create=False: an absent index then yields
    None (so the caller can print "run scan first") instead of silently
    creating an empty database in the wrong place.
    """
    path = db_path_for(root, override)
    if not create and not os.path.exists(path):
        return None
    return Database(path)
