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
from datetime import datetime, timezone

DEFAULT_DB_NAME = ".media-match.db"

# Bump when the schema changes and add the matching step to _MIGRATIONS.
SCHEMA_VERSION = 3

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
    captured_at TEXT NOT NULL DEFAULT '',
    camera TEXT NOT NULL DEFAULT '',
    gps_lat REAL,
    gps_lon REAL,
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

CREATE TABLE IF NOT EXISTS media_tags (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    media_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    facet TEXT
);
CREATE INDEX IF NOT EXISTS idx_mtags_media ON media_tags(kind, media_id);
CREATE INDEX IF NOT EXISTS idx_mtags_tag ON media_tags(tag);
CREATE INDEX IF NOT EXISTS idx_mtags_facet ON media_tags(kind, facet);

CREATE TABLE IF NOT EXISTS clusters (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS trash_log (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'image',
    size INTEGER NOT NULL DEFAULT 0,
    batch TEXT NOT NULL DEFAULT '',
    trashed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_trash_batch ON trash_log(batch);
"""


# Columns added after the first released schema; migration v1 backfills them on
# databases created before schema versioning existed.
_FILE_COLUMNS = (
    ("embedding", "BLOB"),
    ("raw_keywords", "TEXT NOT NULL DEFAULT ''"),
    ("type", "TEXT NOT NULL DEFAULT ''"),
    ("type_score", "REAL NOT NULL DEFAULT 0"),
    ("cluster_id", "INTEGER"),
)
_VIDEO_COLUMNS = (
    ("raw_keywords", "TEXT NOT NULL DEFAULT ''"),
)
_EXIF_COLUMNS = (
    ("captured_at", "TEXT NOT NULL DEFAULT ''"),
    ("camera", "TEXT NOT NULL DEFAULT ''"),
    ("gps_lat", "REAL"),
    ("gps_lon", "REAL"),
)


def _add_missing_columns(conn: sqlite3.Connection, table: str, columns) -> None:
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _migrate_v1(db: "Database") -> None:
    """v1: the columns that predate schema versioning (idempotent)."""
    _add_missing_columns(db.conn, "files", _FILE_COLUMNS)
    _add_missing_columns(db.conn, "videos", _VIDEO_COLUMNS)


def _migrate_v2(db: "Database") -> None:
    """v2: EXIF capture time, camera and GPS on images."""
    _add_missing_columns(db.conn, "files", _EXIF_COLUMNS)


def _migrate_v3(db: "Database") -> None:
    """v3: the trash_log table (created by _SCHEMA; nothing to alter)."""
    db.conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trash_batch ON trash_log(batch)"
    )


# (target_version, step) ascending. Each step upgrades the previous version.
_MIGRATIONS = (
    (1, _migrate_v1),
    (2, _migrate_v2),
    (3, _migrate_v3),
)


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
    raw_keywords: str = ""
    type: str = ""
    type_score: float = 0.0
    captured_at: str = ""
    camera: str = ""
    gps_lat: float | None = None
    gps_lon: float | None = None


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
    raw_keywords: str = ""
    phashes: list[str] = field(default_factory=list)

    @property
    def phash(self) -> str:
        return self.signature


class Database:
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._setup_pragmas()
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self._fts_available = self._setup_fts()
        self.conn.commit()

    def _setup_pragmas(self) -> None:
        """WAL + a busy timeout so a read and a scan can coexist without
        'database is locked'; synchronous=NORMAL is the usual WAL pairing."""
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA busy_timeout=5000",
        ):
            try:
                self.conn.execute(pragma)
            except sqlite3.OperationalError:
                pass  # e.g. a read-only filesystem; keep the defaults

    def _schema_version(self) -> int:
        try:
            return int(self.get_meta("schema_version", "0") or 0)
        except (TypeError, ValueError):
            return 0

    def _migrate(self) -> None:
        """Apply pending migrations, then stamp the schema version.

        Steps are idempotent, so a database created before versioning (version
        0) is upgraded in place without losing data.
        """
        version = self._schema_version()
        for target, step in _MIGRATIONS:
            if version < target:
                step(self)
                version = target
        self._set_meta_no_commit("schema_version", str(version))

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
        self._set_meta_no_commit(key, value)
        self.conn.commit()

    def _set_meta_no_commit(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

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
        captured_at: str = "",
        camera: str = "",
        gps_lat: float | None = None,
        gps_lon: float | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO files(path, size, mtime, sha1, width, height, phash,
                              description, keywords, embedding, captured_at,
                              camera, gps_lat, gps_lon)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size = excluded.size, mtime = excluded.mtime,
                sha1 = excluded.sha1, width = excluded.width,
                height = excluded.height, phash = excluded.phash,
                description = excluded.description,
                keywords = excluded.keywords,
                embedding = excluded.embedding,
                captured_at = excluded.captured_at,
                camera = excluded.camera,
                gps_lat = excluded.gps_lat,
                gps_lon = excluded.gps_lon,
                scanned_at = datetime('now')
            """,
            (path, size, mtime, sha1, width, height, phash, description,
             keywords, embedding, captured_at, camera, gps_lat, gps_lon),
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
                keywords=r["keywords"], raw_keywords=r["raw_keywords"],
                type=r["type"], type_score=r["type_score"],
                captured_at=r["captured_at"], camera=r["camera"],
                gps_lat=r["gps_lat"], gps_lon=r["gps_lon"],
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
            keywords=r["keywords"], raw_keywords=r["raw_keywords"],
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
        self.conn.execute(
            f"DELETE FROM media_tags WHERE kind = 'video' AND media_id IN ({marks})", ids
        )
        if self._fts_available:
            self.conn.execute(f"DELETE FROM videos_fts WHERE rowid IN ({marks})", ids)
        self.conn.commit()

    def _drop_empty_clusters(self, kind: str, table: str) -> None:
        """Remove cluster rows left with no members after media was deleted."""
        self.conn.execute(
            f"DELETE FROM clusters WHERE kind = ? AND id NOT IN "
            f"(SELECT DISTINCT cluster_id FROM {table} WHERE cluster_id IS NOT NULL)",
            (kind,),
        )

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

    def all_paths(self) -> tuple[list[sqlite3.Row], list[sqlite3.Row]]:
        """``(image rows, video rows)`` with only ``id`` and ``path`` selected.

        Cheap enough to load whole tables: just two columns, used by scan to
        prune index rows for media that has since been moved or deleted.
        """
        files = self.conn.execute("SELECT id, path FROM files").fetchall()
        videos = self.conn.execute("SELECT id, path FROM videos").fetchall()
        return files, videos

    def delete_files(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM files WHERE id IN ({marks})", ids)
        self.conn.execute(
            f"DELETE FROM media_tags WHERE kind = 'image' AND media_id IN ({marks})", ids
        )
        if self._fts_available:
            self.conn.execute(
                f"DELETE FROM files_fts WHERE rowid IN ({marks})", ids
            )
        self._drop_empty_clusters("image", "files")
        self.conn.commit()

    # -- search -------------------------------------------------------------

    def search(self, terms: list[str], limit: int = 50, *,
               filters: str = "", params=()) -> list[sqlite3.Row]:
        """AND-search over image description + keywords, plus SQL filters."""
        return self._fts_like_search("files", "files_fts", terms, limit,
                                     filters=filters, params=params)

    def search_videos(self, terms: list[str], limit: int = 50, *,
                      filters: str = "", params=()) -> list[sqlite3.Row]:
        """AND-search over video description + keywords, plus SQL filters."""
        return self._fts_like_search("videos", "videos_fts", terms, limit,
                                     filters=filters, params=params)

    def file_ids_matching(self, filters: str, params=()) -> set[int]:
        """Ids of images matching a filter clause (used to constrain CLIP search)."""
        if not filters:
            return set()
        rows = self.conn.execute(
            f"SELECT id FROM files WHERE {filters}", tuple(params)
        ).fetchall()
        return {r["id"] for r in rows}

    def video_ids_matching(self, filters: str, params=()) -> set[int]:
        if not filters:
            return set()
        rows = self.conn.execute(
            f"SELECT id FROM videos WHERE {filters}", tuple(params)
        ).fetchall()
        return {r["id"] for r in rows}

    def date_range(self) -> tuple[str | None, str | None]:
        """(earliest, latest) capture time, falling back to mtime, as strings."""
        row = self.conn.execute(
            "SELECT MIN(COALESCE(NULLIF(captured_at,''), datetime(mtime,'unixepoch'))) AS lo, "
            "MAX(COALESCE(NULLIF(captured_at,''), datetime(mtime,'unixepoch'))) AS hi "
            "FROM files"
        ).fetchone()
        return (row["lo"], row["hi"])

    def _fts_like_search(
        self, table: str, fts_table: str, terms: list[str], limit: int, *,
        filters: str = "", params=(),
    ) -> list[sqlite3.Row]:
        terms = [t.strip() for t in terms if t.strip()]
        params = list(params)
        if not terms:
            if not filters:
                return []
            return self.conn.execute(
                f"SELECT * FROM {table} WHERE {filters} ORDER BY path LIMIT ?",
                (*params, limit),
            ).fetchall()
        if self._fts_available and all(len(t) >= 3 for t in terms):
            match = " AND ".join(f'"{t}"' for t in terms)
            where = f"{fts_table} MATCH ?"
            if filters:
                where += f" AND {filters}"
            try:
                return self.conn.execute(
                    f"""
                    SELECT f.* FROM {table} f
                    JOIN {fts_table} ON {fts_table}.rowid = f.id
                    WHERE {where}
                    ORDER BY rank LIMIT ?
                    """,
                    (match, *params, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                pass  # fall through to LIKE
        clauses, like_params = [], []
        for t in terms:
            clauses.append("(description LIKE ? OR keywords LIKE ?)")
            like_params.extend([f"%{t}%", f"%{t}%"])
        where = " AND ".join(clauses)
        if filters:
            where += f" AND {filters}"
        return self.conn.execute(
            f"SELECT * FROM {table} WHERE {where} LIMIT ?",
            (*like_params, *params, limit),
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

    # -- scan cost accounting ---------------------------------------------------

    def record_scan(
        self,
        items: int,
        frames: int,
        bytes_read: int,
        wall: float,
        cpu: float,
        gpu: float,
    ) -> None:
        """Accumulate one scan run's measured cost into meta (totals + count).

        Written in a single transaction so a concurrent reader never sees a
        half-updated total, and timestamped in UTC to match ``scanned_at``.
        """
        scanned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.conn:
            for key, value in (
                ("scan_items", items),
                ("scan_frames", frames),
                ("scan_bytes", bytes_read),
                ("scan_wall", wall),
                ("scan_cpu", cpu),
                ("scan_gpu", gpu),
            ):
                total = float(self.get_meta(key, "0") or 0) + value
                self._set_meta_no_commit(key, str(total))
            runs = int(self.get_meta("scans", "0") or 0) + 1
            self._set_meta_no_commit("scans", str(runs))
            self._set_meta_no_commit("last_scan_at", scanned_at)

    def scan_summary(self) -> dict:
        def num(key: str) -> float:
            try:
                return float(self.get_meta(key, "0") or 0)
            except ValueError:
                return 0.0

        return {
            "scans": int(num("scans")),
            "items": int(num("scan_items")),
            "frames": int(num("scan_frames")),
            "bytes": int(num("scan_bytes")),
            "wall": num("scan_wall"),
            "cpu": num("scan_cpu"),
            "gpu": num("scan_gpu"),
            "last_scan_at": self.get_meta("last_scan_at", ""),
        }

    def embedding_dim(self) -> int:
        row = self.conn.execute(
            "SELECT embedding FROM files WHERE embedding IS NOT NULL LIMIT 1"
        ).fetchone()
        return len(row["embedding"]) // 4 if row else 0

    def distinct_tag_count(self, kind: str = "image") -> int:
        return self.conn.execute(
            "SELECT COUNT(DISTINCT tag) AS n FROM media_tags WHERE kind = ?",
            (kind,),
        ).fetchone()["n"]

    def _extensions(self, table: str) -> tuple[list[tuple[str, int]], dict[str, int]]:
        """(ext, count) by count desc, plus ext -> total bytes."""
        from collections import Counter, defaultdict

        counts: Counter[str] = Counter()
        sizes: dict[str, int] = defaultdict(int)
        for r in self.conn.execute(f"SELECT path, size FROM {table}"):
            ext = os.path.splitext(r["path"])[1].lower() or "(none)"
            counts[ext] += 1
            sizes[ext] += r["size"]
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), dict(sizes)

    def image_extensions(self) -> tuple[list[tuple[str, int]], dict[str, int]]:
        return self._extensions("files")

    def video_extensions(self) -> tuple[list[tuple[str, int]], dict[str, int]]:
        return self._extensions("videos")

    # -- tag refinement -----------------------------------------------------

    def rows_for_refinement(self, kind: str) -> list[sqlite3.Row]:
        """id + current/raw keywords + description, for (re)building tags."""
        table = "files" if kind == "image" else "videos"
        return self.conn.execute(
            f"SELECT id, keywords, raw_keywords, description FROM {table} ORDER BY path"
        ).fetchall()

    def apply_refinement(
        self, kind: str, items: list[tuple[int, str, str, str, list[tuple[str, str | None]]]]
    ) -> None:
        """Replace refined tags for one media kind.

        Each item is ``(media_id, refined_csv, raw_keywords, description,
        [(tag, facet), ...])``. ``keywords`` is overwritten with the canonical
        tags (so FTS search benefits) while the original is kept in
        ``raw_keywords``; ``media_tags`` and the FTS row are rebuilt.
        """
        table = "files" if kind == "image" else "videos"
        fts = "files_fts" if kind == "image" else "videos_fts"
        existing = {r["id"] for r in self.conn.execute(f"SELECT id FROM {table}")}
        self.conn.execute("DELETE FROM media_tags WHERE kind = ?", (kind,))
        tag_rows: list[tuple[str, int, str, str | None]] = []
        for media_id, refined, raw, description, pairs in items:
            if media_id not in existing:
                continue
            self.conn.execute(
                f"UPDATE {table} SET keywords = ?, raw_keywords = ? WHERE id = ?",
                (refined, raw, media_id),
            )
            if self._fts_available:
                self.conn.execute(f"DELETE FROM {fts} WHERE rowid = ?", (media_id,))
                self.conn.execute(
                    f"INSERT INTO {fts}(rowid, description, keywords) VALUES(?, ?, ?)",
                    (media_id, description, refined),
                )
            tag_rows.extend((kind, media_id, tag, facet) for tag, facet in pairs)
        self.conn.executemany(
            "INSERT INTO media_tags(kind, media_id, tag, facet) VALUES(?, ?, ?, ?)",
            tag_rows,
        )
        self.conn.commit()

    def has_tags(self, kind: str = "image") -> int:
        return self.conn.execute(
            "SELECT COUNT(*) AS n FROM media_tags WHERE kind = ?", (kind,)
        ).fetchone()["n"]

    def tag_counts(self, limit: int = 40, kind: str = "image") -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT tag, COUNT(*) AS n FROM media_tags WHERE kind = ? "
            "GROUP BY tag ORDER BY n DESC, tag LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [(r["tag"], r["n"]) for r in rows]

    def facet_tag_counts(self, kind: str = "image") -> dict[str, list[tuple[str, int]]]:
        rows = self.conn.execute(
            "SELECT facet, tag, COUNT(*) AS n FROM media_tags "
            "WHERE kind = ? AND facet IS NOT NULL "
            "GROUP BY facet, tag ORDER BY facet, n DESC",
            (kind,),
        ).fetchall()
        out: dict[str, list[tuple[str, int]]] = {}
        for r in rows:
            out.setdefault(r["facet"], []).append((r["tag"], r["n"]))
        return out

    def facet_counts(self, kind: str = "image") -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT facet, COUNT(DISTINCT media_id) AS n FROM media_tags "
            "WHERE kind = ? AND facet IS NOT NULL GROUP BY facet ORDER BY n DESC",
            (kind,),
        ).fetchall()
        return [(r["facet"], r["n"]) for r in rows]

    def media_ids_with_facet(self, facet: str, kind: str = "image") -> list[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT media_id FROM media_tags WHERE kind = ? AND facet = ? "
            "ORDER BY media_id",
            (kind, facet),
        ).fetchall()
        return [r["media_id"] for r in rows]

    # -- image type ---------------------------------------------------------

    def set_types(self, items: list[tuple[int, str, float]]) -> None:
        self.conn.executemany(
            "UPDATE files SET type = ?, type_score = ? WHERE id = ?",
            [(name, score, fid) for fid, name, score in items],
        )
        self.conn.commit()

    def files_needing_type(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, path FROM files WHERE embedding IS NOT NULL AND type = '' "
            "ORDER BY path"
        ).fetchall()

    def files_with_type(self, name: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM files WHERE type = ? ORDER BY path", (name,)
        ).fetchall()

    def type_counts(self) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT type, COUNT(*) AS n FROM files WHERE type != '' "
            "GROUP BY type ORDER BY n DESC, type"
        ).fetchall()
        return [(r["type"], r["n"]) for r in rows]

    # -- clusters (people) --------------------------------------------------

    def set_clusters(
        self,
        kind: str,
        members: dict[int, list[int]],
        labels: dict[int, str] | None = None,
    ) -> None:
        """Replace the cluster assignment for one media kind.

        ``members`` maps cluster id -> media ids; ``labels`` optionally names
        a cluster (a person's name). Only one kind is used at a time, so the
        cluster ids are reused as the primary key.
        """
        table = "files" if kind == "image" else "videos"
        labels = labels or {}
        self.conn.execute(f"UPDATE {table} SET cluster_id = NULL")
        self.conn.execute("DELETE FROM clusters WHERE kind = ?", (kind,))
        for cid, ids in members.items():
            self.conn.execute(
                "INSERT INTO clusters(id, kind, label) VALUES(?, ?, ?)",
                (cid, kind, labels.get(cid, "")),
            )
            self.conn.executemany(
                f"UPDATE {table} SET cluster_id = ? WHERE id = ?",
                [(cid, mid) for mid in ids],
            )
        self.conn.commit()

    def all_clusters(self, kind: str = "image") -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, label FROM clusters WHERE kind = ? ORDER BY id", (kind,)
        ).fetchall()

    def set_cluster_label(self, cluster_id: int, label: str, kind: str = "image") -> None:
        self.conn.execute(
            "UPDATE clusters SET label = ? WHERE id = ? AND kind = ?",
            (label, cluster_id, kind),
        )
        self.conn.commit()

    def files_in_cluster(self, cluster_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM files WHERE cluster_id = ? ORDER BY path", (cluster_id,)
        ).fetchall()

    def clusters_for_label(self, label: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM clusters WHERE label = ? COLLATE NOCASE ORDER BY id", (label,)
        ).fetchall()
        return [r["id"] for r in rows]

    # -- trash log (undo support) -------------------------------------------

    def record_trash(self, records, kind: str = "image", batch: str = "") -> int:
        """Remember media moved to the OS Trash so it can be restored later."""
        rows = [
            (r.path, kind, int(getattr(r, "size", 0) or 0), batch)
            for r in records if getattr(r, "path", None)
        ]
        if not rows:
            return 0
        self.conn.executemany(
            "INSERT INTO trash_log(path, kind, size, batch) VALUES(?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def trash_log(self, batch: str | None = None) -> list[sqlite3.Row]:
        if batch:
            return self.conn.execute(
                "SELECT * FROM trash_log WHERE batch = ? ORDER BY path", (batch,)
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM trash_log ORDER BY batch DESC, path"
        ).fetchall()

    def trash_batches(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT batch, COUNT(*) AS n, SUM(size) AS bytes, MIN(trashed_at) AS at "
            "FROM trash_log GROUP BY batch ORDER BY batch DESC"
        ).fetchall()

    def clear_trash_log(self, ids: list[int]) -> None:
        if not ids:
            return
        marks = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM trash_log WHERE id IN ({marks})", ids)
        self.conn.commit()

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
