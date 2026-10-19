"""Command line interface for media-match."""

from __future__ import annotations

import argparse
import calendar
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
from datetime import datetime, timezone

from .db import Database, db_path_for, open_db
from .groups import find_duplicate_groups, find_similarity_groups
from .pipeline import (
    ScanStats,
    build_jobs,
    prune_missing,
    run_scan,
    run_staged_scan,
    store_result,
)
from .analyze import describe_targets, run_describe_stage
from .check import (
    CORRUPT, NOTABLE, OK, TRUNCATED, diagnose_tree, quarantine, repair_truncated,
)
from .bootstrap import (
    consolidated_path, find_exact_duplicates, fixed_extension_path, walk_tree,
)
from .tags import facet_of, refine
from .vision import DEFAULT_MODEL
from .clip import DEFAULT_CLIP_MODEL
from .video_groups import find_video_duplicate_groups, find_video_similarity_groups
from .video_search import best_moment, format_timestamp

_mp = lambda w, h: f"{w * h / 1_000_000:.1f}MP"

# Per-category cap on how many affected files to name in one report; the
# counts in the summary always cover every file.
_REPORT_LIMIT = 20

# Problems found during one scan/analyze run, grouped by reason so the user
# sees one line per kind of problem instead of one line per file. Guarded by a
# lock: the pipeline's worker threads report concurrently.
_BAD_FILES: dict[str, list[str]] = {}
_BAD_FILES_LOCK = threading.Lock()


def _why(exc: Exception) -> str:
    """Turn an exception or diagnosis into a sentence worth printing."""
    from .diagnose import Diagnosis

    if isinstance(exc, Diagnosis):
        return exc.message
    text = str(exc).strip()
    return text or type(exc).__name__


def _diagnose_failure(path: str, exc: Exception) -> str:
    """Describe why a file could not be indexed, in terms the user can act on.

    Only images go through the image diagnosis. Handing an .mp4 to Pillow
    reports "Pillow cannot identify or decode this file" — true, and useless:
    it hides the real reason, which is usually an ffmpeg probe failure or a
    bug such as a missing import.
    """
    from .diagnose import OK, diagnose_image
    from .pipeline import VIDEO_EXTENSIONS

    if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
        return f"ffmpeg could not read it: {_why(exc)}"
    if os.path.exists(path):
        found = diagnose_image(path)
        if found.status != OK:
            return found.message
    return _why(exc)


def _report_bad_file(path: str, reason: str) -> None:
    """Name the file and what is wrong with it, instead of leaking a traceback.

    Pillow's own warnings carry a site-packages path and no filename; a
    truncation or a malformed MPO is something the user can act on with
    'media-match check', so it has to say which file and why.
    """
    with _BAD_FILES_LOCK:
        _BAD_FILES.setdefault(reason, []).append(path)


def _summarize_bad_files(root: str) -> None:
    with _BAD_FILES_LOCK:
        if not _BAD_FILES:
            return
        found = dict(_BAD_FILES)
        _BAD_FILES.clear()
    for reason, paths in found.items():
        print(f"\n{len(paths)} file(s) could not be indexed: {reason}")
        for path in paths[:_REPORT_LIMIT]:
            try:
                shown = os.path.relpath(path, root)
            except ValueError:  # pragma: no cover - different drive on Windows
                shown = path
            print(f"  {shown}")
        if len(paths) > _REPORT_LIMIT:
            print(f"  ... and {len(paths) - _REPORT_LIMIT} more")
        print(f"  diagnose with: media-match check {root}")


def _dur(seconds: float) -> str:
    return format_timestamp(seconds)


def _fmt_bytes(n: float) -> str:
    """Human-readable byte size: 512.0B / 3.4MB / 1.2GB."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def _fmt_seconds(s: float) -> str:
    """Human duration: 0s / 3.2s / 4m12s / 2h05m."""
    if s <= 0:
        return "0s"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _reclaimable(groups) -> tuple[int, int]:
    """(extra files, bytes) recoverable by keeping only each group's best copy."""
    files = 0
    size = 0
    for group in groups:
        for record, _dist in group.others:
            files += 1
            size += int(getattr(record, "size", 0) or 0)
    return files, size


def _when(value) -> str:
    """'YYYY-MM-DD' from a captured_at string, or an epoch mtime; '' if none."""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return ""
    return str(value or "")[:10]


def _parse_day(text: str | None, *, end: bool) -> str | None:
    """Normalize YYYY / YYYY-MM / YYYY-MM-DD to a comparable boundary string."""
    text = (text or "").strip().replace("/", "-")
    if not text:
        return None
    fmt = ""
    dt = None
    for candidate in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(text, candidate)
            fmt = candidate
            break
        except ValueError:
            continue
    if dt is None:
        return None
    if fmt == "%Y":
        return f"{dt.year:04d}" + ("-12-31T23:59:59" if end else "-01-01")
    if fmt == "%Y-%m":
        if end:
            last = calendar.monthrange(dt.year, dt.month)[1]
            return f"{dt.year:04d}-{dt.month:02d}-{last:02d}T23:59:59"
        return f"{dt.year:04d}-{dt.month:02d}-01"
    return dt.strftime("%Y-%m-%d") + ("T23:59:59" if end else "")


def _date_filters(args) -> tuple[str, list, str, list]:
    """(image SQL, image params, video SQL, video params) for --since/--until/--camera."""
    since = _parse_day(getattr(args, "since", None), end=False)
    until = _parse_day(getattr(args, "until", None), end=True)
    camera = getattr(args, "camera", None)
    img, img_p, vid, vid_p = [], [], [], []
    img_when = "COALESCE(NULLIF(captured_at,''), datetime(mtime,'unixepoch'))"
    vid_when = "datetime(mtime,'unixepoch')"
    if since is not None:
        img.append(f"{img_when} >= ?")
        img_p.append(since)
        vid.append(f"{vid_when} >= ?")
        vid_p.append(since)
    if until is not None:
        img.append(f"{img_when} <= ?")
        img_p.append(until)
        vid.append(f"{vid_when} <= ?")
        vid_p.append(until)
    if camera:
        img.append("camera LIKE ?")
        img_p.append(f"%{camera}%")
    return " AND ".join(img), img_p, " AND ".join(vid), vid_p


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


def _require_ffmpeg() -> bool:
    from shutil import which

    if which("ffmpeg") and which("ffprobe"):
        return True
    print("video support needs 'ffmpeg' and 'ffprobe' on PATH "
          "(e.g. 'brew install ffmpeg')", file=sys.stderr)
    return False


# -- commands ----------------------------------------------------------------


def _run_scan_with_progress(jobs, encoder, args, root, *, interval, max_frames,
                            batch_size, workers, on_result, on_error, stats,
                            label: str = "indexing"):
    """Run the scan, drawing live progress on a terminal and plain lines otherwise.

    The bar needs a TTY to redraw in place. Without one (a pipe, CI, a test)
    it stays off entirely and the old periodic "preparing [n/total]" lines are
    printed instead, so logs keep their one-event-per-line shape.
    """
    from .progress import ProgressBar

    total_jobs = len(jobs)
    bar = ProgressBar(label, root, total_jobs,
                      enabled=_progress_enabled(args)).start()

    def progress(done: int, total: int, _path: str) -> None:
        if bar.enabled or total <= 20:
            return  # the bar is drawing, or the run is too short to bother
        step = max(1, total // 20)
        if done % step == 0 or done == total:
            print(f"  {label} [{done}/{total}]", file=sys.stderr)

    try:
        return run_staged_scan(
            jobs, encoder, interval=interval, max_frames=max_frames,
            batch_size=batch_size, workers=workers,
            io_workers=getattr(args, "io_workers", None) or 2,
            on_result=on_result, progress=progress, on_error=on_error,
            stats=stats, telemetry=bar,
        )
    finally:
        # Always tear the bar down: an exception mid-scan must not leave the
        # cursor sitting in the middle of a rewritten line.
        bar.stop()


def _progress_enabled(args) -> bool:
    """Progress bar unless --no-progress, or stderr is not a terminal.

    stderr is the test, not stdout: the bar is drawn there alongside the other
    progress lines, while stdout keeps carrying only command output.
    """
    if getattr(args, "no_progress", False):
        return False
    if os.environ.get("MEDIA_MATCH_NO_PROGRESS"):
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - detached stderr
        return False


def _add_progress_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-progress", action="store_true",
                   help="no live progress bar (implied when stderr is not a "
                        "terminal)")


def cmd_scan(args: argparse.Namespace) -> int:
    _register_heif()

    from .config import load_config, load_ignore

    cfg = load_config()
    root = os.path.abspath(args.root)
    ignore = load_ignore(root)

    # An explicit flag wins, then the config file, then the built-in default.
    clip_model = args.clip_model or cfg.get("clip_model") or DEFAULT_CLIP_MODEL
    model = args.model or cfg.get("model") or DEFAULT_MODEL
    workers = args.workers if args.workers is not None else (cfg.get("workers") or 4)
    batch = args.batch if args.batch is not None else (cfg.get("batch") or 16)
    interval = args.interval if args.interval is not None else (cfg.get("interval") or 1.0)
    max_frames = (args.max_frames if args.max_frames is not None
                  else (cfg.get("max_frames") or 120))

    db = Database(db_path_for(root, args.db))
    db.set_meta("root", root)
    db.set_meta("clip_model", clip_model)
    if args.describe:
        db.set_meta("vlm_model", model)

    if args.video and not _require_ffmpeg():
        db.close()
        return 2

    # With --prune, drop index rows for media moved or deleted since the last
    # scan, so the index tracks the folder instead of accumulating ghosts.
    if args.prune:
        pruned = prune_missing(root, db, include_video=args.video, ignore=ignore)
        if pruned:
            print(f"removed {pruned} stale index row(s) "
                  f"(moved or deleted since last scan)")
        else:
            print("no stale index rows to remove")

    jobs, total, up_to_date = build_jobs(
        root, db, include_video=args.video, reanalyze=args.reanalyze,
        limit=args.limit, ignore=ignore,
    )
    print(f"{total} media found, {up_to_date} up to date, {len(jobs)} to analyze")

    from .clip import ClipEncoder

    encoder = ClipEncoder(model_id=clip_model, batch_size=batch)
    errors = 0

    def on_error(path: str, exc: Exception) -> None:
        nonlocal errors
        errors += 1
        _report_bad_file(path, exc)

    def on_result(result) -> None:
        nonlocal errors
        try:
            store_result(db, result)
        except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
            errors += 1
            print(f"  ERROR storing {result.job.path}: {exc}", file=sys.stderr)

    scan_stats = ScanStats()
    analyzed = _run_scan_with_progress(
        jobs, encoder, args, root,
        interval=interval, max_frames=max_frames, batch_size=batch,
        workers=workers, on_result=on_result, on_error=on_error,
        stats=scan_stats,
    )
    _summarize_bad_files(root)
    # Accumulate this run's measured cost so `stats` can show it.
    db.record_scan(
        items=scan_stats.items, frames=scan_stats.frames,
        bytes_read=scan_stats.bytes_read, wall=scan_stats.wall,
        cpu=scan_stats.cpu, gpu=scan_stats.gpu,
    )
    if scan_stats.items:
        print(f"  measured: {_fmt_seconds(scan_stats.wall)} wall, "
              f"{_fmt_seconds(scan_stats.gpu)} gpu, "
              f"{_fmt_seconds(scan_stats.cpu)} cpu, "
              f"{_fmt_bytes(scan_stats.bytes_read)} read")

    # Optional keep-tags-fresh step: refine keywords into canonical tags/facets
    # (and, with --classify, type each image) before releasing the CLIP model.
    if args.refine or args.classify:
        refined = _refine_all(db, quiet=True)
        if refined:
            print(f"refined {refined} item(s)")
    if args.classify:
        classified = _classify_with(db, encoder, recompute=False)
        if classified:
            print(f"classified {classified} image(s)")

    encoder.unload()

    # Optional VLM captions, best version of each duplicate group only.
    if args.describe:
        described = _describe_all(db, root, model, args, on_error)
        if described:
            print(f"described {described} item(s)")

    described_count = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE description != ''"
    ).fetchone()["n"]
    vdescribed = db.conn.execute(
        "SELECT COUNT(*) AS n FROM videos WHERE description != ''"
    ).fetchone()["n"]
    print(f"Done: analyzed {analyzed}, errors {errors}. Index: "
          f"{db.count_files()} images ({described_count} described), "
          f"{db.count_videos()} videos ({vdescribed} described) ({db.path})")
    db.close()
    return 0


def _describe_all(db: Database, root: str, model: str, args, on_error) -> int:
    """Caption undescribed images and videos through one pipelined stage.

    ``model`` must already be resolved (flag > config > default) — passing the
    raw ``args.model`` here is what used to make every caption fail with
    "Path(None)" when --model was omitted.
    """
    plan = describe_targets(db, root, include_videos=args.video)
    if plan.missing:
        print(f"{plan.missing} undescribed item(s) skipped: no longer on disk "
              f"(stale index; run 'media-match scan --prune' to refresh)")
    if plan.shadowed:
        print(f"{plan.shadowed} undescribed duplicate(s) skipped; "
              f"their best-resolution version is described instead")
    if not plan.jobs:
        return 0
    counts = " + ".join(f"{n} {kind}(s)" for kind, n in plan.pending.items() if n)
    print(f"describing {len(plan.jobs)} item(s): {counts}")

    from .vision import Describer

    describer = Describer(model_id=model)
    total = len(plan.jobs)
    done = 0

    def progress(job) -> None:
        nonlocal done
        done += 1
        print(f"[{done}/{total}] describing {job.kind} {job.label}",
              file=sys.stderr)

    try:
        described = run_describe_stage(
            plan.jobs, describer, db,
            workers=getattr(args, "describe_workers", None) or 3,
            batch_size=getattr(args, "describe_batch", None) or 1,
            on_error=on_error, progress=progress,
        )
    finally:
        describer.unload()
    return described


def cmd_describe(args: argparse.Namespace) -> int:
    """Describe-only command: caption indexed media that has no caption yet."""
    _register_heif()
    from .config import load_config

    cfg = load_config()
    root = os.path.abspath(args.root)
    model = args.model or cfg.get("model") or DEFAULT_MODEL

    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    errors = 0

    def on_error(path: str, exc: Exception) -> None:
        nonlocal errors
        errors += 1
        print(f"  ERROR {path}: {exc}", file=sys.stderr)

    described = _describe_all(db, root, model, args, on_error)
    print(f"Done: described {described}, errors {errors} ({db.path})")
    db.close()
    return 0


def _list_by_type(db: Database, args) -> int:
    rows = db.files_with_type(args.type)
    if not rows:
        known = ", ".join(name for name, _ in db.type_counts()) or "(none)"
        print(f"No images of type '{args.type}'. Known types: {known}")
        db.close()
        return 1
    for r in rows[: args.limit]:
        when = _when(r["captured_at"])
        print(f"{r['path']}  ({r['width']}x{r['height']}, "
              f"{_mp(r['width'], r['height'])}"
              + (f", {when}" if when else "") + f")  type={r['type']} "
              f"({r['type_score']:.3f})")
    print(f"\n{len(rows)} image(s) of type '{args.type}'"
          + (f", showing {args.limit}" if len(rows) > args.limit else ""))
    db.close()
    return 0


def _list_by_cluster(db: Database, args) -> int:
    cluster_ids = db.clusters_for_label(args.cluster)
    if not cluster_ids:
        print(f"No person cluster named '{args.cluster}'. "
              f"Name one with 'people --label N=Name'.")
        db.close()
        return 1
    rows: list = []
    for cid in cluster_ids:
        rows.extend(db.files_in_cluster(cid))
    for r in rows[: args.limit]:
        when = _when(r["captured_at"])
        print(f"{r['path']}  ({r['width']}x{r['height']}, "
              f"{_mp(r['width'], r['height'])}"
              + (f", {when}" if when else "") + ")")
    print(f"\n{len(rows)} photo(s) of '{args.cluster}'"
          + (f", showing {args.limit}" if len(rows) > args.limit else ""))
    db.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1
    if args.type:
        return _list_by_type(db, args)
    if args.cluster:
        return _list_by_cluster(db, args)
    for name in ("since", "until"):
        value = getattr(args, name, None)
        if value and _parse_day(value, end=(name == "until")) is None:
            print(f"--{name}: expected a date like YYYY-MM-DD")
            db.close()
            return 2

    img_filter, img_params, vid_filter, vid_params = _date_filters(args)

    terms: list[str] = []
    if args.person:
        terms.extend(args.person.split())
    if args.object:
        terms.extend(args.object.split())
    terms.extend(args.query)

    if args.clip and args.fts:
        print("use either --clip or --fts, not both")
        db.close()
        return 2

    include_images = not args.video
    include_videos = not args.image_only

    use_clip = args.clip
    if not use_clip and not args.fts:
        if not terms and (img_filter or vid_filter):
            use_clip = False  # a date/camera-only listing has no text to embed
        else:
            _, blobs = db.all_embeddings()
            use_clip = bool(blobs) or bool(db.video_embeddings())

    if use_clip:
        allowed_images = db.file_ids_matching(img_filter, img_params) if img_filter else None
        allowed_videos = db.video_ids_matching(vid_filter, vid_params) if vid_filter else None
        hits = _clip_search(db, " ".join(terms), args.limit,
                            include_images=include_images,
                            include_videos=include_videos,
                            allowed_images=allowed_images,
                            allowed_videos=allowed_videos)
        if hits is None:
            db.close()
            return 1
        if args.json:
            print(json.dumps(hits, indent=2, ensure_ascii=False))
            db.close()
            return 0 if hits else 1
        if not hits:
            print("No matches.")
            db.close()
            return 1
        print(f"CLIP text->media search for {' '.join(terms)!r}:")
        for h in hits:
            when = _when(h["captured_at"]) if h["kind"] == "image" else _when(h["mtime"])
            if h["kind"] == "image":
                print(f"\n{h['path']}  ({h['width']}x{h['height']}, "
                      f"{_mp(h['width'], h['height'])}"
                      + (f", {when}" if when else "")
                      + f")  similarity={h['similarity']:.3f}")
            else:
                print(f"\n{h['path']}  ({h['width']}x{h['height']}, "
                      f"{_dur(h['duration'])}"
                      + (f", {when}" if when else "")
                      + f")  similarity={h['similarity']:.3f}  "
                      f"@ {format_timestamp(h['timestamp'])}")
            if h["keywords"]:
                print(f"  keywords: {h['keywords']}")
            if h["description"]:
                print(f"  {h['description']}")
        db.close()
        return 0

    results: list[dict] = []
    if include_images:
        for r in db.search(terms, limit=args.limit, filters=img_filter,
                           params=img_params):
            results.append({
                "kind": "image", "path": r["path"], "width": r["width"],
                "height": r["height"], "size": r["size"],
                "captured_at": r["captured_at"], "camera": r["camera"],
                "description": r["description"], "keywords": r["keywords"],
            })
    if include_videos:
        for r in db.search_videos(terms, limit=args.limit, filters=vid_filter,
                                  params=vid_params):
            results.append({
                "kind": "video", "path": r["path"], "width": r["width"],
                "height": r["height"], "size": r["size"], "duration": r["duration"],
                "mtime": r["mtime"], "captured_at": "", "camera": "",
                "description": r["description"], "keywords": r["keywords"],
            })
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        db.close()
        return 0 if results else 1
    if not results:
        print("No matches.")
        db.close()
        return 1
    for r in results:
        when = _when(r["captured_at"]) if r["kind"] == "image" else _when(r["mtime"])
        if r["kind"] == "image":
            print(f"\n{r['path']}  ({r['width']}x{r['height']}, "
                  f"{_mp(r['width'], r['height'])}"
                  + (f", {when}" if when else "") + ")")
        else:
            print(f"\n{r['path']}  ({r['width']}x{r['height']}, "
                  f"{_dur(r['duration'])}"
                  + (f", {when}" if when else "") + ")")
        if r["keywords"]:
            print(f"  keywords: {r['keywords']}")
        if r["description"]:
            print(f"  {r['description']}")
    db.close()
    return 0


def _load_embedding_matrix(db: Database):
    import numpy as np

    ids, blobs = db.all_embeddings()
    if not blobs:
        return ids, None
    return ids, np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])


def _clip_search(db: Database, query_text: str, limit: int, *,
                 include_images: bool = True, include_videos: bool = True,
                 allowed_images: set[int] | None = None,
                 allowed_videos: set[int] | None = None) -> list[dict] | None:
    """Rank media by CLIP similarity; None means 'no embeddings' (error shown)."""
    import numpy as np

    from .clip import ClipEncoder

    if not query_text.strip():
        print("Give a search query (CLIP search needs some text).")
        return None

    ids, matrix = _load_embedding_matrix(db)
    video_embeddings = db.video_embeddings() if include_videos else {}
    if allowed_videos is not None:
        video_embeddings = {v: f for v, f in video_embeddings.items()
                            if v in allowed_videos}
    if (matrix is None or not include_images) and not video_embeddings:
        print("No CLIP embeddings yet. Run 'media-match scan' first.")
        return None

    query = ClipEncoder().embed_texts([query_text])[0]

    hits: list[dict] = []
    if include_images and matrix is not None:
        sims = matrix @ query
        found = 0
        for idx in np.argsort(-sims):
            fid = ids[int(idx)]
            if allowed_images is not None and fid not in allowed_images:
                continue
            row = db.conn.execute(
                "SELECT * FROM files WHERE id = ?", (fid,)
            ).fetchone()
            hits.append({
                "kind": "image", "path": row["path"], "width": row["width"],
                "height": row["height"], "size": row["size"],
                "captured_at": row["captured_at"], "camera": row["camera"],
                "description": row["description"], "keywords": row["keywords"],
                "similarity": round(float(sims[int(idx)]), 4),
            })
            found += 1
            if found >= limit:
                break
    if video_embeddings:
        by_id = {r.id: r for r in db.all_videos()}
        for vid, frames in video_embeddings.items():
            record = by_id.get(vid)
            if record is None:
                continue
            frame_map = {t: np.frombuffer(b, dtype=np.float32) for t, b in frames}
            score, ts = best_moment(query, frame_map)
            hits.append({
                "kind": "video", "path": record.path, "width": record.width,
                "height": record.height, "size": record.size,
                "duration": record.duration, "mtime": record.mtime,
                "captured_at": "", "camera": "",
                "description": record.description, "keywords": record.keywords,
                "similarity": round(float(score), 4), "timestamp": ts,
            })

    hits.sort(key=lambda h: -h["similarity"])
    return hits[:limit]


def cmd_embed(args: argparse.Namespace) -> int:
    import time

    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    if args.recompute:
        rows = db.conn.execute("SELECT id, path FROM files ORDER BY path").fetchall()
    else:
        rows = db.files_missing_embedding()
    if not rows:
        print("All indexed files already have CLIP embeddings.")
        db.close()
        return 0

    from PIL import Image  # noqa: F401  (HEIC support via registration below)

    _register_heif()
    from .clip import ClipEncoder

    encoder = ClipEncoder(model_id=args.clip_model, batch_size=args.batch)
    db.set_meta("clip_model", args.clip_model)
    print(f"embedding {len(rows)} image(s) with {args.clip_model} ...")

    done = 0
    errors = 0
    t0 = time.perf_counter()
    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        ids, paths = [], []
        for r in chunk:
            try:
                if not os.path.exists(r["path"]):
                    raise FileNotFoundError("file missing")
                ids.append(r["id"])
                paths.append(r["path"])
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"  ERROR {r['path']}: {exc}")
        if not paths:
            continue
        try:
            vectors = encoder.embed_image_paths(paths)
        except Exception as exc:  # noqa: BLE001
            errors += len(paths)
            print(f"  ERROR embedding batch: {exc}")
            continue
        db.set_embeddings([(fid, v.tobytes()) for fid, v in zip(ids, vectors)])
        done += len(paths)
        print(f"  {start + len(chunk)}/{len(rows)}", end="\r")

    elapsed = time.perf_counter() - t0
    rate = done / elapsed if elapsed > 0 else 0
    print(f"\nDone: embedded {done} images in {elapsed:.1f}s ({rate:.1f} img/s), "
          f"errors {errors}")
    db.close()
    return 0


def cmd_similar(args: argparse.Namespace) -> int:
    import numpy as np

    _register_heif()
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — run 'media-match scan {root}' first.")
        return 1

    if args.video:
        return _similar_videos(db, args)

    ids, matrix = _load_embedding_matrix(db)
    if matrix is None:
        print("No CLIP embeddings in the index. Run 'media-match embed' first.")
        db.close()
        return 1

    id_to_record = {r.id: r for r in db.all_files()}

    if args.image:
        from .clip import ClipEncoder

        target = os.path.abspath(args.image)
        stored = db.embedding_for_path(target)
        if stored is not None:
            query = np.frombuffer(stored, dtype=np.float32)
            source = "indexed embedding"
        else:
            _register_heif()
            query = ClipEncoder().embed_image_paths([target])[0]
            source = "computed on the fly"
        sims = matrix @ query

        exclude_row = db.conn.execute(
            "SELECT id FROM files WHERE path = ?", (target,)
        ).fetchone()
        exclude_id = exclude_row["id"] if exclude_row else None

        order = np.argsort(-sims)
        shown = 0
        print(f"Images similar to {target} ({source}):")
        for idx in order:
            record = id_to_record[ids[idx]]
            if record.id == exclude_id:
                continue
            print(f"  {sims[idx]:.3f}  {record.path}  "
                  f"({record.width}x{record.height}, {_mp(record.width, record.height)})")
            shown += 1
            if shown >= args.limit:
                break
        db.close()
        return 0

    records = [id_to_record[i] for i in ids]
    groups = find_similarity_groups(records, matrix, threshold=args.threshold)
    if not groups:
        print(f"No semantic duplicate groups at cosine threshold {args.threshold}.")
        db.close()
        return 0

    if args.json:
        print(json.dumps([
            {"best": g.best.path,
             "files": [{"path": r.path, "similarity": round(s, 4)}
                       for r, s in g.others]}
            for g in groups
        ], indent=2, ensure_ascii=False))
        db.close()
        return 0

    total = sum(g.size for g in groups)
    print(f"{len(groups)} semantic group(s), {total} files "
          f"(cosine >= {args.threshold}; best version marked with *)\n")
    for n, group in enumerate(groups, 1):
        b = group.best
        print(f"Group {n} ({group.size} files)")
        print(f"  * {b.path}  ({b.width}x{b.height}, {_mp(b.width, b.height)}) [best]")
        for record, sim in group.others:
            print(f"    {record.path}  ({record.width}x{record.height}, "
                  f"{_mp(record.width, record.height)}) similarity={sim:.3f}")
        print()
    db.close()
    return 0


def _similar_videos(db: Database, args) -> int:
    import numpy as np

    from .clip import ClipEncoder

    embeddings = db.video_embeddings()
    if not embeddings:
        print("No CLIP embeddings for videos. Run 'media-match scan --video' first.")
        db.close()
        return 1
    records = db.all_videos()
    id_to_record = {r.id: r for r in records}

    if args.image:
        target = os.path.abspath(args.image)
        query = ClipEncoder().embed_image_paths([target])[0]
        hits = []
        for vid, frames in embeddings.items():
            frame_map = {t: np.frombuffer(b, dtype=np.float32) for t, b in frames}
            score, ts = best_moment(query, frame_map)
            hits.append((score, vid, ts))
        hits.sort(reverse=True)
        print(f"Videos similar to {target}:")
        for score, vid, ts in hits[: args.limit]:
            record = id_to_record[vid]
            print(f"  {score:.3f}  {record.path}  "
                  f"({record.width}x{record.height}, {_dur(record.duration)})  "
                  f"@ {format_timestamp(ts)}")
        db.close()
        return 0

    stacked = {
        vid: np.stack([np.frombuffer(b, np.float32) for _t, b in frames])
        for vid, frames in embeddings.items() if frames
    }
    groups = find_video_similarity_groups(
        records, stacked, threshold=args.threshold
    )
    if not groups:
        print(f"No semantic video groups at cosine threshold {args.threshold}.")
        db.close()
        return 0

    if args.json:
        print(json.dumps([
            {"best": g.best.path,
             "files": [{"path": r.path, "similarity": round(s, 4)}
                       for r, s in g.others]}
            for g in groups
        ], indent=2, ensure_ascii=False))
        db.close()
        return 0

    total = sum(g.size for g in groups)
    print(f"{len(groups)} semantic video group(s), {total} videos "
          f"(cosine >= {args.threshold}; best version marked with *)\n")
    for n, group in enumerate(groups, 1):
        b = group.best
        print(f"Group {n} ({group.size} videos)")
        print(f"  * {b.path}  ({b.width}x{b.height}, {_dur(b.duration)}) [best]")
        for record, sim in group.others:
            print(f"    {record.path}  ({record.width}x{record.height}, "
                  f"{_dur(record.duration)}) similarity={sim:.3f}")
        print()
    db.close()
    return 0


def cmd_duplicates(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    from .config import load_config

    cfg = load_config()
    threshold = (args.threshold if args.threshold is not None
                 else (cfg.get("threshold") or 8))
    prefer = args.prefer or cfg.get("prefer") or "resolution"

    if args.video:
        groups = find_video_duplicate_groups(
            db.all_videos_with_frames(),
            frame_threshold=args.frame_threshold, merge_ratio=args.ratio,
        )
    groups = find_duplicate_groups(db.all_files(), threshold=threshold,
                                   prefer=prefer, exact_phash=args.exact_phash)
    if not groups:
        print("No duplicate groups found.")
        db.close()
        return 0

    total = sum(g.size for g in groups)

    if args.json:
        payload = []
        for g in groups:
            if args.video:
                best = {
                    "best": g.best.path,
                    "duration": g.best.duration,
                    "resolution": [g.best.width, g.best.height],
                    "duplicates": [
                        {"path": r.path, "duration": r.duration,
                         "resolution": [r.width, r.height], "min_hamming": d}
                        for r, d in g.others
                    ],
                }
            else:
                best = {
                    "best": g.best.path,
                    "best_size": [g.best.width, g.best.height],
                    "duplicates": [
                        {"path": r.path, "size": [r.width, r.height],
                         "hash_distance": d}
                        for r, d in g.others
                    ],
                }
            payload.append(best)
        print(json.dumps(payload, indent=2))
        db.close()
        return 0

    for n, g in enumerate(groups, 1):
        print(f"Group {n} ({g.size} files)")
        b = g.best
        if args.video:
            print(f"  * {b.path}  ({b.width}x{b.height}, {_dur(b.duration)}) [best]")
            for r, dist in g.others:
                print(f"    {r.path}  ({r.width}x{r.height}, {_dur(r.duration)}) "
                      f"hamming_min={dist}")
        else:
            when = _when(b.captured_at)
            print(f"  * {b.path}  ({b.width}x{b.height}, {_mp(b.width, b.height)}"
                  + (f", {when}" if when else "") + ") [best]")
            for r, dist in g.others:
                w = _when(r.captured_at)
                print(f"    {r.path}  ({r.width}x{r.height}, {_mp(r.width, r.height)}"
                      + (f", {w}" if w else "") + f") hamming={dist}")
        print()
    extra, reclaim = _reclaimable(groups)
    print(f"{len(groups)} group(s), {total} files — reclaimable "
          f"{_fmt_bytes(reclaim)} in {extra} extra copy(ies)")
    db.close()
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    from .cleanup import (
        confirm,
        file_size,
        human_size,
        interactive_select,
        print_plan,
        trash_files,
    )

    kind = "video" if args.video else "image"
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    if args.video:
        groups = find_video_duplicate_groups(
            db.all_videos_with_frames(),
            frame_threshold=args.frame_threshold, merge_ratio=args.ratio,
        )
    else:
        groups = find_duplicate_groups(db.all_files(), threshold=args.threshold)
    if not groups:
        print("No duplicate groups found.")
        db.close()
        return 0

    total_files = sum(g.size for g in groups)

    if args.dry_run and args.json:
        trash = print_plan(groups, root, lambda _s: None, trash_word="[would]")
        print(json.dumps([
            {"path": r.path, "kind": kind, "size": r.size,
             "resolution": [r.width, r.height]}
            for r in trash
        ], indent=2, ensure_ascii=False))
        db.close()
        return 0

    print(f"{len(groups)} duplicate group(s), {total_files} files involved")

    if args.dry_run:
        trash = print_plan(groups, root, print, trash_word="[would]")
        size = sum(file_size(r) for r in trash)
        print(f"\nDry run: {len(trash)} file(s) ({human_size(size)}) would go "
              f"to the Trash. Nothing was changed.")
        db.close()
        return 0

    if args.yes:
        trash = print_plan(groups, root, print, trash_word="[trash]")
    else:
        from .tui import run_tui

        session = run_tui(groups, root=root, db=db)
        if session is not None:  # visual UI ran (it updates the index itself)
            moved = session.x_trashed + session.batch_trashed
            for error in session.errors:
                print(f"  {error}")
            if moved:
                db.record_trash(moved, kind=kind, batch=batch)
                size = sum(r.size for r in moved)
                print(f"Moved {len(moved)} file(s) to the Trash "
                      f"({human_size(size)}; recoverable until emptied); "
                      f"removed {len(moved)} entries from the index. "
                      f"{len(session.errors)} failed. "
                      f"Undo with 'media-match restore {root}'.")
            else:
                print("Nothing changed.")
            db.close()
            return 0 if not session.errors else 1
        trash = interactive_select(groups, root)  # no TTY: line-based prompts

    if not trash:
        print("\nNothing selected — no files were changed.")
        db.close()
        return 0

    size = sum(file_size(r) for r in trash)
    print(f"\n{len(trash)} file(s) ({human_size(size)}) selected for the Trash:")
    for record in trash:
        print(f"  {os.path.relpath(record.path, root)}")

    if not args.yes and not confirm("Move to the system Trash? [y/N] "):
        print("Aborted; nothing was changed.")
        db.close()
        return 1

    trashed, failed = trash_files(trash)
    db.delete_media(trashed)
    db.record_trash(trashed, kind=kind, batch=batch)
    print(f"\nMoved {len(trashed)} file(s) to the Trash (recoverable until "
          f"emptied); removed {len(trashed)} entries from the index. "
          f"{len(failed)} failed. Undo with 'media-match restore {root}'.")
    db.close()
    return 0 if not failed else 1


def cmd_restore(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    from .cleanup import confirm, human_size, restore_file

    rows = db.trash_log(args.batch)
    if not rows:
        print("Nothing to restore — no trashed files are logged for this index.")
        db.close()
        return 0

    total = sum(r["size"] for r in rows)
    if args.batch:
        print(f"batch {args.batch}: {len(rows)} file(s), {human_size(total)}")
    else:
        print(f"{len(rows)} file(s), {human_size(total)} in "
              f"{len(db.trash_batches())} batch(es) of trashed files:")
    for r in rows[: args.limit]:
        print(f"  [{r['batch']}] {r['kind']:<5} {r['path']}")
    if len(rows) > args.limit:
        print(f"  ... and {len(rows) - args.limit} more")

    if args.dry_run:
        print("\nDry run: nothing was moved.")
        db.close()
        return 0

    if not args.yes and not confirm(
        f"\nMove {len(rows)} file(s) back from the system Trash? [y/N] "
    ):
        print("Aborted; nothing was changed.")
        db.close()
        return 1

    restored = already = missing = 0
    done_ids: list[int] = []
    for r in rows:
        status = restore_file(r["path"])
        if status == "restored":
            restored += 1
            done_ids.append(r["id"])
        elif status == "already":
            already += 1
            done_ids.append(r["id"])
        else:
            missing += 1
            print(f"  not in the Trash: {r['path']}")

    db.clear_trash_log(done_ids)
    print(f"\nRestored {restored} file(s) from the Trash"
          + (f", {already} were already back" if already else "")
          + (f", {missing} not found" if missing else "")
          + ". Run 'scan' to re-index them.")
    db.close()
    return 0 if not missing else 1


_EXPORT_KEYS = (
    "kind", "path", "width", "height", "size", "captured_at", "camera",
    "description", "keywords", "type", "tags", "cluster", "duration",
)


def _render_export(rows: list[dict], fmt: str) -> str:
    if fmt == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    if fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(_EXPORT_KEYS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _EXPORT_KEYS})
        return buf.getvalue()
    lines = [
        "| path | kind | size | captured | camera | type | tags | cluster |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['path']} | {r['kind']} | {_fmt_bytes(r['size'])} | "
            f"{r['captured_at'][:10]} | {r['camera']} | {r['type']} | "
            f"{r['tags']} | {r['cluster']} |"
        )
    return "\n".join(lines) + "\n"


def cmd_export(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    image_clusters = db.cluster_labels("image")
    rows: list[dict] = []
    for r in db.all_files():
        tags = ", ".join(t for t, _f in db.tags_for("image", r.id))
        cluster = image_clusters.get(r.cluster_id, "") if r.cluster_id else ""
        rows.append({
            "kind": "image", "path": r.path, "width": r.width, "height": r.height,
            "size": r.size, "captured_at": r.captured_at, "camera": r.camera,
            "description": r.description, "keywords": r.keywords, "type": r.type,
            "tags": tags, "cluster": cluster, "duration": "",
        })
    for v in db.all_videos():
        tags = ", ".join(t for t, _f in db.tags_for("video", v.id))
        rows.append({
            "kind": "video", "path": v.path, "width": v.width, "height": v.height,
            "size": v.size, "captured_at": "", "camera": "",
            "description": v.description, "keywords": v.keywords, "type": "",
            "tags": tags, "cluster": "", "duration": v.duration,
        })
    rows.sort(key=lambda r: r["path"])

    text = _render_export(rows, args.format)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {len(rows)} item(s) to {args.out}")
    else:
        print(text, end="")
    db.close()
    return 0


def cmd_clear_metadata(args: argparse.Namespace) -> int:
    """Clear description/keywords/type/tags for listed items — or everything."""
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    targets: dict[str, list[int]] = {}
    if args.items:
        # paths are given as printed by the tool: relative to the root
        paths = [p if os.path.isabs(p) else os.path.join(root, p)
                 for p in args.items]
        found, missing = db.media_ids_for_paths(paths, kind=args.kind)
        for kind, _path, media_id in found:
            targets.setdefault(kind, []).append(media_id)
        for path in missing:
            print(f"not in index: {os.path.relpath(path, root)}")
    else:
        for kind in ("image", "video") if args.kind is None else (args.kind,):
            table = "files" if kind == "image" else "videos"
            rows = db.conn.execute(f"SELECT id FROM {table}").fetchall()
            targets[kind] = [r["id"] for r in rows]

    total = sum(len(ids) for ids in targets.values())
    if not total:
        print("Nothing to clear.")
        db.close()
        return 0

    if not args.yes and sys.stdin.isatty():
        answer = input(f"Clear metadata for {total} item(s)? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted")
            db.close()
            return 1

    for kind, ids in targets.items():
        db.batch_clear_metadata(ids, kind)
        print(f"cleared metadata for {len(ids)} {kind}(s)")
    db.close()
    return 0


def cmd_import_metadata(args: argparse.Namespace) -> int:
    """Import description/keywords/type for indexed items from JSON or CSV."""
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    try:
        with open(args.file, encoding="utf-8-sig") as fh:
            if args.file.lower().endswith(".csv"):
                import csv

                rows = list(csv.DictReader(fh))
            else:
                data = json.load(fh)
                rows = data if isinstance(data, list) else data.get("items", [])
                if not isinstance(rows, list):
                    print("import file must be a JSON list (or CSV)")
                    db.close()
                    return 2
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.file}: {exc}")
        db.close()
        return 2

    fields = [f.strip().lower() for f in args.fields.split(",") if f.strip()]
    unknown = [f for f in fields if f not in ("description", "keywords", "type")]
    if unknown:
        print(f"unknown field(s): {', '.join(unknown)} — use description, keywords, type")
        db.close()
        return 2

    per_kind: dict[str, list[tuple[int, str, str, str]]] = {}
    missing: list[str] = []
    for row in rows:
        rel = (row.get("path") or "").strip()
        if not rel:
            continue
        path = rel if os.path.isabs(rel) else os.path.join(root, rel)
        kind = (row.get("kind") or args.kind or "").strip() or None
        found, _miss = db.media_ids_for_paths([path], kind=kind or None)
        if not found:
            missing.append(rel)
            continue
        _k, _p, media_id = found[0]
        kind = found[0][0]
        per_kind.setdefault(kind, []).append((
            media_id,
            row.get("description") or "" if "description" in fields else "",
            row.get("keywords") or "" if "keywords" in fields else "",
            row.get("type") or "" if ("type" in fields and kind == "image") else "",
        ))

    written = 0
    for kind, items in per_kind.items():
        written += db.set_metadata(kind, items)
    for rel in missing:
        print(f"not in index: {rel}")
    print(f"imported metadata for {written} item(s) from {args.file}")
    db.close()
    return 0 if written else 1


def cmd_export_metadata(args: argparse.Namespace) -> int:
    """Export editable metadata (description/keywords/type) as JSON.

    The output round-trips with 'import-metadata', so it works as a backup
    before 'clear-metadata' or as a hand-edited caption batch.
    """
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    kinds = ("image", "video") if args.kind is None else (args.kind,)
    items: list[dict] = []
    for kind in kinds:
        table = "files" if kind == "image" else "videos"
        for r in db.conn.execute(f"SELECT * FROM {table} ORDER BY path"):
            item = {
                "kind": kind,
                "path": os.path.relpath(r["path"], root),
                "description": r["description"],
                "keywords": r["keywords"],
            }
            if kind == "image":
                item["type"] = r["type"]
            items.append(item)

    text = json.dumps(items, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {len(items)} item(s) to {args.out}")
    else:
        print(text)
    db.close()
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Delete the index database — a full reset. Media files are not touched."""
    root = os.path.abspath(args.root)
    path = db_path_for(root, args.db)
    if not os.path.exists(path):
        print(f"No index at {path} — nothing to reset.")
        return 0

    if not args.yes and sys.stdin.isatty():
        answer = input(
            f"Delete the index at {path}?\n"
            f"(media files are NOT touched; 'scan' rebuilds from scratch) [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    for suffix in ("", "-wal", "-shm"):
        side = path + suffix
        if os.path.exists(side):
            os.unlink(side)
    print(f"removed {path} — index reset; run 'media-match scan' to rebuild")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    """First pass over a tree: walk it, record filename/type/size in the index.

    Dry-run by default — builds the database but touches no files. Add
    --prune to trash exact duplicates (keeping the newest), --media-only to
    prune non-media files and consolidate extensions, and --fix-extensions
    (with --media-only) to add/update missing extensions. With --move-to DIR,
    --media-only moves the non-media files there instead of trashing them.
    """
    from .config import load_ignore

    if getattr(args, "move_to", None) and not args.media_only:
        print("--move-to only applies to --media-only", file=sys.stderr)
        return 2

    root = os.path.abspath(args.root)
    ignore = load_ignore(root)

    # Moving files into a folder inside the tree we are about to walk would
    # make the walk find them again, so keep the destination out of it.
    move_to = os.path.abspath(args.move_to) if args.move_to else None
    if move_to == root:
        print("--move-to must be a folder other than the tree itself",
              file=sys.stderr)
        return 2
    if move_to:
        rel = os.path.relpath(move_to, root)
        if not rel.startswith(os.pardir):
            ignore = list(ignore) + [rel]

    # Walk the whole tree (not just media) so --media-only can see the rest.
    rows = walk_tree(root, include_video=args.video, ignore=ignore)
    db = Database(db_path_for(root, args.db))
    db.replace_bootstrap(rows)

    counts: dict[str, int] = {}
    for _path, _size, _mtime, _ext, kind in rows:
        counts[kind] = counts.get(kind, 0) + 1
    n_img = counts.get("image", 0)
    n_vid = counts.get("video", 0)
    n_aud = counts.get("audio", 0)
    n_other = counts.get("other", 0)
    print(f"indexed {len(rows)} file(s): {n_img} image(s), {n_vid} video(s), "
          f"{n_aud} audio, {n_other} other")
    if n_other:
        print(f"({n_other} non-media file(s) — use --media-only to prune them)")

    # --prune: find exact duplicates, keep newest, trash the rest
    pruned = 0
    trashed_paths: set[str] = set()
    if args.prune:
        groups = find_exact_duplicates(rows)
        to_trash = [rows[i][0] for g in groups for i in g[1:]]
        if to_trash:
            print(f"\n{len(groups)} duplicate group(s): "
                  f"{len(to_trash)} file(s) to trash (keeping the newest of each)")
            for g in groups:
                keep = rows[g[0]]
                print(f"  keep {os.path.relpath(keep[0], root)}")
                for i in g[1:]:
                    print(f"    trash {os.path.relpath(rows[i][0], root)}")
            if args.yes or _confirm(f"Trash {len(to_trash)} duplicate file(s)?"):
                from .cleanup import trash_files

                class _Rec:
                    def __init__(self, p):
                        self.path = p

                trashed, failed = trash_files([_Rec(p) for p in to_trash])
                pruned = len(trashed)
                trashed_paths = {r.path for r in trashed}
                for f in failed:
                    print(f"  FAILED {f.path}")
                print(f"trashed {pruned} duplicate file(s)")
        else:
            print("\nno exact duplicates found")

    # --media-only: fix extensions first, then consolidate, then prune non-media.
    # Fixing must come first: an extensionless JPEG is classified "other" until
    # its extension is added, so pruning first would delete it.
    renamed = 0
    if args.media_only:
        from .bootstrap import canonical_extension

        def _needs_fix(r):
            ext = os.path.splitext(r[0])[1]
            return not ext or canonical_extension(ext) is None

        fixes = []
        if args.fix_extensions:
            fixes = [(r[0], fixed_extension_path(r[0])) for r in rows
                     if _needs_fix(r) and fixed_extension_path(r[0])
                     and r[0] not in trashed_paths]
        consolidations = [(r[0], consolidated_path(r[0])) for r in rows
                          if r[4] != "other" and consolidated_path(r[0])
                          and r[0] not in trashed_paths]

        # after fixing, recompute what is non-media (fixes may promote files)
        fixed_paths = {old: new for old, new in fixes}
        non_media = [r[0] for r in rows
                     if r[4] == "other" and r[0] not in fixed_paths]

        if fixes:
            print(f"\n{len(fixes)} file(s) to fix missing extensions:")
            for old, new in fixes[:20]:
                print(f"  {os.path.relpath(old, root)} -> {os.path.basename(new)}")
            if len(fixes) > 20:
                print(f"  ... and {len(fixes) - 20} more")
        if consolidations:
            print(f"\n{len(consolidations)} file(s) to consolidate extensions:")
            for old, new in consolidations[:20]:
                print(f"  {os.path.relpath(old, root)} -> {os.path.basename(new)}")
            if len(consolidations) > 20:
                print(f"  ... and {len(consolidations) - 20} more")
        if non_media:
            verb = f"move to {move_to}" if move_to else "prune"
            print(f"\n{len(non_media)} non-media file(s) to {verb}:")
            for p in non_media[:20]:
                print(f"  {os.path.relpath(p, root)}")
            if len(non_media) > 20:
                print(f"  ... and {len(non_media) - 20} more")

        total_ops = len(non_media) + len(consolidations) + len(fixes)
        if total_ops and (args.yes or _confirm(
                f"Apply {total_ops} file operation(s) "
                f"({len(fixes)} fix, {len(consolidations)} consolidate, "
                f"{len(non_media)} {('move to ' + move_to) if move_to else 'prune'})?")):
            from .cleanup import move_files, trash_files

            class _Rec:
                def __init__(self, p):
                    self.path = p

            # 1. fix missing extensions first (may promote files out of "other")
            for old, new in fixes:
                try:
                    os.rename(old, new)
                    renamed += 1
                except OSError as exc:
                    print(f"  FAILED {old}: {exc}")
            # 2. consolidate extensions to canonical form
            for old, new in consolidations:
                try:
                    os.rename(old, new)
                    renamed += 1
                except OSError as exc:
                    print(f"  FAILED {old}: {exc}")
            # 3. clear out non-media files (after fixes, so promoted files survive)
            if non_media and move_to:
                moved, _failed = move_files(non_media, root, move_to)
                print(f"moved {len(moved)} non-media file(s) to {move_to}")
            elif non_media:
                trashed, failed = trash_files([_Rec(p) for p in non_media])
                for f in failed:
                    print(f"  FAILED {f.path}")
                print(f"pruned {len(trashed)} non-media file(s)")
            print(f"renamed {renamed} file(s)")
            # refresh the index so paths match the new filenames
            rows = walk_tree(root, include_video=args.video, ignore=ignore)
            db.replace_bootstrap(rows)
            print("index refreshed")

    db.close()
    return 0


def _confirm(prompt: str, *, yes: bool = False) -> bool:
    """Ask for confirmation; --yes or non-interactive means no."""
    if yes:
        return True
    if not sys.stdin.isatty():
        print(prompt, "(not a terminal — skipping)")
        return False
    answer = input(prompt + " [y/N] ")
    return answer.strip().lower() in ("y", "yes")


def cmd_check(args: argparse.Namespace) -> int:
    """Check which media files are loadable, and offer to repair them.

    Every file that is not clean is named with the reason it was flagged and
    what can be done about it:

      * truncated -> re-encoded, the original kept as ``<name>.truncated``
      * corrupt   -> moved to a quarantine folder (never deleted outright)
      * notable   -> loads fine (malformed MPO, very large image); reported
                     for information, no action taken

    The two repairs ask first in a terminal and can be skipped with --yes.
    """
    from .config import load_ignore

    root = os.path.abspath(args.root)
    ignore = load_ignore(root)
    include_video = args.video

    print(f"checking media under {root} ...")
    result = diagnose_tree(root, include_video=include_video, ignore=ignore)
    ok = result[OK]
    truncated = result[TRUNCATED]
    corrupt = result[CORRUPT]
    notable = result[NOTABLE]

    print(f"{len(ok)} ok, {len(truncated)} truncated, {len(corrupt)} corrupt"
          + (f", {len(notable)} notable" if notable else ""))

    # Name the file and say what is wrong with it, rather than letting a
    # Pillow warning print its own site-packages path.
    for label, rows in (("truncated", truncated), ("corrupt", corrupt),
                        ("notable", notable)):
        for path, why in rows[:_REPORT_LIMIT]:
            print(f"  {label}: {os.path.relpath(path, root)}")
            if why:
                print(f"      {why}")
        if len(rows) > _REPORT_LIMIT:
            print(f"  ... and {len(rows) - _REPORT_LIMIT} more {label} file(s)")

    if not truncated and not corrupt:
        if notable:
            print("\nnothing to repair — the notable files above all load fine.")
        else:
            print("\neverything is loadable; nothing to repair.")
        return 0

    repaired = failed = 0
    if truncated:
        print(f"\n{len(truncated)} file(s) can be repaired by re-encoding them "
              f"(the readable part is kept, the original becomes <name>.truncated).")
        if _confirm(f"Repair {len(truncated)} truncated image(s)?",
                    yes=args.yes):
            for path, _why in truncated:
                rel = os.path.relpath(path, root)
                try:
                    print(f"  {rel}: {repair_truncated(path)}")
                    repaired += 1
                except Exception as exc:  # noqa: BLE001 - keep repairing the rest
                    failed += 1
                    print(f"  {rel}: FAILED: {exc}")

    quarantined = 0
    if corrupt:
        qdir = os.path.join(root, args.quarantine)
        print(f"\n{len(corrupt)} file(s) are unreadable and can be moved to "
              f"{qdir} (never deleted outright).")
        if _confirm(f"Quarantine {len(corrupt)} corrupt file(s)?", yes=args.yes):
            for path, _why in corrupt:
                rel = os.path.relpath(path, root)
                try:
                    print(f"  {rel} -> {quarantine(path, qdir)}")
                    quarantined += 1
                except Exception as exc:  # noqa: BLE001 - keep going
                    failed += 1
                    print(f"  {rel}: FAILED: {exc}")

    print(f"\ndone: {len(ok)} ok, {repaired} repaired, "
          f"{len(truncated) - repaired} still truncated, "
          f"{quarantined}/{len(corrupt)} corrupt quarantined"
          + (f", {len(notable)} notable (no action needed)" if notable else "")
          + (f", {failed} failed" if failed else ""))
    if repaired or quarantined:
        print("run 'media-match scan --prune' to sync the index")
    return 1 if failed else 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """One integrated pass: scan, describe, refine and classify everything.

    Images and videos share every stage. The scan pipeline overlaps CPU
    decoding with GPU CLIP batches; the describe stage overlaps CPU frame
    extraction with GPU VLM captions, so the GPU is never waiting on the CPU.
    """
    _register_heif()

    from .config import load_config, load_ignore

    cfg = load_config()
    root = os.path.abspath(args.root)
    ignore = load_ignore(root)

    # An explicit flag wins, then the config file, then the built-in default.
    # Resolved once here and passed down — the raw flag can be None and must
    # never reach the model loaders directly.
    clip_model = args.clip_model or cfg.get("clip_model") or DEFAULT_CLIP_MODEL
    model = args.model or cfg.get("model") or DEFAULT_MODEL
    workers = args.workers if args.workers is not None else (cfg.get("workers") or 4)
    batch = args.batch if args.batch is not None else (cfg.get("batch") or 16)
    interval = args.interval if args.interval is not None else (cfg.get("interval") or 1.0)
    max_frames = (args.max_frames if args.max_frames is not None
                  else (cfg.get("max_frames") or 120))
    describe_workers = (args.describe_workers if args.describe_workers is not None
                        else max(2, min(workers, 4)))

    db = Database(db_path_for(root, args.db))
    db.set_meta("root", root)
    db.set_meta("clip_model", clip_model)
    if args.describe:
        db.set_meta("vlm_model", model)

    if args.video and not _require_ffmpeg():
        db.close()
        return 2

    if args.prune:
        pruned = prune_missing(root, db, include_video=args.video, ignore=ignore)
        if pruned:
            print(f"removed {pruned} stale index row(s) "
                  f"(moved or deleted since last scan)")
        else:
            print("no stale index rows to remove")

    # -- pass 1: scan (pHash + EXIF + CLIP) ----------------------------------
    jobs, total, up_to_date = build_jobs(
        root, db, include_video=args.video, reanalyze=args.reanalyze,
        limit=args.limit, ignore=ignore,
    )
    print(f"{total} media found, {up_to_date} up to date, {len(jobs)} to analyze")

    from .clip import ClipEncoder

    encoder = ClipEncoder(model_id=clip_model, batch_size=batch)
    errors = 0

    def on_error(path: str, exc: Exception) -> None:
        nonlocal errors
        errors += 1
        # Explain it in the file's terms where we can: an unreadable image is
        # "truncated, check can repair it", not "UnidentifiedImageError".
        _report_bad_file(path, _diagnose_failure(path, exc))

    def on_result(result) -> None:
        nonlocal errors
        try:
            store_result(db, result)
        except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
            errors += 1
            print(f"  ERROR storing {result.job.path}: {exc}", file=sys.stderr)

    scan_stats = ScanStats()
    analyzed = _run_scan_with_progress(
        jobs, encoder, args, root,
        interval=interval, max_frames=max_frames, batch_size=batch,
        workers=workers, on_result=on_result, on_error=on_error,
        stats=scan_stats, label="analyzing",
    )
    _summarize_bad_files(root)
    db.record_scan(
        items=scan_stats.items, frames=scan_stats.frames,
        bytes_read=scan_stats.bytes_read, wall=scan_stats.wall,
        cpu=scan_stats.cpu, gpu=scan_stats.gpu,
    )
    if scan_stats.items:
        print(f"  measured: {_fmt_seconds(scan_stats.wall)} wall, "
              f"{_fmt_seconds(scan_stats.gpu)} gpu, "
              f"{_fmt_seconds(scan_stats.cpu)} cpu, "
              f"{_fmt_bytes(scan_stats.bytes_read)} read")

    # -- pass 2: VLM captions (GPU), with CPU frame prefetch -----------------
    # the CLIP encoder stays loaded when --classify needs it afterwards;
    # otherwise it is unloaded first so the VLM has the memory to itself
    if not args.classify:
        encoder.unload()

    described = 0
    if args.describe:
        described = _describe_all(db, root, model, args, on_error)
        if described:
            print(f"described {described} item(s)")

    # -- pass 3: refine keywords into canonical tags -------------------------
    refined = 0
    if args.refine or args.classify:
        refined = _refine_all(db, quiet=True)
        if refined:
            print(f"refined {refined} item(s)")

    # -- pass 4: type each image with the already-loaded CLIP encoder --------
    classified = 0
    if args.classify:
        classified = _classify_with(db, encoder, recompute=False)
        if classified:
            print(f"classified {classified} image(s)")
    encoder.unload()

    described_count = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE description != ''"
    ).fetchone()["n"]
    vdescribed = db.conn.execute(
        "SELECT COUNT(*) AS n FROM videos WHERE description != ''"
    ).fetchone()["n"]
    print(f"Done: analyzed {analyzed}, described {described}, "
          f"refined {refined}, classified {classified}, errors {errors}. "
          f"Index: {db.count_files()} images ({described_count} described), "
          f"{db.count_videos()} videos ({vdescribed} described) ({db.path})")
    db.close()
    return 0


def cmd_refine(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    total = _refine_all(db)
    if not total:
        print("Nothing to refine. Run 'scan' (and optionally '--describe') first.")

    if args.classify:
        _classify_images(db, args)

    facets = db.facet_counts("image")
    if facets:
        print("facets: " + ", ".join(f"{name} {n}" for name, n in facets))
    db.close()
    return 0


def _refine_all(db: Database, *, quiet: bool = False) -> int:
    """Rewrite keywords into canonical tags + facets for images and videos."""
    total = 0
    for kind, label in (("image", "image"), ("video", "video")):
        rows = db.rows_for_refinement(kind)
        items = []
        for r in rows:
            raw = (r["raw_keywords"] or r["keywords"] or "").strip()
            if not raw:
                continue
            keywords = [k for k in raw.split(",") if k.strip()]
            tags, _facets = refine(keywords, r["description"] or "")
            pairs = [(t, facet_of(t)) for t in tags]
            items.append((r["id"], ", ".join(tags), raw, r["description"] or "", pairs))
        if rows:
            db.apply_refinement(kind, items)
        total += len(items)
        if not quiet:
            print(f"  refined {len(items)} {label}(s)")
    return total


def _classify_with(db: Database, encoder, *, recompute: bool = False) -> int:
    """Type every untyped image (or all images) using an already-loaded encoder."""
    import numpy as np

    if recompute:
        rows = db.conn.execute(
            "SELECT id, path FROM files WHERE embedding IS NOT NULL ORDER BY path"
        ).fetchall()
    else:
        rows = db.files_needing_type()
    if not rows:
        return 0

    ids, blobs = db.all_embeddings()
    vec_by_id = {i: np.frombuffer(b, dtype=np.float32) for i, b in zip(ids, blobs)}
    targets = [(r["id"], vec_by_id[r["id"]]) for r in rows if r["id"] in vec_by_id]
    if not targets:
        return 0

    from .imagetype import TypeClassifier

    classifier = TypeClassifier(encoder)
    matrix = np.stack([v for _i, v in targets])
    results = classifier.classify(matrix)
    db.set_types([
        (targets[k][0], name, score) for k, (name, score) in enumerate(results)
    ])
    return len(targets)


def _classify_images(db: Database, args) -> None:
    from .clip import ClipEncoder

    encoder = ClipEncoder(model_id=args.clip_model)
    try:
        classified = _classify_with(db, encoder, recompute=args.recompute)
    finally:
        encoder.unload()
    if classified:
        print(f"classified {classified} image(s)")
    else:
        print("No images need a type (run 'scan'/'embed' first, "
              "or they are all classified).")


def cmd_people(args: argparse.Namespace) -> int:
    import numpy as np

    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1
    kind = "image"

    if args.label:
        return _label_cluster(db, args)

    person_ids = set(db.media_ids_with_facet("person", kind))
    if not person_ids:
        print("No person tags yet. Run 'scan --describe' then 'refine' first.")
        db.close()
        return 1

    ids, blobs = db.all_embeddings()
    vec_by_id = {i: np.frombuffer(b, dtype=np.float32) for i, b in zip(ids, blobs)}
    selected = [i for i in person_ids if i in vec_by_id]
    if len(selected) < 2:
        print(f"{len(selected)} person photo(s) with embeddings — nothing to cluster.")
        db.close()
        return 0

    from .groups import cluster_by_similarity

    matrix = np.stack([vec_by_id[i] for i in selected])
    labels = cluster_by_similarity(matrix, threshold=args.threshold)

    components: dict[int, list[int]] = {}
    for fid, comp in zip(selected, labels):
        components.setdefault(comp, []).append(fid)
    ordered = sorted(components.values(), key=lambda members: (-len(members), min(members)))
    members = {n: ids_ for n, ids_ in enumerate(ordered, 1)}

    # carry a name over when a cluster's membership is unchanged by re-running
    previous: dict[frozenset[int], str] = {}
    for c in db.all_clusters(kind):
        if c["label"]:
            member_set = frozenset(r["id"] for r in db.files_in_cluster(c["id"]))
            previous[member_set] = c["label"]
    carried = {n: previous.get(frozenset(ids_), "") for n, ids_ in members.items()}
    db.set_clusters(kind, members, carried)

    records = {r.id: r for r in db.all_files()}
    if args.json:
        print(json.dumps([
            {"cluster": n, "label": carried.get(n, ""),
             "files": [records[i].path for i in ids_]}
            for n, ids_ in members.items()
        ], indent=2))
        db.close()
        return 0

    print(f"{len(members)} person cluster(s) from {len(selected)} photo(s) "
          f"(cosine >= {args.threshold})")
    shown = 0
    for n, ids_ in members.items():
        if len(ids_) < args.min_size:
            continue
        label = carried.get(n, "")
        tag = f"  [{label}]" if label else ""
        print(f"\nPerson {n}{tag}  ({len(ids_)} photos)")
        for i in ids_[: args.limit]:
            print(f"  {os.path.relpath(records[i].path, root)}")
        shown += 1
    if not shown:
        print(f"No clusters with at least {args.min_size} photo(s).")
    print("\nName a cluster with: media-match people --label N=Name")
    db.close()
    return 0


def _label_cluster(db: Database, args) -> int:
    num, _, name = args.label.partition("=")
    name = name.strip()
    if not num.isdigit() or not name:
        print("--label expects N=Name, e.g. --label 3=Alice")
        db.close()
        return 2
    cid = int(num)
    if cid not in {r["id"] for r in db.all_clusters("image")}:
        print(f"No cluster {cid} yet. Run 'people' first to build clusters.")
        db.close()
        return 1
    db.set_cluster_label(cid, name)
    print(f"cluster {cid} labeled '{name}' "
          f"({len(db.files_in_cluster(cid))} photo(s))")
    db.close()
    return 0


def cmd_tags(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1

    kind = "video" if args.video else "image"
    payload: list | None = None

    if args.types:
        counts = db.type_counts()
        if not counts:
            print("No image types yet. Run 'refine --classify' first.")
            db.close()
            return 1
        payload = [{"type": n, "count": c} for n, c in counts]
        if not args.json:
            width = max(len(name) for name, _ in counts)
            for name, count in counts:
                print(f"{name:<{width}}  {count}")
    elif args.facets:
        grouped = db.facet_tag_counts(kind)
        if not grouped:
            print("No refined tags yet. Run 'refine' first.")
            db.close()
            return 1
        total_by_facet = dict(db.facet_counts(kind))
        payload = [
            {"facet": facet, "total": total_by_facet.get(facet, 0),
             "tags": [{"tag": t, "count": c} for t, c in tags]}
            for facet, tags in grouped.items()
        ]
        if not args.json:
            for facet, tags in grouped.items():
                print(f"{facet} ({total_by_facet.get(facet, 0)} media)")
                width = max(len(t) for t, _ in tags)
                for tag, count in tags[: args.limit]:
                    print(f"  {tag:<{width}}  {count}")
                print()
    else:
        if args.raw:
            counts = db.keyword_counts(limit=args.limit, include_videos=args.video)
            heading = "raw keywords"
        else:
            counts = db.tag_counts(limit=args.limit, kind=kind)
            heading = "refined tags"
        if not counts:
            print(f"No {heading} yet. Run 'scan' then 'refine'.")
            db.close()
            return 1
        payload = [{"tag": k, "count": c} for k, c in counts]
        if not args.json:
            print(f"top {heading}:")
            width = max(len(k) for k, _ in counts)
            for kw, count in counts:
                print(f"{kw:<{width}}  {count}")

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    db.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = open_db(root, args.db, create=False)
    if db is None:
        print(f"No index at {db_path_for(root, args.db)} — "
              f"run 'media-match scan {root}' first.")
        return 1
    total = db.count_files()
    described = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE description != ''"
    ).fetchone()["n"]
    embedded = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE embedding IS NOT NULL"
    ).fetchone()["n"]
    videos = db.count_videos()
    frames = db.conn.execute("SELECT COUNT(*) AS n FROM video_frames").fetchone()["n"]
    vembedded = db.conn.execute(
        "SELECT COUNT(DISTINCT video_id) AS n FROM video_frames "
        "WHERE embedding IS NOT NULL"
    ).fetchone()["n"]
    vdescribed = db.conn.execute(
        "SELECT COUNT(*) AS n FROM videos WHERE description != ''"
    ).fetchone()["n"]
    tagged = db.has_tags("image")
    typed = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE type != ''"
    ).fetchone()["n"]
    people = len(db.all_clusters("image"))
    dim = db.embedding_dim()
    lo, hi = db.date_range()
    summary = db.scan_summary()
    img_counts, img_sizes = db.image_extensions()
    vid_counts, vid_sizes = db.video_extensions()
    img_groups = find_duplicate_groups(db.all_files())
    vid_groups = find_video_duplicate_groups(db.all_videos_with_frames())
    img_extra, img_reclaim = _reclaimable(img_groups)
    vid_extra, vid_reclaim = _reclaimable(vid_groups)

    payload = {
        "index": db.path,
        "root": db.get_meta("root", root),
        "clip_model": db.get_meta("clip_model", ""),
        "vlm_model": db.get_meta("vlm_model", ""),
        "files": {"total": total, "described": described, "embedded": embedded},
        "videos": {"total": videos, "frames": frames, "embedded": vembedded,
                   "described": vdescribed},
        "tags": {"refined": tagged, "typed": typed, "person_clusters": people},
        "embeddings": {"image_vectors": embedded, "dimension": dim,
                       "video_frame_vectors": vembedded},
        "dates": {"earliest": lo, "latest": hi} if lo else None,
        "compute": {
            "scans": summary["scans"], "items": summary["items"],
            "frames": summary["frames"], "bytes": summary["bytes"],
            "wall": summary["wall"], "cpu": summary["cpu"], "gpu": summary["gpu"],
            "last_scan_at": summary["last_scan_at"],
        },
        "formats": {
            "image": [{"extension": e, "count": c, "bytes": img_sizes[e]}
                      for e, c in img_counts],
            "video": [{"extension": e, "count": c, "bytes": vid_sizes[e]}
                      for e, c in vid_counts],
        },
        "duplicates": {
            "image": {"groups": len(img_groups), "extra_files": img_extra,
                      "reclaimable_bytes": img_reclaim},
            "video": {"groups": len(vid_groups), "extra_files": vid_extra,
                      "reclaimable_bytes": vid_reclaim},
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        db.close()
        return 0

    print(f"index:  {db.path}")
    print(f"root:   {db.get_meta('root', root)}")
    print(f"clip:   {db.get_meta('clip_model', '(not set)')}")
    print(f"vlm:    {db.get_meta('vlm_model', '(not used)')}")
    print(f"files:  {total} ({described} described, {embedded} embedded)")
    print(f"videos: {videos} ({vframes_note(frames, vembedded, vdescribed)})")
    if lo:
        print(f"dates:  {lo[:10]} .. {hi[:10]} (captured, else mtime)")
    print(f"tags:   {tagged} refined tags, {typed} typed, {people} person clusters")

    print(f"embeddings: {embedded} image vector(s)"
          + (f", dimension {dim}" if dim else "")
          + f", {vembedded} video frame vector(s)")

    print()
    if summary["scans"]:
        items = summary["items"]
        wall = summary["wall"]
        rate = f"{items / wall:.1f} items/s" if wall > 0 else "n/a"
        io_rate = f"{_fmt_bytes(summary['bytes'] / wall)}/s" if wall > 0 else "n/a"
        print(f"compute: {summary['scans']} scan run(s), {items} items, "
              f"{summary['frames']} frames")
        print(f"  cpu {_fmt_seconds(summary['cpu'])}, "
              f"gpu {_fmt_seconds(summary['gpu'])}, "
              f"wall {_fmt_seconds(wall)}")
        print(f"  I/O: {_fmt_bytes(summary['bytes'])} read ({io_rate}), {rate}")
        if summary["last_scan_at"]:
            print(f"  last scan: {summary['last_scan_at']}")
    else:
        print("compute: no scan runs recorded yet (run 'media-match scan')")

    print()
    print("formats:")
    for ext, count in img_counts:
        print(f"  image {ext:<8} {count:>6}  {_fmt_bytes(img_sizes[ext])}")
    for ext, count in vid_counts:
        print(f"  video {ext:<8} {count:>6}  {_fmt_bytes(vid_sizes[ext])}")
    if not img_counts and not vid_counts:
        print("  (no media indexed yet)")

    print()
    print("duplicates:")
    for label, groups, extra, reclaim in (
        ("image", img_groups, img_extra, img_reclaim),
        ("video", vid_groups, vid_extra, vid_reclaim),
    ):
        print(f"  {label:<5} {len(groups):>5} group(s), {extra:>5} extra file(s), "
              f"{_fmt_bytes(reclaim)} reclaimable")
    if not img_groups and not vid_groups:
        print("  no duplicates found")

    db.close()
    return 0


def vframes_note(frames: int, embedded: int, described: int) -> str:
    return (f"{frames} keyframes, {embedded} embedded, {described} described")


# -- help --------------------------------------------------------------------

# Filled in main(); maps a command name to its argparse parser.
_SUBPARSERS: dict[str, argparse.ArgumentParser] = {}

# One-line summary for the top-level command listing and the `help` command.
_COMMAND_SUMMARIES = {
    "analyze": "one integrated pass: scan, describe, refine and classify",
    "scan": "index new/changed media (hash + pHash + CLIP)",
    "describe": "caption indexed media that has no description yet",
    "search": "find media by person/object/text (CLIP by default)",
    "embed": "compute CLIP embeddings for indexed images",
    "similar": "group the tree by CLIP similarity, or find neighbors of one file",
    "duplicates": "group duplicates; best resolution wins",
    "clean": "review duplicate groups and move rejected copies to the Trash",
    "restore": "move files back from the system Trash (undo 'clean')",
    "refine": "rewrite keywords into canonical tags/facets; type photos",
    "people": "cluster person photos by similarity and name the clusters",
    "tags": "list refined tags, facets or image types",
    "export": "dump the index to JSON, CSV or Markdown",
    "export-metadata": "export editable metadata (description/keywords/type) to JSON",
    "import-metadata": "import metadata from a JSON or CSV file",
    "clear-metadata": "clear description, keywords, type and tags",
    "reset": "delete the index database (media files are never touched)",
    "check": "check which media files are loadable, and offer to repair them",
    "bootstrap": "first pass: walk the tree, record filename/type/size in the index",
    "stats": "show index statistics (media, compute, formats)",
    "about": "show what this program does, its license and project links",
    "version": "print version, license and author",
    "info": "print program information (models, last scan, storage)",
    "help": "show this guide; 'help <cmd>' describes one command",
}

# A runnable example per command, shown in `media-match help <cmd>`.
_EXAMPLES = {
    "analyze": "media-match analyze ~/Pictures --video --describe --classify",
    "scan": "media-match scan ~/Pictures --describe --refine --classify",
    "describe": "media-match describe ~/Pictures --video",
    "search": "media-match search ~/Pictures --object bicycle",
    "embed": "media-match embed ~/Pictures",
    "similar": "media-match similar ~/Pictures --threshold 0.9",
    "duplicates": "media-match duplicates ~/Pictures",
    "clean": "media-match clean ~/Pictures --dry-run",
    "restore": "media-match restore ~/Pictures --yes",
    "refine": "media-match refine ~/Pictures --classify",
    "people": "media-match people ~/Pictures --label 1=Alice",
    "tags": "media-match tags ~/Pictures --facets",
    "export": "media-match export ~/Pictures --format json --out index.json",
    "export-metadata": "media-match export-metadata ~/Pictures --out captions.json",
    "import-metadata": "media-match import-metadata ~/Pictures captions.json",
    "clear-metadata": "media-match clear-metadata ~/Pictures --yes",
    "reset": "media-match reset ~/Pictures --yes",
    "check": "media-match check ~/Pictures",
    "bootstrap": "media-match bootstrap ~/Pictures --video",
    "stats": "media-match stats ~/Pictures",
    "about": "media-match about bench",  # 'about' alone prints the blurb
    "version": "media-match version",
    "info": "media-match info ~/Pictures",
    "help": "media-match help bootstrap",
}

# Project metadata shown by 'about', 'version' and 'info'.
_VERSION = "0.2.26"
_LICENSE = "Non-Commercial Licence"
_AUTHOR = "Timh Bergström"
_REPO = "https://github.com/rnts08/mlx-media-matcher"
_LICENSE_URL = f"{_REPO}/blob/main/LICENSE"
_ISSUES_URL = f"{_REPO}/issues"

# Where people can fund development. Kept in one place so 'info' and the
# license stay in step.
_SUPPORT_LINES = (
    "Support development of media-match",
    "  - EVM: 0x6e8e3c2b31424266e7cff59e910df1587c317427 (ERC-20, BEP-20, TRC-20, Base)",
    "  - BTC: bc1qzzvcguvqjc6qhwe2y5vy38w2zke7hksukjhm68",
    "  - LTC: MPfm5QLKH1r9XxgWmH75Gyps4LDfX5c53L",
    "  - SOL: GEaCMpnyM8tB5BU4RMuLm6tgMr3q9FgMHodxDxxAGby",
    "  - https://buymeacoffee.com/timhbergsta",
    "  - https://ko-fi.com/rnts08",
)


def _flag_lines(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    """[(flags, help), ...] for every option, noting defaults when missing."""
    out: list[tuple[str, str]] = []
    for action in parser._actions:
        if not action.option_strings or action.dest == "help":
            continue
        if action.default is argparse.SUPPRESS:
            continue
        flags = ", ".join(action.option_strings)
        if action.nargs != 0:
            if action.metavar:
                metavar = action.metavar
            elif action.choices:
                metavar = "{" + ",".join(str(c) for c in action.choices) + "}"
            else:
                metavar = action.dest.upper()
            flags = f"{flags} {metavar}"
        # Strip first: a help string written as "batch size " would otherwise
        # render as "batch size   (default: 16)" with a ragged gap.
        text = (action.help or "").strip()
        default = action.default
        if default not in (None, False, argparse.SUPPRESS) and "default" not in text.lower():
            if not text:
                # A flag with no help reads as a bare "(default: 16)", which
                # tells you nothing about what the flag does.
                text = "no description given"
            text += f"  (default: {default})"
        out.append((flags, text))
    return out


def _positional_lines(parser: argparse.ArgumentParser) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for action in parser._actions:
        if action.option_strings or action.dest == "help":
            continue
        if isinstance(action, argparse._SubParsersAction):
            continue
        name = action.metavar or action.dest.upper()
        if action.nargs == "*":
            name += "..."
        out.append((name, action.help or ""))
    return out


def _command_section(name: str) -> str:
    parser = _SUBPARSERS[name]
    lines = [f"{name} — {_COMMAND_SUMMARIES.get(name, '')}", ""]
    if parser.description:
        lines.append(textwrap.dedent(parser.description).strip())
        lines.append("")
    lines.append(parser.format_usage().strip())

    positional = _positional_lines(parser)
    if positional:
        lines += ["", "positional arguments:"]
        width = max(len(n) for n, _ in positional)
        for n, t in positional:
            lines.append(f"  {n:<{width}}  {t}")

    flags = _flag_lines(parser)
    if flags:
        lines += ["", "options:"]
        width = max(len(f) for f, _ in flags)
        for f, t in flags:
            lines.append(f"  {f:<{width}}  {t}")

    if name in _EXAMPLES:
        lines += ["", f"example: {_EXAMPLES[name]}"]
    return "\n".join(lines)


# The guide groups commands by what you are trying to do, in the order you
# would do them, rather than alphabetically. An alphabetical list answers
# "what is this called?"; a grouped one answers "what do I want?", which is
# the question someone actually has. Every command appears in exactly one
# group — verified in main().
#
# Each entry is (heading, one-line purpose, commands in reading order).
_COMMAND_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("get oriented",
     "what this is, what is installed, what the index holds",
     ("help", "about", "info", "version", "stats")),
    ("build the index",
     "first pass over a folder: inventory it, index it, confirm it loads",
     ("bootstrap", "scan", "check")),
    ("go deeper",
     "captions, tags, types and people — what makes search work well",
     ("analyze", "embed", "describe", "refine", "people")),
    ("find things",
     "search and browse what is indexed",
     ("search", "tags", "similar")),
    ("quality and file management",
     "duplicates, cleaning them up, and undoing that",
     ("duplicates", "clean", "restore")),
    ("metadata",
     "captions, keywords and types as data you can edit and move around",
     ("export-metadata", "import-metadata", "clear-metadata")),
    ("index maintenance",
     "the index database itself, not your media files",
     ("export", "reset")),
)


def _print_full_help() -> None:
    lines = [
        "usage: media-match [--help] <command> [command options] [args...]",
        "",
        "commands, grouped by what they are for:",
        "",
    ]
    width = max(len(c) for c in _COMMAND_SUMMARIES)
    for heading, purpose, names in _COMMAND_GROUPS:
        lines.append(f"{heading}: {purpose}")
        for name in names:
            lines.append(f"  {name:<{width}}  {_COMMAND_SUMMARIES[name]}")
        lines.append("")
    lines += [
        "Every command accepts --db to override the index location and takes a",
        "root argument (default: the current directory).",
        "",
        "Run 'media-match <command> --help' or 'media-match help <command>' for",
        "the full flag list of one command, with defaults and an example.",
        "Run 'media-match help' again to re-read this guide.",
    ]
    print("\n".join(lines))


def _cmd_help(args: argparse.Namespace) -> int:
    if args.command:
        if args.command not in _SUBPARSERS:
            print(f"Unknown command: {args.command}")
            print("Commands: " + ", ".join(sorted(_COMMAND_SUMMARIES)))
            return 1
        print(_command_section(args.command))
        return 0
    _print_full_help()
    return 0


# What the program does, shown at the top of 'about' (and nowhere else now
# that 'help' is just the command reference).
_ABOUT_BLURB = (
    "Local photo-library analyzer built on Apple's MLX. Point it at a folder\n"
    "and it builds an index that lets you:\n"
    "\n"
    "  - find photos by person, object or free text (CLIP vector search)\n"
    "  - group duplicates across resolutions, and mark the best version\n"
    "  - group semantic near-duplicates\n"
    "  - refine tags into facets (person/scene/object/...) and type each photo\n"
    "  - cluster photos of the same person and name the clusters\n"
    "  - index video files (keyframes at ~1 fps) with the same machinery"
)


def _pager() -> list[str] | None:
    """The pager to pipe long output through, honouring $PAGER."""
    configured = os.environ.get("PAGER")
    if configured:
        return configured.split()
    for candidate in ("less", "more"):
        if shutil.which(candidate):
            return [candidate, "-R"] if candidate == "less" else [candidate]
    return None


def _page(text: str) -> None:
    """Print text, paging through $PAGER/less/more when stdout is a TTY.

    Outside a terminal (pipes, tests) the text is printed straight through so
    command output stays scriptable; if the pager is missing we fall back too.
    """
    pager = _pager() if sys.stdout.isatty() else None
    if pager:
        try:
            subprocess.run(pager, input=text + "\n", text=True, check=False)
            return
        except OSError:
            pass
    print(text)


def _license_text() -> str:
    """The full license text from the LICENSE file, or a short pointer."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return fh.read().rstrip("\n")
            except OSError:
                break
    return f"{_LICENSE}\nFull text: {_LICENSE_URL}"


def cmd_about(args: argparse.Namespace) -> int:
    # 'about bench' measures this machine instead of printing the blurb, so
    # the tuning advice lives next to what the program says about itself.
    if (getattr(args, "topic", None) or "").lower() in ("bench", "benchmark"):
        return cmd_bench(args)
    text = (
        f"{_ABOUT_BLURB}\n"
        "\n"
        f"Version:    {_VERSION}\n"
        f"Author:     {_AUTHOR}\n"
        f"License:    {_LICENSE}\n"
        f"Repository: {_REPO}\n"
        f"Report:     {_ISSUES_URL}\n"
        f"Support:    media-match info\n"
        "\n"
        "License text\n"
        "------------\n"
        f"{_license_text()}"
    )
    _page(text)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Time this machine's pipeline and recommend --batch/--io-workers/--workers."""
    from .bench import TIE_THRESHOLD, run_benchmark

    root = os.path.abspath(args.root) if getattr(args, "root", None) else None
    count = getattr(args, "files", None) or 32
    repeat = getattr(args, "repeat", None) or 1

    where = root or "a synthetic corpus (pass a folder to use your own media)"
    print(f"benchmarking {where} ...\n")
    report = run_benchmark(
        root=root, count=count, repeat=repeat,
        clip_model=args.clip_model or DEFAULT_CLIP_MODEL,
        video=getattr(args, "video", False),
        ignore=load_bench_ignore(root),
    )

    if getattr(report, "note", ""):
        print(report.note)
        return 1
    print(f"corpus: {report.corpus_items} file(s), "
          f"{_fmt_bytes(report.corpus_bytes)} on disk")

    if report.stages:
        print("\nindividual stages")
        for stage in report.stages:
            print(f"  {stage.name:<12} {stage.seconds:7.3f}s  {stage.detail}")

    if report.configs:
        print("\nend-to-end scan configurations")
        print(f"  {'batch':>5} {'io':>3} {'cpu':>4} {'img/s':>8} {'gpu':>7} "
              f"{'cpu time':>9}  note")
        for cfg in report.configs:
            gpu = f"{cfg.gpu_share * 100:.0f}%" if cfg.ok else "-"
            print(f"  {cfg.batch:>5} {cfg.io_workers:>3} {cfg.workers:>4} "
                  f"{cfg.per_second:>8.1f} {gpu:>7} {cfg.cpu:>8.1f}s  {cfg.note}")

    best = report.best
    if best is None:
        print("\nno configuration completed; nothing to recommend.")
        return 1
    pick = report.recommended or best

    print(f"\nfastest measured: --batch {best.batch} --io-workers "
          f"{best.io_workers} --workers {best.workers}   "
          f"({best.per_second:,.0f} img/s, "
          f"{best.gpu_share * 100:.0f}% of wall time in the GPU)")

    ties = [c for c in report.ties if c is not best]
    if ties:
        others = ", ".join(f"batch {c.batch}/io {c.io_workers}/cpu {c.workers}"
                           for c in ties)
        print(f"\n{len(ties) + 1} configurations measured the same within "
              f"{TIE_THRESHOLD:.0%} on {report.corpus_items} files ({others}) "
              "— run-to-run variation at this corpus size is larger than the "
              "gap between them, so this is a default, not a measured "
              "optimum. Raise --files and --repeat to separate them.")
    print(f"\nrecommended: --batch {pick.batch} --io-workers {pick.io_workers} "
          f"--workers {pick.workers}"
          + ("  (cheapest of the tied configurations)" if ties else ""))

    # Project onto the user's real library so the number means something.
    if root and pick.per_second:
        from .bench import TIE_THRESHOLD, human_seconds
        from .pipeline import iter_media

        real = sum(1 for _ in iter_media(root, False))
        if real:
            print(f"\nat {pick.per_second:,.0f} img/s, {real:,} file(s) in this "
                  f"tree takes about {human_seconds(real / pick.per_second)} "
                  "on this machine")

    print("\nNothing was indexed and no file was changed.")
    return 0


def load_bench_ignore(root: str | None) -> list[str]:
    if not root:
        return []
    from .config import load_ignore

    return load_ignore(root)


def cmd_version(args: argparse.Namespace) -> int:
    print(f"media-match {_VERSION}")
    print(f"Licensed under {_LICENSE}")
    print(f"Developed by {_AUTHOR}")
    print(f"Repository: {_REPO}")
    print(f"Found a bug or want to help? {_ISSUES_URL}")
    return 0


def _free_storage(path: str) -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return "n/a"
    return (f"{_fmt_bytes(usage.free)} free of {_fmt_bytes(usage.total)} "
            f"({path})")


def cmd_info(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    print(f"Version:      {_VERSION}")
    print(f"License:      {_LICENSE}")
    print(f"Author:       {_AUTHOR}")
    print(f"Repository:   {_REPO}")
    print(f"Model engine: CLIP on Apple MLX (local, no cloud)")
    print(f"Models:       {DEFAULT_CLIP_MODEL} (CLIP embeddings)")
    print(f"              {DEFAULT_MODEL} (vision captions)")

    db = open_db(root, args.db, create=False)
    if db is not None:
        summary = db.scan_summary()
        if summary["scans"]:
            print(f"Last scan:    {summary['last_scan_at'] or 'unknown'} "
                  f"({summary['scans']} run(s), {summary['items']} items, "
                  f"{_fmt_seconds(summary['wall'])} wall, "
                  f"{_fmt_seconds(summary['gpu'])} gpu)")
        else:
            print("Last scan:    no scan runs recorded yet")
        db.close()
    else:
        print("Last scan:    no index at this folder yet")

    print(f"Free storage: {_free_storage(root)}")
    print()
    for line in _SUPPORT_LINES:
        print(line)
    return 0


# -- parser ------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("root", nargs="?", default=".",
                   help="media tree to work on (default: current directory)")
    p.add_argument("--db", help="index database path (default: <root>/.media-match.db)")


class _HelpAction(argparse.Action):
    """Print the full guide for -h/--help, exactly like the 'help' command."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest,
                         default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        _print_full_help()
        parser.exit()


class _CommandHelpAction(argparse.Action):
    """-h/--help on a command: print the same page as 'help <cmd>'."""

    def __init__(self, option_strings, command: str, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest,
                         default=default, nargs=0, help=help)
        self.command = command

    def __call__(self, parser, namespace, values, option_string=None):
        print(_command_section(self.command))
        parser.exit()


class _CommandParser(argparse.ArgumentParser):
    """Subparser that routes -h/--help through the shared help page.

    argparse's own --help is a bare option dump with no description and no
    example. Sending it through _command_section keeps 'cmd --help',
    'cmd help' and 'help cmd' byte-identical.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("add_help", False)
        super().__init__(*args, **kwargs)
        name = (self.prog or "").rsplit(" ", 1)[-1]
        self.add_argument("-h", "--help", action=_CommandHelpAction,
                          command=name,
                          help="show this command's help and exit")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Version/info/about are global: accept them anywhere on the line, with or
    # without a command, and with no other positional arguments.
    for i, token in enumerate(argv):
        if token in ("-v", "--version"):
            cmd_version(argparse.Namespace())
            return 0
        if token in ("--info", "--about"):
            root = None
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                root = argv[i + 1]
            if token == "--about":
                cmd_about(argparse.Namespace())
            else:
                cmd_info(argparse.Namespace(root=root or ".", db=None))
            return 0

    parser = argparse.ArgumentParser(
        prog="media-match",
        add_help=False,
        epilog=(
            "Run 'media-match help' (or -h/--help) for this guide, "
            "'media-match help <command>' for one command, "
            "'media-match version' for version/license, and "
            "'media-match info [folder]' for program information."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-h", "--help", action=_HelpAction,
        help="show this guide and exit (same as the 'help' command)",
    )
    parser.add_argument(
        "-v", "--version", action="store_true",
        help="print version, license and author",
    )
    parser.add_argument(
        "--info", action="store_true",
        help="print program information (models, last scan, free storage)",
    )
    parser.add_argument(
        "--about", action="store_true",
        help="show what the program does, its license and project links",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>",
                            parser_class=_CommandParser)

    p_scan = sub.add_parser(
        "scan",
        help="analyze new/changed media in a folder",
        description=(
            "Walk the tree and index every new or changed media file: size, "
            "dimensions, capture date, pHash (for exact duplicates) and a CLIP "
            "embedding. Files already in the index are skipped, so re-running is "
            "cheap. Add --describe to caption them, --refine/--classify to tidy "
            "the keywords, --video to include videos, and --prune to drop index "
            "rows for files that have since been moved or deleted."
        ),
    )
    _add_common(p_scan)
    p_scan.add_argument("--describe", action="store_true",
                        help="also caption media with the vision model "
                             "(best version of each duplicate group only)")
    p_scan.add_argument("--refine", action="store_true",
                        help="after scanning, rewrite keywords into canonical "
                             "tags + facets (see 'refine')")
    p_scan.add_argument("--classify", action="store_true",
                        help="after scanning, also type each image with CLIP "
                             "(selfie/scene/object/...; implies --refine)")
    p_scan.add_argument("--video", action="store_true",
                        help="also index video files (keyframes each)")
    p_scan.add_argument("--model", default=None,
                        help="MLX vision model for --describe "
                             f"(default: {DEFAULT_MODEL}, or config 'model')")
    p_scan.add_argument("--clip-model", default=None,
                        help="CLIP model for embeddings "
                             f"(default: {DEFAULT_CLIP_MODEL}, or config 'clip_model')")
    p_scan.add_argument("--batch", type=int, default=None,
                        help="CLIP batch size (default: 16, or config 'batch')")
    p_scan.add_argument("--workers", type=int, default=None,
                        help="parallel CPU prep threads feeding the GPU "
                             "(default: 4, or config 'workers')")
    p_scan.add_argument("--io-workers", type=int, default=None,
                        help="reader threads feeding the ring buffer from disk "
                             "(default: 2)")
    p_scan.add_argument("--describe-batch", type=int, default=None,
                        help="VLM batch size: caption N frames per forward pass "
                             "(default: 1, higher is faster but uses more memory)")
    p_scan.add_argument("--interval", type=float, default=None,
                        help="target seconds between video keyframes "
                             "(default: 1.0, or config 'interval')")
    p_scan.add_argument("--max-frames", type=int, default=None,
                        help="cap on keyframes per video (default: 120, or config 'max_frames')")
    p_scan.add_argument("--limit", type=int, help="only scan first N media items")
    p_scan.add_argument("--reanalyze", action="store_true",
                        help="re-analyze files even if unchanged")
    p_scan.add_argument("--prune", action="store_true",
                        help="also drop index rows for files moved or deleted "
                             "since the last scan (keeps the index in sync)")
    _add_progress_flag(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_analyze = sub.add_parser(
        "analyze",
        help="one integrated pass: scan, describe, refine and classify",
        description=(
            "Run every analysis pass over a media tree in one go — indexing "
            "(hash + pHash + CLIP), VLM captions, tag refinement and image "
            "typing — for images and videos alike. The scan pipeline keeps the "
            "GPU busy by overlapping CPU decoding with GPU embedding batches, "
            "and the describe stage overlaps CPU frame extraction with GPU "
            "caption generation. Flags map 1:1 onto 'scan'; add --describe, "
            "--refine and/or --classify for the later passes."
        ),
    )
    _add_common(p_analyze)
    p_analyze.add_argument("--describe", action="store_true",
                           help="caption media with the vision model "
                                "(best version of each duplicate group only)")
    p_analyze.add_argument("--refine", action="store_true",
                           help="after scanning, rewrite keywords into canonical "
                                "tags + facets (see 'refine')")
    p_analyze.add_argument("--classify", action="store_true",
                           help="after scanning, also type each image with CLIP "
                                "(selfie/scene/object/...; implies --refine)")
    p_analyze.add_argument("--video", action="store_true",
                           help="also analyze video files (keyframes each)")
    p_analyze.add_argument("--model", default=None,
                           help="MLX vision model for --describe "
                                f"(default: {DEFAULT_MODEL}, or config 'model')")
    p_analyze.add_argument("--clip-model", default=None,
                           help="CLIP model for embeddings "
                                f"(default: {DEFAULT_CLIP_MODEL}, or config 'clip_model')")
    p_analyze.add_argument("--batch", type=int, default=None,
                           help="CLIP batch size (default: 16, or config 'batch')")
    p_analyze.add_argument("--workers", type=int, default=None,
                           help="parallel CPU prep threads feeding the GPU "
                                "(default: 4, or config 'workers')")
    p_analyze.add_argument("--describe-workers", type=int, default=None,
                           metavar="N",
                           help="CPU threads prefetching frames for the VLM "
                                "(default: 2-4, bounded by --workers)")
    p_analyze.add_argument("--describe-batch", type=int, default=None,
                           metavar="N",
                           help="VLM batch size: caption N frames per forward pass "
                                "(default: 1, higher is faster but uses more memory)")
    p_analyze.add_argument("--io-workers", type=int, default=None,
                           metavar="N",
                           help="reader threads feeding the ring buffer from disk "
                                "(default: 2)")
    p_analyze.add_argument("--interval", type=float, default=None,
                           help="target seconds between video keyframes "
                                "(default: 1.0, or config 'interval')")
    p_analyze.add_argument("--max-frames", type=int, default=None,
                           help="cap on keyframes per video (default: 120, or config 'max_frames')")
    p_analyze.add_argument("--limit", type=int, help="only analyze first N media items")
    p_analyze.add_argument("--reanalyze", action="store_true",
                           help="re-analyze files even if unchanged")
    p_analyze.add_argument("--prune", action="store_true",
                           help="also drop index rows for files moved or deleted "
                                "since the last scan (keeps the index in sync)")
    _add_progress_flag(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    p_describe = sub.add_parser(
        "describe",
        help="caption indexed media that has no caption yet",
        description=(
            "Caption indexed images and videos that still lack a description, "
            "using the vision model. Videos are reduced to a mid-clip still "
            "before captioning; the best version of each duplicate group is "
            "described, its copies inherit the match. Frames are prefetched on "
            "CPU workers so VLM generation (GPU) never waits on a decode. "
            "Use after 'scan' (or again later; it only touches items without "
            "a description)."
        ),
    )
    _add_common(p_describe)
    p_describe.add_argument("--model", default=None,
                            help="MLX vision model "
                                 f"(default: {DEFAULT_MODEL}, or config 'model')")
    p_describe.add_argument("--video", action="store_true",
                            help="include videos (needs ffmpeg)")
    p_describe.add_argument("--describe-workers", type=int, default=None,
                            metavar="N",
                            help="CPU threads prefetching frames "
                                 "(default: 3)")
    _add_progress_flag(p_describe)
    p_describe.set_defaults(func=cmd_describe)

    p_search = sub.add_parser(
        "search",
        help="find media by person/object/text",
        description=(
            "Search the index. With words, results come from the captions and "
            "refined tags; with --clip they come from CLIP text-to-image "
            "similarity instead. Narrow the result set with --since/--until/"
            "--camera, --person/--object, --type, --cluster, --image-only or "
            "--video. Add --json for machine-readable output."
        ),
    )
    _add_common(p_search)
    p_search.add_argument("query", nargs="*", metavar="QUERY",
                          help="free-text search terms (ANDed together)")
    p_search.add_argument("--person", help="terms describing a person, e.g. 'woman red jacket'")
    p_search.add_argument("--object", help="terms describing an object, e.g. 'bicycle'")
    p_search.add_argument("--clip", action="store_true",
                          help="force CLIP embedding search (default when embeddings exist)")
    p_search.add_argument("--fts", action="store_true",
                          help="force keyword/full-text search over captions")
    p_search.add_argument("--video", action="store_true",
                          help="search videos only")
    p_search.add_argument("--image-only", action="store_true",
                          help="search images only (skip videos)")
    p_search.add_argument("--type",
                          help="list images of a type, e.g. selfie/scene/object "
                               "(needs 'refine --classify')")
    p_search.add_argument("--cluster",
                          help="list photos of a named person cluster "
                               "(needs 'people --label N=Name')")
    p_search.add_argument("--since", metavar="DATE",
                          help="only photos captured on/after DATE "
                               "(YYYY-MM-DD; falls back to file mtime)")
    p_search.add_argument("--until", metavar="DATE",
                          help="only photos captured on/before DATE (YYYY-MM-DD)")
    p_search.add_argument("--camera", metavar="TEXT",
                          help="only photos whose camera make/model matches TEXT")
    p_search.add_argument("--json", action="store_true",
                          help="machine-readable output")
    p_search.add_argument("--limit", type=int, default=50,
                          help="max results to show")
    p_search.set_defaults(func=cmd_search)

    p_embed = sub.add_parser(
        "embed",
        help="compute CLIP embeddings for indexed images",
        description=(
            "Fill in CLIP embeddings for indexed images that do not have one yet "
            "(the usual case after importing an index from elsewhere). With "
            "--recompute, every indexed image is embedded again, which is how "
            "you switch to a different --clip-model."
        ),
    )
    _add_common(p_embed)
    p_embed.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL,
                         help=f"CLIP model (default: {DEFAULT_CLIP_MODEL})")
    p_embed.add_argument("--batch", type=int, default=16, help="batch size")
    p_embed.add_argument("--recompute", action="store_true",
                         help="recompute embeddings even if present")
    p_embed.set_defaults(func=cmd_embed)

    p_sim = sub.add_parser(
        "similar",
        help="group the tree by CLIP similarity, or find neighbors of one file",
        description=(
            "Group media by CLIP similarity, or show the neighbours of one "
            "file. Two modes: with no arguments it groups the whole tree into "
            "semantic near-duplicate groups (photos of the same scene that "
            "pHash would not catch, shot a moment apart); with --image it ranks "
            "indexed media against that one image instead. --video searches "
            "clips by their keyframes. --threshold tunes the cut-off."
        ),
    )
    _add_common(p_sim)
    p_sim.add_argument("--video", action="store_true",
                       help="group videos instead of images")
    p_sim.add_argument("--image",
                       help="compare a single image against the index instead "
                            "of grouping the whole tree (also valid with --video)")
    p_sim.add_argument("--threshold", type=float, default=0.85,
                       help="min cosine similarity for tree grouping (default: 0.85)")
    p_sim.add_argument("--limit", type=int, default=10,
                       help="max neighbors to show with --image")
    p_sim.add_argument("--json", action="store_true", help="machine-readable output")
    p_sim.set_defaults(func=cmd_similar)

    p_dup = sub.add_parser(
        "duplicates",
        help="group duplicates; best resolution wins",
        description=(
            "Group indexed media that are copies of each other, comparing them "
            "by pHash so re-encodes and resizes are caught. The best copy of "
            "each group (highest resolution, or newest with --prefer mtime) is "
            "marked. This command only reports; 'clean' is what moves the "
            "rejected copies to the Trash."
        ),
    )
    _add_common(p_dup)
    p_dup.add_argument("--video", action="store_true",
                       help="group duplicate/re-encode videos by keyframe pHash")
    p_dup.add_argument("--threshold", type=int, default=None,
                       help="max pHash hamming distance 0-64 for images "
                            "(default: 8, or config 'threshold')")
    p_dup.add_argument("--exact-phash", action="store_true",
                       help="group files that share the exact same phash "
                            "(Hamming distance = 0; faster than general threshold)")
    p_dup.add_argument("--frame-threshold", type=int, default=12,
                       help="max keyframe pHash hamming distance for videos (default: 12)")
    p_dup.add_argument("--ratio", type=float, default=0.6,
                       help="min fraction of matching keyframes for videos (default: 0.6)")
    p_dup.add_argument("--prefer", choices=("resolution", "date"), default=None,
                       help="which copy is 'best': largest (default) or newest "
                            "capture (or config 'prefer')")
    p_dup.add_argument("--json", action="store_true", help="machine-readable output")
    p_dup.set_defaults(func=cmd_duplicates)

    p_clean = sub.add_parser(
        "clean",
        help="review duplicate groups and move rejected copies to the Trash",
        description=(
            "Review duplicate groups and move rejected copies to the OS Trash. "
            "Opens a visual bar UI with the key legend pinned at the top of "
            "the screen: j/k or arrow keys move the bar, h/l jump between "
            "groups, space keeps the file under the bar (unselects the "
            "group's others), a keeps the whole current group, A keeps every "
            "file, v opens it in Quick Look, p in Preview.app, x trashes the "
            "current file, enter executes the whole deduplication (x, enter "
            "and e ask y/n first), e executes only the decisions made so far "
            "up to the bar's group and keeps the review open, q or esc quits "
            "but asks y/n first whenever files are still marked for the "
            "Trash. Without a TTY it falls back to line-based prompts. With "
            "--video the same UI reviews duplicate videos."
        ),
    )
    _add_common(p_clean)
    p_clean.add_argument("--video", action="store_true",
                         help="review duplicate videos instead of images")
    p_clean.add_argument("--threshold", type=int, default=8,
                         help="max pHash hamming distance 0-64 for images (default: 8)")
    p_clean.add_argument("--frame-threshold", type=int, default=12,
                         help="max keyframe pHash hamming distance for videos (default: 12)")
    p_clean.add_argument("--ratio", type=float, default=0.6,
                         help="min fraction of matching keyframes for videos (default: 0.6)")
    p_clean.add_argument("--yes", action="store_true",
                         help="trash the algorithm's worse copies without prompting")
    p_clean.add_argument("--dry-run", action="store_true",
                         help="show what would be trashed, then exit")
    p_clean.add_argument("--json", action="store_true",
                          help="with --dry-run: machine-readable list of files to trash")
    p_clean.set_defaults(func=cmd_clean)

    p_restore = sub.add_parser(
        "restore",
        help="move files back from the system Trash (undo 'clean')",
        description=(
            "List the files 'clean' moved to the system Trash — every batch, "
            "or one with --batch — and move them back to their original paths "
            "on confirmation (best-effort: the OS Trash may have renamed a "
            "colliding name). Run 'scan' afterwards to re-index them."
        ),
    )
    _add_common(p_restore)
    p_restore.add_argument("--batch", help="only this batch id (see the listing)")
    p_restore.add_argument("--limit", type=int, default=50,
                           help="max files to list (default: 50)")
    p_restore.add_argument("--dry-run", action="store_true",
                           help="show what would be restored, then exit")
    p_restore.add_argument("--yes", action="store_true",
                           help="restore without prompting")
    p_restore.set_defaults(func=cmd_restore)

    p_refine = sub.add_parser(
        "refine",
        help="clean keywords into canonical tags + facets; optionally type them",
        description=(
            "Rewrite each item's free-text keywords into canonical tags "
            "(lowercased, singularized, synonyms merged, filler dropped) and "
            "bucket them into facets (person/animal/object/scene/food/plant/"
            "text/activity/color). The original keywords are kept in "
            "raw_keywords. With --classify, also label every image's type "
            "(selfie/portrait/group/scene/city/animal/object/food/document/"
            "art) with zero-shot CLIP."
        ),
    )
    _add_common(p_refine)
    p_refine.add_argument("--classify", action="store_true",
                          help="also label each image's type with zero-shot CLIP")
    p_refine.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL,
                          help=f"CLIP model for --classify (default: {DEFAULT_CLIP_MODEL})")
    p_refine.add_argument("--recompute", action="store_true",
                          help="reclassify types even if already set")
    p_refine.set_defaults(func=cmd_refine)

    p_people = sub.add_parser(
        "people",
        help="cluster person photos by CLIP similarity and name the clusters",
        description=(
            "Cluster every photo tagged with the 'person' facet by CLIP "
            "embedding similarity, so photos of the same person group "
            "together. This groups similar-looking people in similar shots, "
            "not verified identities (CLIP is not a face-recognition model). "
            "Name a cluster with --label N=Name; a named cluster can be listed "
            "with 'search --cluster Name'."
        ),
    )
    _add_common(p_people)
    p_people.add_argument("--threshold", type=float, default=0.80,
                          help="min cosine similarity to link two photos (default: 0.80)")
    p_people.add_argument("--label", help="name a cluster, e.g. --label 3=Alice")
    p_people.add_argument("--min-size", type=int, default=1,
                          help="hide clusters smaller than this (default: 1)")
    p_people.add_argument("--limit", type=int, default=20,
                          help="max photos to list per cluster (default: 20)")
    p_people.add_argument("--json", action="store_true", help="machine-readable output")
    p_people.set_defaults(func=cmd_people)

    p_tags = sub.add_parser(
        "tags",
        help="list refined tags, facets or image types",
        description=(
            "List what 'refine' produced: the refined tags with counts, the "
            "facets those tags belong to, or the classified image types. "
            "Reads the index only; nothing is recomputed."
        ),
    )
    _add_common(p_tags)
    p_tags.add_argument("--video", action="store_true",
                        help="use video tags instead of image tags")
    p_tags.add_argument("--facets", action="store_true",
                        help="group tags into facets (person/scene/object/...)")
    p_tags.add_argument("--types", action="store_true",
                        help="show image type counts (selfie/scene/object/...)")
    p_tags.add_argument("--raw", action="store_true",
                        help="show the original keywords instead of refined tags")
    p_tags.add_argument("--limit", type=int, default=40,
                        help="max tags/facets to list")
    p_tags.add_argument("--json", action="store_true", help="machine-readable output")
    p_tags.set_defaults(func=cmd_tags)
    p_clear = sub.add_parser(
        "clear-metadata",
        help="clear description, keywords, type and tags from media items",
        description=(
            "Clear metadata from one or more indexed items. Items may be "
            "specified by position (paths relative to the root, as printed by "
            "search/export), or every item in the index may be cleared with no "
            "positional arguments. With items, --kind is only a filter; without "
            "items it limits the wipe (default: both kinds). Media files are "
            "never touched."
        ),
    )
    _add_common(p_clear)
    p_clear.add_argument("items", nargs="*", metavar="ITEM",
                         help="paths to clear metadata for (relative to root)")
    p_clear.add_argument("--kind", choices=("image", "video"), default=None,
                         help="restrict to one kind (default: both)")
    p_clear.add_argument("--yes", action="store_true",
                         help="skip confirmation prompt")
    p_clear.set_defaults(func=cmd_clear_metadata)

    p_import = sub.add_parser(
        "import-metadata",
        help="import metadata from JSON or CSV",
        description=(
            "Import description, keywords and/or type for indexed items from a "
            "JSON or CSV file. Rows need a 'path' field (relative to the root, "
            "as written by export-metadata) and optionally 'kind', "
            "'description', 'keywords' and 'type'. Existing metadata is "
            "overwritten only for the fields listed in --fields."
        ),
    )
    _add_common(p_import)
    p_import.add_argument("file", help="path to the import file (JSON or CSV)")
    p_import.add_argument("--kind", choices=("image", "video"), default=None,
                          help="assume this kind when a row has none "
                               "(default: try images, then videos)")
    p_import.add_argument("--fields", default="description,keywords,type",
                          help="comma-separated fields to import "
                               "(default: description,keywords,type)")
    p_import.set_defaults(func=cmd_import_metadata)

    p_export_meta = sub.add_parser(
        "export-metadata",
        help="export editable metadata (description/keywords/type) to JSON",
        description=(
            "Export the human-editable metadata of every indexed item as JSON: "
            "kind, path (relative to root), description, keywords and — for "
            "images — type. The output round-trips with 'import-metadata', so "
            "it works as a backup before 'clear-metadata' or as a batch to "
            "edit and re-import. Writes to --out, or stdout when omitted."
        ),
    )
    _add_common(p_export_meta)
    p_export_meta.add_argument("--kind", choices=("image", "video"), default=None,
                               help="restrict to one kind (default: both)")
    p_export_meta.add_argument("--out", help="write to this file instead of stdout")
    p_export_meta.set_defaults(func=cmd_export_metadata)

    p_reset = sub.add_parser(
        "reset",
        help="delete the index database (media files are never touched)",
        description=(
            "Delete the index database and start over: removes the SQLite "
            "file (with its -wal/-shm sidecars) so the next 'scan' rebuilds "
            "the index from scratch. Media files are NOT touched. With --yes "
            "the confirmation prompt is skipped."
        ),
    )
    _add_common(p_reset)
    p_reset.add_argument("--yes", action="store_true",
                         help="skip confirmation prompt")
    p_reset.set_defaults(func=cmd_reset)

    p_check = sub.add_parser(
        "check",
        help="check which media files are loadable, and offer to repair them",
        description=(
            "Walk the media tree and classify every file: ok, truncated "
            "(recoverable with tolerance) or corrupt (not decodable). "
            "Truncated images can be re-encoded (originals kept as "
            "*.truncated) and corrupt files quarantined to a folder. Both "
            "steps ask first in a terminal; --yes repairs without asking."
        ),
    )
    _add_common(p_check)
    p_check.add_argument("--video", action="store_true",
                         help="also probe videos with ffmpeg")
    p_check.add_argument("--quarantine", default=".media-match-quarantine",
                         metavar="DIR",
                         help="folder for corrupt files, relative to root "
                              "(default: .media-match-quarantine)")
    p_check.add_argument("--yes", action="store_true",
                         help="repair/quarantine without asking")
    p_check.set_defaults(func=cmd_check)

    p_bootstrap = sub.add_parser(
        "bootstrap",
        help="first pass: walk the tree, record filename/type/size in the index",
        description=(
            "Walk the whole tree and record every file's name, type and size in "
            "the index — no decoding or hashing, so it is fast. Dry-run by "
            "default: builds the database but touches no files. Add --prune to "
            "trash exact duplicates (keeping the newest), --media-only to prune "
            "non-media files and consolidate extensions, --move-to DIR (with "
            "--media-only) to move the non-media files to a folder instead of "
            "trashing them, and --fix-extensions "
            "(with --media-only) to add/update missing extensions. Works for "
            "images and videos."
        ),
    )
    _add_common(p_bootstrap)
    p_bootstrap.add_argument("--prune", action="store_true",
                             help="trash exact duplicates, keeping the newest version")
    p_bootstrap.add_argument("--media-only", action="store_true",
                             help="prune non-media files and consolidate extensions "
                                  "to canonical form (.jpeg -> .jpg, etc.)")
    p_bootstrap.add_argument("--move-to", metavar="DIR",
                             help="with --media-only: move the non-media files to DIR "
                                  "(keeping their layout) instead of trashing them")
    p_bootstrap.add_argument("--fix-extensions", action="store_true",
                             help="with --media-only: detect and add/update missing "
                                  "extensions (e.g. IMG_1234 -> IMG_1234.jpg)")
    p_bootstrap.add_argument("--video", action="store_true",
                             help="include videos (probed with ffprobe)")
    p_bootstrap.add_argument("--yes", action="store_true",
                             help="execute file operations without asking")
    p_bootstrap.set_defaults(func=cmd_bootstrap)

    p_export = sub.add_parser(
        "export",
        help="dump the index to JSON, CSV or Markdown",
        description=(
            "Export every indexed item (path, kind, size, capture date, camera, "
            "description, keywords, type, tags, cluster, duration) as JSON, CSV "
            "or a Markdown table. Writes to --out, or stdout when omitted."
        ),
    )
    _add_common(p_export)
    p_export.add_argument("--format", choices=("json", "csv", "md"), default="json",
                          help="output format (default: json)")
    p_export.add_argument("--out", help="write to this file instead of stdout")
    p_export.set_defaults(func=cmd_export)

    p_stats = sub.add_parser(
        "stats",
        help="show index statistics",
        description=(
            "Summarise the index: how many images, videos and captions it "
            "holds, the formats and cameras you have, and what the last few "
            "scans cost in wall time, GPU time and bytes read."
        ),
    )
    _add_common(p_stats)
    p_stats.add_argument("--json", action="store_true", help="machine-readable output")
    p_stats.set_defaults(func=cmd_stats)

    p_help = sub.add_parser(
        "help",
        help="show the full guide (commands, flags, examples)",
        description=(
            "Print this guide. With a command name, print that command's full "
            "help: every positional argument, every flag with its default, "
            "and a runnable example."
        ),
    )
    p_help.add_argument("command", nargs="?", metavar="COMMAND",
                        help="command to describe in detail (default: the whole guide)")
    p_help.set_defaults(func=_cmd_help)

    p_about = sub.add_parser(
        "about",
        help="show what this program does, its license and project links",
        description=(
            "Print what media-match is for, who wrote it, where the source "
            "lives, and the licence it ships under.\n"
            "\n"
            "With 'bench', measure this machine instead: it times disk reads, "
            "CPU decoding and CLIP batches in isolation, then runs several real "
            "scan configurations and recommends --batch, --io-workers and "
            "--workers. Nothing is indexed and no file is changed."
        ),
    )
    p_about.add_argument("topic", nargs="?", metavar="TOPIC",
                         choices=("bench",),
                         help="'bench' to benchmark this machine (default: "
                              "print the blurb)")
    p_about.add_argument("root", nargs="?", metavar="ROOT",
                         help="with 'bench': a folder of real media to benchmark "
                              "against (default: a synthetic corpus)")
    p_about.add_argument("--files", type=int, default=32, metavar="N",
                         help="with 'bench': how many files to use (default: 32)")
    p_about.add_argument("--repeat", type=int, default=1, metavar="N",
                         help="with 'bench': run each configuration N times and "
                              "keep the best (default: 1)")
    p_about.add_argument("--clip-model", default=None,
                         help="with 'bench': CLIP model to time (default: %s)"
                              % DEFAULT_CLIP_MODEL)
    p_about.add_argument("--video", action="store_true",
                         help="with 'bench': include video files")
    p_about.set_defaults(func=cmd_about)

    p_version = sub.add_parser(
        "version",
        help="print version, license and author",
        description=(
            "Print the version, licence and author on one line."
        ),
    )
    p_version.set_defaults(func=cmd_version)

    p_info = sub.add_parser(
        "info",
        help="print program information (models, last scan, free storage)",
        description=(
            "Print what this installation is set up with: the CLIP and VLM "
            "models in use, when the index was last scanned, and how much free "
            "disk space is left. Takes an optional folder to pick the index "
            "from; defaults to the current directory."
        ),
    )
    _add_common(p_info)
    p_info.set_defaults(func=cmd_info)

    # Every subparser must be registered here or 'help <command>' (and
    # '<command> help') report it as unknown.
    for name, parser_obj in (
        ("analyze", p_analyze), ("scan", p_scan), ("describe", p_describe),
        ("search", p_search), ("embed", p_embed), ("similar", p_sim),
        ("duplicates", p_dup), ("clean", p_clean), ("restore", p_restore),
        ("refine", p_refine), ("people", p_people), ("tags", p_tags),
        ("clear-metadata", p_clear), ("import-metadata", p_import),
        ("export-metadata", p_export_meta), ("reset", p_reset),
        ("check", p_check), ("bootstrap", p_bootstrap),
        ("export", p_export),
        ("stats", p_stats), ("about", p_about), ("version", p_version),
        ("info", p_info), ("help", p_help),
    ):
        _SUBPARSERS[name] = parser_obj

    # 'help <command>' must cover everything 'help' lists, or the guide
    # advertises commands it cannot describe.
    missing = sorted(set(_COMMAND_SUMMARIES) - set(_SUBPARSERS))
    if missing:  # pragma: no cover - guards against a parser added above
        raise RuntimeError(f"help summaries without a parser: {missing}")

    # Every command must appear in exactly one guide group: missing means the
    # guide hides it, duplicated means it is listed twice under different
    # headings — which is how 'check' and 'export' came out ambiguous before.
    grouped = [n for _h, _p, names in _COMMAND_GROUPS for n in names]
    unknown = sorted(set(grouped) - set(_COMMAND_SUMMARIES))
    if unknown:
        raise RuntimeError(f"guide groups name unknown commands: {unknown}")
    ungrouped = sorted(set(_COMMAND_SUMMARIES) - set(grouped))
    if ungrouped:
        raise RuntimeError(f"commands missing from the guide groups: {ungrouped}")
    duplicated = sorted({n for n in grouped if grouped.count(n) > 1})
    if duplicated:
        raise RuntimeError(f"commands listed in more than one guide group: {duplicated}")

    # '<command> help' shows that command's page — the same text as
    # 'help <command>'. Without this, 'help' is consumed as the root path and
    # commands that open an index immediately (bootstrap) fail inside a
    # folder that does not exist.
    for i, token in enumerate(argv):
        if token != "help" or i == 0:
            continue
        previous = argv[i - 1]
        if previous.startswith("-") or previous not in _SUBPARSERS:
            continue  # e.g. '--db help': 'help' is an option value, not a request
        print(_command_section(previous))
        return 0

    # No command at all: show the full guide, exactly like 'help'.
    if not argv:
        _print_full_help()
        return 0

    args = parser.parse_args(argv)

    if getattr(args, "command", None) is None:
        _print_full_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
