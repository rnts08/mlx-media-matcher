"""Command line interface for media-match."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

from .db import Database, db_path_for, open_db
from .groups import find_duplicate_groups, find_similarity_groups
from .pipeline import build_jobs, run_scan, store_result
from .tags import facet_of, refine
from .vision import DEFAULT_MODEL
from .clip import DEFAULT_CLIP_MODEL
from .video_groups import find_video_duplicate_groups, find_video_similarity_groups
from .video_search import best_moment, format_timestamp

_mp = lambda w, h: f"{w * h / 1_000_000:.1f}MP"


def _dur(seconds: float) -> str:
    return format_timestamp(seconds)


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


def cmd_scan(args: argparse.Namespace) -> int:
    _register_heif()

    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))
    db.set_meta("root", root)
    db.set_meta("clip_model", args.clip_model)
    if args.describe:
        db.set_meta("vlm_model", args.model)

    if args.video and not _require_ffmpeg():
        db.close()
        return 2

    jobs, total, up_to_date = build_jobs(
        root, db, include_video=args.video, reanalyze=args.reanalyze,
        limit=args.limit,
    )
    print(f"{total} media found, {up_to_date} up to date, {len(jobs)} to analyze")

    from .clip import ClipEncoder

    encoder = ClipEncoder(model_id=args.clip_model, batch_size=args.batch)
    errors = 0

    def on_error(path: str, exc: Exception) -> None:
        nonlocal errors
        errors += 1
        print(f"  ERROR {path}: {exc}", file=sys.stderr)

    def on_result(result) -> None:
        nonlocal errors
        try:
            store_result(db, result)
        except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
            errors += 1
            print(f"  ERROR storing {result.job.path}: {exc}", file=sys.stderr)

    total_jobs = len(jobs)
    step = max(1, total_jobs // 20)

    def progress(done: int, total: int, _path: str) -> None:
        if total > 20 and (done % step == 0 or done == total):
            print(f"  preparing [{done}/{total}]", file=sys.stderr)

    analyzed = run_scan(
        jobs, encoder, interval=args.interval, max_frames=args.max_frames,
        batch_size=args.batch, workers=args.workers,
        on_result=on_result, progress=progress, on_error=on_error,
    )

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
        _describe_images(db, root, args, on_error)
        if args.video:
            _describe_videos(db, root, args, on_error)

    described = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE description != ''"
    ).fetchone()["n"]
    vdescribed = db.conn.execute(
        "SELECT COUNT(*) AS n FROM videos WHERE description != ''"
    ).fetchone()["n"]
    print(f"Done: analyzed {analyzed}, errors {errors}. Index: "
          f"{db.count_files()} images ({described} described), "
          f"{db.count_videos()} videos ({vdescribed} described) ({db.path})")
    db.close()
    return 0


def _describe_images(db: Database, root: str, args, on_error) -> None:
    records = db.all_files()
    group_best: dict[int, int] = {}
    for group in find_duplicate_groups(records):
        for member in [group.best] + [rec for rec, _ in group.others]:
            group_best[member.id] = group.best.id
    pending = [r for r in records if not r.description]
    to_describe = [r for r in pending if group_best.get(r.id, r.id) == r.id]
    shadowed = len(pending) - len(to_describe)
    if shadowed:
        print(f"{shadowed} undescribed image duplicate(s) skipped; "
              f"their best-resolution version is described instead")
    if not to_describe:
        return

    from .vision import Describer

    describer = Describer(model_id=args.model)
    for n, record in enumerate(to_describe, 1):
        print(f"[{n}/{len(to_describe)}] describing "
              f"{os.path.relpath(record.path, root)}", file=sys.stderr)
        try:
            description, keywords = describer.describe(record.path)
            db.set_description(record.id, description, ", ".join(keywords))
        except Exception as exc:  # noqa: BLE001
            on_error(record.path, exc)
    describer.unload()


def _describe_videos(db: Database, root: str, args, on_error) -> None:
    import tempfile

    from .video import extract_still_png

    records = db.all_videos_with_frames()
    group_best: dict[int, int] = {}
    for group in find_video_duplicate_groups(records):
        for member in [group.best] + [rec for rec, _ in group.others]:
            group_best[member.id] = group.best.id
    pending = [r for r in records if not r.description]
    to_describe = [r for r in pending if group_best.get(r.id, r.id) == r.id]
    shadowed = len(pending) - len(to_describe)
    if shadowed:
        print(f"{shadowed} undescribed video duplicate(s) skipped; "
              f"their best-resolution version is described instead")
    if not to_describe:
        return

    from .vision import Describer

    describer = Describer(model_id=args.model)
    for n, record in enumerate(to_describe, 1):
        print(f"[{n}/{len(to_describe)}] describing "
              f"{os.path.relpath(record.path, root)}", file=sys.stderr)
        try:
            png = extract_still_png(record.path, at=record.duration / 2)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(png)
                tmp_path = tmp.name
            try:
                description, keywords = describer.describe(tmp_path)
            finally:
                os.unlink(tmp_path)
            db.set_video_description(record.id, description, ", ".join(keywords))
        except Exception as exc:  # noqa: BLE001
            on_error(record.path, exc)
    describer.unload()


def _list_by_type(db: Database, args) -> int:
    rows = db.files_with_type(args.type)
    if not rows:
        known = ", ".join(name for name, _ in db.type_counts()) or "(none)"
        print(f"No images of type '{args.type}'. Known types: {known}")
        db.close()
        return 1
    for r in rows[: args.limit]:
        print(f"{r['path']}  ({r['width']}x{r['height']}, "
              f"{_mp(r['width'], r['height'])})  type={r['type']} "
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
        print(f"{r['path']}  ({r['width']}x{r['height']}, "
              f"{_mp(r['width'], r['height'])})")
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
        _, blobs = db.all_embeddings()
        use_clip = bool(blobs) or bool(db.video_embeddings())

    if use_clip:
        return _clip_search(db, " ".join(terms), args.limit,
                            include_images=include_images,
                            include_videos=include_videos)

    count = 0
    if include_images:
        for r in db.search(terms, limit=args.limit):
            print(f"\n{r['path']}  ({r['width']}x{r['height']}, "
                  f"{_mp(r['width'], r['height'])})")
            if r["keywords"]:
                print(f"  keywords: {r['keywords']}")
            if r["description"]:
                print(f"  {r['description']}")
            count += 1
    if include_videos:
        for r in db.search_videos(terms, limit=args.limit):
            print(f"\n{r['path']}  ({r['width']}x{r['height']}, "
                  f"{_dur(r['duration'])})")
            if r["keywords"]:
                print(f"  keywords: {r['keywords']}")
            if r["description"]:
                print(f"  {r['description']}")
            count += 1
    if not count:
        print("No matches.")
        db.close()
        return 1
    db.close()
    return 0


def _load_embedding_matrix(db: Database):
    import numpy as np

    ids, blobs = db.all_embeddings()
    if not blobs:
        return ids, None
    return ids, np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])


def _clip_search(db: Database, query_text: str, limit: int, *,
                 include_images: bool = True, include_videos: bool = True) -> int:
    import numpy as np

    from .clip import ClipEncoder

    ids, matrix = _load_embedding_matrix(db)
    video_embeddings = db.video_embeddings() if include_videos else {}
    if (matrix is None or not include_images) and not video_embeddings:
        print("No CLIP embeddings yet. Run 'media-match scan' first.")
        db.close()
        return 1

    query = ClipEncoder().embed_texts([query_text])[0]

    hits: list[tuple[float, str, object, float | None]] = []
    if include_images and matrix is not None:
        sims = matrix @ query
        for idx in np.argsort(-sims)[:limit]:
            row = db.conn.execute(
                "SELECT * FROM files WHERE id = ?", (ids[int(idx)],)
            ).fetchone()
            hits.append((float(sims[int(idx)]), "image", row, None))
    if video_embeddings:
        by_id = {r.id: r for r in db.all_videos()}
        for vid, frames in video_embeddings.items():
            record = by_id.get(vid)
            if record is None:
                continue
            frame_map = {t: np.frombuffer(b, dtype=np.float32) for t, b in frames}
            score, ts = best_moment(query, frame_map)
            hits.append((score, "video", record, ts))

    hits.sort(key=lambda h: -h[0])
    print(f"CLIP text->media search for {query_text!r}:")
    for score, kind, record, ts in hits[:limit]:
        if kind == "image":
            print(f"\n{record['path']}  ({record['width']}x{record['height']}, "
                  f"{_mp(record['width'], record['height'])})  similarity={score:.3f}")
        else:
            print(f"\n{record.path}  ({record.width}x{record.height}, "
                  f"{_dur(record.duration)})  similarity={score:.3f}  "
                  f"@ {format_timestamp(ts)}")
    db.close()
    return 0


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
            print(f"  * {b.path}  ({b.width}x{b.height}, {_mp(b.width, b.height)}) [best]")
            for r, dist in g.others:
                print(f"    {r.path}  ({r.width}x{r.height}, {_mp(r.width, r.height)}) "
                      f"hamming={dist}")
        print()
    print(f"{len(groups)} group(s), {total} files")
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
                size = sum(r.size for r in moved)
                print(f"Moved {len(moved)} file(s) to the Trash "
                      f"({human_size(size)}; recoverable until emptied); "
                      f"removed {len(moved)} entries from the index. "
                      f"{len(session.errors)} failed.")
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
    print(f"\nMoved {len(trashed)} file(s) to the Trash (recoverable until "
          f"emptied); removed {len(trashed)} entries from the index. "
          f"{len(failed)} failed.")
    db.close()
    return 0 if not failed else 1


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

    if args.types:
        counts = db.type_counts()
        if not counts:
            print("No image types yet. Run 'refine --classify' first.")
            db.close()
            return 1
        width = max(len(name) for name, _ in counts)
        for name, count in counts:
            print(f"{name:<{width}}  {count}")
        db.close()
        return 0

    if args.facets:
        grouped = db.facet_tag_counts(kind)
        if not grouped:
            print("No refined tags yet. Run 'refine' first.")
            db.close()
            return 1
        total_by_facet = dict(db.facet_counts(kind))
        for facet, tags in grouped.items():
            print(f"{facet} ({total_by_facet.get(facet, 0)} media)")
            width = max(len(t) for t, _ in tags)
            for tag, count in tags[: args.limit]:
                print(f"  {tag:<{width}}  {count}")
            print()
        db.close()
        return 0

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
    print(f"top {heading}:")
    width = max(len(k) for k, _ in counts)
    for kw, count in counts:
        print(f"{kw:<{width}}  {count}")
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
    print(f"index:  {db.path}")
    print(f"root:   {db.get_meta('root', root)}")
    print(f"clip:   {db.get_meta('clip_model', '(not set)')}")
    print(f"vlm:    {db.get_meta('vlm_model', '(not used)')}")
    print(f"files:  {total} ({described} described, {embedded} embedded)")
    print(f"videos: {videos} ({vframes_note(frames, vembedded, vdescribed)})")
    tagged = db.has_tags("image")
    typed = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE type != ''"
    ).fetchone()["n"]
    people = len(db.all_clusters("image"))
    print(f"tags:   {tagged} refined tags, {typed} typed, {people} person clusters")
    db.close()
    return 0


def vframes_note(frames: int, embedded: int, described: int) -> str:
    return (f"{frames} keyframes, {embedded} embedded, {described} described")


# -- help --------------------------------------------------------------------

# Filled in main(); maps a command name to its argparse parser.
_SUBPARSERS: dict[str, argparse.ArgumentParser] = {}

# One-line summary for the top-level command listing and the `help` command.
_COMMAND_SUMMARIES = {
    "scan": "index new/changed media (hash + pHash + CLIP)",
    "search": "find media by person/object/text (CLIP by default)",
    "embed": "compute CLIP embeddings for indexed images",
    "similar": "group the tree by CLIP similarity, or find neighbors of one file",
    "duplicates": "group duplicates; best resolution wins",
    "clean": "review duplicate groups and move rejected copies to the Trash",
    "refine": "rewrite keywords into canonical tags/facets; type photos",
    "people": "cluster person photos by similarity and name the clusters",
    "tags": "list refined tags, facets or image types",
    "stats": "show index statistics",
    "help": "show this guide; 'help <cmd>' describes one command",
}

# A runnable example per command, shown in `media-match help <cmd>`.
_EXAMPLES = {
    "scan": "media-match scan ~/Pictures --describe --refine --classify",
    "search": "media-match search ~/Pictures --object bicycle",
    "embed": "media-match embed ~/Pictures",
    "similar": "media-match similar ~/Pictures --threshold 0.9",
    "duplicates": "media-match duplicates ~/Pictures",
    "clean": "media-match clean ~/Pictures --dry-run",
    "refine": "media-match refine ~/Pictures --classify",
    "people": "media-match people ~/Pictures --label 1=Alice",
    "tags": "media-match tags ~/Pictures --facets",
    "stats": "media-match stats ~/Pictures",
}


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
            metavar = action.metavar or action.dest.upper()
            flags = f"{flags} {metavar}"
        text = action.help or ""
        default = action.default
        if default not in (None, False, argparse.SUPPRESS) and "default" not in text.lower():
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


def _print_full_help() -> None:
    lines = [
        "Local photo-library analyzer built on Apple's MLX. Point it at a folder",
        "and it builds an index that lets you:",
        "",
        "  - find photos by person, object or free text (CLIP vector search)",
        "  - group duplicates across resolutions, and mark the best version",
        "  - group semantic near-duplicates",
        "  - refine tags into facets (person/scene/object/...) and type each photo",
        "  - cluster photos of the same person and name the clusters",
        "  - index video files (keyframes at ~1 fps) with the same machinery",
        "",
        "usage: media-match [--help] <command> [command options] [args...]",
        "",
        "commands:",
    ]
    width = max(len(c) for c in _COMMAND_SUMMARIES)
    for name in sorted(_COMMAND_SUMMARIES):
        lines.append(f"  {name:<{width}}  {_COMMAND_SUMMARIES[name]}")
    lines += [
        "",
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


# -- parser ------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("root", nargs="?", default=".",
                   help="media tree to work on (default: current directory)")
    p.add_argument("--db", help="index database path (default: <root>/.media-match.db)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="media-match",
        description=(
            "Local photo-library analyzer built on Apple's MLX. Index photos "
            "and videos locally (no cloud, no PyTorch), then find, group, tag "
            "and clean them with CLIP embeddings, perceptual hashes and "
            "optional vision-model captions."
        ),
        epilog=(
            "Run 'media-match help' for the full guide (every command and "
            "flag), or 'media-match <command> --help' for one command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="analyze new/changed media in a folder")
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
    p_scan.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"MLX vision model for --describe (default: {DEFAULT_MODEL})")
    p_scan.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL,
                        help=f"CLIP model for embeddings (default: {DEFAULT_CLIP_MODEL})")
    p_scan.add_argument("--batch", type=int, default=16, help="CLIP batch size")
    p_scan.add_argument("--workers", type=int, default=4,
                        help="parallel CPU prep threads feeding the GPU (default: 4)")
    p_scan.add_argument("--interval", type=float, default=1.0,
                        help="target seconds between video keyframes (default: 1.0)")
    p_scan.add_argument("--max-frames", type=int, default=120,
                        help="cap on keyframes per video (default: 120)")
    p_scan.add_argument("--limit", type=int, help="only scan first N media items")
    p_scan.add_argument("--reanalyze", action="store_true",
                        help="re-analyze files even if unchanged")
    p_scan.set_defaults(func=cmd_scan)

    p_search = sub.add_parser("search", help="find media by person/object/text")
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
    p_search.add_argument("--limit", type=int, default=50)
    p_search.set_defaults(func=cmd_search)

    p_embed = sub.add_parser("embed", help="compute CLIP embeddings for indexed images")
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
    p_sim.set_defaults(func=cmd_similar)

    p_dup = sub.add_parser("duplicates", help="group duplicates; best resolution wins")
    _add_common(p_dup)
    p_dup.add_argument("--video", action="store_true",
                       help="group duplicate/re-encode videos by keyframe pHash")
    p_dup.add_argument("--threshold", type=int, default=8,
                       help="max pHash hamming distance 0-64 for images (default: 8)")
    p_dup.add_argument("--frame-threshold", type=int, default=12,
                       help="max keyframe pHash hamming distance for videos (default: 12)")
    p_dup.add_argument("--ratio", type=float, default=0.6,
                       help="min fraction of matching keyframes for videos (default: 0.6)")
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
    p_clean.set_defaults(func=cmd_clean)

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

    p_tags = sub.add_parser("tags", help="list refined tags, facets or image types")
    _add_common(p_tags)
    p_tags.add_argument("--video", action="store_true",
                        help="use video tags instead of image tags")
    p_tags.add_argument("--facets", action="store_true",
                        help="group tags into facets (person/scene/object/...)")
    p_tags.add_argument("--types", action="store_true",
                        help="show image type counts (selfie/scene/object/...)")
    p_tags.add_argument("--raw", action="store_true",
                        help="show the original keywords instead of refined tags")
    p_tags.add_argument("--limit", type=int, default=40)
    p_tags.set_defaults(func=cmd_tags)

    p_stats = sub.add_parser("stats", help="show index statistics")
    _add_common(p_stats)
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

    for name, parser_obj in (
        ("scan", p_scan), ("search", p_search), ("embed", p_embed),
        ("similar", p_sim), ("duplicates", p_dup), ("clean", p_clean),
        ("refine", p_refine), ("people", p_people), ("tags", p_tags),
        ("stats", p_stats), ("help", p_help),
    ):
        _SUBPARSERS[name] = parser_obj

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
