"""Command line interface for media-match."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from .db import Database, FileRecord, db_path_for
from .groups import find_duplicate_groups
from .hashing import hash_to_hex, phash
from .vision import DEFAULT_MODEL
from .clip import DEFAULT_CLIP_MODEL

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
    ".tif", ".tiff", ".heic", ".heif",
}

_mp = lambda w, h: f"{w * h / 1_000_000:.1f}MP"


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass


def _iter_images(root: str) -> list[str]:
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, name))
    return paths


def _sha1_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- commands ----------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> int:
    _register_heif()
    from PIL import Image

    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))
    db.set_meta("root", root)
    db.set_meta("model", args.model)

    paths = _iter_images(root)
    if args.limit:
        paths = paths[: args.limit]

    to_process: list[str] = []
    for path in paths:
        existing = db.get_file(path)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        if (
            existing
            and not args.reanalyze
            and existing["size"] == stat.st_size
            and abs(existing["mtime"] - stat.st_mtime) < 1e-6
        ):
            continue
        to_process.append(path)

    skipped = len(paths) - len(to_process)
    print(f"{len(paths)} images found, {skipped} up to date, {len(to_process)} to analyze")

    describer = None
    if not args.no_describe:
        from .vision import Describer

        describer = Describer(model_id=args.model)

    errors = 0
    for n, path in enumerate(to_process, 1):
        try:
            stat = os.stat(path)
            with Image.open(path) as img:
                width, height = img.size
                digest_phash = hash_to_hex(phash(img))
            description, keywords = "", []
            if describer:
                print(f"[{n}/{len(to_process)}] describing {os.path.relpath(path, root)}",
                      file=sys.stderr)
                description, keywords = describer.describe(path)
            else:
                print(f"[{n}/{len(to_process)}] hashing {os.path.relpath(path, root)}",
                      file=sys.stderr)
            db.upsert_file(
                path=path,
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha1=_sha1_of(path),
                width=width,
                height=height,
                phash=digest_phash,
                description=description,
                keywords=", ".join(keywords),
            )
        except Exception as exc:  # noqa: BLE001 - keep scanning past bad files
            errors += 1
            print(f"  ERROR {path}: {exc}", file=sys.stderr)

    print(f"Done: analyzed {len(to_process) - errors}, errors {errors}, "
          f"index has {db.count_files()} files ({db.path})")
    db.close()
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))
    terms: list[str] = []
    if args.person:
        terms.extend(args.person.split())
    if args.object:
        terms.extend(args.object.split())
    terms.extend(args.query)

    if args.clip:
        return _clip_search(db, " ".join(terms), args.limit)

    records = db.search(terms, limit=args.limit)
    if not records:
        print("No matches.")
        db.close()
        return 1
    print(f"{len(records)} match(es):")
    for r in records:
        print(f"\n{r['path']}  ({r['width']}x{r['height']}, {_mp(r['width'], r['height'])})")
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


def _clip_search(db: Database, query_text: str, limit: int) -> int:
    import numpy as np

    from .clip import ClipEncoder

    ids, matrix = _load_embedding_matrix(db)
    if matrix is None:
        print("No CLIP embeddings yet. Run 'media-match embed' first.")
        db.close()
        return 1
    query = ClipEncoder().embed_texts([query_text])[0]
    sims = matrix @ query
    order = np.argsort(-sims)[:limit]
    print(f"CLIP text->image search for {query_text!r}:")
    for idx in order:
        r = db.conn.execute("SELECT * FROM files WHERE id = ?", (ids[idx],)).fetchone()
        print(f"\n{r['path']}  ({r['width']}x{r['height']}, "
              f"{_mp(r['width'], r['height'])})  similarity={sims[idx]:.3f}")
    db.close()
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    import time

    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))

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
    db = Database(db_path_for(root, args.db))
    ids, matrix = _load_embedding_matrix(db)
    if matrix is None:
        print("No CLIP embeddings yet. Run 'media-match embed' first.")
        db.close()
        return 1

    from .clip import ClipEncoder

    target = os.path.abspath(args.image)
    stored = db.embedding_for_path(target)
    if stored is not None:
        query = np.frombuffer(stored, dtype=np.float32)
        source = "indexed embedding"
    else:
        query = ClipEncoder().embed_image_paths([target])[0]
        source = "computed on the fly"
    sims = matrix @ query

    target_row = db.conn.execute(
        "SELECT id FROM files WHERE path = ?", (target,)
    ).fetchone()
    exclude_id = target_row["id"] if target_row else None

    order = np.argsort(-sims)
    shown = 0
    print(f"Images similar to {target} ({source}):")
    for idx in order:
        row = db.conn.execute(
            "SELECT * FROM files WHERE id = ?", (ids[idx],)
        ).fetchone()
        if row["id"] == exclude_id:
            continue
        print(f"  {sims[idx]:.3f}  {row['path']}  "
              f"({row['width']}x{row['height']}, {_mp(row['width'], row['height'])})")
        shown += 1
        if shown >= args.limit:
            break
    db.close()
    return 0


def cmd_duplicates(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))
    groups = find_duplicate_groups(db.all_files(), threshold=args.threshold)
    if not groups:
        print("No duplicate groups found.")
        db.close()
        return 0

    total = sum(g.size for g in groups)
    reclaimable = sum(
        r.size for g in groups for r, _ in g.others
    )

    if args.json:
        payload = [
            {
                "best": g.best.path,
                "best_size": [g.best.width, g.best.height],
                "duplicates": [
                    {"path": r.path, "size": [r.width, r.height], "hash_distance": d}
                    for r, d in g.others
                ],
            }
            for g in groups
        ]
        print(json.dumps(payload, indent=2))
        db.close()
        return 0

    for n, g in enumerate(groups, 1):
        print(f"Group {n} ({g.size} files)")
        b = g.best
        print(f"  * {b.path}  ({b.width}x{b.height}, {_mp(b.width, b.height)}) [best]")
        for r, dist in g.others:
            print(f"    {r.path}  ({r.width}x{r.height}, {_mp(r.width, r.height)}) "
                  f"hamming={dist}")
        print()
    db.close()
    return 0


def cmd_tags(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))
    counts = db.keyword_counts(limit=args.limit)
    if not counts:
        print("No keywords indexed yet. Run 'scan' first.")
        db.close()
        return 1
    width = max(len(k) for k, _ in counts)
    for kw, count in counts:
        print(f"{kw:<{width}}  {count}")
    db.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    db = Database(db_path_for(root, args.db))
    total = db.count_files()
    described = db.conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE description != ''"
    ).fetchone()["n"]
    print(f"index:  {db.path}")
    print(f"root:   {db.get_meta('root', root)}")
    print(f"model:  {db.get_meta('model', '(not set)')}")
    print(f"files:  {total} ({described} described)")
    db.close()
    return 0


# -- parser ------------------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("root", nargs="?", default=".",
                   help="media tree to work on (default: current directory)")
    p.add_argument("--db", help="index database path (default: <root>/.media-match.db)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="media-match",
        description="Index photos with a local MLX vision model; search people/"
                    "objects and group duplicates across resolutions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="analyze new/changed images in a folder")
    _add_common(p_scan)
    p_scan.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"MLX vision model (default: {DEFAULT_MODEL})")
    p_scan.add_argument("--limit", type=int, help="only scan first N images")
    p_scan.add_argument("--no-describe", action="store_true",
                        help="skip the vision model; only hash (no search)")
    p_scan.add_argument("--reanalyze", action="store_true",
                        help="re-analyze files even if unchanged")
    p_scan.set_defaults(func=cmd_scan)

    p_search = sub.add_parser("search", help="find images by person/object/text")
    _add_common(p_search)
    p_search.add_argument("query", nargs="*", help="free-text search terms")
    p_search.add_argument("--person", help="terms describing a person, e.g. 'woman red jacket'")
    p_search.add_argument("--object", help="terms describing an object, e.g. 'bicycle'")
    p_search.add_argument("--clip", action="store_true",
                          help="CLIP embedding search instead of keyword/FTS search")
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

    p_sim = sub.add_parser("similar", help="find images visually similar to a given image")
    p_sim.add_argument("image", help="image file to compare against the index")
    p_sim.add_argument("root", nargs="?", default=".",
                       help="media tree index to search (default: current directory)")
    p_sim.add_argument("--db", help="index database path (default: <root>/.media-match.db)")
    p_sim.add_argument("--limit", type=int, default=10)
    p_sim.set_defaults(func=cmd_similar)

    p_dup = sub.add_parser("duplicates", help="group duplicates; best resolution wins")
    _add_common(p_dup)
    p_dup.add_argument("--threshold", type=int, default=8,
                       help="max pHash hamming distance 0-64 (default: 8)")
    p_dup.add_argument("--json", action="store_true", help="machine-readable output")
    p_dup.set_defaults(func=cmd_duplicates)

    p_tags = sub.add_parser("tags", help="list indexed keywords with counts")
    _add_common(p_tags)
    p_tags.add_argument("--limit", type=int, default=40)
    p_tags.set_defaults(func=cmd_tags)

    p_stats = sub.add_parser("stats", help="show index statistics")
    _add_common(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
