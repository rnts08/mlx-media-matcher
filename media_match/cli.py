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
    p_search.add_argument("--limit", type=int, default=50)
    p_search.set_defaults(func=cmd_search)

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
