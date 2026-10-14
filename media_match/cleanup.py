"""Interactive review of duplicate groups; rejected copies go to the OS trash.

The algorithm suggests keeping the best-resolution version of each group.
The user can accept that (per group or for all remaining groups), pick a
different file to keep, view files in the OS image viewer (macOS Quick Look
or Preview.app), skip groups, or quit. Nothing is touched until a
final confirmation; files are then moved to the system trash (recoverable),
never deleted outright.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable

from .db import FileRecord
from .groups import DuplicateGroup


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def file_size(record: FileRecord) -> int:
    try:
        return os.path.getsize(record.path)
    except OSError:
        return 0


def group_files(group: DuplicateGroup) -> list[FileRecord]:
    return [group.best] + [record for record, _ in group.others]


def open_in_viewer(paths: list[str], app: str = "quicklook") -> None:
    """Open image(s) in the OS viewer, without blocking the prompt.

    app="quicklook" is the Finder spacebar panel on macOS (qlmanage -p),
    showing multiple files side by side; app="preview" opens Preview.app.
    """
    if not paths:
        return
    if sys.platform == "darwin":
        if app == "preview":
            command = ["open", "-a", "Preview", *paths]
        else:
            command = ["qlmanage", "-p", *paths]
    else:
        command = ["xdg-open", paths[0]]
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def render_group(group: DuplicateGroup, index: int, total: int, root: str,
                 out: Callable[[str], None], trash_word: str = "[trash]") -> None:
    files = group_files(group)
    extra = sum(file_size(r) for r in files[1:])
    out(f"\nGroup {index}/{total} — {len(files)} files, "
        f"{human_size(extra)} in extra copies")
    for n, record in enumerate(files, 1):
        is_best = record.id == group.best.id
        mark = "[keep*]" if is_best else trash_word
        hamming = ""
        for other, dist in group.others:
            if other.id == record.id:
                hamming = f", hamming={dist}"
        rel = os.path.relpath(record.path, root)
        out(f"  {n}. {mark:8s} {rel}  "
            f"({record.width}x{record.height}, {human_size(file_size(record))}"
            f"{hamming})")
    out("  * algorithm's pick (best resolution)")


def interactive_select(
    groups: list[DuplicateGroup],
    root: str,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    viewer: Callable[[list[str], str], None] = open_in_viewer,
) -> list[FileRecord]:
    """Prompt once per group; return the files chosen for the trash."""
    trash: list[FileRecord] = []
    accept_all = False
    for index, group in enumerate(groups, 1):
        render_group(group, index, len(groups), root, out)
        keep = group.best
        skip = False
        while not accept_all:
            try:
                choice = input_fn(
                    "  keep? [Enter]=best, 1-N=keep that file, "
                    "v [N]=view (Quick Look), p [N]=view in Preview, "
                    "s=skip, a=best for all remaining, q=quit > "
                ).strip().lower()
            except EOFError:
                choice = "q"
            tokens = choice.split()
            verb = tokens[0] if tokens else ""

            if verb in ("v", "p"):
                files = group_files(group)
                picked = files
                if len(tokens) > 1 and tokens[1].isdigit() \
                        and 1 <= int(tokens[1]) <= len(files):
                    picked = [files[int(tokens[1]) - 1]]
                app = "preview" if verb == "p" else "quicklook"
                viewer([f.path for f in picked], app)
                out(f"  opened {len(picked)} file(s) in the OS viewer "
                    f"({'Preview' if app == 'preview' else 'Quick Look'})")
                continue

            if verb == "q":
                out("Stopping here; earlier selections still apply.")
                return trash
            if verb == "a":
                accept_all = True
            elif verb == "s":
                skip = True
                break
            elif verb.isdigit() and 1 <= int(verb) <= len(group_files(group)):
                keep = group_files(group)[int(verb) - 1]
                break
            else:
                break  # Enter or unrecognized -> keep the algorithm's pick
        if not skip:
            trash.extend(r for r in group_files(group) if r.id != keep.id)
    return trash


def print_plan(
    groups: list[DuplicateGroup],
    root: str,
    out: Callable[[str], None],
    trash_word: str = "[trash]",
) -> list[FileRecord]:
    """Show every group with the algorithm's suggestion; return that plan."""
    trash: list[FileRecord] = []
    for index, group in enumerate(groups, 1):
        render_group(group, index, len(groups), root, out, trash_word=trash_word)
        trash.extend(record for record, _ in group.others)
    return trash


def confirm(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> bool:
    try:
        answer = input_fn(prompt).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def trash_files(
    records: list[FileRecord], out: Callable[[str], None] = print
) -> tuple[list[FileRecord], list[FileRecord]]:
    """Move files to the system trash.

    Returns (trashed_or_gone, failed). Files already missing on disk count
    as trashed (they are gone; their index entries still need removing).
    """
    from send2trash import send2trash

    trashed: list[FileRecord] = []
    failed: list[FileRecord] = []
    for record in records:
        if not os.path.exists(record.path):
            trashed.append(record)
            continue
        try:
            send2trash(record.path)
            trashed.append(record)
        except Exception as exc:  # noqa: BLE001 - report per file, keep going
            out(f"  FAILED {record.path}: {exc}")
            failed.append(record)
    return trashed, failed
