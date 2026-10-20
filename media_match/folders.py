"""Folder-level consolidation: find same-named folders and merge or gather them.

Media trees accumulate the same folder name in several places — an export
makes ``2019/christmas``, a phone backup makes ``2020/Christmas``, a manual
sort makes ``2021/CHRISTMAS``. The duplicate detector works on files; this
works one level up, on the folders themselves.

Two operations share the same discovery step:

* **merge** (default) moves every file from the other copies into a chosen
  destination, asking what to do when both sides have a file at the same
  relative path. Decisions are buffered and applied only after confirmation.
* **collapse** (``--collapse``) moves the whole folders into
  ``<root>/<Name>_collapsed/`` as ``<Name>_1``, ``<Name>_2``, … and touches no
  files inside them.

Discovery is case-insensitive (``find -iname``), skips hidden and ignored
directories, never follows directory symlinks, and reads no index: this is a
filesystem tool that happens to live in the same program.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .cleanup import human_size
from .config import is_ignored

Out = Callable[[str], None]
Input = Callable[[str], str]


class QuitReview(Exception):
    """The user asked to stop reviewing (q/Escape), or stdin reached EOF.

    ``eof`` distinguishes "the user chose to quit" from "input ended". Both
    stop the review, but an EOF with nothing decided is an abort (status 1),
    not a successful "nothing to do" run.
    """

    def __init__(self, eof: bool = False) -> None:
        super().__init__("review stopped")
        self.eof = eof


@dataclass(frozen=True)
class FolderGroup:
    """Folders whose names are the same once case is ignored."""

    key: str
    root: str
    members: tuple[str, ...]

    @property
    def name(self) -> str:
        """Display name: the suggested destination's own spelling."""
        return os.path.basename(self.suggested)

    @property
    def suggested(self) -> str:
        return suggest_destination(self.members, self.root)

    @property
    def size(self) -> int:
        return len(self.members)


def _rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:  # pragma: no cover - different drive on Windows
        return path


def suggest_destination(members: tuple[str, ...] | list[str], root: str) -> str:
    """The member closest to the root: fewest components, then shortest path.

    Ties fall back to casefolded and then exact lexicographic order so two runs
    on the same tree always suggest the same folder.
    """
    def rank(path: str) -> tuple[int, int, str, str]:
        rel = _rel(path, root)
        return (rel.count(os.sep), len(rel), rel.casefold(), path)

    return min(members, key=rank)


def find_folder_groups(
    root: str,
    *,
    name: str | None = None,
    ignore: list[str] | tuple[str, ...] = (),
    on_error: Callable[[OSError], None] | None = None,
) -> list[FolderGroup]:
    """Every repeated folder name under ``root``, grouped case-insensitively.

    ``name`` keeps only folders whose name contains it (case-insensitively),
    the ``find -iname '*name*'`` filter. Survivors are still grouped by their
    full casefolded name, so ``--name christmas`` never merges
    ``Christmas 2020`` into ``christmas``.

    An unreadable directory is reported through ``on_error`` and skipped. The
    caller decides whether the partial result is safe to act on; the CLI
    refuses to merge when discovery was incomplete.
    """
    root = os.path.abspath(root)
    needle = name.casefold() if name else None
    buckets: dict[str, list[str]] = {}

    def walk_error(exc: OSError) -> None:
        if on_error is not None:
            on_error(exc)

    for dirpath, dirnames, _filenames in os.walk(root, followlinks=False,
                                                 onerror=walk_error):
        kept: list[str] = []
        for child in sorted(dirnames):
            if child.startswith("."):
                continue
            full = os.path.join(dirpath, child)
            if os.path.islink(full):
                continue  # find -type d does not match symlinks
            if ignore and is_ignored(_rel(full, root).replace(os.sep, "/"), ignore):
                continue
            kept.append(child)
        dirnames[:] = kept

        for child in kept:
            key = child.casefold()
            if needle and needle not in key:
                continue
            buckets.setdefault(key, []).append(os.path.join(dirpath, child))

    groups = [
        FolderGroup(key, root, tuple(sorted(paths)))
        for key, paths in buckets.items()
        if len(paths) >= 2
    ]
    groups.sort(key=lambda g: (g.name.casefold(), g.members))
    return groups


def iter_entries(top: str):
    """Yield ``(path, relative_path)`` for every non-directory entry under top.

    Directory symlinks are yielded as entries (so a merge moves the link rather
    than silently abandoning it) and are never descended into.
    """
    stack = [top]
    while stack:
        directory = stack.pop()
        # Unlike os.walk, do not swallow this: a source that cannot be read
        # must abort its own plan, not silently merge half of it. The caller
        # reports the folder and leaves it untouched.
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
        descend: list[str] = []
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                descend.append(entry.path)
            else:
                yield entry.path, _rel(entry.path, top)
        stack.extend(reversed(descend))


def folder_stats(path: str) -> tuple[int, int]:
    """(file count, total bytes) under a folder, without following symlinks."""
    files = 0
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            files += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return files, total


def numbered_path(directory: str, name: str, *, keep_extension: bool = True,
                  reserved: set[str] | None = None) -> str:
    """``name`` in ``directory`` with the first free ``_N`` suffix.

    Files keep their extension (``photo_1.jpg``); folders do not
    (``christmas_1``), so a dotted folder name is not mistaken for a suffix.
    ``reserved`` holds destinations already claimed by moves that have been
    planned but not applied yet: two source folders can each contribute a
    ``same.txt``, and without this the second plan would reuse the first
    plan's ``same_1.txt`` and fail on apply.
    """
    if keep_extension:
        stem, ext = os.path.splitext(name)
    else:
        stem, ext = name, ""
    n = 1
    while True:
        candidate = os.path.join(directory, f"{stem}_{n}{ext}")
        free = not os.path.lexists(candidate)
        if free and not (reserved and candidate in reserved):
            return candidate
        n += 1


def _within(child: str, parent: str) -> bool:
    """True when ``child`` is strictly inside ``parent``.

    Paths come from one walk, so the comparison is textual; ``normcase`` keeps
    it correct where the platform folds case or separators.
    """
    child_n = os.path.normcase(os.path.abspath(child))
    parent_n = os.path.normcase(os.path.abspath(parent))
    return child_n != parent_n and child_n.startswith(parent_n + os.sep)


def is_inside(child: str, parent: str) -> bool:
    """True when ``child`` is strictly inside ``parent``.

    Moving a folder that another group member already contains would make the
    inner path vanish, so groups with nested members process the outer one and
    skip the inner.
    """
    return _within(child, parent)


def is_nested(first: str, second: str) -> bool:
    """True when either folder contains the other."""
    return _within(first, second) or _within(second, first)


def collapsed_dir(group: FolderGroup) -> str:
    """Where ``--collapse`` gathers a group's non-destination members."""
    return os.path.join(group.root, f"{group.name}_collapsed")


@dataclass
class Move:
    source: str
    destination: str
    action: str = "move"  # move | replace | rename | skip | ask
    reason: str = ""
    base: str = ""  # the un-numbered target, for renumbering at apply time


def _is_real_dir(path: str) -> bool:
    return os.path.isdir(path) and not os.path.islink(path)


def plan_folder(
    source: str,
    destination: str,
    *,
    policy: str | None = None,
    resolve: Callable[[str, str], str] | None = None,
    reserved: set[str] | None = None,
) -> list[Move]:
    """Plan the moves that merge ``source`` into ``destination``.

    ``policy`` is ``replace``, ``rename`` or ``skip`` for conflicts.
    ``policy="ask"`` marks conflicts for a dry-run display. With a policy of
    ``None`` and a ``resolve`` callback, each conflict is resolved by calling
    it; ``resolve`` may raise :class:`QuitReview`. Without either, conflicts
    default to skip, which is the safe non-interactive answer.
    """
    if policy not in (None, "replace", "rename", "skip", "ask"):
        raise ValueError(f"unknown conflict policy: {policy!r}")

    moves: list[Move] = []
    for path, rel in iter_entries(source):
        target = os.path.join(destination, rel)
        claimed = bool(reserved and target in reserved)
        if not os.path.lexists(target) and not claimed:
            moves.append(Move(path, target))
            if reserved is not None:
                reserved.add(target)
            continue
        if _is_real_dir(target) or _is_real_dir(path):
            moves.append(Move(path, target, "skip", "file/directory mismatch"))
            continue

        action = policy
        if action is None:
            action = resolve(path, target) if resolve is not None else "skip"
        if action not in ("replace", "rename", "skip", "ask"):
            raise ValueError(f"unknown conflict action: {action!r}")

        if action == "rename":
            original = target
            target = numbered_path(os.path.dirname(target),
                                   os.path.basename(target),
                                   reserved=reserved)
            moves.append(Move(path, target, "rename", base=original))
        elif action == "replace":
            moves.append(Move(path, target, "replace"))
        elif action == "ask":
            moves.append(Move(path, target, "ask", "conflict"))
            continue
        else:
            moves.append(Move(path, target, "skip", "conflict"))
            continue
        if reserved is not None:
            reserved.add(target)
    return moves


def _move_file(source: str, target: str, *, overwrite: bool) -> None:
    """Move one file, replacing the target when asked and clearing EXDEV."""
    if overwrite:
        if _is_real_dir(target):
            raise OSError("target is a directory")
        try:
            os.replace(source, target)
            return
        except OSError:
            # Cross-device replace is not atomic; drop the old file first so
            # shutil.move does not nest the source inside it.
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass
    elif os.path.lexists(target):
        raise OSError("target appeared during the run")
    try:
        os.rename(source, target)
    except OSError:
        shutil.move(source, target)


@dataclass
class ApplyResult:
    moved: list[Move]
    failed: list[tuple[Move, str]]

    @property
    def replaced(self) -> list[Move]:
        return [m for m in self.moved if m.action == "replace"]

    @property
    def renamed(self) -> list[Move]:
        return [m for m in self.moved if m.action == "rename"]


def apply_moves(moves: list[Move], *, root: str, out: Out = print) -> ApplyResult:
    """Carry out planned moves; report failures and keep going."""
    result = ApplyResult([], [])
    labels = {"replace": "replaced", "rename": "renamed"}
    for move in moves:
        if move.action in ("skip", "ask"):
            continue
        # Planning is buffered, so a numbered name chosen for an earlier
        # folder can be taken by the time this one applies (or by a file that
        # appeared during the run). Renumber rather than fail.
        if move.action in ("move", "rename") and os.path.lexists(move.destination):
            # Renumber from the un-numbered name: renumbering "photo_1.jpg"
            # would produce "photo_1_1.jpg" instead of "photo_2.jpg".
            original = move.base or move.destination
            move.destination = numbered_path(
                os.path.dirname(move.destination),
                os.path.basename(original),
            )
        parent = os.path.dirname(move.destination)
        try:
            os.makedirs(parent, exist_ok=True)
            _move_file(move.source, move.destination,
                       overwrite=move.action == "replace")
        except OSError as exc:
            out(f"  FAILED {_rel(move.source, root)}: {exc}")
            result.failed.append((move, str(exc)))
            continue
        result.moved.append(move)
        out(f"  {labels.get(move.action, 'moved')} "
            f"{_rel(move.source, root)} -> {_rel(move.destination, root)}")
    return result


def prune_empty_dirs(top: str) -> int:
    """Remove directories under (and including) ``top`` that are now empty.

    Files the user skipped keep their parent directories alive, because they
    are still there. Returns the number of directories removed.
    """
    removed = 0
    for dirpath, _dirnames, _filenames in os.walk(top, topdown=False):
        try:
            os.rmdir(dirpath)
        except OSError:
            continue
        removed += 1
    return removed


def prune_empty_ancestors(path: str, stop: str) -> int:
    """Remove empty parents of a merged-away folder, stopping before ``stop``.

    Merging ``2020/Christmas`` leaves an empty ``2020/``. It is tidy to remove
    it, but never climb past the root the user pointed us at.
    """
    removed = 0
    stop = os.path.abspath(stop)
    current = os.path.dirname(os.path.abspath(path))
    while current != stop and current.startswith(stop + os.sep):
        try:
            os.rmdir(current)
        except OSError:
            break
        removed += 1
        current = os.path.dirname(current)
    return removed


def move_folder(source: str, target: str) -> None:
    try:
        os.rename(source, target)
    except OSError:
        shutil.move(source, target)


def collapse_group(
    group: FolderGroup,
    *,
    out: Out = print,
    dry_run: bool = False,
) -> tuple[str, str, list[tuple[str, str]], list[tuple[str, str]]]:
    """Move every non-destination member into ``<root>/<Name>_collapsed``.

    Returns ``(destination, collapsed_dir, moved, failed)`` where moved and
    failed are ``(source, target)`` pairs. Numbering continues past anything
    already in the collapsed folder, so a second run does not overwrite the
    first run's folders.
    """
    root = group.root
    destination = group.suggested
    name = os.path.basename(destination)
    collapsed = collapsed_dir(group)

    moved: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    reserved: set[str] = set()
    if not dry_run:
        try:
            os.makedirs(collapsed, exist_ok=True)
        except OSError as exc:
            out(f"  FAILED {_rel(collapsed, root)}: {exc}")
            return destination, collapsed, moved, [(collapsed, str(exc))]

    for member in group.members:
        if member == destination:
            continue
        if is_nested(member, destination):
            out(f"  skipped {_rel(member, root)}: nested inside the destination")
            continue
        if any(is_inside(member, other) for other in group.members
               if other != member):
            out(f"  skipped {_rel(member, root)}: inside another member; "
                f"it moves with that folder")
            continue
        target = numbered_path(collapsed, name, keep_extension=False,
                               reserved=reserved)
        reserved.add(target)
        if dry_run:
            out(f"  would move {_rel(member, root)} -> {_rel(target, root)}")
            moved.append((member, target))
            continue
        try:
            move_folder(member, target)
        except OSError as exc:
            out(f"  FAILED {_rel(member, root)}: {exc}")
            failed.append((member, str(exc)))
            continue
        out(f"  moved {_rel(member, root)} -> {_rel(target, root)}")
        moved.append((member, target))
        parent = os.path.dirname(member)
        prune_empty_dirs(parent)
        if not os.path.exists(parent):
            prune_empty_ancestors(parent, root)
    return destination, collapsed, moved, failed


# -- review UI ---------------------------------------------------------------


@dataclass
class ReviewState:
    """Choices that persist across groups ("replace all", "skip all")."""

    policy: str | None = None


def _files_summary(path: str) -> str:
    files, total = folder_stats(path)
    return f"{files} files, {human_size(total)}"


def render_group(group: FolderGroup, index: int, total: int, out: Out) -> None:
    """Print one group in the same shape as a duplicate group."""
    destination = group.suggested
    out(f"\nGroup {index}/{total} — {group.name} ({group.size} folders)")
    for n, member in enumerate(group.members, 1):
        mark = "[dest*]" if member == destination else "       "
        out(f"  {n}. {mark} {_rel(member, group.root)}  "
            f"({_files_summary(member)})")
    out("  * suggested destination (shortest path)")


def choose_destination(
    group: FolderGroup,
    index: int,
    total: int,
    *,
    input_fn: Input | None = None,
    out: Out = print,
) -> str:
    """Ask which member receives the others; Enter accepts the suggestion."""
    input_fn = input_fn or input
    destination = group.suggested
    render_group(group, index, total, out)
    n = len(group.members)
    while True:
        try:
            answer = input_fn(
                f"  destination? [Enter]={_rel(destination, group.root)}, "
                f"1-{n}, q=quit > "
            ).strip()
        except EOFError:
            raise QuitReview(eof=True) from None
        if answer == "":
            return destination
        if answer in ("q", "Q") or answer.startswith("\x1b"):
            raise QuitReview
        if answer.isdigit() and 1 <= int(answer) <= n:
            return group.members[int(answer) - 1]
        out(f"  pick 1-{n}, Enter for the suggestion, or q to quit")


def _mtime(path: str) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime(
            "%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def _stat_line(path: str) -> str:
    """``size, modified when`` for one side of a conflict."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "unreadable"
    return f"{human_size(size)}, modified {_mtime(path)}"


def _short_error(exc: Exception) -> str:
    """The last stderr line of a failed subprocess, or a short message."""
    stderr = getattr(exc, "stderr", None)
    if isinstance(stderr, bytes):
        lines = stderr.decode("utf-8", "replace").strip().splitlines()
        if lines:
            return lines[-1]
    text = str(exc).strip()
    return text or type(exc).__name__


def _compare_facts(path: str) -> list[str]:
    """Size/mtime plus whatever EXIF or video metadata can be read."""
    facts = [_stat_line(path)]
    if facts[0] == "unreadable":
        return facts

    ext = os.path.splitext(path)[1].lower()
    if ext in _IMAGE_EXTS:
        try:
            from PIL import Image

            from .exif import read_exif

            with Image.open(path) as img:
                facts.append(f"{img.width}x{img.height}")
                exif = read_exif(img)
            if exif.get("captured_at"):
                facts.append(f"captured {exif['captured_at']}")
            if exif.get("camera"):
                facts.append(exif["camera"])
            if exif.get("gps_lat") is not None:
                facts.append(f"gps {exif['gps_lat']:.5f}, {exif['gps_lon']:.5f}")
        except Exception as exc:  # noqa: BLE001 - compare is best-effort
            facts.append(f"no EXIF ({type(exc).__name__})")
    elif ext in _VIDEO_EXTS:
        if not shutil.which("ffprobe"):
            facts.append("no video metadata (ffprobe not on PATH)")
        else:
            try:
                from .video import probe

                meta = probe(path)
                facts.append(f"{meta.width}x{meta.height}, "
                             f"{meta.duration:.1f}s {meta.codec}".strip())
            except Exception as exc:  # noqa: BLE001 - compare is best-effort
                facts.append(f"no video metadata ({_short_error(exc)})")
    return facts


_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp",
    ".heic", ".heif", ".avif", ".jxl",
}
_VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def compare_files(source: str, destination: str, *, root: str,
                  out: Out = print) -> None:
    """Show both sides of a conflict, then return to the prompt."""
    out(f"  compare {_rel(source, root)}")
    for label, path in (("destination", destination), ("source", source)):
        out(f"    {label:<12}{_rel(path, root)}")
        for fact in _compare_facts(path):
            out(f"      {fact}")
    out("")


def ask_conflict(
    source: str,
    destination: str,
    *,
    root: str,
    state: ReviewState,
    input_fn: Input | None = None,
    out: Out = print,
) -> str:
    """Ask one conflict; return replace/rename/skip, or raise QuitReview."""
    input_fn = input_fn or input
    while True:
        out(f"\n  conflict  {os.path.basename(destination)}")
        out(f"    destination  {_rel(destination, root)}  "
            f"({_stat_line(destination)})")
        out(f"    source       {_rel(source, root)}  "
            f"({_stat_line(source)})")
        try:
            answer = input_fn(
                "  [r]eplace [R]eplace all [n]ame [N]ame all [s]kip "
                "[S]kip all [c]ompare [q]uit > "
            ).strip()
        except EOFError:
            raise QuitReview(eof=True) from None

        if answer in ("q", "Q") or answer.startswith("\x1b"):
            raise QuitReview
        if answer == "":
            return "skip"  # Enter is always the safe answer
        if answer == "c":
            compare_files(source, destination, root=root, out=out)
            continue
        if answer == "R":
            state.policy = "replace"
            return "replace"
        if answer == "N":
            state.policy = "rename"
            return "rename"
        if answer == "S":
            state.policy = "skip"
            return "skip"
        lowered = answer.lower()
        if lowered in ("r", "n", "s"):
            return {"r": "replace", "n": "rename", "s": "skip"}[lowered]
        out("  r/R replace, n/N rename, s/S skip, c compare, q quit")


def make_resolver(*, root: str, state: ReviewState,
                  input_fn: Input | None = None, out: Out = print):
    """A per-conflict resolver that honours an earlier "all" choice."""
    input_fn = input_fn or input
    def resolve(source: str, destination: str) -> str:
        if state.policy is not None:
            return state.policy
        return ask_conflict(source, destination, root=root, state=state,
                            input_fn=input_fn, out=out)

    return resolve
