"""Full-screen visual UI for reviewing duplicate groups.

A movable highlight bar (hjkl / arrow keys) over the files of every duplicate
group. Keys:

    j / down, k / up      move the bar
    h / left, l / right   jump to the previous / next group
    space                 keep the file under the bar (unselects the group's
                          others; pressing again deselects -> group untouched)
    v                     open the file under the bar in Quick Look
    p                     open the file under the bar in Preview.app
    x                     move the file under the bar to the Trash (y/n)
    enter                 execute: trash every unkept file of every group (y/n)
    q / esc               quit without executing

The algorithm's best-resolution file of each group starts selected. Selected
files survive `enter`; everything else goes to the OS Trash (never unlinked).
Groups whose keeper was deselected (or whose keeper was trashed with x) are
left untouched.

`CleanSession` is a pure state machine (keys in, state out) so it can be
unit-tested without a terminal; `run_tui` drives it in a real terminal.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field

from .cleanup import file_size, human_size, open_in_viewer
from .db import Database, FileRecord
from .groups import DuplicateGroup

_RESET = "\x1b[0m"
_REVERSE = "\x1b[7m"
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"


def _default_trasher(record: FileRecord) -> None:
    if not os.path.exists(record.path):
        return
    from send2trash import send2trash

    send2trash(record.path)


@dataclass
class GroupState:
    files: list[FileRecord]
    keeper_id: int | None

    @property
    def size(self) -> int:
        return len(self.files)


def _ordered_files(group: DuplicateGroup) -> list[FileRecord]:
    files = [group.best] + [record for record, _ in group.others]
    return sorted(files, key=lambda r: (r.width * r.height, r.size), reverse=True)


class CleanSession:
    """State machine behind the visual dedup UI."""

    def __init__(
        self,
        groups: list[DuplicateGroup],
        root: str,
        db: Database | None = None,
        opener=open_in_viewer,
        trasher=_default_trasher,
    ):
        self.root = root
        self.db = db
        self.opener = opener
        self.trasher = trasher
        self.groups = [
            GroupState(_ordered_files(g), keeper_id=g.best.id) for g in groups
        ]
        self.cursor = 0
        self.confirm: tuple[str, list[FileRecord]] | None = None
        self.done = False
        self.x_trashed: list[FileRecord] = []
        self.batch_trashed: list[FileRecord] = []
        self.errors: list[str] = []
        self.status = "Ready. The algorithm's best copy of each group is selected."

    # -- geometry -------------------------------------------------------------

    def flat_files(self) -> list[FileRecord]:
        return [record for group in self.groups for record in group.files]

    def group_of_flat(self, flat_index: int) -> int:
        seen = 0
        for gi, group in enumerate(self.groups):
            seen += group.size
            if flat_index < seen:
                return gi
        return len(self.groups) - 1

    def _flat_group_ranges(self) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        for group in self.groups:
            ranges.append((start, start + group.size))
            start += group.size
        return ranges

    # -- actions ---------------------------------------------------------------

    def move(self, delta: int) -> None:
        total = len(self.flat_files())
        if total:
            self.cursor = max(0, min(total - 1, self.cursor + delta))

    def jump_group(self, direction: int) -> None:
        if not self.groups:
            return
        gi = self.group_of_flat(self.cursor)
        target = gi + direction
        if 0 <= target < len(self.groups):
            self.cursor = self._flat_group_ranges()[target][0]

    def toggle_keep(self) -> None:
        files = self.flat_files()
        if not files:
            return
        record = files[self.cursor]
        gi = self.group_of_flat(self.cursor)
        group = self.groups[gi]
        group.keeper_id = None if group.keeper_id == record.id else record.id
        name = os.path.basename(record.path)
        self.status = (
            f"Keeping {name}." if group.keeper_id == record.id
            else f"Deselected {name}; this group will be left untouched."
        )

    def open_viewer(self, app: str) -> None:
        files = self.flat_files()
        if not files:
            return
        record = files[self.cursor]
        self.opener([record.path], app)
        label = "Preview" if app == "preview" else "Quick Look"
        self.status = f"Opened {os.path.basename(record.path)} in {label}."

    def pending_trash(self) -> list[FileRecord]:
        return [
            record
            for group in self.groups
            if group.keeper_id is not None  # deselected groups are left untouched
            for record in group.files
            if record.id != group.keeper_id
        ]

    def _forget(self, record: FileRecord) -> None:
        """Remove a record from the display state; fix cursor/keeper."""
        files = self.flat_files()
        flat_index = next(i for i, f in enumerate(files) if f.id == record.id)
        gi = self.group_of_flat(flat_index)
        group = self.groups[gi]
        group.files = [f for f in group.files if f.id != record.id]
        if group.keeper_id == record.id:
            group.keeper_id = None
        if not group.files:
            self.groups.pop(gi)
        total = len(self.flat_files())
        self.cursor = max(0, min(self.cursor, total - 1)) if total else 0

    def _trash_now(self, record: FileRecord) -> None:
        try:
            self.trasher(record)
        except Exception as exc:  # noqa: BLE001 - keep going, report per file
            self.errors.append(f"{record.path}: {exc}")
            self.status = f"FAILED {os.path.basename(record.path)}: {exc}"
            return
        self.x_trashed.append(record)
        if self.db is not None:
            self.db.delete_files([record.id])
        self._forget(record)
        self.status = (
            f"Moved {os.path.basename(record.path)} to the Trash "
            f"(recoverable until emptied)."
        )

    def _execute(self, pending: list[FileRecord]) -> None:
        for record in pending:
            try:
                self.trasher(record)
                self.batch_trashed.append(record)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{record.path}: {exc}")
        if self.db is not None and self.batch_trashed:
            self.db.delete_files([r.id for r in self.batch_trashed])
        self.done = True

    def handle(self, key: str) -> None:
        if self.confirm:
            kind, payload = self.confirm
            self.confirm = None
            if key == "y":
                if kind == "x":
                    self._trash_now(payload[0])
                else:
                    self._execute(payload)
            else:
                self.status = "Cancelled; nothing changed."
            return

        files = self.flat_files()
        if key in ("j", "down"):
            self.move(1)
        elif key in ("k", "up"):
            self.move(-1)
        elif key in ("h", "left"):
            self.jump_group(-1)
        elif key in ("l", "right"):
            self.jump_group(1)
        elif key == "space":
            self.toggle_keep()
        elif key == "v":
            self.open_viewer("quicklook")
        elif key == "p":
            self.open_viewer("preview")
        elif key == "x" and files:
            record = files[self.cursor]
            self.confirm = ("x", [record])
        elif key == "enter":
            pending = self.pending_trash()
            if not pending:
                self.status = "Nothing to trash — every group has a keeper."
            else:
                self.confirm = ("execute", pending)
        elif key == "q":
            self.done = True

    # -- rendering --------------------------------------------------------------

    def render(self, width: int = 100, height: int = 30) -> str:
        rows: list[str] = []  # (style, text) flattened to styled strings
        total_files = sum(g.size for g in self.groups)
        rows.append(
            f"{_BOLD}media-match clean{_RESET} — {os.path.abspath(self.root)}  "
            f"({len(self.groups)} group(s), {total_files} file(s))"
        )
        rows.append("")

        flat = self.flat_files()
        cursor_record_id = flat[self.cursor].id if flat else None
        for gi, group in enumerate(self.groups):
            extra = sum(file_size(r) for r in group.files if r.id != group.keeper_id)
            rows.append(
                f"{_DIM}Group {gi + 1}/{len(self.groups)} — {group.size} file(s), "
                f"{human_size(extra)} pending{_RESET}"
            )
            for record in group.files:
                is_cursor = record.id == cursor_record_id
                is_keeper = record.id == group.keeper_id
                if is_keeper:
                    mark, style = f"{_GREEN}✔ keep{_RESET}      ", _GREEN
                else:
                    mark, style = f"{_RED}→ trash{_RESET}    ", _RED
                arrow = "❯" if is_cursor else " "
                name = os.path.relpath(record.path, self.root)
                meta = (f"{record.width}x{record.height}, "
                        f"{human_size(file_size(record))}")
                line = f" {arrow} {mark} {name}  ({meta})"
                if is_cursor:
                    line = f" {arrow} {_REVERSE}{mark} {name}  ({meta}){_RESET}"
                rows.append(line)
        rows.append("")

        if self.confirm:
            kind, payload = self.confirm
            if kind == "x":
                rows.append(
                    f"{_BOLD}Move {os.path.basename(payload[0].path)} to the "
                    f"Trash? [y/N]{_RESET}"
                )
            else:
                size = sum(file_size(r) for r in payload)
                rows.append(
                    f"{_BOLD}Trash {len(payload)} unkept file(s) "
                    f"({human_size(size)})? [y/N]{_RESET}"
                )
        else:
            rows.append(f"{_DIM}j/k or ↑/↓ move · h/l or ←/→ group · "
                        f"space keep · v Quick Look · p Preview · "
                        f"x trash file · enter dedupe · q quit{_RESET}")
        rows.append(f"{self.status}")

        # window the rows to the terminal height, keeping the cursor visible
        avail = max(4, height - 4)
        body = rows
        offset = 0
        if len(body) > avail:
            cursor_row = next(
                (i for i, r in enumerate(body)
                 if "❯" in r), 0
            )
            if cursor_row >= avail:
                offset = cursor_row - avail + 1
            body = body[offset : offset + avail]
        clipped = [r if len(r) < width else r[: width - 1] for r in body]
        return "\r\n".join(clipped)


# -- terminal plumbing ----------------------------------------------------------


def _read_key() -> str:
    ch = sys.stdin.read(1)
    if ch == "":
        return "q"
    if ch == "\x1b":
        rest = sys.stdin.read(2)
        if rest == "[":
            return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                sys.stdin.read(1), ""
            )
        return "q"  # bare ESC
    if ch == " ":
        return "space"
    if ch in ("\r", "\n"):
        return "enter"
    return ch.lower()


def run_tui(
    groups: list[DuplicateGroup],
    root: str,
    db: Database | None = None,
) -> CleanSession | None:
    """Drive a CleanSession in the terminal. Returns None if not a TTY."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    import termios
    import tty

    session = CleanSession(groups, root=root, db=db)
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not session.done:
            size = shutil.get_terminal_size((100, 30))
            screen = "\x1b[H\x1b[2J" + session.render(size.columns, size.lines)
            sys.stdout.write(screen)
            sys.stdout.flush()
            key = _read_key()
            if key:
                session.handle(key)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
    return session
