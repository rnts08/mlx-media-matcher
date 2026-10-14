"""Full-screen visual UI for reviewing duplicate groups.

A movable highlight bar (hjkl / arrow keys) over the files of every duplicate
group. Keys:

    j / down, k / up      move the bar
    h / left, l / right   jump to the previous / next group
    space                 keep the file under the bar (unselects the group's
                          others; pressing again deselects -> group untouched)
    a                     keep every file of the current group (nothing from
                          it will be trashed)
    A                     keep every file of every group
    v                     open the file under the bar in Quick Look
    p                     open the file under the bar in Preview.app
    x                     move the file under the bar to the Trash (y/n)
    enter                 execute: trash every unkept file of every group (y/n)
    e                     execute the decisions made so far: trash the unkept
                          files of every group from the first up to and
                          including the group under the bar (y/n), then stay
                          in the review so the remaining groups can continue
    y / n                 answer the prompt shown at the bottom
                          (x/enter/e/quit)
    q / esc               quit without executing — but asks y/N first whenever
                          files are still marked for the Trash

The full key legend is pinned at the top of the screen and the status line at
the bottom; only the group list scrolls. The algorithm's best-resolution file
of each group starts selected. Selected files survive `enter`; everything else
goes to the OS Trash (never unlinked). Groups marked as kept with `a` / `A`
(or whose keeper was deselected, or whose keeper was trashed with x) are left
untouched; `space` on a file reactivates its group. `e` commits the work done
so far (useful when quitting halfway through a long list): the already-made
decisions are executed, their groups become "done", and the review continues.

`CleanSession` is a pure state machine (keys in, state out) so it can be
unit-tested without a terminal; `run_tui` drives it in a real terminal.
"""

from __future__ import annotations

import os
import select
import shutil
import sys
from dataclasses import dataclass

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
    done: bool = False  # its decisions were already executed (key e / enter)

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
        self.confirm_gi: int | None = None  # group cutoff for the `e` confirm
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
        was_kept = group.keeper_id is None
        group.keeper_id = None if group.keeper_id == record.id else record.id
        name = os.path.basename(record.path)
        if group.keeper_id == record.id:
            self.status = (
                f"Keeping {name}; the group's other {group.size - 1} file(s) "
                "are now marked for trash."
                if was_kept
                else f"Keeping {name}."
            )
        else:
            self.status = f"Deselected {name}; this group will be left untouched."

    def keep_group(self) -> None:
        """Mark every file of the group under the bar as kept."""
        if not self.groups:
            return
        group = self.groups[self.group_of_flat(self.cursor)]
        group.keeper_id = None
        self.status = (
            f"Keeping all {group.size} file(s) of this group; "
            "nothing here will be trashed."
        )

    def keep_all(self) -> None:
        """Mark every file of every group as kept."""
        for group in self.groups:
            group.keeper_id = None
        self.status = "Keeping every file; nothing will be trashed."

    def open_viewer(self, app: str) -> None:
        files = self.flat_files()
        if not files:
            return
        record = files[self.cursor]
        self.opener([record.path], app)
        label = "Preview" if app == "preview" else "Quick Look"
        self.status = f"Opened {os.path.basename(record.path)} in {label}."

    def pending_trash(
        self, groups: list[GroupState] | None = None
    ) -> list[FileRecord]:
        return [
            record
            for group in (self.groups if groups is None else groups)
            if group.keeper_id is not None  # deselected groups are untouched
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

    def _execute_to_cursor(self, gi: int, pending: list[FileRecord]) -> None:
        """Execute the decisions of groups[0..gi] and keep reviewing.

        Used by the `e` key: commit the work done so far (e.g. before
        quitting halfway through a long list) without ending the session.
        """
        trashed = 0
        size = 0
        before = self.flat_files()
        anchor = before[self.cursor].id if before else None
        for record in pending:
            try:
                self.trasher(record)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{record.path}: {exc}")
                continue
            self.batch_trashed.append(record)
            self._forget(record)
            trashed += 1
            size += record.size  # file is already in the Trash; use index size
        if self.db is not None and trashed:
            self.db.delete_files([r.id for r in self.batch_trashed[-trashed:]])
        for group in self.groups[: gi + 1]:
            # only groups with nothing pending left are actually completed
            group.done = not self.pending_trash([group])
        # re-anchor the bar: it stays on its file if that file survived;
        # otherwise (it was one of the trashed copies) jump to the first
        # group still waiting for a decision, so the review continues
        flat = self.flat_files()
        if flat:
            survivors = [r.id for r in flat]
            if anchor in survivors:
                self.cursor = survivors.index(anchor)
            else:
                first_open = next(
                    (i for i, g in enumerate(self.groups) if not g.done), None
                )
                target = (
                    first_open
                    if first_open is not None
                    else len(self.groups) - 1
                )
                self.cursor = min(
                    self._flat_group_ranges()[target][0], len(flat) - 1
                )
        failed = len(pending) - trashed
        note = f" ({failed} failed)" if failed else ""
        remaining = sum(1 for g in self.groups if self.pending_trash([g]))
        self.status = (
            f"Trashed {trashed} file(s) ({human_size(size)}){note} from the "
            f"first {gi + 1} group(s); {remaining} group(s) still marked."
        )

    def handle(self, key: str) -> None:
        if self.confirm:
            kind, payload = self.confirm
            self.confirm = None
            if key.lower() == "y":
                if kind == "x":
                    self._trash_now(payload[0])
                elif kind == "quit":
                    self.done = True
                elif kind == "partial":
                    self._execute_to_cursor(self.confirm_gi or 0, payload)
                else:
                    self._execute(payload)
            elif kind == "quit":
                self.status = (
                    f"Not exiting — {len(payload)} file(s) still marked for "
                    "the Trash. e = to the bar, enter = all, q = quit."
                )
            else:
                self.status = "Cancelled; nothing changed."
            return

        files = self.flat_files()
        if key != "A":  # A (keep everything) is case-sensitive; the rest is not
            key = key.lower()
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
        elif key == "a":
            self.keep_group()
        elif key == "A":
            self.keep_all()
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
                self.status = "Nothing to trash — every file is kept."
            else:
                self.confirm = ("execute", pending)
        elif key == "e":
            gi = self.group_of_flat(self.cursor) if files else -1
            pending = self.pending_trash(self.groups[: gi + 1])
            if gi < 0 or not pending:
                self.status = (
                    "Nothing pending in the groups up to the bar — "
                    "nothing to execute."
                )
            else:
                self.confirm_gi = gi
                self.confirm = ("partial", pending)
        elif key == "q":
            pending = self.pending_trash()
            if pending:
                # files are still marked for the Trash: leaving now would
                # silently discard the session's decisions — confirm first
                self.confirm = ("quit", pending)
            else:
                self.done = True

    # -- rendering --------------------------------------------------------------

    def render(self, width: int = 100, height: int = 30) -> str:
        total_files = sum(g.size for g in self.groups)
        # Fixed chrome: title + the key legend, pinned at the top of the screen
        # so the hotkeys never scroll out of view.
        top = [
            f"{_BOLD}media-match clean{_RESET} — {os.path.abspath(self.root)}  "
            f"({len(self.groups)} group(s), {total_files} file(s))",
            "",
            f"{_DIM}j/k or ↑/↓ move · h/l or ←/→ group · space keep file · "
            f"a keep group · A keep all{_RESET}",
            f"{_DIM}v Quick Look · p Preview · x trash file · "
            f"enter execute all · e execute to bar{_RESET}",
            f"{_DIM}q/esc quit · y/n answer prompts — e commits the work done "
            f"so far, then keeps reviewing{_RESET}",
            "",
        ]

        # Scrolling region: the duplicate groups.
        body: list[str] = []
        flat = self.flat_files()
        cursor_record_id = flat[self.cursor].id if flat else None
        for gi, group in enumerate(self.groups):
            kept = group.keeper_id is None
            extra = (
                0
                if kept
                else sum(
                    file_size(r) for r in group.files if r.id != group.keeper_id
                )
            )
            if group.done and extra == 0:
                tail = "done"
            elif kept:
                tail = "kept"
            else:
                tail = f"{human_size(extra)} pending"
            body.append(
                f"{_DIM}Group {gi + 1}/{len(self.groups)} — {group.size} "
                f"file(s), {tail}{_RESET}"
            )
            for record in group.files:
                is_cursor = record.id == cursor_record_id
                is_keeper = kept or record.id == group.keeper_id
                mark = (
                    f"{_GREEN}✔ keep{_RESET}      "
                    if is_keeper
                    else f"{_RED}→ trash{_RESET}    "
                )
                arrow = "❯" if is_cursor else " "
                name = os.path.relpath(record.path, self.root)
                meta = (f"{record.width}x{record.height}, "
                        f"{human_size(file_size(record))}")
                line = f" {arrow} {mark} {name}  ({meta})"
                if is_cursor:
                    line = f" {arrow} {_REVERSE}{mark} {name}  ({meta}){_RESET}"
                body.append(line)

        # Fixed footer: prompt (if confirming) + status line, pinned at bottom.
        footer: list[str] = []
        if self.confirm:
            kind, payload = self.confirm
            if kind == "x":
                footer.append(
                    f"{_BOLD}Move {os.path.basename(payload[0].path)} to the "
                    f"Trash? [y/N]{_RESET}"
                )
            elif kind == "quit":
                size = sum(file_size(r) for r in payload)
                footer.append(
                    f"{_BOLD}Exit without executing? {len(payload)} file(s) "
                    f"({human_size(size)}) are still marked for the Trash. "
                    f"[y/N]{_RESET}"
                )
            elif kind == "partial":
                size = sum(file_size(r) for r in payload)
                footer.append(
                    f"{_BOLD}Trash {len(payload)} unkept file(s) "
                    f"({human_size(size)}) from the first "
                    f"{(self.confirm_gi or 0) + 1} group(s) (up to the bar)? "
                    f"[y/N]{_RESET}"
                )
            else:
                size = sum(file_size(r) for r in payload)
                footer.append(
                    f"{_BOLD}Trash {len(payload)} unkept file(s) "
                    f"({human_size(size)})? [y/N]{_RESET}"
                )
        footer.append(self.status)

        # window the group list to the terminal height, keeping the bar visible
        avail = max(1, height - len(top) - 1 - len(footer))
        if len(body) > avail:
            cursor_row = next(
                (i for i, r in enumerate(body)
                 if "❯" in r), 0
            )
            offset = 0 if cursor_row < avail else cursor_row - avail + 1
            body = body[offset : offset + avail]

        rows = top + body + [""] + footer
        clipped = [r if len(r) < width else r[: width - 1] for r in rows]
        return "\r\n".join(clipped)


# -- terminal plumbing ----------------------------------------------------------

_pending = b""  # bytes an ESC probe swallowed, replayed by the next _read_key


def _read_key() -> str:
    global _pending
    # Read bytes straight from the fd: raw mode + no Python buffering, and a
    # bare ESC can be distinguished from an escape sequence with a short
    # select (read(2) would block forever waiting for bytes that never come).
    fd = sys.stdin.fileno()
    if _pending:
        data, _pending = _pending[:1], _pending[1:]
    else:
        data = os.read(fd, 1)
    if not data:
        return "q"
    if data == b"\x1b":
        rest = b""
        if select.select([fd], [], [], 0.05)[0]:
            rest = os.read(fd, 2)
        if rest[:1] == b"[":  # e.g. "[B" for the down arrow
            return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                rest[1:2].decode("ascii", "replace"), ""
            )
        # not an escape sequence: bare ESC quits, but any bytes it swallowed
        # (esc pressed right before another key) are replayed on the next call
        _pending = rest
        return "q"
    ch = data.decode("utf-8", "replace")
    if ch == " ":
        return "space"
    if ch in ("\r", "\n"):
        return "enter"
    return ch  # case is preserved: handle() decides what is case-sensitive


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
