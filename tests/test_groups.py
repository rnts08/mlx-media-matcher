"""--exact-phash, and the lint that would have caught its crash.

Two bugs lived in the same block of find_duplicate_groups: it appended to a
`groups` list that is only defined further down the function, so the flag
raised UnboundLocalError on every call; and it sorted a group's rejects by a
loop variable leaked from a comprehension. Neither was caught because no test
exercised exact_phash=True.
"""

import numpy as np
import pytest
from PIL import Image

from media_match.db import Database
from media_match.groups import find_duplicate_groups


def _write(path, size, color=(120, 30, 200)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


class _FakeEncoder:
    def embed_images(self, arrays):
        return np.stack([np.ones(4, dtype=np.float32) for _ in arrays])


def _records_with_same_content(tmp_path, sizes):
    """Several resolutions of one image: identical pHash, different bytes."""
    from media_match.db import FileRecord
    from media_match.hashing import hash_to_hex, phash
    from media_match.pipeline import load_image_tolerant

    records = []
    base = tmp_path / "base.png"
    _write(base, (300, 200))
    with load_image_tolerant(str(base)) as img:
        shared = hash_to_hex(phash(img))
    for i, size in enumerate(sizes):
        p = tmp_path / f"v{i}.png"
        Image.open(base).resize(size).save(p)
        with load_image_tolerant(str(p)) as img:
            h = hash_to_hex(phash(img))
        records.append(FileRecord(
            id=i + 1, path=str(p), width=size[0], height=size[1],
            size=p.stat().st_size, mtime=0, sha1=f"sha{i}", phash=h,
            captured_at="", camera="", description="", keywords="",
        ))
    return records


def test_exact_phash_groups_identical_hashes(tmp_path):
    records = _records_with_same_content(tmp_path, [(400, 300), (800, 600)])
    groups = find_duplicate_groups(records, exact_phash=True)
    assert len(groups) == 1
    assert groups[0].size == 2


def test_exact_phash_does_not_crash_on_an_empty_index():
    assert find_duplicate_groups([], exact_phash=True) == []


def test_exact_phash_marks_the_largest_as_best(tmp_path):
    records = _records_with_same_content(
        tmp_path, [(400, 300), (1600, 1200), (800, 600)])
    group = find_duplicate_groups(records, exact_phash=True)[0]
    assert group.best.width == 1600
    assert group.best.height == 1200


def test_exact_phash_orders_rejects_largest_first(tmp_path):
    records = _records_with_same_content(
        tmp_path, [(400, 300), (1600, 1200), (800, 600)])
    group = find_duplicate_groups(records, exact_phash=True)[0]
    widths = [r.width for r, _d in group.others]
    assert widths == sorted(widths, reverse=True), \
        "rejects must be ordered by resolution, not by a leaked loop variable"


def test_exact_phash_reports_each_file_once(tmp_path):
    # The old code appended exact-phash groups and then fell through to the
    # Hamming pass, which grouped the same files a second time.
    records = _records_with_same_content(tmp_path, [(400, 300), (800, 600)])
    groups = find_duplicate_groups(records, threshold=8, exact_phash=True)
    seen = [g.best.path for g in groups] + [
        r.path for g in groups for r, _d in g.others]
    assert len(seen) == len(set(seen)), "a file appeared in two groups"


def test_exact_phash_ignores_near_but_not_identical_hashes(tmp_path):
    from media_match.db import FileRecord

    records = [
        FileRecord(id=1, path="/a.png", width=10, height=10, size=1, mtime=0,
                   sha1="a", phash="0" * 16, captured_at="", camera="",
                   description="", keywords=""),
        FileRecord(id=2, path="/b.png", width=10, height=10, size=1, mtime=0,
                   sha1="b", phash="1" * 16, captured_at="", camera="",
                   description="", keywords=""),
    ]
    # threshold=0 would match these by Hamming distance, but --exact-phash
    # promises distance exactly 0 and nothing else.
    assert find_duplicate_groups(records, threshold=0, exact_phash=True) == []


def test_threshold_path_still_groups_near_duplicates(tmp_path):
    records = _records_with_same_content(tmp_path, [(400, 300), (800, 600)])
    groups = find_duplicate_groups(records, threshold=8)
    assert len(groups) == 1 and groups[0].size == 2


def test_no_module_has_an_undefined_name():
    """pyflakes F821: the bug class that broke every video and --exact-phash.

    A missing import inside a function is invisible until that exact branch
    runs. `scan` with --video and `duplicates --exact-phash` were both broken
    this way while the test suite stayed green.
    """
    import ast
    import builtins
    import pathlib

    import media_match

    # Names that exist at module scope without being imported or assigned.
    implicit = {"__file__", "__name__", "__doc__", "__package__",
                "__spec__", "__loader__", "__builtins__", "__annotations__"}

    root = pathlib.Path(media_match.__file__).parent
    offenders = []
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        known = set(dir(builtins)) | implicit
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    known.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    known.add(alias.asname or alias.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef)):
                known.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                known.add(node.id)
            elif isinstance(node, ast.arg):
                known.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                known.add(node.name)
            elif isinstance(node, (ast.comprehension, ast.Global, ast.Nonlocal)):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Name):
                        known.add(sub.id)

        for node in ast.walk(tree):
            if (isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id not in known):
                offenders.append(
                    f"{path.name}:{node.lineno} undefined name '{node.id}'")
    assert not offenders, "undefined names:\n  " + "\n  ".join(offenders)