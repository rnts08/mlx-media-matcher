"""Folder-level consolidation: discovery, planning, merging and collapse."""

import io
import json
import os

import pytest
from PIL import Image

from media_match import cli
from media_match.folders import (
    QuitReview,
    ReviewState,
    apply_moves,
    ask_conflict,
    choose_destination,
    collapse_group,
    collapsed_dir,
    compare_files,
    find_folder_groups,
    folder_stats,
    make_resolver,
    numbered_path,
    plan_folder,
    prune_empty_ancestors,
    prune_empty_dirs,
    suggest_destination,
)


def _write(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _christmas_tree(tmp_path):
    """a/christmas is the shallowest, b/c/d are case variants."""
    root = tmp_path / "tree"
    _write(root / "a" / "christmas" / "keep.txt", "dest")
    _write(root / "b" / "Christmas" / "keep.txt", "src1")
    _write(root / "b" / "Christmas" / "only1.txt", "1")
    _write(root / "c" / "CHRISTMAS" / "keep.txt", "src2")
    _write(root / "c" / "CHRISTMAS" / "only2.txt", "2")
    return root


def _scripted(answers):
    iterator = iter(answers)

    def answer(_prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return answer


# -- discovery ---------------------------------------------------------------


def test_finds_same_named_folders_regardless_of_case(tmp_path):
    root = _christmas_tree(tmp_path)
    groups = find_folder_groups(str(root))
    assert len(groups) == 1
    group = groups[0]
    assert group.key == "christmas"
    assert group.size == 3
    assert [os.path.relpath(m, root) for m in group.members] == [
        "a/christmas", "b/Christmas", "c/CHRISTMAS"]


def test_singletons_are_not_groups(tmp_path):
    root = tmp_path / "tree"
    _write(root / "a" / "holiday" / "x.txt")
    _write(root / "a" / "christmas" / "x.txt")
    assert find_folder_groups(str(root)) == []


def test_name_filter_matches_substrings_but_groups_exact_names(tmp_path):
    root = tmp_path / "tree"
    _write(root / "a" / "christmas" / "x.txt")
    _write(root / "b" / "CHRISTMAS" / "x.txt")
    # a different folder name that merely contains the needle
    _write(root / "c" / "Christmas 2020" / "x.txt")

    groups = find_folder_groups(str(root), name="christm")
    assert [g.name for g in groups] == ["christmas"]
    assert groups[0].size == 2, "Christmas 2020 is a different folder name"

    assert find_folder_groups(str(root), name="2020") == []
    assert find_folder_groups(str(root), name="summer") == []


def test_hidden_and_ignored_directories_are_skipped(tmp_path):
    root = tmp_path / "tree"
    _write(root / "a" / "christmas" / "x.txt")
    _write(root / "b" / "Christmas" / "x.txt")
    _write(root / ".cache" / "Christmas" / "x.txt")
    _write(root / "vendor" / "CHRISTMAS" / "x.txt")

    groups = find_folder_groups(str(root), ignore=["vendor"])
    assert len(groups) == 1
    assert all(".cache" not in m and "vendor" not in m for m in groups[0].members)


def test_directory_symlinks_are_not_followed(tmp_path):
    root = tmp_path / "tree"
    _write(root / "a" / "christmas" / "x.txt")
    _write(root / "b" / "Christmas" / "x.txt")
    try:
        os.symlink(root / "a" / "christmas", root / "c" / "CHRISTMAS")
    except OSError:  # pragma: no cover - platform without symlink support
        pytest.skip("symlinks unavailable")

    assert find_folder_groups(str(root))[0].size == 2


def test_suggested_destination_is_the_shallowest(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]
    assert group.suggested == str(root / "a" / "christmas")
    assert group.name == "christmas"


def test_suggestion_breaks_ties_deterministically(tmp_path):
    root = tmp_path / "tree"
    _write(root / "aa" / "x" / "one.txt")
    _write(root / "bb" / "X" / "two.txt")
    group = find_folder_groups(str(root))[0]
    # Same depth and same relative length; lexicographic decides.
    assert group.suggested == str(root / "aa" / "x")
    assert suggest_destination(list(reversed(group.members)), str(root)) == \
        group.suggested


def test_folder_stats_counts_files_and_bytes(tmp_path):
    folder = tmp_path / "f"
    _write(folder / "a.txt", "12345")
    _write(folder / "sub" / "b.txt", "12")
    assert folder_stats(str(folder)) == (2, 7)


# -- planning and applying ---------------------------------------------------


def test_plan_preserves_subdirectories_and_moves_clean_files(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]
    plan = plan_folder(group.members[1], group.members[0])

    assert {(os.path.relpath(m.source, group.members[1]),
             os.path.relpath(m.destination, group.members[0]), m.action)
            for m in plan} == {
        ("keep.txt", "keep.txt", "skip"),
        ("only1.txt", "only1.txt", "move"),
    }


def test_plan_actions_replace_rename_and_skip(tmp_path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    _write(source / "sub" / "same.txt", "src")
    _write(destination / "sub" / "same.txt", "dst")

    replace = plan_folder(str(source), str(destination), policy="replace")
    assert replace[0].action == "replace"
    assert replace[0].destination == str(destination / "sub" / "same.txt")

    rename = plan_folder(str(source), str(destination), policy="rename")
    assert rename[0].action == "rename"
    assert rename[0].destination == str(destination / "sub" / "same_1.txt")

    skip = plan_folder(str(source), str(destination), policy="skip")
    assert skip[0].action == "skip" and skip[0].reason == "conflict"


def test_ask_policy_marks_conflicts_for_dry_run(tmp_path):
    source = tmp_path / "src"
    destination = tmp_path / "dst"
    _write(source / "same.txt")
    _write(destination / "same.txt")
    plan = plan_folder(str(source), str(destination), policy="ask")
    assert plan[0].action == "ask"


def test_numbering_keeps_file_extensions_but_not_folder_suffixes(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert numbered_path(str(d), "photo.jpg") == str(d / "photo_1.jpg")
    assert numbered_path(str(d), "v1.2", keep_extension=False) == \
        str(d / "v1.2_1")
    _write(d / "photo_1.jpg")
    assert numbered_path(str(d), "photo.jpg") == str(d / "photo_2.jpg")


def test_plans_reserve_destinations_across_source_folders(tmp_path):
    """Two sources with the same name must not both plan the same _1 name."""
    destination = tmp_path / "dst"
    first = tmp_path / "one"
    second = tmp_path / "two"
    for folder in (destination, first, second):
        folder.mkdir()
    _write(destination / "same.txt", "dest")
    _write(first / "same.txt", "one")
    _write(second / "same.txt", "two")

    reserved: set[str] = set()
    p1 = plan_folder(str(first), str(destination), policy="rename",
                     reserved=reserved)
    p2 = plan_folder(str(second), str(destination), policy="rename",
                     reserved=reserved)
    assert p1[0].destination.endswith("same_1.txt")
    assert p2[0].destination.endswith("same_2.txt")


def test_apply_moves_replaces_renames_and_reports(tmp_path):
    destination = tmp_path / "dst"
    source = tmp_path / "src"
    destination.mkdir()
    source.mkdir()
    _write(destination / "keep.txt", "dest")
    _write(source / "keep.txt", "src")
    _write(source / "new.txt", "new")

    plan = plan_folder(str(source), str(destination), policy="replace")
    result = apply_moves(plan, root=str(tmp_path), out=lambda _t: None)
    assert not result.failed
    assert (destination / "keep.txt").read_text() == "src"
    assert (destination / "new.txt").read_text() == "new"
    assert len(result.replaced) == 1


def test_apply_renumbers_if_the_chosen_name_appeared_late(tmp_path):
    destination = tmp_path / "dst"
    source = tmp_path / "src"
    destination.mkdir()
    source.mkdir()
    _write(destination / "photo.jpg", "dest")
    _write(source / "photo.jpg", "src")

    plan = plan_folder(str(source), str(destination), policy="rename")
    _write(destination / "photo_1.jpg", "appeared")  # race after planning
    result = apply_moves(plan, root=str(tmp_path), out=lambda _t: None)

    assert not result.failed
    assert (destination / "photo_1.jpg").read_text() == "appeared"
    assert (destination / "photo_2.jpg").read_text() == "src"


def test_file_directory_mismatch_is_skipped_not_forced(tmp_path):
    destination = tmp_path / "dst"
    source = tmp_path / "src"
    _write(destination / "thing" / "inside.txt")
    _write(source / "thing", "a file where the destination has a folder")

    plan = plan_folder(str(source), str(destination), policy="replace")
    assert len(plan) == 1
    assert plan[0].action == "skip"
    assert "mismatch" in plan[0].reason
    assert (source / "thing").is_file()


def test_prune_removes_only_empty_directories(tmp_path):
    root = tmp_path / "src"
    _write(root / "moved" / "a.txt")
    _write(root / "kept" / "b.txt")
    os.unlink(root / "moved" / "a.txt")

    assert prune_empty_dirs(str(root)) == 1
    assert not (root / "moved").exists()
    assert (root / "kept" / "b.txt").exists()


def test_prune_ancestors_stops_at_the_root(tmp_path):
    root = tmp_path / "tree"
    empty_chain = root / "2020" / "christmas"
    empty_chain.mkdir(parents=True)
    os.rmdir(empty_chain)  # what prune_empty_dirs does to the source folder

    assert prune_empty_ancestors(str(empty_chain), str(root)) == 1
    assert not (root / "2020").exists()
    assert root.exists(), "the root itself must never be removed"


# -- collapse ----------------------------------------------------------------


def test_collapse_moves_non_destination_folders_into_a_numbered_gather(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]

    destination, collapsed, moved, failed = collapse_group(
        group, out=lambda _t: None)

    assert destination == str(root / "a" / "christmas")
    assert collapsed == str(root / "christmas_collapsed")
    assert not failed and len(moved) == 2
    assert (root / "a" / "christmas" / "keep.txt").exists()
    assert (root / "christmas_collapsed" / "christmas_1" / "keep.txt").read_text() \
        == "src1"
    assert (root / "christmas_collapsed" / "christmas_2" / "keep.txt").read_text() \
        == "src2"
    # the empty parents are gone, the destination is untouched
    assert not (root / "b").exists()
    assert not (root / "c").exists()


def test_collapse_continues_numbering_on_a_second_run(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]
    collapse_group(group, out=lambda _t: None)

    # a third copy appears after the first collapse
    _write(root / "d" / "christmas" / "keep.txt", "src3")
    group2 = find_folder_groups(str(root))[0]
    _destination, _collapsed, moved, failed = collapse_group(
        group2, out=lambda _t: None)

    assert not failed
    assert moved[0][1].endswith("christmas_3")


def test_collapse_dry_run_numbers_without_writing(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]
    lines: list[str] = []
    _destination, _collapsed, moved, failed = collapse_group(
        group, out=lines.append, dry_run=True)

    assert not failed and len(moved) == 2
    assert [os.path.basename(target) for _src, target in moved] == [
        "christmas_1", "christmas_2"]
    assert not os.path.exists(collapsed_dir(group))
    assert (root / "b" / "Christmas" / "keep.txt").exists()


# -- review ------------------------------------------------------------------


def test_choose_destination_accepts_default_and_numbers(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]

    assert choose_destination(group, 1, 1, input_fn=_scripted([""]),
                              out=lambda _t: None) == group.suggested
    assert choose_destination(group, 1, 1, input_fn=_scripted(["3"]),
                              out=lambda _t: None) == group.members[2]


def test_choose_destination_quits_on_q_and_eof(tmp_path):
    root = _christmas_tree(tmp_path)
    group = find_folder_groups(str(root))[0]
    with pytest.raises(QuitReview):
        choose_destination(group, 1, 1, input_fn=_scripted(["q"]),
                           out=lambda _t: None)
    with pytest.raises(QuitReview):
        choose_destination(group, 1, 1, input_fn=_scripted([]),
                           out=lambda _t: None)


@pytest.mark.parametrize("answer,expected", [
    ("r", "replace"), ("n", "rename"), ("s", "skip"), ("", "skip"),
])
def test_conflict_single_keys(tmp_path, answer, expected):
    state = ReviewState()
    assert ask_conflict("s", "d", root=str(tmp_path), state=state,
                        input_fn=_scripted([answer]),
                        out=lambda _t: None) == expected


@pytest.mark.parametrize("answer,policy", [
    ("R", "replace"), ("N", "rename"), ("S", "skip"),
])
def test_conflict_all_keys_set_the_run_policy(tmp_path, answer, policy):
    state = ReviewState()
    ask_conflict("s", "d", root=str(tmp_path), state=state,
                 input_fn=_scripted([answer]), out=lambda _t: None)
    assert state.policy == policy


def test_conflict_compare_then_choose(tmp_path):
    source = _write(tmp_path / "src" / "a.jpg", "x")
    destination = _write(tmp_path / "dst" / "a.jpg", "y")
    lines: list[str] = []
    answer = ask_conflict(str(source), str(destination), root=str(tmp_path),
                          state=ReviewState(), input_fn=_scripted(["c", "s"]),
                          out=lines.append)

    assert answer == "skip"
    assert any("compare" in line for line in lines)
    assert any("destination" in line for line in lines)


def test_conflict_quit_raises(tmp_path):
    with pytest.raises(QuitReview):
        ask_conflict("s", "d", root=str(tmp_path), state=ReviewState(),
                     input_fn=_scripted(["\x1b"]), out=lambda _t: None)


def test_resolver_honours_replace_all_without_asking(tmp_path):
    state = ReviewState(policy="replace")

    def never(_prompt):  # pragma: no cover - must not be reached
        raise AssertionError("prompted after 'replace all'")

    resolve = make_resolver(root=str(tmp_path), state=state,
                            input_fn=never, out=lambda _t: None)
    assert resolve("s", "d") == "replace"


def test_compare_prints_exif_when_present(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b.jpg"
    for path, make in ((a, "Apple"), (b, "Canon")):
        img = Image.new("RGB", (40, 30))
        exif = img.getexif()
        exif[271] = make
        exif[272] = "iPhone 12" if make == "Apple" else "EOS"
        exif[306] = "2020:12:24 18:30:12"
        img.save(path, exif=exif)

    lines: list[str] = []
    compare_files(str(b), str(a), root=str(tmp_path), out=lines.append)
    text = "\n".join(lines)
    assert "iPhone 12" in text and "EOS" in text
    assert "captured 2020-12-24T18:30:12" in text


def test_compare_handles_a_non_image(tmp_path):
    a = _write(tmp_path / "a", "hello")
    b = _write(tmp_path / "b", "hi")
    lines: list[str] = []
    compare_files(str(b), str(a), root=str(tmp_path), out=lines.append)
    assert any("2 B" in line for line in lines)


# -- CLI ---------------------------------------------------------------------


def _run(argv):
    return cli.main(argv)


def test_cli_list_reports_the_groups(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--list"]) == 0
    out = capsys.readouterr().out
    assert "Group 1/1 — christmas (3 folders)" in out
    assert "[dest*] a/christmas" in out
    assert "2 redundant folder(s)" in out


def test_cli_list_json_is_machine_readable(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "christmas"
    assert payload[0]["suggested"] == "a/christmas"
    assert [m["path"] for m in payload[0]["members"]] == [
        "a/christmas", "b/Christmas", "c/CHRISTMAS"]


def test_cli_replace_yes_merges_and_overwrites(tmp_path):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--replace", "--yes"]) == 0
    assert (root / "a" / "christmas" / "keep.txt").read_text() == "src2"
    assert (root / "a" / "christmas" / "only1.txt").exists()
    assert (root / "a" / "christmas" / "only2.txt").exists()
    assert not (root / "b").exists()
    assert not (root / "c").exists()


def test_cli_rename_yes_keeps_every_conflicting_file(tmp_path):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--rename", "--yes"]) == 0
    base = root / "a" / "christmas"
    assert (base / "keep.txt").read_text() == "dest"
    assert (base / "keep_1.txt").read_text() == "src1"
    assert (base / "keep_2.txt").read_text() == "src2"


def test_cli_yes_without_a_policy_skips_conflicts(tmp_path):
    # 'yes to everything' must never guess between overwrite and rename.
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--yes"]) == 0
    base = root / "a" / "christmas"
    assert (base / "keep.txt").read_text() == "dest"
    assert (base / "only1.txt").exists()   # unambiguous files still move
    assert (root / "b" / "Christmas" / "keep.txt").read_text() == "src1"
    assert (root / "c" / "CHRISTMAS" / "keep.txt").read_text() == "src2"


def test_cli_dry_run_changes_nothing(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert _run(["folders", str(root), "--dry-run"]) == 0
    after = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert before == after
    out = capsys.readouterr().out
    assert "would ask" in out
    assert "Dry run: nothing was moved." in out


def test_cli_dry_run_json_reports_actions(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--replace", "--dry-run", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "christmas"
    actions = {move["action"] for entry in payload[0]["sources"]
               for move in entry["moves"]}
    assert "replace" in actions and "move" in actions


def test_cli_collapse_numbers_the_gather(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--collapse"]) == 0
    assert (root / "christmas_collapsed" / "christmas_1" / "keep.txt").exists()
    assert (root / "christmas_collapsed" / "christmas_2" / "keep.txt").exists()
    assert "2 folder(s) gathered" in capsys.readouterr().out


def test_cli_collapse_json(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--collapse", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["collapsed"] == "christmas_collapsed"
    assert [os.path.basename(m["to"]) for m in payload[0]["moved"]] == [
        "christmas_1", "christmas_2"]


def test_cli_rejects_collapse_with_a_file_policy(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--collapse", "--replace"]) == 2
    assert "never touches files" in capsys.readouterr().out


def test_cli_rejects_json_without_a_reporting_mode(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--json", "--yes"]) == 2
    assert "--json needs" in capsys.readouterr().out


def test_cli_reports_no_groups(tmp_path, capsys):
    root = tmp_path / "tree"
    _write(root / "only" / "x.txt")
    assert _run(["folders", str(root)]) == 0
    assert "No same-named folders" in capsys.readouterr().out


def test_cli_name_filter(tmp_path, capsys):
    root = _christmas_tree(tmp_path)
    assert _run(["folders", str(root), "--list", "--name", "christm"]) == 0
    assert "christmas" in capsys.readouterr().out
    assert _run(["folders", str(root), "--list", "--name", "summer"]) == 0
    assert "No repeated folder names containing 'summer'" in \
        capsys.readouterr().out


def test_cli_quit_keeps_completed_folders_only(tmp_path, monkeypatch,
                                              capsys):
    root = tmp_path / "tree"
    _write(root / "a" / "christmas" / "one.txt", "1")
    _write(root / "b" / "Christmas" / "two.txt", "2")
    _write(root / "a" / "summer" / "x.txt", "x")
    _write(root / "b" / "Summer" / "y.txt", "y")

    monkeypatch.setattr("sys.stdin", io.StringIO("1\nq\ny\n"))
    assert _run(["folders", str(root)]) == 0

    out = capsys.readouterr().out
    assert "excluding the current folder" not in out  # quit at a group, not a file
    assert "Execute everything decided so far?" in out
    assert (root / "a" / "christmas" / "two.txt").exists()
    assert (root / "b" / "Summer" / "y.txt").exists(), \
        "the group the user quit on must be untouched"


def test_cli_quit_mid_folder_excludes_it(tmp_path, monkeypatch, capsys):
    root = _christmas_tree(tmp_path)
    # destination 1, replace the first conflict, quit at the second, keep work
    monkeypatch.setattr("sys.stdin", io.StringIO("1\nr\nq\ny\n"))
    assert _run(["folders", str(root)]) == 0

    out = capsys.readouterr().out
    assert "excluding the current folder (c/CHRISTMAS)" in out
    assert (root / "a" / "christmas" / "only1.txt").exists()
    assert (root / "a" / "christmas" / "keep.txt").read_text() == "src1"
    assert (root / "c" / "CHRISTMAS" / "keep.txt").read_text() == "src2"


def test_cli_eof_leaves_everything_alone(tmp_path, monkeypatch):
    root = _christmas_tree(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("1\n"))
    assert _run(["folders", str(root)]) == 0
    assert (root / "b" / "Christmas" / "keep.txt").read_text() == "src1"
    assert not (root / "a" / "christmas" / "keep_1.txt").exists()


def test_nested_group_member_is_moved_with_its_parent(tmp_path):
    root = tmp_path / "tree"
    _write(root / "a" / "x" / "one.txt", "1")
    _write(root / "b" / "x" / "top.txt", "top")
    _write(root / "b" / "x" / "x" / "inner.txt", "inner")

    group = find_folder_groups(str(root))[0]
    assert group.size == 3  # a/x, b/x and b/x/x

    lines: list[str] = []
    _destination, collapsed, moved, failed = collapse_group(
        group, out=lines.append)
    assert not failed
    assert len(moved) == 1, "b/x/x travels inside b/x"
    assert (root / "x_collapsed" / "x_1" / "top.txt").exists()
    assert (root / "x_collapsed" / "x_1" / "x" / "inner.txt").exists()
    assert any("inside another member" in line for line in lines)


def test_nested_group_member_merges_recursively(tmp_path):
    root = tmp_path / "tree"
    _write(root / "a" / "x" / "one.txt", "1")
    _write(root / "b" / "x" / "top.txt", "top")
    _write(root / "b" / "x" / "x" / "inner.txt", "inner")

    assert _run(["folders", str(root), "--yes"]) == 0
    destination = root / "a" / "x"
    assert (destination / "top.txt").exists()
    assert (destination / "x" / "inner.txt").exists()
    assert not (root / "b").exists()
