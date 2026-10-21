"""User mistakes get one neat line, never a stack trace.

A typo in the root, a `--db` in a folder that does not exist, a `--db` that is
not a database, or Ctrl-C must all answer with a message and an exit code.
Anything that would previously escaped as a traceback from `main()` is either
guarded before the filesystem is touched (root checks) or converted by the
dispatch net (`_run_command`).
"""

import contextlib
import io
import sqlite3

import pytest
from PIL import Image

from media_match import cli


def _tree(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(root / "a.png")
    return root


def _main(argv):
    """Run main() capturing both streams; return (code, out, err)."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
    except SystemExit as exc:  # argparse exits for usage errors
        code = exc.code
    return code, out.getvalue(), err.getvalue()


MISSPELLED = "definitely-not-here"


@pytest.mark.parametrize("argv", [
    ["scan"],
    ["analyze"],
    ["check"],
    ["bootstrap"],
    ["folders"],
    ["about", "bench"],
])
def test_misspelled_root_names_the_folder(argv, tmp_path):
    code, out, err = _main([*argv, str(tmp_path / MISSPELLED)])
    assert code == 2
    assert f"No folder at {tmp_path / MISSPELLED}." in err
    assert out == ""  # nothing that --json callers could mistake for data
    assert "Traceback" not in err


def test_scan_root_that_is_a_file(tmp_path):
    code, out, err = _main(["scan", str(tmp_path / "a.png")])
    assert code == 2
    assert "No folder at" in err
    assert out == ""


def test_bad_db_location(tmp_path):
    root = _tree(tmp_path)
    bad = tmp_path / "no" / "such" / "dir" / "idx.db"
    code, out, err = _main(["scan", str(root), "--db", str(bad)])
    assert code == 2
    assert "cannot open the index database" in err
    assert "unable to open database file" in err
    assert out == "" and "Traceback" not in err


def test_db_that_is_not_a_database(tmp_path):
    root = _tree(tmp_path)
    bogus = tmp_path / "bogus.db"
    bogus.write_text("not a database")
    code, _out, err = _main(["duplicates", str(root), "--db", str(bogus)])
    assert code == 2
    assert "cannot open the index database" in err
    assert "file is not a database" in err
    assert "Traceback" not in err


def test_no_index_failure_keeps_json_stdout_empty(tmp_path):
    """A text failure on stdout would corrupt `--json` output."""
    root = tmp_path / "tree"
    root.mkdir()
    code, out, err = _main(["stats", str(root), "--json"])
    assert code == 1
    assert out == ""
    assert "No index at" in err


def test_unwritable_out_file(tmp_path):
    root = _tree(tmp_path)
    assert cli.main(["scan", str(root), "--quick"]) == 0
    bad = tmp_path / "no" / "such" / "dir" / "out.json"
    code, _out, err = _main(["export", str(root), "--out", str(bad)])
    assert code == 1
    assert "No such file or directory" in err
    assert str(bad) in err
    assert "Traceback" not in err


def test_keyboard_interrupt_exits_130(tmp_path, monkeypatch):
    root = _tree(tmp_path)

    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_export", interrupted)
    code, _out, err = _main(["export", str(root)])
    assert code == 130
    assert "Interrupted." in err


def test_oserror_from_command_becomes_a_message(tmp_path, monkeypatch):
    root = _tree(tmp_path)

    def denied(_args):
        raise PermissionError(13, "Permission denied", "/locked/x.db")

    monkeypatch.setattr(cli, "cmd_export", denied)
    code, _out, err = _main(["export", str(root)])
    assert code == 1
    assert err == "Permission denied: /locked/x.db\n"


def test_database_bug_still_raises(tmp_path, monkeypatch):
    """A constraint failure is a bug, not bad input; it must stay loud."""

    def buggy(_args):
        raise sqlite3.IntegrityError("UNIQUE constraint failed: files.path")

    monkeypatch.setattr(cli, "cmd_export", buggy)
    with pytest.raises(sqlite3.IntegrityError):
        cli.main(["export", str(_tree(tmp_path))])


def test_unknown_command_is_argparse_neat(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["scna"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "Traceback" not in err


def test_unknown_flag_is_argparse_neat(tmp_path):
    root = _tree(tmp_path)
    code, _out, err = _main(["scan", str(root), "--limitt", "5"])
    assert code == 2
    assert "unrecognized arguments" in err
    assert "Traceback" not in err
