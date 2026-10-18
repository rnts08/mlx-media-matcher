import io
import contextlib

import pytest

from media_match import cli


def _run(argv):
    """Run main() and return (exit_code, stdout) without letting it escape."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = cli.main(argv)
    except SystemExit as exc:  # argparse exits for --help
        rc = exc.code
    return rc, buf.getvalue()


def _commands():
    cli.main(["help"])  # populates the parser registry
    return sorted(cli._SUBPARSERS)


def test_help_prints_full_guide(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert "commands:" in out
    for name in ("scan", "search", "duplicates", "clean", "refine", "people"):
        assert name in out


def test_help_shows_one_command(capsys):
    assert cli.main(["help", "scan"]) == 0
    out = capsys.readouterr().out
    assert "scan —" in out
    assert "positional arguments:" in out
    assert "options:" in out
    assert "example:" in out
    assert "--refine" in out


def test_help_unknown_command(capsys):
    assert cli.main(["help", "bogus"]) == 1
    assert "Unknown command" in capsys.readouterr().out


def test_top_level_help_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "scan" in out
    assert "usage: media-match" in out


def test_every_command_answers_help_in_all_three_forms():
    # 'cmd --help', 'cmd help' and 'help cmd' must be interchangeable.
    for name in _commands():
        forms = [_run([name, "--help"]), _run([name, "help"]), _run(["help", name])]
        assert [rc for rc, _ in forms] == [0, 0, 0], name
        assert len({out for _, out in forms}) == 1, f"{name}: forms differ"


def test_every_command_help_page_is_complete():
    # A page that is only a usage line is not a help page: it needs prose
    # explaining what the command does and a runnable example.
    for name in _commands():
        if name == "help":
            continue  # describes the help command, not a folder operation
        _rc, out = _run(["help", name])
        assert f"{name} —" in out, name
        assert "example:" in out, f"{name}: no example"
        assert "usage: media-match" in out, name


def test_no_command_is_listed_without_a_parser():
    # main() rebuilds the registry on every run, so asserting on the dict
    # after the fact proves nothing. Instead plant a summary with no parser
    # behind it and check that main() refuses to start.
    cli._COMMAND_SUMMARIES["no-such-command"] = "not a real command"
    try:
        with pytest.raises(RuntimeError, match="no-such-command"):
            cli.main(["help"])
    finally:
        cli._COMMAND_SUMMARIES.pop("no-such-command")


def test_command_help_flag_does_not_open_an_index(tmp_path):
    # Regression: the -h action used to run the command, so 'info --help'
    # would go looking for a database instead of printing help.
    root = tmp_path / "tree"
    root.mkdir()
    assert _run(["info", "--help", str(root)])[1].startswith("info —")
    assert _run(["info", str(root)])[0] == 0
