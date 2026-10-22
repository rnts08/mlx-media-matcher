import io
import contextlib
import pathlib
import re

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
    assert "grouped by what they are for" in out
    for name in ("scan", "search", "duplicates", "clean", "refine", "people"):
        assert name in out


def test_help_groups_commands_by_workflow_not_alphabetically(capsys):
    # An alphabetical list answers "what is this called?"; the point of the
    # grouping is to answer "what do I want?", so the order must not be sorted.
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    positions = [out.index(f"  {n} ") for n in
                 ("help", "about", "info", "version", "stats")]
    assert positions == sorted(positions), "first group is not in reading order"
    # a command whose name sorts early must not appear first
    assert out.index("  scan ") > out.index("  help ")


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
    assert "Unknown command" in capsys.readouterr().err


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


def test_readme_documents_only_flags_that_exist():
    # The README quotes flags in backticks; a flag that does not exist is a
    # documentation bug that sends people to a command that cannot work.
    _commands()
    real = {
        option
        for parser in cli._SUBPARSERS.values()
        for action in parser._actions
        for option in action.option_strings
    }
    readme = pathlib.Path(__file__).resolve().parent.parent / "README.md"
    quoted = set(re.findall(r"`(--[a-z0-9-]+)`", readme.read_text()))
    assert not sorted(quoted - real), f"README documents flags that do not exist: {sorted(quoted - real)}"


def test_every_flag_explains_itself():
    # A flag whose help is empty renders as "--limit LIMIT (default: 40)",
    # which tells you nothing about what the flag does.
    _commands()
    for name in _commands():
        for action in cli._SUBPARSERS[name]._actions:
            if not action.option_strings or action.dest == "help":
                continue
            assert (action.help or "").strip(), \
                f"{name} {action.option_strings[0]} has no help text"


def test_flag_lines_have_no_ragged_spacing():
    # "batch size " + "  (default: 16)" renders with a three-space gap.
    _commands()
    for name in _commands():
        for flags, text in cli._flag_lines(cli._SUBPARSERS[name]):
            assert text == text.strip(), f"{name} {flags}: {text!r}"
            assert "  (" not in text or "   (" not in text, f"{name} {flags}"


def test_every_command_appears_in_exactly_one_guide_group():
    grouped = [n for _h, _p, names in cli._COMMAND_GROUPS for n in names]
    assert sorted(grouped) == sorted(cli._COMMAND_SUMMARIES)
    # 'check' and 'export' each appeared under two headings in the first
    # draft of the grouping, which left it ambiguous where to look for them.
    assert len(grouped) == len(set(grouped)), "a command is listed twice"
    assert not set(grouped) - set(cli._COMMAND_SUMMARIES)


def test_main_refuses_to_start_with_a_misgrouped_command(monkeypatch):
    # The grouping is data next to the parsers, so it can silently drift. Make
    # a mistake fail loudly at startup rather than hide a command.
    monkeypatch.setitem(cli._COMMAND_SUMMARIES, "phantom-command", "not real")
    try:
        with pytest.raises(RuntimeError, match="phantom-command"):
            cli.main(["help"])
    finally:
        cli._COMMAND_SUMMARIES.pop("phantom-command")


def test_main_refuses_to_start_with_a_duplicated_command():
    saved = cli._COMMAND_GROUPS
    groups = list(saved)
    groups.append(("duplicated on purpose", "", ("scan",)))
    cli._COMMAND_GROUPS = tuple(groups)
    try:
        with pytest.raises(RuntimeError, match="more than one guide group"):
            cli.main(["help"])
    finally:
        cli._COMMAND_GROUPS = saved


def test_readme_commands_table_is_grouped_like_the_cli():
    # The README and `media-match help` must agree on the grouping, or one of
    # them is showing people a map of a tool the other one does not have.
    readme = (pathlib.Path(__file__).resolve().parent.parent
              / "README.md").read_text()
    section = readme[readme.index("## Commands"):
                     readme.index("All commands take `--db`")]

    readme_groups = [line[4:].lower() for line in section.splitlines()
                     if line.startswith("### ")]
    # The README uses markdown Title Case for its headings; the guide prints
    # them inline in lower case. The order and membership must still match.
    guide_groups = [h.lower() for h, _p, _n in cli._COMMAND_GROUPS]
    assert readme_groups == guide_groups

    # every command the guide lists has a README row
    for _h, _p, names in cli._COMMAND_GROUPS:
        for name in names:
            assert f"| `{name}" in section, f"{name} has no README row"


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


def test_analyze_help_says_every_stage_runs_by_default(capsys):
    """`analyze` is the complete pass; the old summary promised that but the
    implementation made each later stage opt-in. It now runs them all."""
    assert cli.main(["help", "analyze"]) == 0
    out = capsys.readouterr().out
    assert "complete analysis pass" in out
    assert "every undescribed item is captioned" in out
    assert "every untyped image is classified" in out
    assert "No stage flags are needed" in out
    # compatibility flags are accepted but not advertised
    assert "--describe " not in out
    assert "--refine " not in out
    assert "--classify" not in out


def test_classify_has_its_own_command(capsys):
    assert cli.main(["help", "classify"]) == 0
    out = capsys.readouterr().out
    assert "classify —" in out
    assert "zero-shot CLIP" in out
    assert "--recompute" in out


def test_search_help_matches_its_clip_default(capsys):
    assert cli.main(["help", "search"]) == 0
    out = capsys.readouterr().out
    assert "by default" in out
    assert "--fts switches" in out
    assert "With words, results come from the captions" not in out
