import pytest

from media_match import cli


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
