import pytest

from media_match import cli


def test_info_lists_support_details(capsys, tmp_path):
    assert cli.main(["info", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Support development of media-match" in out
    assert "0x6e8e3c2b31424266e7cff59e910df1587c317427" in out
    assert "bc1qzzvcguvqjc6qhwe2y5vy38w2zke7hksukjhm68" in out
    assert "MPfm5QLKH1r9XxgWmH75Gyps4LDfX5c53L" in out
    assert "GEaCMpnyM8tB5BU4RMuLm6tgMr3q9FgMHodxDxxAGby" in out
    assert "buymeacoffee.com/timhbergsta" in out
    assert "ko-fi.com/rnts08" in out


def test_about_leads_with_blurb_and_license(capsys):
    assert cli.main(["about"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Local photo-library analyzer built on Apple's MLX.")
    assert "CLIP vector search" in out
    assert "License text" in out


def test_pager_honours_PAGER(monkeypatch):
    monkeypatch.setenv("PAGER", "cat -v")
    assert cli._pager() == ["cat", "-v"]


def test_page_prints_without_tty(capsys, monkeypatch):
    monkeypatch.delenv("PAGER", raising=False)
    cli._page("hello page")
    assert "hello page" in capsys.readouterr().out
