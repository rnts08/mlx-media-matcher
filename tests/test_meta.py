import pytest

from media_match import cli


def test_info_lists_support_details(capsys, tmp_path):
    # 0.4.x legacy alias: still runnable, but the guide points at 'about'.
    assert cli.main(["info", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Support development of media-match" in out
    assert "0x6e8e3c2b31424266e7cff59e910df1587c317427" in out
    assert "bc1qzzvcguvqjc6qhwe2y5vy38w2zke7hksukjhm68" in out
    assert "MPfm5QLKH1r9XxgWmH75Gyps4LDfX5c53L" in out
    assert "GEaCMpnyM8tB5BU4RMuLm6tgMr3q9FgMHodxDxxAGby" in out
    assert "buymeacoffee.com/timhbergsta" in out
    assert "ko-fi.com/rnts08" in out


def test_about_merges_version_info_and_support(capsys, tmp_path):
    # One command answers everything the old about/version/info trio did.
    assert cli.main(["about", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Local photo-library analyzer built on Apple's MLX.")
    assert "Version:      " in out
    assert "Model engine: CLIP on Apple MLX" in out
    assert "Last scan:    no index at this folder yet" in out
    assert "Free storage: " in out
    assert "Support development of media-match" in out
    assert "License text" in out


def test_about_rejects_two_paths(capsys, tmp_path):
    # 'about bench <folder>' is the benchmark form (covered in test_bench);
    # two folder paths is a mistake and must not silently ignore the second.
    assert cli.main(["about", str(tmp_path), str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "not two" in err


def test_about_names_a_typod_folder(capsys):
    # Same rule as 'scan <typo>': a missing folder is an error, not an
    # "no index yet" report.
    assert cli.main(["about", "hekp-typo"]) == 2
    assert "No folder at" in capsys.readouterr().err


def test_global_info_about_flags_show_merged_about(capsys, tmp_path):
    for argv in (["--about"], ["--info"], ["--info", str(tmp_path)]):
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        assert out.startswith(
            "Local photo-library analyzer built on Apple's MLX.")
        assert "Free storage: " in out


def test_version_alias_still_prints_short_form(capsys):
    # 0.4.x scripts parse this output; keep it stable and short.
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("media-match ")
    assert "License text" not in out


def test_guide_hides_legacy_aliases_and_lists_stats_in_maintenance(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert "  info " not in out
    assert "  version " not in out
    maintenance = out[out.index("index maintenance"):]
    assert "  export " in maintenance
    assert "  reset " in maintenance
    assert "  stats " in maintenance


def test_about_leads_with_blurb_and_license(capsys):
    assert cli.main(["about"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Local photo-library analyzer built on Apple's MLX.")
    assert "CLIP vector search" in out
    assert "License text" in out


def test_help_has_no_program_blurb(capsys):
    assert cli.main(["help"]) == 0
    out = capsys.readouterr().out
    assert "grouped by what they are for" in out
    assert "Local photo-library analyzer" not in out
    assert "group duplicates across resolutions" not in out


def test_pager_honours_PAGER(monkeypatch):
    monkeypatch.setenv("PAGER", "cat -v")
    assert cli._pager() == ["cat", "-v"]


def test_page_prints_without_tty(capsys, monkeypatch):
    monkeypatch.delenv("PAGER", raising=False)
    cli._page("hello page")
    assert "hello page" in capsys.readouterr().out
