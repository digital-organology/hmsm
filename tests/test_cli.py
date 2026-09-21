# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import pytest

from hmsm.cli import main


def test_no_command_prints_help_and_fails(capsys):
    assert main(["hmsm"]) == 1
    assert "roll2midi" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["hmsm", "--version"])
    assert "hmsm" in capsys.readouterr().out


def test_profiles_lists_the_presets(capsys):
    assert main(["hmsm", "profiles"]) == 0
    out = capsys.readouterr().out
    assert "phonola" in out and "ariston" in out


def test_profiles_can_be_filtered(capsys):
    assert main(["hmsm", "profiles", "-m", "cluster"]) == 0
    out = capsys.readouterr().out
    assert "ariston" in out and "phonola" not in out


def test_a_bad_profile_name_fails_cleanly(tmp_path, caplog):
    code = main(
        ["hmsm", "roll2midi", "-c", "nope", "in.tif", str(tmp_path / "out.mid")]
    )
    assert code == 1
    assert "nope" in caplog.text


def test_a_missing_input_file_fails_cleanly(tmp_path, caplog):
    code = main(
        [
            "hmsm",
            "roll2midi",
            "-c",
            "phonola",
            str(tmp_path / "absent.tif"),
            str(tmp_path / "out.mid"),
        ]
    )
    assert code == 1
    assert "absent.tif" in caplog.text


def test_roll2midi_writes_a_midi_file(roll_scan, tmp_path):
    out = tmp_path / "out.mid"
    code = main(
        ["hmsm", "roll2midi", "-c", "phonola", "-t", "60", str(roll_scan), str(out)]
    )
    assert code == 0
    assert out.exists() and out.stat().st_size > 0


def test_legacy_entry_points_reach_the_same_command(capsys):
    from hmsm.cli import roll2midi

    with pytest.raises(SystemExit):
        roll2midi(["roll2midi", "--help"])
    assert "Digitize a piano roll" in capsys.readouterr().out
