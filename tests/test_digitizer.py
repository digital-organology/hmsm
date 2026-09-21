# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import json

import numpy as np
import pytest

from hmsm.io import ArraySource
from hmsm.profiles import RollProfile
from hmsm.rolls import PaperModel, RollDigitizer
from hmsm.rolls.digitizer import find_roll_start
from hmsm.rolls.edges import RollEdges


@pytest.fixture
def profile():
    """A profile matching the ``synthetic_roll`` fixture.

    The paper spans columns 20 to 179, so a hole at column 60 sits at
    (60 - 20) / 160 = 0.25 of the roll width and one at 120 sits at 0.625.
    Measurements are given against a nominal 160 mm roll so that a millimetre
    is a pixel.
    """
    return RollProfile.from_dict(
        {
            "roll_width_mm": 160.0,
            "binarization_method": "paper_relative",
            "hole_width_mm": 1.0,
            "track_measurements": [
                {"left": 40.0, "right": 52.0, "tone": 60},
                {"left": 100.0, "right": 112.0, "tone": 72},
            ],
        }
    )


def test_a_synthetic_roll_transcribes_to_the_expected_notes(synthetic_roll, profile):
    transcription = RollDigitizer(profile, band_height=150).run(
        ArraySource(synthetic_roll)
    )
    assert set(transcription.notes[:, 2]) == {60, 72}
    assert len(transcription.notes) == 8
    assert transcription.notes[:, 0].min() == 0


def test_band_height_does_not_change_the_result(synthetic_roll, profile):
    runs = [
        RollDigitizer(profile, band_height=h).run(ArraySource(synthetic_roll))
        for h in (23, 73, 150, 400, 10_000)
    ]
    for other in runs[1:]:
        np.testing.assert_array_equal(runs[0].notes, other.notes)


def test_read_ahead_does_not_change_the_result(synthetic_roll, profile):
    eager = RollDigitizer(profile, band_height=97, read_ahead=True)
    lazy = RollDigitizer(profile, band_height=97, read_ahead=False)
    np.testing.assert_array_equal(
        eager.run(ArraySource(synthetic_roll)).notes,
        lazy.run(ArraySource(synthetic_roll)).notes,
    )


def test_skipping_rows_drops_the_holes_above_them(synthetic_roll, profile):
    whole = RollDigitizer(profile, band_height=150).run(ArraySource(synthetic_roll))
    trimmed = RollDigitizer(profile, band_height=150).run(
        ArraySource(synthetic_roll), skip_rows=200
    )
    assert len(trimmed.notes) < len(whole.notes)


def test_a_scan_with_no_holes_is_an_error(profile):
    blank = np.zeros((400, 200, 3), np.uint8)
    blank[:, 20:180] = 200
    with pytest.raises(ValueError, match="No holes"):
        RollDigitizer(profile, band_height=150).run(ArraySource(blank))


def test_an_invalid_background_is_rejected(profile):
    with pytest.raises(ValueError, match="black"):
        RollDigitizer(profile, background="chartreuse")


def test_an_invalid_band_height_is_rejected(profile):
    with pytest.raises(ValueError, match="positive"):
        RollDigitizer(profile, band_height=0)


def test_transcription_renders_to_midi(synthetic_roll, profile, tmp_path):
    transcription = RollDigitizer(profile, band_height=150).run(
        ArraySource(synthetic_roll)
    )
    path = str(tmp_path / "out.mid")
    transcription.to_midi(tempo=60).write(path)

    import mido

    notes = [m for m in mido.MidiFile(path).tracks[0] if m.type == "note_on"]
    assert len(notes) == len(transcription.notes)


def test_background_detection_reads_the_scan(synthetic_roll):
    from hmsm.rolls.paper import sample_scan

    dark = PaperModel.estimate(sample_scan(ArraySource(synthetic_roll)))
    light = PaperModel.estimate(sample_scan(ArraySource(255 - synthetic_roll)))
    assert dark.background_name == "black"
    assert light.background_name == "white"


def test_a_white_background_scan_transcribes_to_the_same_notes(synthetic_roll, profile):
    # The same roll photographed over a white bed rather than a black one:
    # the holes are now the lightest thing in the scan instead of the
    # darkest, and nothing about the format has changed.
    inverted = 255 - synthetic_roll
    on_black = RollDigitizer(profile, band_height=150).run(ArraySource(synthetic_roll))
    on_white = RollDigitizer(profile, band_height=150).run(ArraySource(inverted))
    np.testing.assert_array_equal(on_black.notes, on_white.notes)


def test_roll_start_is_zero_for_a_straight_roll():
    edges = RollEdges(np.full(1000, 20, np.int32), np.full(1000, 180, np.int32))
    assert find_roll_start(edges, 200) == 0


def test_roll_start_skips_a_tapering_head():
    left = np.concatenate((np.linspace(95, 20, 200), np.full(800, 20)))
    right = np.concatenate((np.linspace(105, 180, 200), np.full(800, 180)))
    edges = RollEdges(left.astype(np.int32), right.astype(np.int32))
    start = find_roll_start(edges, 200)
    assert start is not None and 100 <= start <= 400


def test_roll_start_misses_a_head_that_tapers_too_gently():
    # Known weakness of the current heuristic: it only looks at how far the
    # edges move within a hundred row window, so a long shallow taper reads as
    # straight roll. This is what puts stray notes at the start of a
    # transcription, and is scheduled for rework.
    left = np.concatenate((np.linspace(95, 20, 2000), np.full(2000, 20)))
    right = np.concatenate((np.linspace(105, 180, 2000), np.full(2000, 180)))
    edges = RollEdges(left.astype(np.int32), right.astype(np.int32))
    assert find_roll_start(edges, 200) == 0


def test_debug_artefacts_are_written_when_asked(synthetic_roll, profile, tmp_path):
    debug = tmp_path / "debug"
    RollDigitizer(profile, band_height=150, debug_dir=str(debug)).run(
        ArraySource(synthetic_roll)
    )
    assert (debug / "notes_raw.csv").exists()
    assert (debug / "notes_merged.csv").exists()
