# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest

from hmsm.rolls.notes import END, START, TONE, ControlCode, merge_notes, rebase


def table(rows):
    return np.array(rows, dtype=np.int64)


def test_empty_input_gives_an_empty_table():
    assert merge_notes(table([]).reshape(0, 3)).shape == (0, 3)


def test_close_holes_on_one_track_become_a_single_note():
    # At 300 dpi a 2.7 mm hole is 31 px, so the merge gap is about 54 px.
    merged = merge_notes(table([[0, 20, 60], [40, 60, 60]]), hole_width_mm=2.7)
    np.testing.assert_array_equal(merged, table([[0, 60, 60]]))


def test_distant_holes_stay_separate():
    merged = merge_notes(table([[0, 20, 60], [500, 520, 60]]), hole_width_mm=2.7)
    assert len(merged) == 2


def test_holes_on_different_tracks_never_merge():
    merged = merge_notes(table([[0, 20, 60], [30, 50, 61]]), hole_width_mm=2.7)
    assert len(merged) == 2
    assert sorted(merged[:, TONE]) == [60, 61]


def test_a_run_of_holes_merges_into_one_note():
    rows = [[start, start + 20, 60] for start in range(0, 300, 40)]
    merged = merge_notes(table(rows), hole_width_mm=2.7)
    np.testing.assert_array_equal(merged, table([[0, 300, 60]]))


def test_control_codes_merge_like_any_other_track():
    pedal = int(ControlCode.PEDAL)
    merged = merge_notes(table([[0, 20, pedal], [40, 60, pedal]]), hole_width_mm=2.7)
    np.testing.assert_array_equal(merged, table([[0, 60, pedal]]))


def test_merging_never_shortens_a_note():
    # A hole wholly inside another must not truncate it.
    merged = merge_notes(table([[0, 100, 60], [10, 20, 60]]), hole_width_mm=2.7)
    assert merged[0, END] == 100


def test_output_is_sorted_by_start():
    merged = merge_notes(table([[500, 520, 60], [0, 20, 61]]), hole_width_mm=2.7)
    assert list(merged[:, START]) == sorted(merged[:, START])


def test_merge_threshold_is_estimated_when_no_hole_width_is_given():
    merged = merge_notes(table([[0, 20, 60], [25, 45, 60]]))
    assert len(merged) == 1


def test_rebase_moves_the_first_note_to_zero():
    notes, _ = rebase(table([[1000, 1020, 60], [1100, 1120, 61]]))
    assert notes[:, START].min() == 0
    assert notes[0, END] == 20


def test_rebase_shifts_the_dynamics_line_by_the_same_amount():
    dynamics = np.array([[1000, 50], [1001, 51]], dtype=np.int64)
    notes, shifted = rebase(table([[1000, 1020, 60]]), dynamics)
    assert notes[0, START] == 0
    np.testing.assert_array_equal(shifted[:, 0], [0, 1])
    # The originals are left alone.
    assert dynamics[0, 0] == 1000


def test_rebase_without_notes_is_a_no_op():
    empty = table([]).reshape(0, 3)
    notes, dynamics = rebase(empty, None)
    assert len(notes) == 0 and dynamics is None
