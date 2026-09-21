# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np

from hmsm.rolls.edges import RollEdges
from hmsm.rolls.holes import (
    assign_tracks,
    extract_notes,
    filter_components,
    find_components,
)


def mask_with(boxes, shape=(200, 200)):
    mask = np.zeros(shape, np.uint8)
    for top, left, height, width in boxes:
        mask[top : top + height, left : left + width] = 1
    return mask


def test_bounding_boxes_are_inclusive_extents():
    # A 10x20 block spans 9 rows and 19 columns from its top left corner.
    components = find_components(mask_with([(5, 7, 10, 20)]))
    assert len(components) == 1
    assert (components.top[0], components.left[0]) == (5, 7)
    assert (components.height[0], components.width[0]) == (9, 19)
    assert components.area[0] == 200
    assert components.bottom[0] == 14 and components.right[0] == 26


def test_separate_blocks_are_separate_components():
    assert len(find_components(mask_with([(0, 0, 5, 5), (50, 50, 5, 5)]))) == 2


def test_diagonally_touching_blocks_are_one_component():
    assert len(find_components(mask_with([(0, 0, 5, 5), (5, 5, 5, 5)]))) == 1


def test_an_empty_mask_has_no_components():
    assert len(find_components(np.zeros((10, 10), np.uint8))) == 0


def test_width_filter_keeps_only_plausible_holes():
    components = find_components(
        mask_with([(0, 0, 10, 3), (30, 0, 10, 12), (60, 0, 10, 90)])
    )
    kept = filter_components(components, width_bounds=(5, 20))
    assert len(kept) == 1
    assert kept.width[0] == 11


def test_filters_combine():
    components = find_components(mask_with([(0, 0, 10, 12), (30, 0, 80, 12)]))
    kept = filter_components(components, width_bounds=(5, 20), height_bounds=(0, 20))
    assert len(kept) == 1


def test_tracks_are_assigned_by_position_relative_to_the_roll_edges():
    grid = np.array([[0.1, 0.2, 60.0], [0.5, 0.6, 72.0], [0.8, 0.9, 84.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    components = find_components(
        mask_with([(10, 10, 5, 5), (50, 50, 5, 5), (100, 80, 5, 5)])
    )
    tracks, accepted = assign_tracks(components, edges, grid)
    assert accepted.all()
    np.testing.assert_array_equal(grid[tracks, 2], [60, 72, 84])


def test_track_assignment_follows_a_roll_that_drifts_sideways():
    grid = np.array([[0.1, 0.2, 60.0], [0.5, 0.6, 72.0]])
    # The roll slides 40 px to the right over the band.
    shift = np.arange(200, dtype=np.int32) // 5
    edges = RollEdges(shift, shift + 100)
    # A hole that drifts with the roll stays on the same track.
    boxes = [(10, 10 + 10 // 5, 5, 5), (150, 10 + 150 // 5, 5, 5)]
    components = find_components(mask_with(boxes))
    tracks, accepted = assign_tracks(components, edges, grid)
    assert accepted.all()
    np.testing.assert_array_equal(grid[tracks, 2], [60, 60])


def test_a_component_beyond_the_outermost_track_is_not_a_hole():
    # Two tracks a fifth of the roll apart. A speck in the margin, more than
    # half that again beyond the last one, belongs to neither and used to be
    # snapped onto whichever was nearest.
    grid = np.array([[0.2, 0.25, 60.0], [0.4, 0.45, 72.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    components = find_components(mask_with([(10, 22, 5, 5), (100, 80, 5, 5)]))
    tracks, accepted = assign_tracks(components, edges, grid)
    np.testing.assert_array_equal(accepted, [True, False])
    assert grid[tracks[0], 2] == 60


def test_a_hole_is_matched_by_its_middle_not_its_left_side():
    # A hole scanned wider than nominal still belongs to its own track: its
    # middle is what is compared, so the extra width falls either side.
    grid = np.array([[0.20, 0.30, 60.0], [0.50, 0.60, 72.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    components = find_components(mask_with([(10, 22, 5, 12)]))
    tracks, accepted = assign_tracks(components, edges, grid)
    assert accepted.all() and grid[tracks[0], 2] == 60


def test_density_of_a_solid_rectangle_is_one():
    components = find_components(mask_with([(5, 7, 10, 20)]))
    np.testing.assert_allclose(components.density, [1.0])


def test_long_components_are_rejected_when_the_format_bounds_hole_length():
    # A fold running the length of the band is the right width for a hole
    # and nothing like the right length.
    grid = np.array([[0.1, 0.2, 60.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    mask = mask_with([(0, 10, 200, 6)])
    assert len(extract_notes(mask, edges, grid, (3, 10))) == 1
    assert len(extract_notes(mask, edges, grid, (3, 10), height_bounds=(3, 60))) == 0


def test_extract_notes_produces_start_end_and_tone():
    grid = np.array([[0.1, 0.2, 60.0], [0.5, 0.6, 72.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    notes = extract_notes(
        mask_with([(20, 10, 15, 6)]), edges, grid, width_bounds=(3, 10)
    )
    np.testing.assert_array_equal(notes, [[20, 34, 60]])


def test_extract_notes_offsets_rows_by_the_band_position():
    grid = np.array([[0.1, 0.2, 60.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    notes = extract_notes(
        mask_with([(20, 10, 15, 6)]), edges, grid, (3, 10), row_offset=8000
    )
    assert notes[0, 0] == 8020


def test_extract_notes_returns_an_empty_table_when_nothing_qualifies():
    grid = np.array([[0.1, 0.2, 60.0]])
    edges = RollEdges(np.zeros(200, np.int32), np.full(200, 100, np.int32))
    notes = extract_notes(np.zeros((200, 200), np.uint8), edges, grid, (3, 10))
    assert notes.shape == (0, 3)
