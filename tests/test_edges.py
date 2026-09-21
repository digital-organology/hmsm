# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest

from hmsm.rolls.edges import RollEdges, detect_edges


def strip(height=200, width=100, left=20, right=80):
    mask = np.zeros((height, width), bool)
    mask[:, left:right] = True
    return mask


def test_straight_roll_is_found_exactly():
    edges = detect_edges(strip(), smooth=False)
    assert set(edges.left.tolist()) == {20}
    assert set(edges.right.tolist()) == {79}
    assert len(edges) == 200


def test_result_has_one_entry_per_row():
    mask = strip(height=137)
    assert len(detect_edges(mask)) == 137


def test_inverted_masks_are_recognised():
    assert np.array_equal(
        detect_edges(strip(), smooth=False).left,
        detect_edges(~strip(), smooth=False).left,
    )


def test_rows_without_any_roll_are_interpolated_not_dropped():
    mask = strip()
    mask[100:110] = False
    edges = detect_edges(mask, smooth=False)
    # Length is preserved, which is what keeps row indices lining up with the
    # band the edges came from.
    assert len(edges) == 200
    assert edges.left[105] == 20 and edges.right[105] == 79


def test_a_mask_with_no_roll_at_all_is_an_error():
    with pytest.raises(ValueError, match="No roll"):
        detect_edges(np.zeros((10, 10), bool))


def test_travel_measures_how_far_the_edges_move():
    assert detect_edges(strip(), smooth=False).travel == 0

    tapered = np.zeros((200, 100), bool)
    for row in range(200):
        tapered[row, 20 : 80 - row // 10] = True
    assert detect_edges(tapered, smooth=False).travel >= 19


def test_a_roll_reaching_the_image_border_stays_in_bounds():
    # The top left corner of a scan is background, which is how an inverted
    # mask is recognised; below it the roll may run right to the border.
    mask = np.zeros((200, 100), bool)
    mask[1:, :] = True
    edges = detect_edges(mask)
    assert edges.left.min() >= 0
    assert edges.right.max() <= 99
    assert edges.left[100] == 0 and edges.right[100] == 99


def test_smoothing_cannot_push_an_edge_out_of_the_image():
    # A sharp step makes the smoothing overshoot; the result must still be a
    # valid column index.
    mask = np.zeros((400, 100), bool)
    mask[:200, 0:5] = True
    mask[200:, 0:99] = True
    edges = detect_edges(mask)
    assert edges.right.max() <= 99 and edges.left.min() >= 0


def test_crop_takes_a_slice_of_the_band():
    edges = detect_edges(strip(), smooth=False)
    cropped = edges.crop(50, 100)
    assert len(cropped) == 50
    assert cropped.left[0] == edges.left[50]


def test_width_is_the_distance_between_the_edges():
    edges = RollEdges(np.array([10, 10]), np.array([90, 92]))
    np.testing.assert_array_equal(edges.width, [80, 82])
