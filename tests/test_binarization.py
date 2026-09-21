# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest

from hmsm.rolls.binarization import (
    BinarizationError,
    available_methods,
    binarizer,
    segment,
)


def test_v_channel_is_registered():
    assert "v_channel" in available_methods()


def test_unknown_method_is_reported_with_the_alternatives():
    with pytest.raises(BinarizationError, match="v_channel"):
        segment("no_such_method", np.zeros((10, 10, 3), np.uint8), "black", x=1)


def test_unusable_options_are_reported():
    with pytest.raises(BinarizationError, match="v_channel"):
        segment("v_channel", np.zeros((10, 10, 3), np.uint8), "black", nonsense=1)


def test_methods_can_be_registered_from_outside():
    sentinel = object()

    @binarizer("test_method")
    def _method(band, bg_color, **options):
        return sentinel

    assert "test_method" in available_methods()
    assert segment("test_method", np.zeros((4, 4, 3), np.uint8), "black") is sentinel


def test_holes_are_found_on_a_black_background(synthetic_roll):
    masks = segment("v_channel", synthetic_roll, "black", threshold=0.1)
    # The punched columns are masked, the paper between them is not.
    assert masks.holes[50, 65] == 1
    assert masks.holes[50, 100] == 0


def test_the_roll_edges_are_found(synthetic_roll):
    masks = segment("v_channel", synthetic_roll, "black", threshold=0.1)
    assert len(masks.edges) == synthetic_roll.shape[0]
    assert abs(int(masks.edges.left.mean()) - 20) <= 2
    assert abs(int(masks.edges.right.mean()) - 179) <= 2


def test_everything_outside_the_roll_is_masked(synthetic_roll):
    masks = segment("v_channel", synthetic_roll, "black", threshold=0.1)
    # Including column zero: a roll whose edge sits at the very left of the
    # scan must not wrap around and swallow the whole row.
    assert masks.holes[50, 0] == 1
    assert masks.holes[50, -1] == 1


def test_an_edge_at_column_zero_does_not_swallow_the_row():
    # The old implementation computed `left - 1` on an unsigned type, so an
    # edge sitting at column zero wrapped around and marked the entire row as
    # being outside the roll. Rolls really do touch column zero in places.
    from hmsm.rolls.binarization import _mask_outside_roll
    from hmsm.rolls.edges import RollEdges

    mask = np.zeros((4, 50), np.uint8)
    edges = RollEdges(
        np.array([0, 0, 5, 5], np.int32), np.array([40, 40, 40, 40], np.int32)
    )
    _mask_outside_roll(mask, edges)

    # Rows whose left edge is at zero keep their whole interior unmasked.
    assert mask[0, :40].sum() == 0
    assert mask[0, 40:].all()
    # Rows with a left edge further in have exactly that margin masked.
    assert mask[2, :5].all() and mask[2, 5:40].sum() == 0


def test_white_and_black_backgrounds_are_handled_symmetrically():
    dark = np.zeros((200, 100, 3), np.uint8)
    dark[:, 10:90] = 200
    dark[50:80, 40:52] = 0

    light = np.full((200, 100, 3), 255, np.uint8)
    light[:, 10:90] = 55
    light[50:80, 40:52] = 255

    on_black = segment("v_channel", dark, "black", threshold=0.1)
    on_white = segment("v_channel", light, "white", threshold=0.1)
    np.testing.assert_array_equal(on_black.holes, on_white.holes)


def test_annotations_are_separated_from_holes():
    scan = np.zeros((200, 100, 3), np.uint8)
    scan[:, 10:90] = 230
    scan[50:80, 20:32] = 0  # a hole: as dark as the background
    scan[100:160, 60:72] = 90  # printed ink: darker than paper, lighter than a hole

    masks = segment("v_channel", scan, "black", threshold=0.1, upper_threshold=0.6)
    assert masks.annotations is not None
    assert masks.holes[60, 26] == 1 and masks.annotations[60, 26] == 0
    assert masks.annotations[130, 66] == 1 and masks.holes[130, 66] == 0


def test_no_annotation_mask_without_an_upper_threshold(synthetic_roll):
    assert (
        segment("v_channel", synthetic_roll, "black", threshold=0.1).annotations is None
    )
