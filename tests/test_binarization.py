# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest

from hmsm.rolls.binarization import (
    BinarizationError,
    available_methods,
    binarizer,
    segment,
)
from hmsm.rolls.paper import PaperModel

BLACK = PaperModel(paper=(200, 200, 200), background=(0, 0, 0))
WHITE = PaperModel(paper=(55, 55, 55), background=(255, 255, 255))

DYNAMICS = [{"name": "dynamics", "region": [0.0, 1.0]}]


@pytest.fixture
def paper(synthetic_roll):
    return PaperModel.estimate(synthetic_roll)


def test_both_methods_are_registered():
    assert {"paper_relative", "v_channel"} <= set(available_methods())


def test_unknown_method_is_reported_with_the_alternatives():
    with pytest.raises(BinarizationError, match="v_channel"):
        segment("no_such_method", np.zeros((10, 10, 3), np.uint8), BLACK, x=1)


def test_unusable_options_are_reported():
    with pytest.raises(BinarizationError, match="v_channel"):
        segment("v_channel", np.zeros((10, 10, 3), np.uint8), BLACK, nonsense=1)


def test_methods_can_be_registered_from_outside():
    sentinel = object()

    @binarizer("test_method")
    def _method(band, paper, **options):
        return sentinel

    assert "test_method" in available_methods()
    assert segment("test_method", np.zeros((4, 4, 3), np.uint8), BLACK) is sentinel


@pytest.mark.parametrize("method", ["paper_relative", "v_channel"])
def test_holes_are_found_on_a_black_background(method, synthetic_roll, paper):
    masks = segment(method, synthetic_roll, paper, **_options(method))
    # The punched columns are masked, the paper between them is not.
    assert masks.holes[50, 65] == 1
    assert masks.holes[50, 100] == 0


@pytest.mark.parametrize("method", ["paper_relative", "v_channel"])
def test_the_roll_edges_are_found(method, synthetic_roll, paper):
    masks = segment(method, synthetic_roll, paper, **_options(method))
    assert len(masks.edges) == synthetic_roll.shape[0]
    assert abs(int(masks.edges.left.mean()) - 20) <= 2
    assert abs(int(masks.edges.right.mean()) - 179) <= 2


def test_nothing_outside_the_roll_is_taken_for_a_hole(synthetic_roll, paper):
    masks = segment("paper_relative", synthetic_roll, paper)
    # The scanner background either side of the paper is as dark as a hole
    # and must not be read as one.
    assert masks.holes[50, 0] == 0
    assert masks.holes[50, -1] == 0


def test_v_channel_marks_everything_outside_the_roll(synthetic_roll, paper):
    masks = segment("v_channel", synthetic_roll, paper, threshold=0.1)
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


@pytest.mark.parametrize("method", ["paper_relative", "v_channel"])
def test_white_and_black_backgrounds_give_the_same_holes(method):
    dark = np.zeros((200, 100, 3), np.uint8)
    dark[:, 10:90] = 200
    dark[50:80, 40:52] = 0

    light = 255 - dark

    on_black = segment(method, dark, PaperModel.estimate(dark), **_options(method))
    on_white = segment(method, light, PaperModel.estimate(light), **_options(method))
    np.testing.assert_array_equal(on_black.holes, on_white.holes)


def test_ink_is_separated_from_holes():
    scan = _scan_with_ink()
    masks = segment(
        "paper_relative", scan, PaperModel.estimate(scan), ink_layers=DYNAMICS
    )

    ink = masks.ink["dynamics"]
    assert masks.holes[60, 26] == 1
    assert ink.mask[60, 26 - ink.column_offset] == 0
    assert ink.mask[130, 66 - ink.column_offset] == 1
    assert masks.holes[130, 66] == 0


def test_ink_as_dark_as_a_hole_is_still_ink():
    # A Hupfeld dynamics line is printed heavily enough to be as dark as the
    # scanner background behind a hole. What sets it apart is its colour: a
    # hole shows the background itself, this is grey ink on beige paper.
    scan = np.zeros((200, 100, 3), np.uint8)
    scan[:, 10:90] = (220, 200, 165)
    scan[50:80, 20:32] = 0  # a hole: the black background, showing through
    scan[100:160, 60:72] = (30, 30, 30)  # ink: as dark, but a neutral grey

    masks = segment(
        "paper_relative", scan, PaperModel.estimate(scan), ink_layers=DYNAMICS
    )
    ink = masks.ink["dynamics"]

    assert masks.holes[60, 26] == 1 and masks.holes[130, 66] == 0
    assert ink.mask[130, 66 - ink.column_offset] == 1


def test_ink_layers_split_the_printing_by_where_it_sits():
    scan = _scan_with_ink()
    scan[20:40, 30:42] = 90  # more ink, on the bass half of the roll

    masks = segment(
        "paper_relative",
        scan,
        PaperModel.estimate(scan),
        ink_layers=[
            {"name": "bass", "region": [0.0, 0.5]},
            {"name": "discant", "region": [0.5, 1.0]},
        ],
    )

    assert set(masks.ink) == {"bass", "discant"}
    bass, discant = masks.ink["bass"], masks.ink["discant"]
    assert bass.mask[30, 35 - bass.column_offset] == 1
    assert discant.mask[130, 66 - discant.column_offset] == 1
    # Neither lane holds the other's printing.
    assert discant.mask[30].sum() == 0
    assert bass.mask[130].sum() == 0


def test_no_ink_masks_without_declared_layers(synthetic_roll, paper):
    assert segment("paper_relative", synthetic_roll, paper).ink == {}


def test_ink_lanes_are_cut_down_to_their_own_columns():
    scan = _scan_with_ink()
    masks = segment(
        "paper_relative",
        scan,
        PaperModel.estimate(scan),
        ink_layers=[{"name": "narrow", "region": [0.5, 0.7]}],
    )
    lane = masks.ink["narrow"]
    assert lane.column_offset > 0
    assert lane.mask.shape[1] < scan.shape[1]
    assert lane.mask.shape[0] == scan.shape[0]


@pytest.mark.parametrize(
    "options",
    [
        {"hole_seed": 0.4, "hole_grow": 0.5},
        {"hole_seed": 1.5},
        {"ink_shade": 0.0},
        {"ink_shade": 2.0},
    ],
)
def test_inconsistent_thresholds_are_rejected(options, synthetic_roll, paper):
    with pytest.raises(BinarizationError):
        segment("paper_relative", synthetic_roll, paper, **options)


def test_hysteresis_keeps_the_whole_of_a_soft_edged_hole():
    # A scanned hole has a penumbra: the paper edge is not sharp at the pixel
    # level. Seeding on the dark core and growing out to the half-way point
    # recovers the hole's true extent, where thresholding once at the seed
    # level would cut two pixels off every side.
    scan = np.zeros((60, 60, 3), np.uint8)
    scan[:, 5:55] = 200
    scan[20:40, 24:36] = 60  # penumbra: past half way to the background
    scan[22:38, 26:34] = 0  # core: the background itself

    masks = segment("paper_relative", scan, BLACK)

    rows = np.flatnonzero(masks.holes.any(axis=1))
    columns = np.flatnonzero(masks.holes.any(axis=0))
    assert (rows.min(), rows.max()) == (20, 39)
    assert (columns.min(), columns.max()) == (24, 35)


def _options(method):
    return {"threshold": 0.1} if method == "v_channel" else {}


def _scan_with_ink():
    scan = np.zeros((200, 100, 3), np.uint8)
    scan[:, 10:90] = 230
    scan[50:80, 20:32] = 0  # a hole: as dark as the background
    scan[100:160, 60:72] = 90  # printed ink: darker than paper, lighter than a hole
    return scan
