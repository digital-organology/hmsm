# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest

from hmsm.rolls.paper import (
    SHADE,
    TRANSMISSION,
    PaperError,
    PaperModel,
    localize_channels,
    sample_scan,
)

BEIGE = (210, 192, 158)


def scan_of(paper, background, shape=(300, 200)):
    """A scan of a roll on a background, with one hole punched through it."""
    band = np.empty((*shape, 3), np.uint8)
    band[:] = background
    band[:, 20:-20] = paper
    band[100:140, 60:80] = background
    return band


def test_the_paper_is_the_colour_the_scan_is_mostly_made_of():
    model = PaperModel.estimate(scan_of(BEIGE, (10, 10, 10)))
    np.testing.assert_allclose(model.paper, BEIGE, atol=2)
    np.testing.assert_allclose(model.background, (10, 10, 10), atol=2)
    assert model.background_is_dark and model.background_name == "black"


def test_a_white_background_is_recognised():
    model = PaperModel.estimate(scan_of(BEIGE, (250, 250, 250)))
    np.testing.assert_allclose(model.paper, BEIGE, atol=2)
    assert not model.background_is_dark and model.background_name == "white"


def test_the_background_can_be_forced():
    # A Welte T100 on dark red paper over a white bed, with a dark stain on
    # the paper. Left to itself the estimate takes the white bed, which is
    # what it should do; being told otherwise it takes the stain.
    scan = scan_of((96, 22, 26), (250, 250, 250))
    scan[200:260, 100:160] = (6, 5, 5)

    assert not PaperModel.estimate(scan).background_is_dark
    assert PaperModel.estimate(scan, background="black").background_is_dark
    assert not PaperModel.estimate(scan, background="white").background_is_dark
    with pytest.raises(PaperError, match="black"):
        PaperModel.estimate(scan, background="chartreuse")


@pytest.mark.parametrize(
    "paper, background",
    [
        (BEIGE, (10, 10, 10)),  # beige on black, as the project scans
        (BEIGE, (250, 250, 250)),  # the same roll over a white bed
        ((96, 22, 26), (8, 6, 6)),  # a Welte T100 on dark red paper
        ((205, 215, 180), (10, 10, 10)),  # a green roll
    ],
)
def test_paper_reads_as_zero_and_a_hole_as_one(paper, background):
    scan = scan_of(paper, background)
    channels = PaperModel.estimate(scan).channels(scan)

    on_paper = channels[200, 100]
    in_hole = channels[120, 70, TRANSMISSION]

    assert abs(on_paper[TRANSMISSION]) < 0.02
    assert abs(on_paper[SHADE]) < 0.02
    assert in_hole > 0.95


def test_ink_shades_the_paper_whichever_way_the_background_goes():
    # The same printing on the same paper, scanned over a black and over a
    # white bed. Shade is what annotation extraction thresholds, so it has to
    # read the same both times; transmission does not, and cannot.
    for background in ((10, 10, 10), (250, 250, 250)):
        scan = scan_of(BEIGE, background)
        scan[200:220, 100:130] = (150, 137, 113)  # 70% of the paper's brightness
        channels = PaperModel.estimate(scan).channels(scan)
        assert 0.25 < channels[210, 115, SHADE] < 0.35


def test_a_hole_sits_on_the_paper_background_line_and_grey_ink_does_not():
    scan = scan_of(BEIGE, (10, 10, 10))
    scan[200:220, 100:130] = (60, 60, 60)  # neutral grey ink, nearly as dark
    model = PaperModel.estimate(scan)

    offsets = np.hypot(*np.moveaxis(model.chroma(scan), 2, 0))
    assert offsets[120, 70] < 0.01  # inside the hole
    assert offsets[210, 115] > 0.03  # inside the ink


def test_chroma_at_matches_the_full_plane():
    scan = scan_of(BEIGE, (10, 10, 10))
    scan[200:220, 100:130] = (60, 60, 60)
    model = PaperModel.estimate(scan)

    full = np.hypot(*np.moveaxis(model.chroma(scan), 2, 0))
    where = np.flatnonzero(full.ravel() > 0.01)
    np.testing.assert_allclose(
        model.chroma_at(scan, where), full.ravel()[where], atol=1e-5
    )


def test_the_local_level_takes_out_a_brightness_gradient():
    # A stretch of roll lit unevenly across its width, as a scanner lights a
    # thirty centimetre bed. Uncorrected, the dim side reads as printing.
    gradient = np.linspace(1.0, 0.8, 1200)[None, :, None]
    scan = (np.array(BEIGE) * gradient * np.ones((400, 1, 1))).astype(np.uint8)
    model = PaperModel(paper=BEIGE, background=(10, 10, 10))

    uncorrected = model.channels(scan)
    assert uncorrected[200, 1100, SHADE] > 0.08

    corrected = localize_channels(uncorrected)
    for column in (20, 600, 1100):
        assert abs(corrected[200, column, SHADE]) < 0.02


def test_the_local_level_leaves_real_ink_alone():
    # The correction must not swallow what it is there to expose: a block
    # with printing in it is still levelled against its paper.
    scan = np.empty((400, 400, 3), np.uint8)
    scan[:] = BEIGE
    scan[190:210, 100:140] = (147, 134, 111)  # ink at 70% of the paper

    model = PaperModel(paper=BEIGE, background=(10, 10, 10))
    corrected = localize_channels(model.channels(scan))

    assert 0.25 < corrected[200, 120, SHADE] < 0.35
    assert abs(corrected[200, 300, SHADE]) < 0.02


def test_two_colours_too_close_together_are_refused():
    with pytest.raises(PaperError, match="too close"):
        PaperModel(paper=(200, 200, 200), background=(202, 201, 200))


def test_an_empty_sample_is_refused():
    with pytest.raises(PaperError, match="empty"):
        PaperModel.estimate(np.empty((0, 3), np.uint8))


def test_sampling_spreads_over_the_whole_scan():
    from hmsm.io import ArraySource

    # Paper that darkens down the length of the roll: a sample taken from the
    # top alone would set the reference too light for the bottom.
    scan = np.empty((20_000, 100, 3), np.uint8)
    scan[:] = (np.linspace(230, 150, 20_000)[:, None, None] * np.ones(3)).astype(
        np.uint8
    )

    taken = sample_scan(ArraySource(scan), rows=800)
    assert taken.max() > 220 and taken.min() < 160
