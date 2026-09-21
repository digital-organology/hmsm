# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import pytest

from hmsm.units import DEFAULT_DPI, mm_to_px, px_to_mm


def test_an_inch_is_the_resolution_in_pixels():
    assert mm_to_px(25.4, 300) == pytest.approx(300)


def test_conversion_round_trips():
    assert px_to_mm(mm_to_px(2.7, DEFAULT_DPI), DEFAULT_DPI) == pytest.approx(2.7)


def test_resolution_scales_the_result():
    assert mm_to_px(10, 600) == pytest.approx(2 * mm_to_px(10, 300))
