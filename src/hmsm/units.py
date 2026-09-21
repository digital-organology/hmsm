# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Conversion between physical measurements and pixels.

Profiles describe media in millimetres so that a profile is a property of the
physical format rather than of a particular scan.  Everything downstream works
in pixels, so the conversion happens once, at profile resolution time.
"""

from __future__ import annotations

MM_PER_INCH = 25.4

#: Resolution the project scans its media at.  Profiles are resolution
#: independent; this is only the default assumed when a scan does not declare
#: its own resolution.
DEFAULT_DPI = 300


def mm_to_px(mm: float, dpi: int = DEFAULT_DPI) -> float:
    """Convert a millimetre measurement to pixels at the given resolution."""
    return mm / MM_PER_INCH * dpi


def px_to_mm(px: float, dpi: int = DEFAULT_DPI) -> float:
    """Convert a pixel measurement to millimetres at the given resolution."""
    return px * MM_PER_INCH / dpi
