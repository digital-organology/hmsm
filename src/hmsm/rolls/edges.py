# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Tracking where the roll sits within the scan.

Rolls are never scanned perfectly straight and are rarely perfectly straight to
begin with, so the left and right edge of the paper are tracked per row.  Hole
positions are then expressed relative to those edges, which compensates for
gentle curvature and for the roll drifting sideways across a long scan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

#: Window and polynomial order for smoothing the raw per-row edge positions.
#: The raw positions are noisy at the scale of single pixels; the physical edge
#: of a paper roll is not.
_SMOOTHING_WINDOW = 20
_SMOOTHING_ORDER = 3


@dataclass(frozen=True)
class RollEdges:
    """Position of the left and right roll edge for every row of a band.

    Attributes:
        left: Column index of the left edge, one entry per row.
        right: Column index of the right edge, one entry per row.
    """

    left: np.ndarray
    right: np.ndarray

    def __len__(self) -> int:
        return len(self.left)

    @property
    def width(self) -> np.ndarray:
        """Width of the roll in pixels, per row."""
        return self.right - self.left

    @property
    def travel(self) -> int:
        """How far either edge moves across the band, in pixels.

        A large value means the band is not a straight stretch of roll: it is
        the roll head, a tear, or the end of the paper.
        """
        return int(
            max(
                self.left.max() - self.left.min(),
                self.right.max() - self.right.min(),
            )
        )

    def crop(self, start: int, stop: int | None = None) -> "RollEdges":
        """Restrict the edges to rows ``[start, stop)`` of the band."""
        return RollEdges(self.left[start:stop], self.right[start:stop])


def detect_edges(
    mask: np.ndarray, smooth: bool = True, invert: bool | None = None
) -> RollEdges:
    """Find the roll edges on a binary mask of the roll body.

    Args:
        mask: Boolean or uint8 mask of a band, True where the roll paper is.
        smooth: Whether to smooth the raw positions. Off is useful for diagnostics.
        invert: Whether the mask is the other way round, marking everything
            that is *not* roll. ``None`` guesses from the top left pixel,
            which is what callers handing over a hole mask want; pass
            ``False`` where the mask really is the roll body, since a roll
            whose edge touches column zero defeats the guess.

    Raises:
        ValueError: If the mask contains no roll at all.

    Returns:
        The detected edges, one entry per row of the mask.
    """
    mask = mask.astype(bool, copy=False)

    if invert is None:
        invert = bool(mask[0, 0])
    if invert:
        mask = ~mask

    # argmax on a boolean row returns the first set column and stops there, so
    # this finds both edges in two linear passes instead of materialising the
    # coordinates of every roll pixel in the band.
    populated = mask.any(axis=1)
    if not populated.any():
        raise ValueError("No roll detected in this band")

    left = mask.argmax(axis=1).astype(np.int32)
    right = (mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)).astype(np.int32)

    if not populated.all():
        # Rows with no roll pixels at all carry no information, and leaving
        # them at argmax's fallback of zero would drag the edge across the
        # band. Interpolate them from the rows that do have an edge.
        left, right = _fill_gaps(left, right, populated)

    if smooth and len(left) > _SMOOTHING_WINDOW:
        left = _smooth(left)
        right = _smooth(right)

    # Guard against the smoothing overshooting past the image bounds, which
    # would otherwise produce out-of-range slice bounds downstream.
    np.clip(left, 0, mask.shape[1] - 1, out=left)
    np.clip(right, 0, mask.shape[1] - 1, out=right)

    return RollEdges(left, right)


def _fill_gaps(
    left: np.ndarray, right: np.ndarray, populated: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate edge positions for rows that contain no roll pixels."""
    logger.debug(
        "Interpolating roll edges for %d row(s) without roll pixels",
        int((~populated).sum()),
    )
    rows = np.arange(len(left))
    known = rows[populated]
    return (
        np.interp(rows, known, left[populated]).round().astype(np.int32),
        np.interp(rows, known, right[populated]).round().astype(np.int32),
    )


def _smooth(edge: np.ndarray) -> np.ndarray:
    """Smooth a raw edge trace, keeping it integral."""
    return (
        scipy.signal.savgol_filter(edge, _SMOOTHING_WINDOW, _SMOOTHING_ORDER)
        .round()
        .astype(np.int32)
    )
