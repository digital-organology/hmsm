# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Reading the markings printed or drawn onto a roll.

Some formats carry a continuous dynamics line along one side of the roll and,
next to it, discrete pedal markers.  Neither is punched, so they come from the
annotation mask rather than the hole mask, and both have to be reassembled
from the fragments the mask breaks them into.

The two are told apart by where they sit across the roll: a profile's
``pedal_cutoff`` says which side of the roll the pedal markers occupy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.signal
import scipy.spatial

from hmsm.rolls.edges import RollEdges
from hmsm.rolls.holes import find_components
from hmsm.rolls.notes import ControlCode

logger = logging.getLogger(__name__)

#: Fragments smaller than this are scanning noise rather than ink.
_MIN_AREA = 200
_MIN_AREA_WITH_PEDAL = 250

#: Below this many fragments the "annotations" are noise and the roll simply
#: has no printed dynamics line.
_MIN_DYNAMICS_POINTS = 200

#: A pedal marker is a solid printed block; smaller blobs are line fragments.
_MIN_PEDAL_AREA = 3000

#: Pedal marker fragments closer together than this are one marker.
_PEDAL_MERGE_DISTANCE = 100

#: Multiple of the typical nearest-neighbour spacing within which a fragment
#: must have company to be considered part of the line rather than a speck.
_NEIGHBOUR_FACTOR = 2.5
_MIN_NEIGHBOURS = 2

#: Window and order for smoothing the reconstructed dynamics line.
_SMOOTHING_WINDOW = 500
_SMOOTHING_ORDER = 2


@dataclass
class AnnotationCollector:
    """Accumulates annotation fragments across the bands of a scan.

    Fragments are gathered band by band and only assembled into a dynamics
    line and pedal events once the whole scan has been read, because both need
    to reason about the annotation as a whole.

    Attributes:
        pedal_cutoff: Fraction of the roll width separating pedal markers from
            the dynamics line, or None if the format has no printed pedal
            markers. Values of 0.5 or below put the markers on the left.
    """

    pedal_cutoff: Optional[float] = None
    _line: List[np.ndarray] = field(default_factory=list, repr=False)
    _pedal: List[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def min_area(self) -> int:
        return _MIN_AREA if self.pedal_cutoff is None else _MIN_AREA_WITH_PEDAL

    def add_band(self, mask: np.ndarray, edges: RollEdges, row_offset: int = 0) -> None:
        """Collect the annotation fragments in one band.

        Args:
            mask: The band's annotation mask.
            edges: Roll edges for the band.
            row_offset: Row the band starts at, added to the recorded positions.
        """
        if self.pedal_cutoff is not None:
            # Pedal markers are solid blocks; closing the gaps the scan leaves
            # in them keeps a single marker from splitting into fragments that
            # each fall below the area threshold.
            mask = cv2.dilate(mask, _DIAMOND_5)

        components = find_components(mask)
        if len(components) == 0:
            return

        # Reject specks, and reject anything large enough to be the roll
        # margin rather than ink.
        max_area = mask.shape[0] + mask.shape[1] * 10
        keep = (components.area >= self.min_area) & (components.area <= max_area)
        components = components.select(keep)
        if len(components) == 0:
            return

        centers = components.centroid
        rows = np.clip(np.trunc(centers[:, 0]).astype(np.int64), 0, len(edges) - 1)

        if self.pedal_cutoff is None:
            self._line.append(centers + (row_offset, 0))
            return

        boundary = edges.left[rows] + edges.width[rows] * self.pedal_cutoff
        columns = np.trunc(centers[:, 1]).astype(np.int64)
        is_pedal = (
            columns < boundary if self.pedal_cutoff <= 0.5 else columns > boundary
        )

        if is_pedal.any():
            pedal = components.select(is_pedal)
            pedal_rows = rows[is_pedal]
            self._pedal.append(
                np.column_stack(
                    (
                        pedal_rows + row_offset,
                        pedal.area,
                        pedal.right - edges.left[pedal_rows],
                        pedal.left - edges.left[pedal_rows],
                    )
                ).astype(np.int64)
            )

        if (~is_pedal).any():
            self._line.append(centers[~is_pedal] + (row_offset, 0))

    def dynamics_line(self) -> Optional[np.ndarray]:
        """Reconstruct the dynamics line, or None if the roll has none.

        Returns:
            An ``(n, 2)`` array of ``[row, column]``, one entry for every row
            the line spans, or None.
        """
        if not self._line:
            return None
        return reconstruct_dynamics_line(np.vstack(self._line))

    def pedal_events(self) -> Optional[np.ndarray]:
        """Turn the collected pedal markers into note table rows, or None.

        Returns:
            An ``(n, 3)`` note table of pedal-down spans, or None.
        """
        if not self._pedal:
            return None
        return reconstruct_pedal_events(np.vstack(self._pedal))


def reconstruct_dynamics_line(points: np.ndarray) -> Optional[np.ndarray]:
    """Fit a continuous dynamics line through the collected fragments.

    The line arrives as a cloud of fragment centres with gaps where the ink is
    faint. Isolated points are dropped as noise, points sharing a row are
    averaged, the gaps are filled by linear interpolation, and the result is
    smoothed.

    Args:
        points: ``(n, 2)`` array of ``[row, column]`` fragment centres.

    Returns:
        An ``(n, 2)`` int array of ``[row, column]`` covering every row between
        the first and last fragment, or None if there is no line to be found.
    """
    if len(points) < _MIN_DYNAMICS_POINTS:
        logger.info(
            "Found only %d annotation fragments, too few for a dynamics line; "
            "this roll most likely has none and these are noise. Skipping.",
            len(points),
        )
        return None

    points = np.trunc(points).astype(np.int64)
    points = _drop_isolated(points)

    if len(points) == 0:
        logger.info("No annotation fragments survived noise filtering, skipping.")
        return None

    rows, columns = _average_per_row(points)

    if len(rows) < 2:
        logger.info("Dynamics line spans a single row only, skipping.")
        return None

    dense_rows = np.arange(rows[0], rows[-1] + 1)
    dense_columns = np.interp(dense_rows, rows, columns)

    window = _odd_at_most(min(_SMOOTHING_WINDOW, len(dense_rows)))
    if window > _SMOOTHING_ORDER:
        dense_columns = scipy.signal.savgol_filter(
            dense_columns, window, _SMOOTHING_ORDER
        )
    else:
        logger.debug("Dynamics line too short to smooth (%d rows)", len(dense_rows))

    # Smoothing can overshoot at the ends; a column index outside the scan is
    # meaningless and used to wrap around when cast to an unsigned type.
    np.clip(dense_columns, 0, None, out=dense_columns)

    return np.column_stack((dense_rows, np.rint(dense_columns).astype(np.int64)))


def reconstruct_pedal_events(markers: np.ndarray) -> Optional[np.ndarray]:
    """Pair up pedal markers into the spans over which the pedal is down.

    The markers alternate: a wide one opens a span, a narrow one closes it.
    Where one is missed, the neighbouring markers decide how to read the one
    in hand.

    Args:
        markers: ``(n, 4)`` array of ``[row, area, right, left]``, the last two
            measured from the left roll edge.

    Returns:
        An ``(n, 3)`` note table of pedal-down spans, or None if the markers
        look like noise.
    """
    markers = markers[markers[:, 1] > _MIN_PEDAL_AREA]
    if len(markers) == 0:
        logger.info("No pedal markers large enough to be genuine, skipping.")
        return None

    markers = markers[markers[:, 0].argsort()]
    markers = _merge_close_markers(markers)

    if len(markers) < 10:
        logger.info(
            "Found only %d pedal markers, which likely means the roll has no "
            "pedal annotations and these are noise. Skipping.",
            len(markers),
        )
        return None

    widths = markers[:, 2] - markers[:, 3]
    is_wide = widths > widths.mean()

    spans = []
    opened_at = None

    for i, wide in enumerate(is_wide):
        last = i + 1 >= len(is_wide)
        if wide:
            # A wide marker opens a span, unless we are already inside one and
            # the next marker is also wide, in which case this one is really
            # the missing close.
            if not last and (opened_at is None or not is_wide[i + 1]):
                opened_at = markers[i, 0]
            elif opened_at is not None:
                spans.append((opened_at, markers[i, 0]))
                opened_at = None
        else:
            if opened_at is not None:
                spans.append((opened_at, markers[i, 0]))
                opened_at = None
            elif not last and not is_wide[i + 1]:
                opened_at = markers[i, 0]

    if not spans:
        logger.info("Pedal markers could not be paired into any spans, skipping.")
        return None

    return np.array(
        [(start, end, int(ControlCode.PEDAL)) for start, end in spans], dtype=np.int64
    )


def _drop_isolated(points: np.ndarray) -> np.ndarray:
    """Drop fragments that have too few neighbours to be part of a line.

    A tree query rather than a full distance matrix, so this stays usable on
    the tens of thousands of fragments a long roll produces.
    """
    tree = scipy.spatial.cKDTree(points)

    # Distance to the nearest fragment that is not at the same spot.
    coincident = tree.query_ball_point(points, r=0, return_length=True)
    nearest = tree.query(points, k=min(len(points), 16))[0]
    nearest = np.where(nearest > 0, nearest, np.inf).min(axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if len(nearest) == 0:
        return points

    radius = _NEIGHBOUR_FACTOR * nearest.mean()
    within = tree.query_ball_point(points, r=radius, return_length=True)
    return points[(within - coincident) > _MIN_NEIGHBOURS]


def _average_per_row(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse fragments sharing a row into one column position per row."""
    order = points[:, 0].argsort(kind="stable")
    points = points[order]
    rows, first = np.unique(points[:, 0], return_index=True)
    counts = np.diff(np.append(first, len(points)))
    columns = np.add.reduceat(points[:, 1], first) / counts
    return rows, columns


def _merge_close_markers(markers: np.ndarray) -> np.ndarray:
    """Fuse marker fragments that are close enough to be one marker."""
    group = np.zeros(len(markers), dtype=np.int64)
    if len(markers) > 1:
        group[1:] = np.cumsum(np.diff(markers[:, 0]) > _PEDAL_MERGE_DISTANCE)

    first = np.flatnonzero(np.append(True, np.diff(group) != 0))
    return np.column_stack(
        (
            markers[first, 0],
            np.add.reduceat(markers[:, 1], first),
            np.maximum.reduceat(markers[:, 2], first),
            np.minimum.reduceat(markers[:, 3], first),
        )
    )


def _odd_at_most(value: int) -> int:
    """The largest odd number not greater than ``value``."""
    return value if value % 2 else value - 1


_DIAMOND_5 = np.array(
    [
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ],
    dtype=np.uint8,
)
