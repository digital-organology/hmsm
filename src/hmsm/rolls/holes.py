# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Finding the punched holes in a band and assigning them to tracks.

Every hole is a connected component of the hole mask.  What the pipeline needs
from each one is its bounding box: where it starts and ends vertically becomes
the note's timing, and where it starts horizontally, relative to the roll
edges, decides which track it belongs to.

Connected component analysis gives us all of that directly, so nothing here
ever materialises the pixel coordinates of a component.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from hmsm.rolls.edges import RollEdges

logger = logging.getLogger(__name__)

# Column layout of the stats array from cv2.connectedComponentsWithStats.
_LEFT, _TOP, _WIDTH, _HEIGHT, _AREA = range(5)


@dataclass(frozen=True)
class Components:
    """Bounding boxes of the connected components of a mask.

    Widths and heights are inclusive pixel extents, so a component occupying a
    single column has a width of zero. That matches how hole sizes are
    expressed in profiles, where a width is the distance between the two sides
    of a hole.

    Attributes:
        left: Leftmost column of each component.
        top: Topmost row of each component.
        width: Horizontal extent, inclusive.
        height: Vertical extent, inclusive.
        area: Number of set pixels in each component.
        centroid: Mean position of each component's pixels, as ``[row, column]``.
            Unlike the centre of the bounding box this follows the mass of an
            irregular shape, which matters for tracing the dynamics line.
        labels: The label image, retained only for diagnostics.
    """

    left: np.ndarray
    top: np.ndarray
    width: np.ndarray
    height: np.ndarray
    area: np.ndarray
    centroid: np.ndarray
    labels: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.left)

    @property
    def right(self) -> np.ndarray:
        return self.left + self.width

    @property
    def bottom(self) -> np.ndarray:
        return self.top + self.height

    @property
    def center_row(self) -> np.ndarray:
        """Row through the middle of each component."""
        return np.rint(self.top + self.height / 2).astype(np.int32)

    @property
    def density(self) -> np.ndarray:
        """Fraction of each bounding box that is actually set."""
        box = np.maximum(self.width, 1) * np.maximum(self.height, 1)
        return self.area / box

    def select(self, keep: np.ndarray) -> "Components":
        """Return only the components ``keep`` selects."""
        return Components(
            left=self.left[keep],
            top=self.top[keep],
            width=self.width[keep],
            height=self.height[keep],
            area=self.area[keep],
            centroid=self.centroid[keep],
            labels=self.labels,
        )


def find_components(mask: np.ndarray, keep_labels: bool = False) -> Components:
    """Find the connected components of a binary mask, eight-connected.

    Args:
        mask: uint8 or boolean mask.
        keep_labels: Retain the label image on the result, for diagnostics.

    Returns:
        The components, excluding the background.
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = np.ascontiguousarray(mask)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    # Label 0 is the background.
    stats = stats[1:count]
    return Components(
        left=stats[:, _LEFT],
        top=stats[:, _TOP],
        width=stats[:, _WIDTH] - 1,
        height=stats[:, _HEIGHT] - 1,
        area=stats[:, _AREA],
        # OpenCV reports centroids as (x, y); everything here is (row, column).
        centroid=centroids[1:count, ::-1],
        labels=labels if keep_labels else None,
    )


def filter_components(
    components: Components,
    width_bounds: Optional[Tuple[float, float]] = None,
    height_bounds: Optional[Tuple[float, float]] = None,
    density_bounds: Optional[Tuple[float, float]] = None,
    aspect_bounds: Optional[Tuple[float, float]] = None,
) -> Components:
    """Keep only components whose shape is plausible for a hole.

    Args:
        components: The components to filter.
        width_bounds: Inclusive bounds on horizontal extent, in pixels.
        height_bounds: Inclusive bounds on vertical extent, in pixels.
        density_bounds: Inclusive bounds on how filled the bounding box is.
        aspect_bounds: Inclusive bounds on the height to width ratio.

    Returns:
        The components that fall within every bound given.
    """
    keep = np.ones(len(components), dtype=bool)

    for bounds, value in (
        (width_bounds, components.width),
        (height_bounds, components.height),
        (density_bounds, components.density),
    ):
        if bounds is not None:
            keep &= (value >= bounds[0]) & (value <= bounds[1])

    if aspect_bounds is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = components.height / components.width
        keep &= (ratio >= aspect_bounds[0]) & (ratio <= aspect_bounds[1])

    return components.select(keep)


def assign_tracks(
    components: Components,
    edges: RollEdges,
    alignment_grid: np.ndarray,
) -> np.ndarray:
    """Match each component to the track it sits on.

    Position is taken relative to the roll edges at the component's own centre
    row, which is what makes this robust against the roll drifting sideways or
    curving over the length of a scan.

    Args:
        components: Components already filtered down to plausible holes.
        edges: Roll edges for the band the components came from.
        alignment_grid: ``(n, 3)`` array of ``[left, right, tone]``, positions
            as fractions of the roll width.

    Returns:
        Index into ``alignment_grid`` for each component.
    """
    rows = np.clip(components.center_row, 0, len(edges) - 1)
    roll_width = np.maximum(edges.width[rows], 1)
    offset = (components.left - edges.left[rows]) / roll_width

    # Nearest track by the left side of the hole. Tracks are sorted, so a
    # binary search finds the two candidates and we take the closer one.
    track_left = alignment_grid[:, 0]
    right = np.searchsorted(track_left, offset)
    left = np.clip(right - 1, 0, len(track_left) - 1)
    right = np.clip(right, 0, len(track_left) - 1)
    closer_right = np.abs(track_left[right] - offset) < np.abs(
        track_left[left] - offset
    )
    return np.where(closer_right, right, left)


def extract_notes(
    masks_holes: np.ndarray,
    edges: RollEdges,
    alignment_grid: np.ndarray,
    width_bounds: Tuple[float, float],
    row_offset: int = 0,
) -> np.ndarray:
    """Extract the note table for one band.

    Args:
        masks_holes: The band's hole mask.
        edges: Roll edges for the band.
        alignment_grid: ``(n, 3)`` array of ``[left, right, tone]``.
        width_bounds: Inclusive width range, in pixels, for a component to count as a hole.
        row_offset: Row the band starts at, added to the output timings.

    Returns:
        An ``(n, 3)`` int64 array of ``[start_row, end_row, tone]``. Negative
        tones are control codes, not pitches; see ``docs/FORMATS.md``. The
        array is empty if the band holds no holes.
    """
    components = filter_components(
        find_components(masks_holes), width_bounds=width_bounds
    )

    if len(components) == 0:
        return np.empty((0, 3), dtype=np.int64)

    tracks = assign_tracks(components, edges, alignment_grid)

    return np.column_stack(
        (
            components.top.astype(np.int64) + row_offset,
            components.bottom.astype(np.int64) + row_offset,
            alignment_grid[tracks, 2].astype(np.int64),
        )
    )
