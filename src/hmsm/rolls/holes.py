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
        """Fraction of each bounding box that is actually set.

        The box is one pixel wider and taller than the extents, which are
        inclusive, so a solid rectangle comes to exactly one.
        """
        box = (self.width + 1) * (self.height + 1)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Match each component to the track it sits on.

    Position is taken relative to the roll edges at the component's own centre
    row, which is what makes this robust against the roll drifting sideways or
    curving over the length of a scan, and from the *middle* of the hole
    rather than its left side, so that a hole the scan has widened or narrowed
    still lands on its own track.

    A component nowhere near any track is not a hole. The grid divides the
    roll into one cell per track, bounded half way to each neighbour and, at
    the two ends, half a track pitch beyond the outermost track. Anything
    falling outside every cell is rejected rather than snapped to whichever
    track happens to be nearest, which is how dirt in the margin used to
    become a note on the outermost track.

    Args:
        components: Components already filtered down to plausible holes.
        edges: Roll edges for the band the components came from.
        alignment_grid: ``(n, 3)`` array of ``[left, right, tone]``, positions
            as fractions of the roll width.

    Returns:
        The index into ``alignment_grid`` for each component, and a boolean
        mask of the components that fell within a track's cell at all.
    """
    rows = np.clip(components.center_row, 0, len(edges) - 1)
    roll_width = np.maximum(edges.width[rows], 1)
    middle = (components.left + components.right) / 2
    offset = (middle - edges.left[rows]) / roll_width

    centers = alignment_grid[:, 0:2].mean(axis=1)
    boundaries = (centers[:-1] + centers[1:]) / 2

    track = np.searchsorted(boundaries, offset)

    # Between the outermost tracks every component is within half a cell of
    # one of them by construction; beyond them it has to be within half the
    # neighbouring pitch to count as a hole at all.
    if len(centers) > 1:
        first = centers[0] - (centers[1] - centers[0]) / 2
        last = centers[-1] + (centers[-1] - centers[-2]) / 2
        accepted = (offset >= first) & (offset <= last)
    else:
        accepted = np.ones(len(offset), dtype=bool)

    return track, accepted


def extract_notes(
    masks_holes: np.ndarray,
    edges: RollEdges,
    alignment_grid: np.ndarray,
    width_bounds: Tuple[float, float],
    height_bounds: Optional[Tuple[float, float]] = None,
    density_bounds: Optional[Tuple[float, float]] = None,
    row_offset: int = 0,
) -> np.ndarray:
    """Extract the note table for one band.

    Args:
        masks_holes: The band's hole mask.
        edges: Roll edges for the band.
        alignment_grid: ``(n, 3)`` array of ``[left, right, tone]``.
        width_bounds: Inclusive width range, in pixels, for a component to count as a hole.
        height_bounds: Inclusive length range, in pixels, likewise. A hole can
            be as long as the note it sounds, but not longer than the roll's
            longest note, which is what separates a chain of holes from a fold
            or a tear running down the paper.
        density_bounds: Inclusive bounds on how much of its bounding box a
            component fills. A punched hole is convex and fills most of it.
        row_offset: Row the band starts at, added to the output timings.

    Returns:
        An ``(n, 3)`` int64 array of ``[start_row, end_row, tone]``. Negative
        tones are control codes, not pitches; see ``docs/FORMATS.md``. The
        array is empty if the band holds no holes.
    """
    components = filter_components(
        find_components(masks_holes),
        width_bounds=width_bounds,
        height_bounds=height_bounds,
        density_bounds=density_bounds,
    )

    if len(components) == 0:
        return np.empty((0, 3), dtype=np.int64)

    tracks, accepted = assign_tracks(components, edges, alignment_grid)
    if not accepted.all():
        logger.debug(
            "Dropped %d component(s) that sit on no track", int((~accepted).sum())
        )
        components, tracks = components.select(accepted), tracks[accepted]

    return np.column_stack(
        (
            components.top.astype(np.int64) + row_offset,
            components.bottom.astype(np.int64) + row_offset,
            alignment_grid[tracks, 2].astype(np.int64),
        )
    )
