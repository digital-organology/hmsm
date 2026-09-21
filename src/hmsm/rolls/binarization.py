# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Turning a band of a roll scan into the masks the pipeline works on.

A band is segmented into up to three things: the *holes* punched into the
paper, the *printed annotations* on it where the format has any, and the
*edges* of the paper itself.

Methods are registered by name with :func:`binarizer` and selected by a
profile's ``binarization_method``.  Adding a method means writing a function
that takes a band and returns :class:`BandMasks`; nothing else has to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import skimage.filters

from hmsm.rolls.edges import RollEdges, detect_edges

logger = logging.getLogger(__name__)

#: Luminance weights scikit-image uses for RGB to grayscale (ITU-R BT.709).
_LUMINANCE = np.array([[0.2125, 0.7154, 0.0721]], dtype=np.float32)

_REGISTRY: Dict[str, Callable[..., "BandMasks"]] = {}


class BinarizationError(RuntimeError):
    """Raised when a band cannot be segmented, e.g. because it holds no roll."""


@dataclass(frozen=True)
class BandMasks:
    """The masks extracted from one band of a scan.

    Attributes:
        holes: uint8 mask, non-zero where the paper is perforated. Regions
            outside the roll are marked too, so that nothing beyond the paper
            edge can be mistaken for a small hole; they are discarded later by
            the hole width filter.
        edges: Where the roll sits in each row of the band.
        annotations: uint8 mask of printed annotations, or None for formats
            that have none.
    """

    holes: np.ndarray
    edges: RollEdges
    annotations: Optional[np.ndarray] = None

    def crop(self, start: int, stop: Optional[int] = None) -> "BandMasks":
        """Restrict the masks to rows ``[start, stop)`` of the band.

        Used to drop the context rows either side of a band, and to skip past
        a roll head.
        """
        return BandMasks(
            holes=self.holes[start:stop],
            edges=self.edges.crop(start, stop),
            annotations=(
                None if self.annotations is None else self.annotations[start:stop]
            ),
        )


def binarizer(name: str) -> Callable:
    """Register a segmentation method under ``name``."""

    def register(func: Callable[..., BandMasks]) -> Callable[..., BandMasks]:
        _REGISTRY[name] = func
        return func

    return register


def available_methods() -> tuple[str, ...]:
    """Names of all registered segmentation methods."""
    return tuple(sorted(_REGISTRY))


def segment(method: str, band: np.ndarray, bg_color: str, **options) -> BandMasks:
    """Segment a band using the named method.

    Args:
        method: Name of a registered method, from a profile's ``binarization_method``.
        band: ``(rows, cols, 3)`` uint8 band of the scan.
        bg_color: Colour of the scan background, ``"black"`` or ``"white"``.
        **options: Method specific options, from a profile's ``binarization_options``.

    Raises:
        BinarizationError: If the method is unknown or its options do not fit.

    Returns:
        The masks for this band.
    """
    try:
        func = _REGISTRY[method]
    except KeyError:
        raise BinarizationError(
            f"Unknown binarization method '{method}'. "
            f"Available: {', '.join(available_methods())}"
        ) from None

    try:
        return func(band, bg_color, **options)
    except TypeError as exc:
        raise BinarizationError(
            f"Options {sorted(options)} do not fit binarization method '{method}': {exc}"
        ) from exc


@binarizer("v_channel")
def v_channel(
    band: np.ndarray,
    bg_color: str,
    threshold: float,
    upper_threshold: Optional[float] = None,
    roll_detection_threshold: Optional[str | float] = None,
) -> BandMasks:
    """Segment by how dark each pixel is, ignoring hue and saturation.

    Holes show the scanner background through the paper, so on a black
    background they are the darkest thing in the scan and a single threshold
    separates them.  Printed annotations sit between the two, which is what
    ``upper_threshold`` picks out.

    Args:
        band: ``(rows, cols, 3)`` uint8 band of the scan.
        bg_color: Colour of the scan background, ``"black"`` or ``"white"``.
        threshold: Darkness below which a pixel counts as a hole, in ``[0, 1]``.
        upper_threshold: Darkness below which a pixel counts as a printed
            annotation, in ``[0, 1]``. None for formats without annotations.
        roll_detection_threshold: Find the paper edges by thresholding the
            scan itself rather than by taking the extent of the hole mask.
            Needed where the background is noisy. ``"auto"`` picks a threshold
            with Otsu's method.

    Returns:
        The masks for this band.
    """
    # Only the value channel of HSV matters here, and that is just the per
    # pixel maximum over the channels. Computing it directly on uint8 avoids
    # converting the whole band to float.
    value = np.maximum(np.maximum(band[:, :, 0], band[:, :, 1]), band[:, :, 2])

    # Work in "darkness" so both background colours follow the same code path.
    darkness = value if bg_color == "black" else np.subtract(255, value)

    holes = _threshold(darkness, threshold)
    holes = _open(holes)
    holes = _close(holes)

    if roll_detection_threshold is not None:
        edges = detect_edges(_roll_body(band, roll_detection_threshold))
    else:
        edges = detect_edges(holes)

    if upper_threshold is None:
        _mask_outside_roll(holes, edges)
        return BandMasks(holes=holes, edges=edges)

    # Anything darker than the paper but lighter than a hole is printed on the
    # roll. Dilating the holes first keeps their soft edges out of the result.
    annotations = cv2.bitwise_and(
        _threshold(darkness, upper_threshold),
        cv2.bitwise_not(_dilate(holes)),
    )
    annotations = _open(annotations)

    _mask_outside_roll(holes, edges)
    _mask_outside_roll(annotations, edges)

    return BandMasks(holes=holes, edges=edges, annotations=annotations)


def _threshold(darkness: np.ndarray, threshold: float) -> np.ndarray:
    """Mask pixels darker than a threshold given as a fraction of full scale."""
    return (darkness < threshold * 255).view(np.uint8)


def _roll_body(band: np.ndarray, threshold: str | float) -> np.ndarray:
    """Mask the paper itself by thresholding the scan.

    Used where the background is too noisy for the extent of the hole mask to
    be a reliable indicator of where the paper is.
    """
    gray = cv2.transform(band, _LUMINANCE)

    if threshold == "auto":
        cutoff, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        logger.debug("Otsu threshold for roll detection: %.1f/255", cutoff)
    else:
        mask = (gray > float(threshold) * 255).view(np.uint8)

    if mask[0, 0]:
        mask = cv2.bitwise_not(mask) & 1

    return _close(mask, radius=5)


def _mask_outside_roll(mask: np.ndarray, edges: RollEdges) -> None:
    """Fill everything outside the paper edges, in place.

    Marking it rather than clearing it means stray background structures next
    to the paper merge into one huge component, which the hole size filter
    then rejects outright instead of being fooled by its fragments.
    """
    columns = np.arange(mask.shape[1], dtype=np.int32)
    outside = (columns[None, :] < edges.left[:, None]) | (
        columns[None, :] >= edges.right[:, None]
    )
    mask[outside] = 1


def _footprint(radius: int) -> np.ndarray:
    """A diamond shaped structuring element, as scikit-image's ``diamond``."""
    size = 2 * radius + 1
    offsets = np.abs(np.arange(size) - radius)
    return (offsets[:, None] + offsets[None, :] <= radius).astype(np.uint8)


_FOOTPRINTS = {radius: _footprint(radius) for radius in (3, 5)}


def _open(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _FOOTPRINTS[radius])


def _close(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _FOOTPRINTS[radius])


def _dilate(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return cv2.dilate(mask, _FOOTPRINTS[radius])
