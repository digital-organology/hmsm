# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Turning a band of a roll scan into the masks the pipeline works on.

A band is segmented into three things: the *holes* punched into the paper, the
*ink* printed on it where the format carries any, and the *edges* of the paper
itself.  Ink comes back split into the named layers a profile declares, so a
format with a dynamics line down one side and pedal markings down the other
hands the next stage two masks rather than one it has to disentangle.

Methods are registered by name with :func:`binarizer` and selected by a
profile's ``binarization_method``.  Adding a method means writing a function
that takes a band and a :class:`~hmsm.rolls.paper.PaperModel` and returns
:class:`BandMasks`; nothing else has to change.

Two methods ship:

``paper_relative``
    The default. Works in the paper-relative channels of
    :mod:`hmsm.rolls.paper`, so its thresholds mean the same thing whatever
    colour the paper and the scanner background are.

``v_channel``
    The original method: fixed thresholds on absolute pixel brightness. Kept
    because it is cheaper and perfectly adequate on a clean, high-contrast
    scan of beige paper, and because it is the honest baseline to compare
    against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from hmsm.rolls.edges import RollEdges, detect_edges
from hmsm.rolls.paper import SHADE, TRANSMISSION, PaperModel, localize_channels
from hmsm.units import mm_to_px

logger = logging.getLogger(__name__)

#: Luminance weights scikit-image uses for RGB to grayscale (ITU-R BT.709).
_LUMINANCE = np.array([[0.2125, 0.7154, 0.0721]], dtype=np.float32)

#: Transmission at which a pixel is unambiguously a hole, and the lower level
#: the hole is then grown out to. Half way between paper and background is
#: where the edge of a blurred hole actually lies, so growing to it recovers
#: the true extent, while seeding higher keeps paper speckle from starting a
#: hole of its own.
HOLE_SEED = 0.75
HOLE_GROW = 0.5

#: Shade above which a pixel counts as ink. Clean paper sits within about
#: three percent of zero once the local level is taken out, so this is several
#: times the noise and still well below the faintest printing measured on the
#: Hupfeld rolls, which reaches a shade of roughly 0.25.
INK_SHADE = 0.10

#: How far beyond a hole its penumbra reaches, in millimetres. Scanned holes
#: have soft edges; without excluding a margin around them every hole
#: contributes a ring of "ink" and the annotation masks are mostly rings.
HOLE_PENUMBRA_MM = 0.5

#: How far inside the paper edge ink is looked for, in millimetres. The very
#: edge of the paper is a shadow, not printing.
EDGE_INSET_MM = 1.5

#: How far off the paper-background line a dark region's average colour may
#: sit and still be a hole, as a fraction of the paper's brightness.
#: Everything a hole can be made of -- paper, background, and the soft rim
#: where the two blend -- lies on that line, so this only has to clear the
#: scanner's own noise, which measures about 0.008 on the project's scans.
#: The Hupfeld dynamics line, which is what makes the distinction necessary,
#: sits at about 0.033.
HOLE_NEUTRALITY = 0.015

_REGISTRY: Dict[str, Callable[..., "BandMasks"]] = {}


class BinarizationError(RuntimeError):
    """Raised when a band cannot be segmented, e.g. because it holds no roll."""


@dataclass(frozen=True)
class InkMask:
    """The printing found in one lane of a band.

    A lane is a narrow strip of a wide scan, so the mask is cut down to the
    columns the lane can reach rather than kept at the width of the band.
    Everything downstream that reads a position out of it has to add
    :attr:`column_offset` back.

    Attributes:
        mask: uint8 mask of the printing, ``(rows, lane width)``.
        column_offset: Column of the band the mask's first column is.
    """

    mask: np.ndarray
    column_offset: int

    def crop(self, start: int, stop: Optional[int] = None) -> "InkMask":
        """Restrict the mask to rows ``[start, stop)``."""
        return InkMask(self.mask[start:stop], self.column_offset)


@dataclass(frozen=True)
class BandMasks:
    """The masks extracted from one band of a scan.

    Attributes:
        holes: uint8 mask, non-zero where the paper is perforated.
        edges: Where the roll sits in each row of the band.
        ink: One :class:`InkMask` per declared ink layer, keyed by layer name.
            Empty for formats that carry no printed annotations.
    """

    holes: np.ndarray
    edges: RollEdges
    ink: Mapping[str, InkMask] = field(default_factory=dict)

    def crop(self, start: int, stop: Optional[int] = None) -> "BandMasks":
        """Restrict the masks to rows ``[start, stop)`` of the band.

        Used to drop the context rows either side of a band, and to skip past
        a roll head.
        """
        return BandMasks(
            holes=self.holes[start:stop],
            edges=self.edges.crop(start, stop),
            ink={name: mask.crop(start, stop) for name, mask in self.ink.items()},
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


def segment(method: str, band: np.ndarray, paper: PaperModel, **options) -> BandMasks:
    """Segment a band using the named method.

    Args:
        method: Name of a registered method, from a profile's ``binarization_method``.
        band: ``(rows, cols, 3)`` uint8 band of the scan.
        paper: The scan's paper and background colours, and its resolution.
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
        return func(band, paper, **options)
    except TypeError as exc:
        raise BinarizationError(
            f"Options {sorted(options)} do not fit binarization method '{method}': {exc}"
        ) from exc


@binarizer("paper_relative")
def paper_relative(
    band: np.ndarray,
    paper: PaperModel,
    ink_layers: Sequence[Mapping[str, object]] = (),
    hole_seed: float = HOLE_SEED,
    hole_grow: float = HOLE_GROW,
    ink_shade: float = INK_SHADE,
    hole_neutrality: float = HOLE_NEUTRALITY,
) -> BandMasks:
    """Segment a band by how far each pixel departs from the paper.

    Holes are found on the transmission channel, which runs from zero on paper
    to one where the scanner background shows through, so the same thresholds
    hold for a black and for a white background and for paper of any colour.
    Ink is found on the shade channel, because printing darkens paper whatever
    the background is doing.

    Darkness alone does not separate the two: a Hupfeld dynamics line is
    printed heavily enough in places to be as dark as a punched hole. Colour
    does. A hole can only ever show a mixture of paper and scanner
    background, so it stays on the line between those two colours; the ink
    does not, and is held back from the hole mask on that ground.

    Args:
        band: ``(rows, cols, 3)`` uint8 band of the scan.
        paper: The scan's paper and background colours, and its resolution.
        ink_layers: The layers to split the ink mask into, as
            ``{"name": str, "region": (left, right)}`` with the region given
            as fractions of the roll width. Ink outside every declared region
            is dropped. No layers means the format carries no printing.
        hole_seed: Transmission at which a pixel certainly belongs to a hole.
        hole_grow: Transmission out to which that hole is then grown.
        ink_shade: Shade above which a pixel counts as printing.
        hole_neutrality: How far off the paper-background line a region's
            average colour may sit and still be a hole rather than printing.

    Raises:
        BinarizationError: If the thresholds are inconsistent, or the band
            holds no roll.

    Returns:
        The masks for this band.
    """
    if not 0.0 < hole_grow < hole_seed <= 1.0:
        raise BinarizationError(
            f"Hole thresholds must satisfy 0 < hole_grow < hole_seed <= 1, "
            f"got hole_grow={hole_grow}, hole_seed={hole_seed}"
        )
    if not 0.0 < ink_shade < 1.0:
        raise BinarizationError(
            f"'ink_shade' must be a fraction between 0 and 1, got {ink_shade}"
        )

    channels = paper.channels(band)

    # Where the paper is, before any level correction: the raw transmission
    # already separates paper from background by a factor of twenty, and the
    # correction needs the edges rather than the other way round.
    edges = detect_edges(channels[:, :, TRANSMISSION] < hole_grow, invert=False)

    # Flatten everything outside the paper onto the paper value. It is not
    # paper and must not be levelled against, it must not be mistaken for ink,
    # and a hole cannot be found there either.
    _fill_outside_roll(channels, edges)
    localize_channels(channels)

    holes = _find_holes(
        channels[:, :, TRANSMISSION],
        band,
        paper,
        hole_seed,
        hole_grow,
        hole_neutrality,
    )
    holes = _close(_open(holes))

    if not ink_layers:
        return BandMasks(holes=holes, edges=edges)

    ink = (channels[:, :, SHADE] > ink_shade).view(np.uint8)
    del channels
    # Every hole is ringed by a penumbra that shades exactly like faint ink.
    # Cutting a margin around the holes out is what keeps the layers below
    # made of printing rather than of rings.
    penumbra = max(1, int(round(mm_to_px(HOLE_PENUMBRA_MM, paper.dpi))))
    cv2.bitwise_and(ink, cv2.bitwise_not(_grow(holes, penumbra)), dst=ink)
    ink = _open(ink)

    inset = int(round(mm_to_px(EDGE_INSET_MM, paper.dpi)))
    layers = _split_layers(ink, edges, ink_layers, inset)

    return BandMasks(holes=holes, edges=edges, ink=layers)


@binarizer("v_channel")
def v_channel(
    band: np.ndarray,
    paper: PaperModel,
    threshold: float,
    upper_threshold: Optional[float] = None,
    roll_detection_threshold: Optional[str | float] = None,
    ink_layers: Sequence[Mapping[str, object]] = (),
) -> BandMasks:
    """Segment by how dark each pixel is, ignoring hue and saturation.

    The original method, kept as a baseline and for scans the paper-relative
    thresholds do not suit. Holes show the scanner background through, so on a
    black background they are the darkest thing in the scan and a single fixed
    threshold separates them; printed annotations sit between the two, which
    is what ``upper_threshold`` picks out. Both thresholds are absolute, so
    they have to be retuned for every paper colour.

    Args:
        band: ``(rows, cols, 3)`` uint8 band of the scan.
        paper: The scan's paper and background colours. Only which of the two
            is darker is used.
        threshold: Darkness below which a pixel counts as a hole, in ``[0, 1]``.
        upper_threshold: Darkness below which a pixel counts as printing, in
            ``[0, 1]``. None for formats without annotations.
        roll_detection_threshold: Find the paper edges by thresholding the
            scan itself rather than by taking the extent of the hole mask.
            Needed where the background is noisy. ``"auto"`` picks a threshold
            with Otsu's method.
        ink_layers: As for :func:`paper_relative`.

    Returns:
        The masks for this band.
    """
    # Only the value channel of HSV matters here, and that is just the per
    # pixel maximum over the channels. Computing it directly on uint8 avoids
    # converting the whole band to float.
    value = np.maximum(np.maximum(band[:, :, 0], band[:, :, 1]), band[:, :, 2])

    # Work in "darkness" so both background colours follow the same code path.
    darkness = value if paper.background_is_dark else np.subtract(255, value)

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
    ink = cv2.bitwise_and(
        _threshold(darkness, upper_threshold),
        cv2.bitwise_not(_dilate(holes)),
    )
    ink = _open(ink)

    _mask_outside_roll(holes, edges)
    inset = int(round(mm_to_px(EDGE_INSET_MM, paper.dpi)))
    layers = _split_layers(ink, edges, ink_layers, inset)

    return BandMasks(holes=holes, edges=edges, ink=layers)


def _find_holes(
    transmission: np.ndarray,
    band: np.ndarray,
    paper: PaperModel,
    seed: float,
    grow: float,
    neutrality: float,
) -> np.ndarray:
    """Mask the punched holes on the transmission channel.

    Thresholding once would have to choose between cutting into holes and
    letting paper speckle through. Thresholding twice does not: the low level
    puts the boundary where the edge of a blurred hole really is, and the high
    one decides which of the regions so found are holes at all.

    Each surviving region then has to be the right *colour* as well as the
    right darkness. What shows through a hole is the scanner background, so
    every pixel of a hole is a mixture of background and paper; a region whose
    average colour sits off that line is printing dark enough to pass for a
    hole, and is left for the ink mask instead.
    """
    weak = (transmission > grow).view(np.uint8)
    count, labels = cv2.connectedComponents(weak, connectivity=8)
    if count <= 1:
        return weak

    keep = np.zeros(count, dtype=np.uint8)
    keep[labels[transmission > seed]] = 1
    keep[0] = 0

    # Averaging the colour offset over a whole region rather than testing
    # pixel by pixel: printing is not uniformly dark, and its thinner parts
    # are as neutral as any hole.
    inside = np.flatnonzero(weak)
    if len(inside):
        of_region = labels.ravel()[inside]
        offset = paper.chroma_at(band, inside)
        size = np.bincount(of_region, minlength=count)
        total = np.bincount(of_region, weights=offset, minlength=count)
        coloured = total > neutrality * size
        if coloured.any():
            logger.debug(
                "Rejected %d dark region(s) as printing rather than holes",
                int((coloured & keep.astype(bool)).sum()),
            )
        keep[coloured] = 0

    return keep[labels]


def _fill_outside_roll(channels: np.ndarray, edges: RollEdges) -> None:
    """Set everything outside the paper edges to the paper value, in place.

    Multiplying by a mask of the inside rather than assigning through a mask
    of the outside: it comes to the same thing, since the paper value is
    zero on every channel, and it is twice as quick over a whole band.
    """
    columns = np.arange(channels.shape[1], dtype=np.int32)
    inside = (columns[None, :] >= edges.left[:, None]) & (
        columns[None, :] <= edges.right[:, None]
    )
    channels *= inside.view(np.uint8)[:, :, None]


def _split_layers(
    ink: np.ndarray,
    edges: RollEdges,
    layers: Sequence[Mapping[str, object]],
    inset: int,
) -> Dict[str, InkMask]:
    """Cut the ink mask into one mask per declared layer.

    Layers are strips running the length of the roll, given as fractions of
    the roll width, so a marking's position across the roll is what decides
    which layer it belongs to. Positions are taken relative to the paper edges
    row by row, so a roll that drifts sideways across a long scan keeps its
    layers lined up with its printing.

    Each layer comes back cut down to the columns its strip can reach. A
    dynamics lane is a tenth of the width of the scan it was cut from, and
    everything that follows -- connected components above all -- then costs a
    tenth as much.
    """
    if not layers:
        return {}

    left = edges.left.astype(np.float32)[:, None]
    width = np.maximum(edges.width, 1).astype(np.float32)[:, None]
    usable_left = left + inset
    usable_right = edges.right.astype(np.float32)[:, None] - inset

    out: Dict[str, InkMask] = {}
    for layer in layers:
        name = str(layer["name"])
        start, end = (float(x) for x in layer["region"])  # type: ignore[union-attr]

        lane_left = np.maximum(left + start * width, usable_left)
        lane_right = np.minimum(left + end * width, usable_right)

        first = int(max(np.floor(lane_left.min()), 0))
        last = int(min(np.ceil(lane_right.max()), ink.shape[1]))
        if last <= first:
            logger.debug("Ink layer '%s' falls outside the paper in this band", name)
            out[name] = InkMask(np.zeros((ink.shape[0], 0), np.uint8), first)
            continue

        columns = np.arange(first, last, dtype=np.float32)[None, :]
        within = (columns >= lane_left) & (columns < lane_right)
        out[name] = InkMask(
            cv2.bitwise_and(
                np.ascontiguousarray(ink[:, first:last]), within.view(np.uint8)
            ),
            first,
        )
    return out


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


def _grow(mask: np.ndarray, radius: int) -> np.ndarray:
    """Widen a mask by a margin given in millimetre-derived pixels.

    A square margin rather than a round one: this only ever cuts a little
    extra out around a hole, and OpenCV takes a rectangle apart into two
    one-dimensional passes where it has to walk a disc pixel by pixel.
    """
    size = 2 * radius + 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.dilate(mask, element)
