# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Measuring a scan against the paper it is a scan of.

Everything on a roll is defined by how it differs from the paper around it: a
hole lets the scanner background through, ink darkens the paper without
letting anything through.  Thresholding absolute brightness only works as long
as the paper is the one colour the thresholds were tuned for, which holds
neither across a collection whose paper is beige, pink, red or green and has
aged for a century, nor across scanners with black and with white backgrounds.

So rather than absolute brightness this module measures every pixel against
two reference colours, the *paper* and the *background*, and reports three
paper-relative channels:

``transmission``
    Position on the line from paper to background: zero on clean paper, one
    where the background shows through unobstructed.  That is what a hole is,
    whichever colour either of them happens to be.

``shade``
    How much darker than the paper a pixel is, as a fraction.  Ink darkens the
    paper whatever the background does, so this is what printed annotations
    are found with, on black and white background scans alike.

``chroma``
    How far a pixel's colour lies *off* that line, in the two directions
    across it.  Segmentation uses only whether the offset is small, to tell
    heavily printed ink from a punched hole; the signed channels themselves
    are what would tell a red Metrostyle line from a grey dynamics line
    printed on the same roll, and nothing reads them yet.

Transmission and shade are corrected against a *local* estimate of the paper
level, so a scan that is unevenly lit, or paper that has darkened towards one
edge, does not shift the thresholds downstream.  That correction is what lets
one set of thresholds serve every format.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from hmsm.units import DEFAULT_DPI

logger = logging.getLogger(__name__)

#: Side length, in pixels, of the blocks the local paper level is estimated
#: on. Large enough that the biggest piece of ink on a roll cannot dominate a
#: block, small enough to follow the illumination gradients scanners produce.
LEVEL_BLOCK = 128

#: Percentile of each block taken as its paper level. Paper is the bulk of
#: any block once the region outside the roll has been flattened onto it, so
#: the median lands on paper unless a block is more than half hole.
LEVEL_PERCENTILE = 50

#: Every n-th pixel of a block goes into that percentile. A paper level does
#: not need more than a thousand samples and this keeps the temporary array
#: an order of magnitude smaller than the band.
LEVEL_SUBSAMPLE = 4

#: How far the local paper level may be corrected. A block lying mostly
#: outside the roll holds no paper to measure, and without this its "level"
#: would be the background and would rescale the whole block.
LEVEL_LIMIT = 0.25

#: Rows converted to float at a time when projecting a band onto the
#: paper-relative channels. Bounds the temporary the conversion needs.
_TRANSFORM_CHUNK = 256

#: Rows sampled when estimating the reference colours from a whole scan.
_SAMPLE_ROWS = 4000

#: Pixels the reference colours are estimated from, at most.
_ESTIMATE_PIXELS = 2_000_000

#: Indices of the two channels :meth:`PaperModel.channels` returns.
TRANSMISSION, SHADE = 0, 1


class PaperError(RuntimeError):
    """Raised when a scan's reference colours cannot be established."""


@dataclass(frozen=True)
class PaperModel:
    """The two reference colours a roll scan is built out of.

    Attributes:
        paper: Colour of the roll paper, as a ``(3,)`` float32 RGB triple.
        background: Colour the scanner shows through a hole, likewise.
        dpi: Resolution the scan is interpreted at, so that callers can
            express tolerances in millimetres.
    """

    paper: np.ndarray
    background: np.ndarray
    dpi: float = DEFAULT_DPI

    def __post_init__(self) -> None:
        for name in ("paper", "background"):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (3,):
                raise PaperError(f"'{name}' must be an RGB triple, got {value.shape}")
            object.__setattr__(self, name, value)

        if self.contrast < 8.0:
            raise PaperError(
                f"Paper {self.paper} and background {self.background} are too "
                "close to tell apart; this scan holds no usable roll"
            )
        if float(self.paper @ self.paper) <= 0.0:
            raise PaperError("Paper colour is black; nothing can be measured from it")

    @property
    def background_is_dark(self) -> bool:
        """Whether holes read darker than the paper, as on a black background."""
        return bool(self.background.max() < self.paper.max())

    @property
    def background_name(self) -> str:
        """``"black"`` or ``"white"``, for methods that only know the two."""
        return "black" if self.background_is_dark else "white"

    @property
    def contrast(self) -> float:
        """Distance between the two reference colours, in RGB units."""
        return float(np.linalg.norm(self.background - self.paper))

    def channels(self, band: np.ndarray) -> np.ndarray:
        """Compute transmission and shade for a band.

        Both are measured against the reference colours as they stand, which
        leaves in whatever the scanner's lighting and a century of ageing did
        to the paper locally. Taking that out is :func:`localize_channels`,
        and it is a separate step because it needs to be told where the roll
        is first: a block straddling the paper edge would otherwise be
        levelled against the scanner bed rather than against paper.

        Args:
            band: ``(rows, cols, 3)`` uint8 band of the scan.

        Returns:
            A ``(rows, cols, 2)`` float32 array indexed by :data:`TRANSMISSION`
            and :data:`SHADE`.
        """
        # Both channels are affine in a pixel's RGB triple, so one 2x4 matrix
        # produces them together; the fourth column is the constant term.
        direction = self.background - self.paper
        towards_background = direction / float(direction @ direction)
        towards_black = self.paper / float(self.paper @ self.paper)

        matrix = np.array(
            [
                [*towards_background, -float(self.paper @ towards_background)],
                [*(-towards_black), 1.0],
            ],
            dtype=np.float32,
        )

        return _project(band, matrix)

    def chroma(self, band: np.ndarray) -> np.ndarray:
        """How far each pixel's colour lies off the paper-background line.

        Transmission and shade both measure position *along* the line from
        paper to background. This measures the two directions across it: a
        pixel that is paper, or paper with some background showing through,
        sits at zero whatever the mixture, and only a colour that is neither
        does not.

        That is what separates printed ink from a punched hole where darkness
        alone cannot. A hole shows the scanner's own neutral background, so
        however dark it is it stays on the line. Heavily printed ink on beige
        paper is a neutral grey, which is not a mixture of beige and black
        and so sits well off it.

        Both channels are given as a fraction of the paper's own brightness,
        so the scale means the same thing on a light and on a dark paper.
        Segmentation only asks how long this is at the few pixels that might
        be holes, which :meth:`chroma_at` answers far more cheaply; the two
        signed channels are what a format with a red line beside a grey one
        needs to tell them apart, and nothing in the pipeline reads them yet.

        Args:
            band: ``(rows, cols, 3)`` uint8 band of the scan.

        Returns:
            A ``(rows, cols, 2)`` float32 array.
        """
        basis = self._chroma_basis()
        matrix = np.hstack((basis, -(basis @ self.paper).reshape(2, 1)))
        return _project(band, matrix.astype(np.float32))

    def chroma_at(self, band: np.ndarray, where: np.ndarray) -> np.ndarray:
        """The length of :meth:`chroma` at selected pixels only.

        Segmentation asks how far off the line a pixel lies for the handful
        of pixels that are candidate holes, which is a percent or two of a
        band. Asking only for those costs a hundredth of what a full plane
        would.

        Args:
            band: ``(rows, cols, 3)`` uint8 band of the scan.
            where: Flat indices into the band's pixels, as
                :func:`numpy.flatnonzero` of a mask gives them.

        Returns:
            A ``(len(where),)`` float32 array of non-negative offsets.
        """
        basis = self._chroma_basis()
        pixels = band.reshape(-1, 3)[where].astype(np.float32)
        offsets = pixels @ basis.T.astype(np.float32) - (basis @ self.paper).astype(
            np.float32
        )
        return np.hypot(offsets[:, 0], offsets[:, 1])

    def _chroma_basis(self) -> np.ndarray:
        """Orthonormal basis across the paper-background line, scaled to it.

        Scaling by the paper's brightness rather than by the distance to the
        background is what keeps the channel comparable between a black and a
        white background scan of the same roll.
        """
        basis = _perpendicular_basis(self.background - self.paper)
        return basis / float(np.linalg.norm(self.paper))

    @classmethod
    def estimate(
        cls,
        sample: np.ndarray,
        dpi: float = DEFAULT_DPI,
        background: Optional[str] = None,
    ) -> "PaperModel":
        """Read the reference colours off a sample of a scan.

        The paper is the colour of the bulk of the scan and the background the
        colour of one of its extremes, so both fall out of the distribution of
        pixel brightness without anything having to be segmented first.

        Args:
            sample: ``(rows, cols, 3)`` uint8 pixels from the scan. A few
                thousand rows are plenty; see :func:`sample_scan`.
            dpi: Resolution the scan is interpreted at.
            background: Force a ``"black"`` or ``"white"`` background instead
                of taking whichever extreme lies further from the paper.

        Raises:
            PaperError: If the sample is empty, or holds nothing to tell apart.

        Returns:
            The estimated model.
        """
        pixels = np.asarray(sample).reshape(-1, 3)
        if len(pixels) == 0:
            raise PaperError("Cannot estimate paper colour from an empty sample")

        if len(pixels) > _ESTIMATE_PIXELS:
            pixels = pixels[:: len(pixels) // _ESTIMATE_PIXELS + 1]

        value = pixels.max(axis=1)

        # The paper is whatever the scan is mostly made of: the median colour
        # of everything close to the median brightness. Holes, ink and the
        # scanner margins all sit away from it and drop out.
        nearby = np.abs(value.astype(np.int16) - int(np.median(value))) <= 12
        paper = np.median(pixels[nearby] if nearby.any() else pixels, axis=0)

        dark = np.median(pixels[value <= np.percentile(value, 0.5)], axis=0)
        light = np.median(pixels[value >= np.percentile(value, 99.5)], axis=0)

        if background == "black":
            chosen = dark
        elif background == "white":
            chosen = light
        elif background is None:
            # Whichever extreme the paper is further from is the background;
            # the other is just the lightest or darkest paper in the scan.
            chosen = (
                dark
                if np.linalg.norm(dark - paper) > np.linalg.norm(light - paper)
                else light
            )
        else:
            raise PaperError(
                f"Background must be 'black', 'white' or None, got '{background}'"
            )

        model = cls(paper=paper, background=chosen, dpi=dpi)
        logger.debug(
            "Paper colour %s against a %s background %s",
            np.round(model.paper).astype(int),
            model.background_name,
            np.round(model.background).astype(int),
        )
        return model


def sample_scan(source, rows: int = _SAMPLE_ROWS) -> np.ndarray:
    """Read a representative sample of a scan's pixels.

    Slices spread over the whole length of the scan, so that a roll whose
    paper darkens towards one end is represented by both ends.

    Args:
        source: An open :class:`~hmsm.io.ImageSource`.
        rows: Roughly how many rows to read in total.

    Raises:
        PaperError: If the scan holds no pixels at all.

    Returns:
        An ``(n, cols, 3)`` uint8 array.
    """
    slices = 8
    per_slice = max(1, rows // slices)
    starts = dict.fromkeys(
        np.linspace(0, max(source.height - per_slice, 0), slices, dtype=int).tolist()
    )

    read = [source.read_rows(start, start + per_slice) for start in starts]
    read = [chunk for chunk in read if len(chunk)]
    if not read:
        raise PaperError("Scan is empty; cannot sample it")
    return np.vstack(read)


def _project(band: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply an affine colour transform, in chunks of rows.

    ``cv2.transform`` gives its result the depth of its input, so the band has
    to be float for the result to be. Converting it whole would triple the
    memory a band costs; converting a few hundred rows at a time does not.
    """
    rows, columns = band.shape[:2]
    out = np.empty((rows, columns, len(matrix)), dtype=np.float32)

    for start in range(0, rows, _TRANSFORM_CHUNK):
        stop = min(start + _TRANSFORM_CHUNK, rows)
        chunk = band[start:stop].astype(np.float32)
        out[start:stop] = cv2.transform(chunk, matrix)

    return out


def localize_channels(channels: np.ndarray) -> np.ndarray:
    """Re-zero transmission and shade against a local estimate of paper level.

    Paper is never quite one colour: scanners light a 30 cm wide bed
    unevenly, and a roll that spent a century half-exposed is darker down one
    edge than the other. Both shift every channel by a few percent, which is
    the same order as the ink the thresholds have to find, so the level is
    measured per block and taken out.

    Rescaling rather than merely subtracting keeps the far end of each channel
    pinned, so a hole still reads as a transmission of one however dark the
    paper immediately around it happens to be.

    The estimate is the median of each block, which is the paper as long as a
    block is not more than half hole. Callers should flatten everything
    outside the roll onto the paper value first: the scanner background is not
    paper, and a block straddling the roll edge would otherwise be levelled
    against it.

    Args:
        channels: A ``(rows, cols, n)`` float32 array, modified in place.

    Returns:
        The same array.
    """
    level = _block_percentile(channels, LEVEL_BLOCK, LEVEL_PERCENTILE)
    np.clip(level, -LEVEL_LIMIT, LEVEL_LIMIT, out=level)

    # Widen the grid to the band's full width, which leaves it only as tall as
    # the grid, then take it to full height a chunk of rows at a time. Blown
    # up in one go it would be another array the size of the channels, and a
    # band's worth of float allocated and freed for every band is what makes
    # a long scan spend its time in the kernel rather than on the roll.
    rows, columns = channels.shape[:2]
    wide = cv2.resize(level, (columns, len(level)), interpolation=cv2.INTER_LINEAR)
    wide = wide.reshape(len(level), columns, -1)

    for start, stop, block in _row_chunks(rows):
        window = _interpolate_rows(wide, block)
        target = channels[start:stop]
        np.subtract(target, window, out=target)
        np.subtract(1.0, window, out=window)
        np.divide(target, window, out=target)

    return channels


def _row_chunks(rows: int):
    """Chunks of output rows, with where each one falls in a coarse grid."""
    for start in range(0, rows, _TRANSFORM_CHUNK):
        stop = min(start + _TRANSFORM_CHUNK, rows)
        yield start, stop, (np.arange(start, stop) + 0.5) / rows


def _interpolate_rows(grid: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Sample a coarse grid at fractional row positions, linearly.

    ``at`` gives positions as fractions of the full height; the grid's rows
    sit at the centres of the blocks they were measured over, which is where
    ``cv2.resize`` would place them too.
    """
    high = len(grid)
    position = np.clip(at * high - 0.5, 0, high - 1)
    lower = position.astype(np.int64)
    upper = np.minimum(lower + 1, high - 1)
    weight = (position - lower).astype(np.float32)[:, None, None]

    below, above = grid[lower], grid[upper]
    return below + (above - below) * weight


def _block_percentile(planes: np.ndarray, block: int, percentile: float) -> np.ndarray:
    """Percentile of every ``block`` by ``block`` tile of a stack of planes.

    Only every :data:`LEVEL_SUBSAMPLE`-th pixel of a tile is considered, which
    is ample for a paper level and keeps the reshaped copy small.
    """
    step = LEVEL_SUBSAMPLE
    side = max(block // step, 1)

    sparse = planes[::step, ::step]
    rows, columns, count = sparse.shape
    high, wide = max(-(-rows // side), 1), max(-(-columns // side), 1)

    padded = np.pad(
        sparse,
        ((0, high * side - rows), (0, wide * side - columns), (0, 0)),
        mode="edge",
    )
    tiles = padded.reshape(high, side, wide, side, count)

    return np.percentile(tiles, percentile, axis=(1, 3)).astype(np.float32)


def _perpendicular_basis(direction: np.ndarray) -> np.ndarray:
    """An orthonormal basis of the plane perpendicular to ``direction``.

    Returns a ``(2, 3)`` array spanning what is left of colour space once
    movement along the given direction, and with it every mixture of the two
    colours that direction runs between, is taken out.
    """
    _, _, vectors = np.linalg.svd(np.asarray(direction, dtype=np.float64).reshape(1, 3))
    return vectors[1:3]
