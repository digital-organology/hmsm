# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Row-wise access to scans that are too large to hold in memory.

Roll scans routinely reach two gigabytes decoded, but the pipeline only ever
looks at a few thousand rows at a time.  An :class:`ImageSource` exposes a scan
as something you can read horizontal bands out of, so the pipeline never needs
the whole thing resident.

:func:`open_source` picks an implementation.  Strip and tile based TIFFs get
:class:`TiffSource`, which decodes only the segments a requested band touches.
Anything else is decoded once into an :class:`ArraySource`, which has the same
interface so callers never need to care which one they got.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))

import logging
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Segments are decoded concurrently; the decoders release the GIL, so this
#: scales with cores rather than being bound by Python.
DEFAULT_WORKERS = min(8, (os.cpu_count() or 1))


class ImageSource(ABC):
    """A scan that can be read one horizontal band at a time.

    Attributes:
        height: Number of rows in the scan, after any cropping applied by the source.
        width: Number of columns in the scan.
        channels: Number of colour channels; always 3 for the scans we process.
        dpi: Resolution of the scan, taken from its metadata where available.
    """

    height: int
    width: int
    channels: int
    dpi: Optional[float] = None

    @abstractmethod
    def read_rows(self, start: int, stop: int) -> np.ndarray:
        """Read rows ``[start, stop)`` as an ``(n, width, channels)`` uint8 array."""

    def bands(
        self, band_height: int, start: int = 0, stop: Optional[int] = None
    ) -> Iterator[Tuple[Tuple[int, int], np.ndarray]]:
        """Iterate over the scan in bands of at most ``band_height`` rows.

        Args:
            band_height: Maximum number of rows per band.
            start: First row to read.
            stop: Row to stop before. Defaults to the end of the scan.

        Yields:
            ``((start_row, stop_row), pixels)`` for each band, in order. Row
            indices are absolute, counted from the top of the scan.
        """
        stop = self.height if stop is None else min(stop, self.height)
        for row in range(start, stop, band_height):
            end = min(row + band_height, stop)
            yield (row, end), self.read_rows(row, end)

    def context_bands(
        self,
        band_height: int,
        margin: int,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> Iterator[Tuple[Tuple[int, int], np.ndarray, int]]:
        """Iterate over bands, each padded with rows of surrounding context.

        Operations with a spatial footprint, such as morphology and smoothing,
        give different answers at the edge of a band than they would in the
        middle of an uninterrupted scan. Handing them a few rows either side
        of the band and discarding those rows afterwards removes that
        difference, so results no longer depend on where the band boundaries
        happen to fall.

        Args:
            band_height: Rows per band, not counting the context.
            margin: Rows of context to include either side, where available.
            start: First row to process.
            stop: Row to stop before. Defaults to the end of the scan.

        Yields:
            ``((start_row, stop_row), pixels, lead)`` where ``start_row`` and
            ``stop_row`` bound the band proper, ``pixels`` also covers the
            context, and ``lead`` is how many context rows precede the band
            within ``pixels``.
        """
        stop = self.height if stop is None else min(stop, self.height)
        for row in range(start, stop, band_height):
            end = min(row + band_height, stop)
            lead = min(margin, row - start)
            trail = min(margin, stop - end)
            yield (row, end), self.read_rows(row - lead, end + trail), lead

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.height, self.width, self.channels)

    def close(self) -> None:
        """Release any resources held by the source."""

    def __enter__(self) -> "ImageSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class ArraySource(ImageSource):
    """An :class:`ImageSource` backed by an array that is already in memory."""

    def __init__(self, image: np.ndarray, dpi: Optional[float] = None) -> None:
        image = _as_rgb(image)
        self._image = image
        self.height, self.width, self.channels = image.shape
        self.dpi = dpi

    def read_rows(self, start: int, stop: int) -> np.ndarray:
        return self._image[start : min(stop, self.height)]

    @property
    def array(self) -> np.ndarray:
        """The backing array. Only available on this implementation."""
        return self._image


class TiffSource(ImageSource):
    """Streams bands out of a strip or tile based TIFF.

    Only the segments that overlap a requested band are read from disk and
    decoded, which keeps memory proportional to the band rather than to the
    scan.  Decoding runs on a thread pool because the codecs release the GIL.
    """

    def __init__(self, path: str, workers: int = DEFAULT_WORKERS) -> None:
        import tifffile

        self._file = tifffile.TiffFile(path)
        page = self._file.pages[0]
        self._page = page

        # (rows, cols, samples) per segment and how many segments there are
        # along each of those axes.  This addresses strips and tiles alike:
        # a strip is just a tile that spans the full width.
        self._seg_shape = tuple(page.chunks)
        self._seg_grid = tuple(page.chunked)

        self.height, self.width = int(page.shape[0]), int(page.shape[1])
        self.channels = int(page.shape[2]) if len(page.shape) > 2 else 1
        self.dtype = page.dtype
        self.dpi = _tiff_dpi(page)

        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers))
        self._handle = self._file.filehandle

    @staticmethod
    def supports(path: str) -> bool:
        """Whether ``path`` is a TIFF this class can stream out of."""
        try:
            import tifffile
        except ImportError:  # pragma: no cover - tifffile is a hard dependency
            return False
        try:
            with tifffile.TiffFile(path) as handle:
                page = handle.pages[0]
                # Planar (channel separated) and non-8-bit scans are rare enough
                # in practice that decoding them wholesale is acceptable.
                return (
                    len(handle.pages) >= 1
                    and page.dtype == np.uint8
                    and len(page.shape) == 3
                    and page.shape[2] in (3, 4)
                    and len(page.chunked) == 3
                    and page.chunked[2] == 1
                    and page.chunks[2] == page.shape[2]
                )
        except Exception:
            return False

    def _segment(self, index: int) -> Tuple[int, int, np.ndarray]:
        """Read and decode segment ``index``; returns its origin and pixels."""
        with self._lock:
            self._handle.seek(self._page.dataoffsets[index])
            data = self._handle.read(self._page.databytecounts[index])
        decoded, _position, shape = self._page.decode(data, index)
        rows, cols, _ = np.unravel_index(index, self._seg_grid)
        return (
            rows * self._seg_shape[0],
            cols * self._seg_shape[1],
            decoded.reshape(shape)[0],
        )

    def read_rows(self, start: int, stop: int) -> np.ndarray:
        start = max(0, start)
        stop = min(stop, self.height)
        if stop <= start:
            return np.empty((0, self.width, self.channels), self.dtype)

        seg_rows = self._seg_shape[0]
        first, last = start // seg_rows, (stop - 1) // seg_rows + 1
        cols = self._seg_grid[1]

        # Decode into a buffer aligned to segment boundaries, then hand back the
        # requested slice of it, so partial segments at either end just work.
        buffer = np.empty(
            ((last - first) * seg_rows, self.width, self.channels), self.dtype
        )
        indices = [
            row * cols + col for row in range(first, last) for col in range(cols)
        ]

        for y, x, pixels in self._pool.map(self._segment, indices):
            y -= first * seg_rows
            # Segments on the right and bottom edge are padded to full size by
            # the format, so clip rather than trusting the segment shape.
            rows = min(pixels.shape[0], buffer.shape[0] - y)
            width = min(pixels.shape[1], self.width - x)
            buffer[y : y + rows, x : x + width] = pixels[:rows, :width]

        band = buffer[start - first * seg_rows : stop - first * seg_rows]
        return band[:, :, :3] if self.channels == 4 else band

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._file.close()


def open_source(path: str, workers: int = DEFAULT_WORKERS) -> ImageSource:
    """Open ``path`` as an :class:`ImageSource`.

    Args:
        path: Path to the scan to read.
        workers: Number of threads to decode segments with, where streaming applies.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Returns:
        A streaming source where the file format allows it, an in-memory one otherwise.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such image file: '{path}'")

    if TiffSource.supports(path):
        source = TiffSource(path, workers=workers)
        logger.info(
            "Streaming %s (%d x %d, %s segments of %d rows)",
            path,
            source.height,
            source.width,
            source._seg_grid[0] * source._seg_grid[1],
            source._seg_shape[0],
        )
        return source

    import skimage.io

    logger.info("Reading %s into memory (format does not support streaming)", path)
    image = skimage.io.imread(path)
    return ArraySource(image)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Coerce a decoded image to three channel RGB."""
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        logger.info("Image appears to have an alpha channel which will be dropped")
        return image[:, :, :3]
    return image


def _tiff_dpi(page) -> Optional[float]:
    """Read the horizontal resolution off a TIFF page, in dots per inch."""
    try:
        numerator, denominator = page.tags["XResolution"].value
        unit = page.tags["ResolutionUnit"].value
        dpi = numerator / denominator
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    if int(unit) == 3:  # centimetre
        dpi *= 2.54
    return float(dpi)
