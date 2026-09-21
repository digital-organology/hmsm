# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Driving a roll scan through segmentation, hole detection and MIDI output.

A scan is read one band at a time and each band is segmented, searched for
holes and, where the format has them, for printed annotations.  Bands are
independent once the alignment grid is known, so the reader runs ahead of the
processing rather than taking turns with it.

Nothing here holds more than one band of pixels at a time, which is what makes
multi-gigabyte scans workable.
"""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import numpy as np

import hmsm.midi
from hmsm.io import ImageSource, open_source
from hmsm.profiles import RollProfile
from hmsm.rolls import annotations as annotations_module
from hmsm.rolls import notes as notes_module
from hmsm.rolls.binarization import BandMasks, segment
from hmsm.rolls.edges import RollEdges
from hmsm.rolls.holes import extract_notes
from hmsm.units import DEFAULT_DPI

logger = logging.getLogger(__name__)

#: Default number of scan rows processed at a time. Large enough that the
#: per-band overhead disappears, small enough to stay comfortably in cache-
#: friendly territory and bound memory.
DEFAULT_BAND_HEIGHT = 4000

#: Edge travel, in pixels, above which a band is not a straight stretch of
#: roll. Used to recognise the roll head.
_HEAD_TRAVEL_THRESHOLD = 250

#: Edge travel tolerated within a segment while scanning for where the head
#: ends and the playable roll begins.
_STRAIGHT_TRAVEL_THRESHOLD = 20

#: Granularity of that scan, in rows.
_HEAD_SEARCH_STEP = 100

#: Rows of surrounding scan handed to each band and discarded afterwards, so
#: that morphology and edge smoothing give the same answer regardless of where
#: the band boundaries fall. Comfortably covers both the largest structuring
#: element and the edge smoothing window.
BAND_MARGIN = 32


@dataclass
class Transcription:
    """The musical content recovered from a roll scan.

    Attributes:
        notes: ``(n, 3)`` note table of ``[start_row, end_row, tone]``, in scan
            rows. Negative tones are control codes; see :mod:`hmsm.rolls.notes`.
        dynamics: ``(n, 2)`` array of ``[row, column]`` tracing the printed
            dynamics line, or None where the format has none.
        profile: The profile the scan was read with.
        dpi: Resolution the scan was interpreted at.
        rows_processed: How many rows of the scan were read before the roll ended.
    """

    notes: np.ndarray
    dynamics: Optional[np.ndarray]
    profile: RollProfile
    dpi: float = DEFAULT_DPI
    rows_processed: int = 0

    def to_midi(self, tempo: int = 50) -> "hmsm.midi.MidiGenerator":
        """Render the transcription to MIDI.

        Args:
            tempo: Roll tempo in feet per minute times ten, as annotated on
                most rolls.

        Returns:
            A :class:`hmsm.midi.MidiGenerator` holding the rendered file.
        """
        generator = hmsm.midi.MidiGenerator(tempo, dpi=self.dpi)
        generator.make_midi(
            self.notes[:, notes_module.START].tolist(),
            (
                self.notes[:, notes_module.END] - self.notes[:, notes_module.START]
            ).tolist(),
            self.notes[:, notes_module.TONE].tolist(),
            self.profile.primary_hole_width_mm,
            self.dynamics,
        )
        return generator


@dataclass
class RollDigitizer:
    """Converts roll scans into note tables.

    Attributes:
        profile: The format profile describing the roll.
        band_height: Number of scan rows processed at a time.
        background: Colour of the scan background: ``"black"``, ``"white"`` or
            ``"guess"`` to detect it from the scan.
        dpi: Resolution to interpret physical measurements at. Taken from the
            scan's own metadata when not given.
        read_ahead: Read the next band while the current one is processed.
        band_margin: Rows of context handed to each band, so that results do
            not depend on where the band boundaries fall.
        debug_dir: Where to write diagnostic artefacts, if any.
    """

    profile: RollProfile
    band_height: int = DEFAULT_BAND_HEIGHT
    background: str = "guess"
    dpi: Optional[float] = None
    read_ahead: bool = True
    band_margin: int = BAND_MARGIN
    debug_dir: Optional[str] = None

    _alignment_grid: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.background not in ("black", "white", "guess"):
            raise ValueError(
                f"Background must be one of 'black', 'white' or 'guess', "
                f"got '{self.background}'"
            )
        if self.band_height < 1:
            raise ValueError(f"Band height must be positive, got {self.band_height}")
        self._alignment_grid = self.profile.alignment_grid()

    def run(self, source: ImageSource | str, skip_rows: int = 0) -> Transcription:
        """Digitize a roll scan.

        Args:
            source: An open :class:`~hmsm.io.ImageSource`, or a path to open.
            skip_rows: Rows to skip from the top of the scan, to step past a
                roll head the automatic detection cannot handle.

        Raises:
            ValueError: If no holes could be found anywhere in the scan.

        Returns:
            The recovered notes and dynamics.
        """
        opened = isinstance(source, str)
        source = open_source(source) if opened else source
        try:
            return self._run(source, skip_rows)
        finally:
            if opened:
                source.close()

    def _run(self, source: ImageSource, skip_rows: int) -> Transcription:
        dpi = self.dpi or source.dpi or DEFAULT_DPI
        if self.dpi is None and source.dpi and source.dpi != DEFAULT_DPI:
            logger.info("Scan declares a resolution of %.0f dpi", dpi)

        background = self.background
        if background == "guess":
            background = guess_background(source)
            logger.info("Detected a %s scan background", background)

        if self.debug_dir:
            pathlib.Path(self.debug_dir).mkdir(parents=True, exist_ok=True)

        width_bounds = self.profile.hole_width_bounds_px(dpi)
        collector = annotations_module.AnnotationCollector(self.profile.pedal_cutoff)

        note_bands: list[np.ndarray] = []
        in_playable_roll = False
        rows_processed = skip_rows

        for (start, stop), band, lead in self._bands(source, skip_rows):
            masks = self._segment(band, background, start, stop)

            if masks is not None:
                # Drop the context rows; from here on the masks line up with
                # rows `start` to `stop` of the scan.
                masks = masks.crop(lead, lead + (stop - start))

            if masks is None:
                if in_playable_roll:
                    logger.info(
                        "Band %d-%d could not be segmented, taking this as the "
                        "end of the roll",
                        start,
                        stop,
                    )
                    break
                logger.info(
                    "Band %d-%d could not be segmented; still in the roll head, "
                    "skipping",
                    start,
                    stop,
                )
                continue

            if not in_playable_roll and masks.edges.travel > _HEAD_TRAVEL_THRESHOLD:
                head_end = find_roll_start(masks.edges, source.width)
                if head_end is None:
                    logger.info(
                        "Band %d-%d is not a straight stretch of roll, most "
                        "likely the roll head; skipping",
                        start,
                        stop,
                    )
                    continue
                logger.info("Roll appears to start at row %d", start + head_end)
                masks = masks.crop(head_end)
                start += head_end

            if masks.annotations is not None:
                collector.add_band(masks.annotations, masks.edges, start)

            band_notes = extract_notes(
                masks.holes, masks.edges, self._alignment_grid, width_bounds, start
            )

            if len(band_notes):
                note_bands.append(band_notes)
                in_playable_roll = True

            rows_processed = stop

        if not note_bands:
            raise ValueError(
                "No holes were found anywhere in this scan. Check that the "
                "profile matches the roll and that the background colour is right."
            )

        notes = np.vstack(note_bands)
        self._dump(notes, "notes_raw.csv")

        logger.info("Found %d holes; post-processing", len(notes))

        pedal = collector.pedal_events()
        if pedal is not None:
            logger.info("Recovered %d printed pedal spans", len(pedal))
            notes = np.vstack((notes, pedal))

        dynamics = collector.dynamics_line()
        if dynamics is not None:
            logger.info("Recovered a dynamics line over %d rows", len(dynamics))

        notes = notes_module.merge_notes(notes, self.profile.primary_hole_width_mm, dpi)
        # Notes and the dynamics line have to share a coordinate system, so
        # both move together when the music is shifted to start at row zero.
        notes, dynamics = notes_module.rebase(notes, dynamics)
        self._dump(notes, "notes_merged.csv")

        logger.info("Post-processing left %d notes", len(notes))

        return Transcription(
            notes=notes,
            dynamics=dynamics,
            profile=self.profile,
            dpi=dpi,
            rows_processed=rows_processed,
        )

    def _segment(
        self, band: np.ndarray, background: str, start: int, stop: int
    ) -> Optional[BandMasks]:
        """Segment one band, or None if it cannot be segmented.

        A band that fails to segment carries meaning to the caller: before the
        first holes are seen it is the roll head, afterwards it is the end of
        the paper.
        """
        try:
            return segment(
                self.profile.binarization_method,
                band,
                background,
                **self.profile.binarization_options,
            )
        except Exception as exc:
            logger.debug("Segmentation of band %d-%d failed: %s", start, stop, exc)
            return None

    def _bands(
        self, source: ImageSource, skip_rows: int
    ) -> Iterator[Tuple[Tuple[int, int], np.ndarray, int]]:
        bands = source.context_bands(
            self.band_height, self.band_margin, start=skip_rows
        )
        return _read_ahead(bands) if self.read_ahead else bands

    def _dump(self, array: np.ndarray, name: str) -> None:
        """Write an intermediate table out, when diagnostics are enabled."""
        if not self.debug_dir:
            return
        np.savetxt(os.path.join(self.debug_dir, name), array, delimiter=",", fmt="%d")


def guess_background(source: ImageSource, samples: int = 8) -> str:
    """Work out whether a scan has a black or a white background.

    Reads thin slices spread over the scan and looks at the margins either
    side of the roll, which is where the background shows.

    Args:
        source: The scan to inspect.
        samples: How many slices to look at.

    Returns:
        ``"black"`` or ``"white"``.
    """
    margin = max(1, min(10, source.width // 100))
    rows = np.linspace(0, max(source.height - 1, 0), samples, dtype=int)

    values = []
    for row in rows:
        slice_ = source.read_rows(int(row), int(row) + 1)
        if not len(slice_):
            continue
        edges = np.concatenate((slice_[:, :margin], slice_[:, -margin:]), axis=1)
        values.append(edges.max(axis=2).mean())

    if not values:
        raise ValueError("Scan is empty; cannot determine its background colour")

    brightness = float(np.mean(values)) / 255
    if 0.2 <= brightness <= 0.8:
        logger.warning(
            "Scan margins have an inconclusive brightness of %.2f. Automatic "
            "background detection may be wrong; pass the background explicitly "
            "if the results look off.",
            brightness,
        )

    return "black" if brightness < 0.5 else "white"


def find_roll_start(edges: RollEdges, image_width: int) -> Optional[int]:
    """Find where the roll head ends and the straight, playable roll begins.

    Works backwards from the end of the band, extending the stretch under
    consideration until the roll edges stop being straight.

    Note:
        This is imperfect: rolls whose label extends past the triangular head
        defeat it, which is why stray notes sometimes appear at the start of a
        transcription. Scheduled for rework.

    Args:
        edges: Roll edges for the band.
        image_width: Width of the scan, in pixels.

    Returns:
        Row within the band at which the roll starts, or None if the whole
        band is roll head.
    """
    end = len(edges) - 1

    for start in reversed(range(0, len(edges), _HEAD_SEARCH_STEP)):
        segment_ = edges.crop(start, end)
        if len(segment_) == 0:
            end = start
            continue
        not_straight = segment_.travel > _STRAIGHT_TRAVEL_THRESHOLD
        spans_full_width = (
            segment_.left.mean() < 5 and segment_.right.mean() > image_width * 0.99
        )
        if not_straight or spans_full_width:
            return start if end != len(edges) - 1 else None
        end = start

    return 0


def _read_ahead(bands: Iterator, depth: int = 1) -> Iterator:
    """Decode the next band while the caller is busy with the current one.

    Decoding releases the GIL, so this genuinely overlaps with processing
    rather than just deferring it. The reader is shut down cleanly when the
    caller stops early, which it does at the end of every roll.
    """
    pending: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()
    sentinel = object()

    def hand_over(item) -> bool:
        """Block until the consumer takes ``item``, or until it gives up."""
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def pump() -> None:
        try:
            for item in bands:
                if not hand_over(item):
                    return
        except Exception as exc:  # surface reader failures on the main thread
            hand_over(exc)
        else:
            hand_over(sentinel)

    thread = threading.Thread(target=pump, daemon=True, name="hmsm-band-reader")
    thread.start()

    try:
        while True:
            item = pending.get()
            if item is sentinel:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # The caller may have stopped at the end of the roll with bands still
        # queued; let the reader notice and let go of the scan.
        stop.set()
        while True:
            try:
                pending.get_nowait()
            except queue.Empty:
                break
        thread.join(timeout=5.0)
