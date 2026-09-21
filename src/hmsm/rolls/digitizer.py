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
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

import hmsm.midi
from hmsm.io import ImageSource, open_source
from hmsm.profiles import HOLE_DENSITY_BOUNDS, RollProfile
from hmsm.rolls import annotations as annotations_module
from hmsm.rolls import notes as notes_module
from hmsm.rolls.binarization import BandMasks, segment
from hmsm.rolls.edges import RollEdges
from hmsm.rolls.holes import extract_notes
from hmsm.rolls.paper import PaperModel, sample_scan
from hmsm.units import DEFAULT_DPI, mm_to_px

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
class RollUpdate:
    """A synchronous preview, valid only during the callback.

    Notes remain in absolute scan rows and include open notes that may grow in
    later updates. Playback must not advance past ``safe_row``. Pixels are a
    view of the current band; consumers should resize or persist them rather
    than retaining full bands. Expression is resolved in the final transcription.
    """

    start: int
    stop: int
    pixels: np.ndarray
    edges: Optional[RollEdges]
    notes: np.ndarray
    safe_row: int


@dataclass
class Transcription:
    """The musical content recovered from a roll scan.

    Attributes:
        notes: ``(n, 3)`` note table of ``[start_row, end_row, tone]``, in scan
            rows. Negative tones are control codes; see :mod:`hmsm.rolls.notes`.
        dynamics: ``(n, 2)`` array of ``[row, column]`` tracing the printed
            dynamics line, or None where the format has none. The column is
            measured from the left paper edge rather than from the edge of the
            scan, so a roll that drifts sideways down a long scan does not
            read as a slow crescendo.
        profile: The profile the scan was read with.
        paper: The paper and background colours the scan was read against.
        dpi: Resolution the scan was interpreted at.
        rows_processed: How many rows of the scan were read before the roll ended.
        origin_row: Absolute scan row subtracted when rebasing the notes.
    """

    notes: np.ndarray
    dynamics: Optional[np.ndarray]
    profile: RollProfile
    paper: Optional[PaperModel] = None
    dpi: float = DEFAULT_DPI
    rows_processed: int = 0
    origin_row: int = 0

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

    def run(
        self,
        source: ImageSource | str,
        skip_rows: int = 0,
        on_update: Optional[Callable[[RollUpdate], None]] = None,
    ) -> Transcription:
        """Digitize a roll scan.

        Args:
            source: An open :class:`~hmsm.io.ImageSource`, or a path to open.
            skip_rows: Rows to skip from the top of the scan, to step past a
                roll head the automatic detection cannot handle.
            on_update: Optional synchronous callback after each band. Exceptions
                propagate, allowing a consumer to cancel processing.

        Raises:
            ValueError: If no holes could be found anywhere in the scan.

        Returns:
            The recovered notes and dynamics.
        """
        opened = isinstance(source, str)
        source = open_source(source) if opened else source
        try:
            return self._run(source, skip_rows, on_update)
        finally:
            if opened:
                source.close()

    def _run(
        self, source: ImageSource, skip_rows: int, on_update=None
    ) -> Transcription:
        dpi = self.dpi or source.dpi or DEFAULT_DPI
        if self.dpi is None and source.dpi and source.dpi != DEFAULT_DPI:
            logger.info("Scan declares a resolution of %.0f dpi", dpi)

        paper = PaperModel.estimate(
            sample_scan(source),
            dpi=dpi,
            background=None if self.background == "guess" else self.background,
        )
        logger.info(
            "Reading %s paper against a %s background",
            _describe(paper.paper),
            paper.background_name,
        )

        if self.debug_dir:
            pathlib.Path(self.debug_dir).mkdir(parents=True, exist_ok=True)

        width_bounds = self.profile.hole_width_bounds_px(dpi)
        height_bounds = self.profile.hole_length_bounds_px(dpi)
        collector = annotations_module.AnnotationCollector(
            self.profile.ink_layers, dpi=dpi
        )

        note_bands: list[np.ndarray] = []
        in_playable_roll = False
        rows_processed = skip_rows

        def publish(start, stop, pixels, edges=None):
            if on_update is None:
                return
            merged = notes_module.merge_notes(
                np.vstack(note_bands) if note_bands else notes_module.empty(),
                self.profile.primary_hole_width_mm,
                dpi,
            )
            # Withhold a full band as well as the merge gap. This also covers
            # fragments rejected at a boundary by the minimum hole length.
            gap = notes_module.MERGE_FACTOR * np.floor(
                mm_to_px(self.profile.primary_hole_width_mm or 0, dpi)
            )
            # Without a nominal width, merge_notes estimates its threshold
            # from the whole scan. Future holes could then change past merges.
            safe_row = (
                max(skip_rows, int(stop - self.band_height - gap))
                if self.profile.primary_hole_width_mm is not None
                else skip_rows
            )
            on_update(
                RollUpdate(
                    start,
                    stop,
                    pixels,
                    edges,
                    merged,
                    safe_row,
                )
            )

        for (start, stop), band, lead in self._bands(source, skip_rows):
            masks = self._segment(band, paper, start, stop)

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
                publish(start, stop, band[lead : lead + stop - start])
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
                    publish(start, stop, band[lead : lead + stop - start])
                    continue
                logger.info("Roll appears to start at row %d", start + head_end)
                masks = masks.crop(head_end)
                start += head_end
                lead += head_end

            if masks.ink:
                collector.add_band(masks.ink, masks.edges, start)

            band_notes = extract_notes(
                masks.holes,
                masks.edges,
                self._alignment_grid,
                width_bounds,
                height_bounds,
                HOLE_DENSITY_BOUNDS,
                start,
            )

            if len(band_notes):
                note_bands.append(band_notes)
                in_playable_roll = True

            rows_processed = stop
            publish(start, stop, band[lead : lead + stop - start], masks.edges)

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

        collector.report_uninterpreted()

        notes = notes_module.merge_notes(notes, self.profile.primary_hole_width_mm, dpi)
        origin_row = int(notes[:, 0].min())
        # Notes and the dynamics line have to share a coordinate system, so
        # both move together when the music is shifted to start at row zero.
        notes, dynamics = notes_module.rebase(notes, dynamics)
        self._dump(notes, "notes_merged.csv")

        logger.info("Post-processing left %d notes", len(notes))

        return Transcription(
            notes=notes,
            dynamics=dynamics,
            profile=self.profile,
            paper=paper,
            dpi=dpi,
            rows_processed=rows_processed,
            origin_row=origin_row,
        )

    def _segment(
        self, band: np.ndarray, paper: PaperModel, start: int, stop: int
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
                paper,
                **self.profile.segmentation_options(),
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


def _describe(colour: np.ndarray) -> str:
    """A colour as an ``#rrggbb`` string, for log messages."""
    return "#%02x%02x%02x" % tuple(int(np.clip(c, 0, 255)) for c in colour)


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
