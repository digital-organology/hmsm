# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Reading the markings printed onto a roll.

Where a punched hole is a hole, printing is a shape that has to be recognised
for what it is.  A Hupfeld Phonola carries the word "Ped." and a printer's
flower down its bass edge, opening and closing the sustain pedal between
them, and a dotted line just inside them whose distance from the edge is the
dynamic the pianolist is asked for.  An Aeolian Themodist Metrostyle adds a
red tempo line in a lane of its own.

Each of those lanes arrives here as its own mask, cut out by the profile's
:class:`~hmsm.profiles.InkLayer` declarations, and is interpreted according to
the layer's role.  Fragments are gathered band by band and only assembled at
the end of the scan, because both a line and a sequence of pedal markers only
make sense whole.

Everything here is measured in millimetres and converted with the scan's own
resolution, so the same thresholds hold for a 300 dpi and a 600 dpi scan.
Positions across the roll are kept relative to the left paper edge, so a roll
that wanders sideways down a fifty foot scan does not read as a crescendo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.signal

from hmsm.midi.controls import ControlCode
from hmsm.profiles import InkLayer
from hmsm.rolls.binarization import InkMask
from hmsm.rolls.edges import RollEdges
from hmsm.rolls.holes import find_components
from hmsm.units import DEFAULT_DPI, mm_to_px

logger = logging.getLogger(__name__)

#: Smallest and largest piece of ink taken seriously, in square millimetres.
#: Below the first is scanner noise and paper grain, above the second is a
#: label or a title rather than a marking.
MIN_INK_MM2 = 0.3
MAX_INK_MM2 = 500.0

#: Longest gap along the roll between two marks of the same dynamics line, in
#: millimetres. A dotted line leaves a few millimetres between dots and rather
#: more where the ink has failed; beyond this the line has ended.
DYNAMICS_MAX_GAP_MM = 45.0

#: What one mark is worth to a trace, in millimetres of sideways wander. A
#: trace takes a mark in only if the detour to reach it costs less than this,
#: which is what keeps an accent mark or a stray word off the line. It is the
#: only thing that decides what belongs to the line: a step of any length is
#: allowed and simply costs what it is long, so the line may change direction
#: as sharply as it likes and still be followed, while a mark far off it is
#: never worth the detour there and back.
DYNAMICS_MARK_VALUE_MM = 15.0

#: Marks a trace may look back over for its predecessor. Only reached where
#: several pieces of ink share a stretch of roll with the line.
DYNAMICS_LOOKBACK = 48

#: Marks the traced line needs before it is believed to be a line at all.
MIN_DYNAMICS_MARKS = 40

#: Fraction of the inked length of the lane the traced line has to span. A
#: printed line runs the length of the roll; a run of accent marks that
#: happens to trace nicely does not.
MIN_DYNAMICS_COVERAGE = 0.5

#: Window and order for smoothing the reconstructed dynamics line, in
#: millimetres of roll and polynomial degree.
DYNAMICS_SMOOTHING_MM = 42.0
DYNAMICS_SMOOTHING_ORDER = 2

#: Smallest printed pedal marker, in square millimetres. Hupfeld's flower is
#: the small one of the pair and covers rather more than this.
MIN_PEDAL_MM2 = 8.0

#: Fragments of printing closer together than this along the roll are one
#: marker: the dot of a "Ped." is not a marker of its own.
PEDAL_MERGE_MM = 8.0

#: Pedal markers needed before the sequence is believed to be a sequence.
MIN_PEDAL_MARKERS = 6


@dataclass(frozen=True)
class Fragments:
    """Pieces of ink collected from one layer over a whole scan.

    Attributes:
        row: Position along the roll of each fragment's centre, in scan rows.
        column: Position across the roll of each fragment's centre, in pixels
            from the left paper edge.
        left: Left side of each fragment, likewise.
        right: Right side of each fragment, likewise.
        area: Set pixels in each fragment.
    """

    row: np.ndarray
    column: np.ndarray
    left: np.ndarray
    right: np.ndarray
    area: np.ndarray

    def __len__(self) -> int:
        return len(self.row)

    @property
    def width(self) -> np.ndarray:
        return self.right - self.left


@dataclass
class AnnotationCollector:
    """Accumulates printed markings across the bands of a scan.

    Attributes:
        layers: The ink layers the profile declares, which decide both what
            masks arrive here and what is made of each.
        dpi: Resolution the scan is interpreted at.
    """

    layers: Sequence[InkLayer] = ()
    dpi: float = DEFAULT_DPI
    _fragments: Dict[str, List[np.ndarray]] = field(default_factory=dict, repr=False)

    def add_band(
        self, ink: Mapping[str, InkMask], edges: RollEdges, row_offset: int = 0
    ) -> None:
        """Collect the ink fragments in one band.

        Args:
            ink: One mask per layer, as segmentation produced them.
            edges: Roll edges for the band.
            row_offset: Row the band starts at, added to the recorded positions.
        """
        low = mm_to_px(1.0, self.dpi) ** 2 * MIN_INK_MM2
        high = mm_to_px(1.0, self.dpi) ** 2 * MAX_INK_MM2

        for name, lane in ink.items():
            if lane.mask.size == 0:
                continue

            components = find_components(lane.mask)
            if len(components) == 0:
                continue

            keep = (components.area >= low) & (components.area <= high)
            components = components.select(keep)
            if len(components) == 0:
                continue

            rows = np.clip(
                np.rint(components.centroid[:, 0]).astype(np.int64), 0, len(edges) - 1
            )
            # Positions come out of the lane's own coordinates; put them back
            # into the band's, then measure them from the left paper edge.
            origin = edges.left[rows] - lane.column_offset
            self._fragments.setdefault(name, []).append(
                np.column_stack(
                    (
                        rows + row_offset,
                        components.centroid[:, 1] - origin,
                        components.left - origin,
                        components.right - origin,
                        components.area,
                    )
                ).astype(np.float64)
            )

    def fragments(self, name: str) -> Optional[Fragments]:
        """Everything collected from one layer, or None if it stayed empty."""
        parts = self._fragments.get(name)
        if not parts:
            return None
        stacked = np.vstack(parts)
        order = stacked[:, 0].argsort(kind="stable")
        stacked = stacked[order]
        return Fragments(
            row=stacked[:, 0],
            column=stacked[:, 1],
            left=stacked[:, 2],
            right=stacked[:, 3],
            area=stacked[:, 4],
        )

    def _first_with_role(self, role: str) -> Optional[Fragments]:
        for layer in self.layers:
            if layer.role == role:
                found = self.fragments(layer.name)
                if found is None:
                    logger.info(
                        "Layer '%s' (%s) held no printing at all", layer.name, role
                    )
                return found
        return None

    def dynamics_line(self) -> Optional[np.ndarray]:
        """Reconstruct the dynamics line, or None if the roll has none.

        Returns:
            An ``(n, 2)`` array of ``[row, column]``, one entry for every row
            the line spans, with the column measured from the left paper edge.
        """
        found = self._first_with_role("dynamics")
        if found is None:
            return None
        return trace_line(found, self.dpi)

    def pedal_events(self) -> Optional[np.ndarray]:
        """Turn the collected pedal markers into note table rows, or None.

        Returns:
            An ``(n, 3)`` note table of pedal-down spans, or None.
        """
        found = self._first_with_role("pedal")
        if found is None:
            return None
        return pedal_spans(found, self.dpi)

    def report_uninterpreted(self) -> None:
        """Log what was found in layers the pipeline does not read."""
        for layer in self.layers:
            if layer.is_interpreted:
                continue
            found = self.fragments(layer.name)
            logger.info(
                "Layer '%s' has role '%s', which this pipeline does not "
                "interpret; its %d fragment(s) were segmented and discarded",
                layer.name,
                layer.role,
                0 if found is None else len(found),
            )


def trace_line(fragments: Fragments, dpi: float = DEFAULT_DPI) -> Optional[np.ndarray]:
    """Follow a printed line through the marks it is made of.

    A dynamics line is single valued along the roll: at any point down the
    paper it is at one distance from the edge, and from one mark to the next
    it has not moved far. On a Hupfeld roll it shares the lane with accent
    marks, printer's ornament and the maker's own watermark, so which ink is
    the line cannot be decided one mark at a time; it is decided by which
    *sequence* of marks makes a line.

    So this picks the best such sequence outright. Every mark may follow any
    earlier mark within reach along the roll, at a cost of however far
    sideways that step is, and every mark taken in is worth a fixed credit
    against that cost. The best-scoring chain is the line: it takes in the
    dots because they are cheap to reach and leaves out an accent mark a
    thousand pixels off the line because no credit covers the detour there
    and back.

    Args:
        fragments: Ink collected from the layer the line is printed in.
        dpi: Resolution the scan is interpreted at.

    Returns:
        An ``(n, 2)`` int array of ``[row, column]`` covering every row from
        the first mark to the last, or None if there is no line to be found.
    """
    if len(fragments) < MIN_DYNAMICS_MARKS:
        logger.info(
            "Found only %d piece(s) of ink in the dynamics lane, too few for a "
            "line; this roll most likely has none. Skipping.",
            len(fragments),
        )
        return None

    chosen = _trace_marks(
        fragments.row,
        fragments.column,
        max_gap=mm_to_px(DYNAMICS_MAX_GAP_MM, dpi),
        mark_value=mm_to_px(DYNAMICS_MARK_VALUE_MM, dpi),
    )

    if len(chosen) < MIN_DYNAMICS_MARKS:
        logger.info(
            "Only %d of %d piece(s) of ink in the dynamics lane form a line; "
            "treating them as noise and skipping.",
            len(chosen),
            len(fragments),
        )
        return None

    rows = fragments.row[chosen]
    columns = fragments.column[chosen]

    inked = fragments.row[-1] - fragments.row[0]
    coverage = (rows[-1] - rows[0]) / inked if inked > 0 else 0.0
    if coverage < MIN_DYNAMICS_COVERAGE:
        logger.info(
            "The best line through the dynamics lane spans only %.0f%% of the "
            "inked length of the roll, which is too little to be the printed "
            "line. Skipping.",
            coverage * 100,
        )
        return None

    dense_rows = np.arange(int(rows[0]), int(rows[-1]) + 1)
    dense_columns = np.interp(dense_rows, rows, columns)

    window = _odd_at_most(
        min(int(mm_to_px(DYNAMICS_SMOOTHING_MM, dpi)), len(dense_rows))
    )
    if window > DYNAMICS_SMOOTHING_ORDER:
        dense_columns = scipy.signal.savgol_filter(
            dense_columns, window, DYNAMICS_SMOOTHING_ORDER
        )
    else:
        logger.debug("Dynamics line too short to smooth (%d rows)", len(dense_rows))

    logger.debug(
        "Dynamics line traced over %d rows through %d of %d mark(s)",
        len(dense_rows),
        len(chosen),
        len(fragments),
    )

    return np.column_stack((dense_rows, np.rint(dense_columns).astype(np.int64)))


def _trace_marks(
    rows: np.ndarray,
    columns: np.ndarray,
    max_gap: float,
    mark_value: float,
    lookback: int = DYNAMICS_LOOKBACK,
) -> np.ndarray:
    """Find the best-scoring chain of marks through a lane.

    A chain scores ``mark_value`` for every mark it takes in, less the
    sideways distance it travels between them, and may only step between
    marks close enough together along the roll to be the same line. Since
    every step goes forward, the best chain ending at each mark can be built
    up in one pass over the marks in order.

    Args:
        rows: Position of each mark along the roll, ascending.
        columns: Position of each mark across the roll.
        max_gap: Furthest apart along the roll two consecutive marks may be.
        mark_value: What one mark is worth, in pixels of sideways wander.
        lookback: How many earlier marks each mark may follow. Reached only
            where several pieces of ink share a stretch of roll with the line.

    Returns:
        The indices of the chosen marks, ascending.
    """
    count = len(rows)
    score = np.full(count, mark_value, dtype=np.float64)
    came_from = np.full(count, -1, dtype=np.int64)

    for index in range(1, count):
        first = max(0, index - lookback)
        within_reach = rows[index] - rows[first:index] <= max_gap
        if not within_reach.any():
            continue

        step = np.abs(columns[index] - columns[first:index])
        candidates = np.where(within_reach, score[first:index] - step, -np.inf)

        best = int(candidates.argmax())
        if candidates[best] > 0.0:
            score[index] += candidates[best]
            came_from[index] = first + best

    chain = []
    index = int(score.argmax())
    while index >= 0:
        chain.append(index)
        index = int(came_from[index])

    return np.array(chain[::-1], dtype=np.int64)


def pedal_spans(fragments: Fragments, dpi: float = DEFAULT_DPI) -> Optional[np.ndarray]:
    """Pair printed pedal markers into the spans over which the pedal is down.

    The markers alternate: a wide one, the word "Ped.", opens a span and a
    narrow one, a printer's flower, closes it. Where one is missed the
    neighbouring markers decide how to read the one in hand.

    Args:
        fragments: Ink collected from the layer the markers are printed in.
        dpi: Resolution the scan is interpreted at.

    Returns:
        An ``(n, 3)`` note table of pedal-down spans, or None if the markers
        look like noise.
    """
    markers = _merge_into_markers(fragments, mm_to_px(PEDAL_MERGE_MM, dpi))

    big_enough = markers.area >= mm_to_px(1.0, dpi) ** 2 * MIN_PEDAL_MM2
    rows, widths = markers.row[big_enough], markers.width[big_enough]

    if len(rows) < MIN_PEDAL_MARKERS:
        logger.info(
            "Found only %d printed pedal marker(s), which most likely means "
            "the roll has none and these are noise. Skipping.",
            len(rows),
        )
        return None

    is_wide = widths > _split_point(widths)
    spans = _pair_markers(rows, is_wide)

    if not spans:
        logger.info("Pedal markers could not be paired into any spans, skipping.")
        return None

    logger.debug(
        "Paired %d pedal marker(s) into %d span(s)", int(big_enough.sum()), len(spans)
    )
    return np.array(
        [(start, end, int(ControlCode.PEDAL)) for start, end in spans], dtype=np.int64
    )


def _merge_into_markers(fragments: Fragments, distance: float) -> Fragments:
    """Fuse fragments close enough along the roll to be one printed marker."""
    if len(fragments) == 0:
        return fragments

    group = np.zeros(len(fragments), dtype=np.int64)
    if len(fragments) > 1:
        group[1:] = np.cumsum(np.diff(fragments.row) > distance)

    first = np.flatnonzero(np.append(True, np.diff(group) != 0))
    return Fragments(
        row=fragments.row[first],
        column=np.add.reduceat(fragments.column * fragments.area, first)
        / np.add.reduceat(fragments.area, first),
        left=np.minimum.reduceat(fragments.left, first),
        right=np.maximum.reduceat(fragments.right, first),
        area=np.add.reduceat(fragments.area, first),
    )


def _split_point(widths: np.ndarray) -> float:
    """Where to cut a set of marker widths into a narrow and a wide group.

    One step of two-means, seeded on either side of the median. That is
    robust where the plain mean is not: pedal markings come in long runs of
    one kind, and a run of wide ones pulls the mean above the narrow ones.
    """
    middle = np.median(widths)
    narrow = widths[widths <= middle]
    wide = widths[widths > middle]
    if len(narrow) == 0 or len(wide) == 0:
        return float(middle)
    return float((np.median(narrow) + np.median(wide)) / 2)


def _pair_markers(rows: np.ndarray, is_wide: np.ndarray) -> List[Tuple[int, int]]:
    """Read an alternating sequence of wide and narrow markers into spans."""
    spans: List[Tuple[int, int]] = []
    opened_at: Optional[int] = None

    for i, wide in enumerate(is_wide):
        last = i + 1 >= len(is_wide)
        if wide:
            # A wide marker opens a span, unless we are already inside one and
            # the next marker is also wide, in which case this one is really
            # the missing close.
            if not last and (opened_at is None or not is_wide[i + 1]):
                opened_at = int(rows[i])
            elif opened_at is not None:
                spans.append((opened_at, int(rows[i])))
                opened_at = None
        else:
            if opened_at is not None:
                spans.append((opened_at, int(rows[i])))
                opened_at = None
            elif not last and not is_wide[i + 1]:
                opened_at = int(rows[i])

    return spans


def _odd_at_most(value: int) -> int:
    """The largest odd number not greater than ``value``."""
    return value if value % 2 else value - 1
