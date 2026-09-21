# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""The note table, and turning detected holes into sounding notes.

Between hole detection and MIDI generation the music is carried as an
``(n, 3)`` integer array of ``[start_row, end_row, tone]``, in scan rows.

A negative ``tone`` is a control code rather than a pitch; the codes are
listed in :class:`ControlCode` and documented in ``docs/FORMATS.md``. Any code
that touches a note table has to preserve that convention.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from hmsm.midi.controls import ControlCode
from hmsm.units import DEFAULT_DPI, mm_to_px

logger = logging.getLogger(__name__)

#: Holes separated by less than this multiple of the hole size sound as one
#: note on the original playback mechanism.
MERGE_FACTOR = 1.75

START, END, TONE = 0, 1, 2

__all__ = [
    "ControlCode",
    "MERGE_FACTOR",
    "START",
    "END",
    "TONE",
    "empty",
    "merge_notes",
    "rebase",
]


def empty() -> np.ndarray:
    """An empty note table."""
    return np.empty((0, 3), dtype=np.int64)


def merge_notes(
    notes: np.ndarray,
    hole_width_mm: Optional[float] = None,
    dpi: float = DEFAULT_DPI,
) -> np.ndarray:
    """Join holes on the same track that sound as a single note.

    A pneumatic playback mechanism cannot re-articulate between two holes that
    are close together, so a run of closely spaced holes on one track is one
    long note rather than several short ones.

    Args:
        notes: Note table to merge.
        hole_width_mm: Nominal hole size, used to derive the gap below which
            two holes belong to the same note. Estimated from the note
            durations themselves if not given.
        dpi: Resolution of the scan the notes were detected on.

    Returns:
        A new note table, sorted by start row. Row positions are left as they
        are; use :func:`rebase` to move the start of the music to row zero.
    """
    if len(notes) == 0:
        return empty()

    notes = notes.astype(np.int64, copy=True)

    if hole_width_mm is not None:
        threshold = MERGE_FACTOR * np.floor(mm_to_px(hole_width_mm, dpi))
    else:
        threshold = round(np.mean(notes[:, END] - notes[:, START]) * MERGE_FACTOR)
        logger.debug("Estimated note merging threshold of %d rows", threshold)

    # Sort by track, then by position along the roll, so each track's holes
    # end up in one contiguous, ordered run and the whole merge is a single
    # pass of vector operations rather than a loop per track.
    order = np.lexsort((notes[:, START], notes[:, TONE]))
    notes = notes[order]

    starts_run = np.ones(len(notes), dtype=bool)
    if len(notes) > 1:
        gap = notes[1:, START] - notes[:-1, END]
        same_track = notes[1:, TONE] == notes[:-1, TONE]
        starts_run[1:] = ~(same_track & (gap < threshold))

    run_start = np.flatnonzero(starts_run)
    merged = np.column_stack(
        (
            notes[run_start, START],
            np.maximum.reduceat(notes[:, END], run_start),
            notes[run_start, TONE],
        )
    )

    return merged[merged[:, START].argsort(kind="stable")]


def rebase(
    notes: np.ndarray, dynamics: Optional[np.ndarray] = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Move the start of the music to row zero.

    Scan rows are counted from the top of the image, which includes however
    much roll head and leader the scan happens to contain. Timings should be
    counted from the first note instead.

    The dynamics line is shifted by the same amount, because it has to stay in
    the same coordinate system as the notes for velocities to line up with the
    notes they belong to.

    Args:
        notes: Note table to rebase.
        dynamics: Dynamics line to shift along with it, if there is one.

    Returns:
        The rebased note table and dynamics line.
    """
    if len(notes) == 0:
        return notes, dynamics

    origin = int(notes[:, START].min())
    notes = notes.copy()
    notes[:, START : END + 1] -= origin

    if dynamics is not None:
        dynamics = dynamics.copy()
        dynamics[:, 0] -= origin

    return notes, dynamics
