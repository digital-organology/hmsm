# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Rendering note tables to MIDI.

The note table produced by the roll pipeline is in scan rows, so the first
thing that happens here is converting rows to seconds, which is what the roll
tempo is for: a roll advertised at "60" travels six feet per minute, and at
300 dpi a foot is 3600 rows.

Beyond that this layer applies the things that are not pitches: velocities
from the printed dynamics line, the velocity boost some Hupfeld formats
encode on dedicated tracks, and sustain pedal spans. See ``docs/FORMATS.md``
for the control codes involved.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import mido
import numpy as np

import hmsm.midi.utils
from hmsm.midi.controls import ControlCode
from hmsm.units import DEFAULT_DPI, MM_PER_INCH

logger = logging.getLogger(__name__)

#: MIDI resolution of the files we write, and the tempo they declare. The
#: music is not beat aligned, so the tempo is just a unit of time.
TICKS_PER_BEAT = 960
BPM = 120

#: Velocity applied where a roll carries no dynamics information at all.
VELOCITY_BASE = 64

#: Range the printed dynamics line is mapped onto.
VELOCITY_MIN = 40
VELOCITY_MAX = 100

#: Added on top where a roll's dynamics tracks call for a boost.
VELOCITY_BOOST = 18

#: Highest MIDI note counted as bass for the purpose of the boost tracks.
BASS_RANGE_TOP = 64

#: Hupfeld rolls hold the sustain pedal down by default and lift it while the
#: pedal track is punched. Set this where a format works the other way round.
INVERT_PEDAL = False

#: Ordering of simultaneous events. Releases go out before anything else so a
#: repeated note is not cut off by its own predecessor, and control changes
#: land before the notes they apply to.
_EVENT_ORDER = {"note_off": 0, "control_change": 1, "note_on": 2}

_BASS_BOOST = (int(ControlCode.BASS_BOOST_ON), int(ControlCode.BASS_BOOST_OFF))
_DISCANT_BOOST = (
    int(ControlCode.DISCANT_BOOST_ON),
    int(ControlCode.DISCANT_BOOST_OFF),
)


class MidiGenerator:
    """Builds a MIDI file from roll timings, pitches and control information.

    Attributes:
        fpm: Roll tempo in feet per minute times ten, as annotated on most
            rolls. A roll marked "60" plays at six feet per minute.
        dpi: Resolution the scan the notes came from was read at.
    """

    def __init__(self, fpm: int = 50, dpi: float = DEFAULT_DPI) -> None:
        if fpm <= 0:
            raise ValueError(f"Roll tempo must be positive, got {fpm}")
        self.fpm = fpm
        self.dpi = dpi
        self._midi_file: Optional[mido.MidiFile] = None

    @property
    def seconds_per_row(self) -> float:
        """How long one row of the scan takes to pass the tracker bar."""
        inches_per_minute = self.fpm / 10 * 12
        return 60.0 / (inches_per_minute * self.dpi)

    @property
    def midi_file(self) -> mido.MidiFile:
        """The rendered file.

        Raises:
            RuntimeError: If nothing has been rendered yet.
        """
        if self._midi_file is None:
            raise RuntimeError("Nothing has been rendered yet; call make_midi() first")
        return self._midi_file

    def write(self, path: str) -> None:
        """Write the rendered file to ``path``."""
        self.midi_file.save(path)

    def make_midi(
        self,
        note_start: Sequence[float],
        note_length: Sequence[float],
        midi_note: Sequence[int],
        hole_size_mm: Optional[float] = None,
        dynamics_line: Optional[np.ndarray] = None,
    ) -> mido.MidiFile:
        """Render notes and control information to MIDI.

        Args:
            note_start: Start of each note, in scan rows.
            note_length: Duration of each note, in scan rows.
            midi_note: Pitch of each note, or a negative control code.
            hole_size_mm: Nominal hole size, used when reassembling pedal spans.
            dynamics_line: ``(n, 2)`` array of ``[row, column]`` tracing the
                printed dynamics line, in the same coordinate system as the
                notes. Velocities are derived from it where it is given.

        Returns:
            The rendered file, also retained on the generator.
        """
        if not len(note_start) == len(note_length) == len(midi_note):
            raise ValueError(
                "note_start, note_length and midi_note must be equal length"
            )

        music = np.column_stack(
            (
                np.asarray(note_start, dtype=np.int64),
                np.asarray(note_length, dtype=np.int64),
                np.asarray(midi_note, dtype=np.int64),
            )
        )

        self._warn_about_unknown_codes(music)

        music, has_pedal = self._resolve_pedal(music, hole_size_mm)
        boosts = _boost_spans(music)
        velocities = self._velocities(music, dynamics_line, boosts)

        events = _build_events(music, velocities, has_pedal)
        self._midi_file = _render(events, self.seconds_per_row)
        return self._midi_file

    def _resolve_pedal(
        self, music: np.ndarray, hole_size_mm: Optional[float]
    ) -> tuple[np.ndarray, bool]:
        """Reassemble the punched pedal track into pedal-down spans."""
        pedal_rows = music[:, 2] == int(ControlCode.PEDAL)
        if not pedal_rows.any():
            return music, False

        threshold = (
            1.75 * np.floor(hole_size_mm / MM_PER_INCH * self.dpi)
            if hole_size_mm is not None
            else None
        )
        pedal = hmsm.midi.utils.preprocess_hupfeld_pedal(music[pedal_rows], threshold)
        music = np.vstack((music[~pedal_rows], pedal))
        return music[music[:, 0].argsort(kind="stable")], True

    def _velocities(
        self,
        music: np.ndarray,
        dynamics_line: Optional[np.ndarray],
        boosts: dict,
    ) -> np.ndarray:
        """Work out the velocity for every note in the table."""
        starts = music[:, 0]

        if dynamics_line is None or len(dynamics_line) == 0:
            velocities = np.full(len(music), VELOCITY_BASE, dtype=np.int64)
        else:
            rows, values = dynamics_line[:, 0], dynamics_line[:, 1].astype(np.float64)
            low, high = values.min(), values.max()
            scale = (VELOCITY_MAX - VELOCITY_MIN) / (high - low) if high > low else 0.0
            # The line covers a contiguous run of rows, so a binary search
            # lands on the exact row inside it and clamps to the nearest end
            # outside it, which is what notes before or after the line get.
            index = np.clip(np.searchsorted(rows, starts), 0, len(rows) - 1)
            velocities = ((values[index] - low) * scale + VELOCITY_MIN).astype(np.int64)

        for codes, in_range in (
            (_BASS_BOOST, music[:, 2] <= BASS_RANGE_TOP),
            (_DISCANT_BOOST, music[:, 2] > BASS_RANGE_TOP),
        ):
            spans = boosts.get(codes)
            if spans is None:
                continue
            boosted = in_range & _covered_by(starts, spans)
            velocities[boosted] += VELOCITY_BOOST

        return np.clip(velocities, 1, 127)

    @staticmethod
    def _warn_about_unknown_codes(music: np.ndarray) -> None:
        present = set(np.unique(music[music[:, 2] < 0, 2]).tolist())
        unknown = present - {int(code) for code in ControlCode}
        if unknown:
            logger.warning(
                "No handling is defined for the following control codes in the "
                "provided data; they will be ignored: %s",
                ", ".join(str(code) for code in sorted(unknown)),
            )


def _boost_spans(music: np.ndarray) -> dict:
    """Collect the spans over which each velocity boost track is punched."""
    spans = {}
    for codes in (_BASS_BOOST, _DISCANT_BOOST):
        rows = music[np.isin(music[:, 2], codes)]
        if len(rows):
            spans[codes] = np.column_stack((rows[:, 0], rows[:, 0] + rows[:, 1]))
    return spans


def _covered_by(positions: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Whether each position falls strictly inside any of the spans.

    Counting how many spans have started against how many have already ended
    answers this for every position at once, whether or not the spans overlap.
    """
    started = np.searchsorted(np.sort(spans[:, 0]), positions, side="left")
    ended = np.searchsorted(np.sort(spans[:, 1]), positions, side="right")
    return started > ended


def _build_events(
    music: np.ndarray, velocities: np.ndarray, has_pedal: bool
) -> List[tuple]:
    """Turn the note table into timed MIDI events, in the order to write them.

    Returns:
        ``(row, kind, data1, data2)`` tuples sorted by time.
    """
    events: List[tuple] = []

    if has_pedal:
        # Hupfeld rolls assume the pedal is down; state that explicitly at the
        # top of the file so the file does not depend on the player's defaults.
        events.append((0, "control_change", 64, 127 if INVERT_PEDAL else 0))

    for (start, length, tone), velocity in zip(music, velocities):
        if tone > 0:
            events.append((start, "note_on", tone, int(velocity)))
            events.append((start + length, "note_off", tone, 0))
        elif tone == int(ControlCode.PEDAL):
            events.append((start, "control_change", 64, 0 if INVERT_PEDAL else 127))
            events.append(
                (start + length, "control_change", 64, 127 if INVERT_PEDAL else 0)
            )

    events.sort(key=lambda event: (event[0], _EVENT_ORDER[event[1]]))
    return events


def _render(events: Sequence[tuple], seconds_per_row: float) -> mido.MidiFile:
    """Assemble timed events into a single track MIDI file.

    Args:
        events: ``(row, kind, data1, data2)`` tuples, sorted by time.
        seconds_per_row: How long one scan row takes to play.

    Returns:
        A one track MIDI file.
    """
    tempo = mido.bpm2tempo(BPM)

    midi_file = mido.MidiFile()
    midi_file.ticks_per_beat = TICKS_PER_BEAT
    track = mido.MidiTrack()
    midi_file.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    if not events:
        logger.warning("No events to write; the resulting MIDI file will be empty")
        return midi_file

    # Round absolute positions to ticks and difference them afterwards, so
    # rounding error cannot accumulate over the length of a roll.
    rows = np.fromiter(
        (event[0] for event in events), dtype=np.float64, count=len(events)
    )
    ticks = np.rint(
        [mido.second2tick(row * seconds_per_row, TICKS_PER_BEAT, tempo) for row in rows]
    ).astype(np.int64)
    deltas = np.diff(ticks, prepend=0)

    for (_row, kind, data, value), delta in zip(events, deltas):
        if kind == "control_change":
            track.append(mido.Message(kind, control=data, value=value, time=int(delta)))
        else:
            track.append(mido.Message(kind, note=data, velocity=value, time=int(delta)))

    return midi_file


# Legacy path, still used by the disc pipeline. Discs have no dynamics, pedal
# or control tracks, so they never needed most of MidiGenerator; they will be
# ported onto it when the disc pipeline is reworked.


def create_midi(
    note_start: List[float],
    note_length: List[float],
    midi_note: List[int],
    scaling_factor: float = 1,
) -> mido.MidiFile:
    """Create a MIDI file from timing and pitch information.

    Args:
        note_start: Time each note starts sounding.
        note_length: How long each note sounds.
        midi_note: Pitch of each note, as a MIDI note number.
        scaling_factor: Multiplier from the input time unit to MIDI ticks.

    Returns:
        A one track MIDI file holding the provided music.
    """
    if not len(note_start) == len(note_length) == len(midi_note):
        raise ValueError("note_start, note_length and midi_note must be equal length")

    events = []
    for start, length, note in zip(note_start, note_length, midi_note):
        events.append((start * scaling_factor, "note_on", note))
        events.append(((start + length) * scaling_factor, "note_off", note))

    events.sort(key=lambda event: (event[0], _EVENT_ORDER[event[1]]))

    midi_file = mido.MidiFile()
    midi_file.ticks_per_beat = TICKS_PER_BEAT

    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    previous = 0.0
    for time, kind, note in events:
        track.append(mido.Message(kind, note=note, time=round(time - previous)))
        previous = time

    return midi_file
