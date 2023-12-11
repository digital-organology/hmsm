# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import logging
import math
from typing import List, Optional, Self

import mido
import numpy as np

import hmsm.midi.utils


class MidiGenerator:
    _scaling_factor: float = None

    _midi_file: mido.MidiFile = None

    _control_codes: dict = {
        -3: None,
        -10: None,
        -11: None,
        -20: None,
        -21: None,
    }

    def __init__(self, fpm: Optional[int] = 50) -> Self:
        self._scaling_factor = 1 / ((fpm * 12 * 300) / 600)

    def write_midi(self, path: str) -> None:
        self._midi_file.save(path)

    def make_midi(
        self,
        note_start: List[float],
        note_length: List[float],
        midi_note: List[int],
        hole_size_mm: Optional[float] = None,
        dynamics_line: Optional[np.ndarray] = None,
    ) -> None:
        # TODO: Scheduled for rewrite once we get a better grasp of what we actually need
        assert len(note_start) == len(note_length) == len(midi_note)

        INVERT_PEDAL = False
        VELOCITY_BASE = 64
        VELOCITY_BOOST_AMOUNT = 18
        VELOCITY_BASE_MIN = 40
        VELOCITY_BASE_MAX = 100

        note_start = np.array(note_start)
        note_length = np.array(note_length)
        midi_note = np.array(midi_note)

        music_data = np.column_stack((note_start, note_length, midi_note))

        control_codes = np.unique(music_data[music_data[:, 2] < 0, 2])

        if not all(code in self._control_codes for code in control_codes):
            logging.warning(
                f"Could not find configuration on how to handle the following control codes contained in the provided data, they will be ignored: {', '.join(str(s) for s in (set(control_codes) - self._control_codes.keys()))}"
            )

        events = list()

        if dynamics_line is not None:
            min_vel = int(dynamics_line[:, 1].min())
            max_vel = int(dynamics_line[:, 1].max())

            velocity_factor = (VELOCITY_BASE_MAX - VELOCITY_BASE_MIN) / (
                int(dynamics_line[:, 1].max()) - min_vel
            )

        if -3 in control_codes:
            merging_threshold = (
                1.75 * math.floor(hole_size_mm / 25.4 * 300)
                if hole_size_mm is not None
                else None
            )

            pedal_notes = hmsm.midi.utils.preprocess_hupfeld_pedal(
                music_data[music_data[:, 2] == -3], merging_threshold
            )

            music_data = np.delete(music_data, np.where(music_data[:, 2] == -3), axis=0)

            music_data = np.vstack((music_data, pedal_notes))

            music_data = music_data[music_data[:, 0].argsort()]

            events.append(
                list(
                    [
                        "control_change",
                        0,
                        64,
                        127 if INVERT_PEDAL else 0,
                    ]
                )
            )

        # Retrieve the velocity control information

        has_velocity_control = False

        if np.isin(music_data[:, 2], [-10, -11, -20, -21]).any():
            has_velocity_control = True
            velocity_boost = music_data[
                np.isin(music_data[:, 2], [-10, -11, -20, -21]), :
            ]
            velocity_boost[:, 1] = velocity_boost[:, 0] + velocity_boost[:, 1]
            music_data = np.delete(
                music_data, np.isin(music_data[:, 2], [-10, -11]), axis=0
            )

        # Transform the data into midi events

        for row in music_data:
            if row[2] > 0:
                # Set velocity according to dynamics line or a flat value if there is none
                if dynamics_line is not None:
                    if row[0] < dynamics_line[0, 0]:
                        velocity = int(
                            (dynamics_line[0, 1] - min_vel) * velocity_factor
                            + VELOCITY_BASE_MIN
                        )
                    elif row[0] > dynamics_line[-1, 0]:
                        velocity = int(
                            (dynamics_line[-1, 1] - min_vel) * velocity_factor
                            + VELOCITY_BASE_MIN
                        )
                    else:
                        velocity = int(
                            (dynamics_line[dynamics_line[:, 0] == row[0], 1] - min_vel)
                            * velocity_factor
                            + VELOCITY_BASE_MIN
                        )
                else:
                    velocity = VELOCITY_BASE

                # Apply boost if roll has one

                if has_velocity_control:
                    if row[2] <= 64:
                        velocity = (
                            min(velocity + VELOCITY_BOOST_AMOUNT, 127)
                            if (
                                np.isin(velocity_boost[:, 2], [-10, -11])
                                & (velocity_boost[:, 0] < row[0])
                                & (velocity_boost[:, 1] > row[0])
                            ).any()
                            else velocity
                        )
                    elif row[2] > 64:
                        velocity = (
                            min(velocity + VELOCITY_BOOST_AMOUNT, 127)
                            if (
                                np.isin(velocity_boost[:, 2], [-20, -21])
                                & (velocity_boost[:, 0] < row[0])
                                & (velocity_boost[:, 1] > row[0])
                            ).any()
                            else velocity
                        )

                events.append(
                    list(["note_on", row[0] * self._scaling_factor, row[2], velocity])
                )
                events.append(
                    list(
                        [
                            "note_off",
                            (row[0] + row[1]) * self._scaling_factor,
                            row[2],
                            0,
                        ]
                    )
                )
            elif row[2] == -3:
                events.append(
                    list(
                        [
                            "control_change",
                            row[0] * self._scaling_factor,
                            64,
                            0 if INVERT_PEDAL else 127,
                        ]
                    )
                )
                events.append(
                    list(
                        [
                            "control_change",
                            (row[0] + row[1]) * self._scaling_factor,
                            64,
                            127 if INVERT_PEDAL else 0,
                        ]
                    )
                )
            else:
                continue

        # Sort by event time

        events.sort(key=lambda x: x[1])

        # Convert into time deltas and also round to ticks

        tempo = mido.bpm2tempo(120)
        TICKS_PER_BEAT = 960

        events = [
            [
                events[n][0],
                round(
                    mido.second2tick(
                        events[n][1] - events[n - 1][1],
                        TICKS_PER_BEAT,
                        tempo,
                    )
                )
                if n > 0
                else round(
                    mido.second2tick(
                        events[n][1],
                        TICKS_PER_BEAT,
                        tempo,
                    )
                ),
                events[n][2],
                *(list() if len(events[n]) <= 3 else events[n][3:]),
            ]
            for n in range(0, len(events))
        ]

        # Initialize midi file

        midi_file = mido.MidiFile()
        midi_file.ticks_per_beat = TICKS_PER_BEAT

        track = mido.MidiTrack()
        midi_file.tracks.append(track)

        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

        for event in events:
            if event[0] == "control_change":
                track.append(
                    mido.Message(
                        event[0], value=event[3], control=event[2], time=event[1]
                    )
                )
            else:
                track.append(
                    mido.Message(
                        event[0], note=event[2], time=event[1], velocity=event[3]
                    )
                )

        self._midi_file = midi_file


# Maintained for backwards compatability until we port the disc processing to the new generator class


def create_midi(
    note_start: List[float],
    note_length: List[float],
    midi_note: List[int],
    scaling_factor: Optional[float] = 1,
) -> mido.MidiFile:
    """Create a midi file from timing and pitch information

    This method will take care of converting the provided information into midi events as well as transforming absolute timings into time deltas.

    Args:
        note_start (List[float]): List containing the time each note is supposed to start sounding
        note_length (List[float]): List containing the duration each note is supposed to sound
        midi_note (List[int]): List containing the pitch information for each note, as midi number

    Returns:
        mido.MidiFile: MidiFile object containing the provided musical information
    """

    assert len(note_start) == len(note_length) == len(midi_note)

    # Transform the data into midi events

    events = list()

    for i in range(len(note_start)):
        events.append(list(["note_on", note_start[i] * scaling_factor, midi_note[i]]))
        events.append(
            list(
                [
                    "note_off",
                    (note_start[i] + note_length[i]) * scaling_factor,
                    midi_note[i],
                ]
            )
        )

    # Sort by event time

    events.sort(key=lambda x: x[1])

    # Convert into time deltas and also round to ticks

    events = [
        [
            events[n][0],
            round(events[n][1] - events[n - 1][1]) if n > 0 else round(events[n][1]),
            events[n][2],
        ]
        for n in range(0, len(events))
    ]

    # Initialize midi file

    midi_file = mido.MidiFile()
    midi_file.ticks_per_beat = 960

    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    for event in events:
        track.append(mido.Message(event[0], note=event[2], time=event[1]))

    return midi_file
