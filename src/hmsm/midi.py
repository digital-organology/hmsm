# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

from typing import List, Optional

import mido


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
