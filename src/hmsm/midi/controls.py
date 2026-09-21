# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Control codes carried in note tables alongside pitches.

Rolls encode more than notes: dedicated tracks hold sustain pedal and
velocity information. Those tracks travel through the pipeline in the same
note table as the pitches, distinguished by a negative tone value.

The numbering is historical and documented in ``docs/FORMATS.md``.
"""

from __future__ import annotations

from enum import IntEnum


class ControlCode(IntEnum):
    """Negative tone values that carry control information, not pitch."""

    #: Sustain pedal. Hupfeld rolls hold the pedal down by default and lift it
    #: while this track is punched.
    PEDAL = -3

    #: Velocity boost for the bass half of the keyboard. Used in pairs.
    BASS_BOOST_ON = -10
    BASS_BOOST_OFF = -11

    #: Velocity boost for the discant half of the keyboard. Used in pairs.
    DISCANT_BOOST_ON = -20
    DISCANT_BOOST_OFF = -21
