# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Digitization of piano rolls.

The pipeline reads a scan in horizontal bands and runs each through four
stages: :mod:`~hmsm.rolls.binarization` separates holes, printed annotations
and the paper edges; :mod:`~hmsm.rolls.edges` tracks where the paper sits;
:mod:`~hmsm.rolls.holes` turns the holes into a note table; and
:mod:`~hmsm.rolls.annotations` reassembles the dynamics line and pedal marks.
:class:`~hmsm.rolls.digitizer.RollDigitizer` drives all of it.

Typical use::

    from hmsm.profiles import load_profile
    from hmsm.rolls import RollDigitizer

    transcription = RollDigitizer(load_profile("phonola")).run("scan.tif")
    transcription.to_midi(tempo=60).write("out.mid")
"""

from hmsm.rolls.annotations import AnnotationCollector
from hmsm.rolls.binarization import (
    BandMasks,
    BinarizationError,
    available_methods,
    binarizer,
    segment,
)
from hmsm.rolls.digitizer import (
    DEFAULT_BAND_HEIGHT,
    RollDigitizer,
    Transcription,
    find_roll_start,
    guess_background,
)
from hmsm.rolls.edges import RollEdges, detect_edges
from hmsm.rolls.holes import (
    Components,
    assign_tracks,
    extract_notes,
    filter_components,
    find_components,
)
from hmsm.rolls.notes import ControlCode, merge_notes, rebase

__all__ = [
    "AnnotationCollector",
    "BandMasks",
    "BinarizationError",
    "Components",
    "ControlCode",
    "DEFAULT_BAND_HEIGHT",
    "RollDigitizer",
    "RollEdges",
    "Transcription",
    "assign_tracks",
    "available_methods",
    "binarizer",
    "detect_edges",
    "extract_notes",
    "filter_components",
    "find_components",
    "find_roll_start",
    "guess_background",
    "merge_notes",
    "rebase",
    "roll_to_midi",
    "segment",
]


def roll_to_midi(
    input_path: str,
    output_path: str,
    profile,
    background: str = "guess",
    band_height: int = DEFAULT_BAND_HEIGHT,
    skip_rows: int = 0,
    tempo: int = 50,
    debug_dir: str | None = None,
) -> Transcription:
    """Digitize a roll scan and write the result to a MIDI file.

    Args:
        input_path: Path to the roll scan.
        output_path: Path to write the MIDI file to.
        profile: The :class:`~hmsm.profiles.RollProfile` describing the format.
        background: ``"black"``, ``"white"`` or ``"guess"``.
        band_height: Number of scan rows to process at a time.
        skip_rows: Rows to skip from the top of the scan.
        tempo: Roll tempo in feet per minute times ten.
        debug_dir: Where to write diagnostic artefacts, if wanted.

    Returns:
        The transcription that was written.
    """
    digitizer = RollDigitizer(
        profile=profile,
        band_height=band_height,
        background=background,
        debug_dir=debug_dir,
    )
    transcription = digitizer.run(input_path, skip_rows=skip_rows)
    transcription.to_midi(tempo).write(output_path)
    return transcription
