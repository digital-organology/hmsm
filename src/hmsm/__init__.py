# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""HMSM-Tools: image based digitization of historical musical storage media.

Piano rolls and cardboard discs are scanned, read for their punched holes and
printed annotations, and written out as MIDI; discs can also be rendered back
into images from MIDI.

The pieces are:

- :mod:`hmsm.profiles` -- what a physical format looks like, in millimetres.
- :mod:`hmsm.io` -- reading scans band by band, without holding them in memory.
- :mod:`hmsm.rolls` -- the roll pipeline.
- :mod:`hmsm.discs` -- the disc pipeline.
- :mod:`hmsm.midi` -- rendering note tables to MIDI.
"""

__version__ = "0.10.0"

__all__ = ["__version__"]
