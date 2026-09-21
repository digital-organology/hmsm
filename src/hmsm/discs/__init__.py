# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Digitization of disc shaped media, such as Ariston cardboard discs.

Unlike rolls, discs are small enough to hold in memory whole, so this pipeline
reads the scan in one go. It has not yet been reworked onto the streaming and
profile machinery the roll pipeline uses; :meth:`~hmsm.profiles.DiscProfile.as_dict`
is the seam between the two.
"""

from __future__ import annotations

import logging

import hmsm.discs.cluster
from hmsm.io import open_source
from hmsm.profiles import DiscProfile

logger = logging.getLogger(__name__)

__all__ = ["process_disc"]


def process_disc(
    input_path: str,
    output_path: str,
    method: str,
    profile: DiscProfile,
    offset: int = 0,
) -> None:
    """Digitize a disc scan to MIDI.

    Args:
        input_path: Path to the scan of the disc.
        output_path: Path to write the MIDI file to.
        method: Digitization method; only ``"cluster"`` exists.
        profile: Geometry of the disc format.
        offset: Rotational offset of the disc's starting position, in degrees.

    Raises:
        ValueError: If ``method`` is not a known digitization method.
    """
    if method != "cluster":
        raise ValueError(f"Unknown disc digitization method '{method}'")

    logger.info("Reading disc scan from %s", input_path)
    with open_source(input_path) as source:
        image = source.read_rows(0, source.height)

    hmsm.discs.cluster.process_disc(
        image, output_path, config=profile.as_dict(), offset=offset
    )
