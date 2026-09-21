# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Reading scans without holding them in memory in their entirety."""

from hmsm.io.source import (
    ArraySource,
    ImageSource,
    TiffSource,
    open_source,
)

__all__ = ["ArraySource", "ImageSource", "TiffSource", "open_source"]
