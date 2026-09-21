# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Allows the CLI to be run as ``python -m hmsm``.

Useful when the console scripts are not on PATH, and it is what a debugger
configuration should point at.
"""

import sys

from hmsm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["hmsm", *sys.argv[1:]]))
