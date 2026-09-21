# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Shared fixtures.

Tests that need a real scan are skipped where the scan is not present, so the
suite runs on a clean checkout: the example scans are large and not all of
them are in the repository.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


def require(path: pathlib.Path) -> pathlib.Path:
    """Skip the calling test if ``path`` is not available."""
    if not path.exists():
        pytest.skip(f"requires {path.relative_to(REPO_ROOT)}, which is not present")
    return path


@pytest.fixture
def roll_scan() -> pathlib.Path:
    return require(EXAMPLES / "phonola_roll_playback.tif")


@pytest.fixture
def synthetic_roll() -> np.ndarray:
    """A small black-background roll with two punched tracks.

    Built rather than loaded so the tests that use it state exactly what the
    pipeline is supposed to find.
    """
    height, width = 400, 200
    scan = np.zeros((height, width, 3), np.uint8)

    # The paper: light, inset from both sides of the scan.
    scan[:, 20:180] = 200

    # Two tracks of holes, punched through to the dark background.
    for column in (60, 120):
        for row in range(40, height - 40, 80):
            scan[row : row + 30, column : column + 12] = 0

    return scan
