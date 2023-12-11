# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import logging
from typing import List, Optional

import numpy as np
import skimage
import skimage.color
import skimage.filters
import skimage.io
import skimage.measure


def get_initial_alignment_grid(
    roll_width_mm: int,
    track_measurements: List,
) -> np.ndarray:
    """Gets the initial alignment grid from the provided track measurements and converts it into relative positions

    Args:
        roll_width_mm (int): Widht of the roll in mm
        track_measurements (List): List containing information about each track on the roll

    Returns:
        np.ndarray: Initial alignment grid with relative positions
    """
    alignment_grid = np.array([list(v.values()) for v in track_measurements])
    alignment_grid[:, 0:2] = alignment_grid[:, 0:2] / roll_width_mm
    return alignment_grid


def guess_background_color(
    image: np.ndarray,
    n_points: Optional[int] = 1000,
) -> str:
    """Detect the background color of the provided image

    Args:
        image (np.ndarray): Input image to get the background color of
        n_points (Optional[int], optional): Number of sample points to use for detection. Defaults to 1000.

    Returns:
        str: Detected background color, will be either "black" or "white"
    """
    sample_points_y = np.random.choice(
        image.shape[0] - 1,
        n_points,
        replace=False if n_points >= image.shape[0] else True,
    )

    sample_points_x = np.concatenate(
        [
            np.random.choice(10, int(n_points / 2), replace=True),
            np.random.choice(
                np.arange(image.shape[1] - 11, image.shape[1] - 1),
                int(n_points / 2),
                replace=True,
            ),
        ]
    )

    sample_points = image[sample_points_y, sample_points_x]

    sample_points = skimage.color.rgb2hsv(sample_points)

    mean_value = np.mean(sample_points, axis=0)[2]

    if 0.2 <= mean_value <= 0.8:
        logging.warning(
            f"Found inconclusive blackness value of {mean_value} when trying to determine the background color. This might result in problems during further processing."
        )

    return "black" if mean_value < 0.5 else "white"
