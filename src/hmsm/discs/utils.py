# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import scipy.spatial
import skimage.measure

import hmsm.utils


def transform_to_rectangle(
    image: np.ndarray, offset: Optional[int] = 0, binarize: Optional[bool] = False
) -> np.ndarray:
    """Transforms an image of a circular music storage medium to the shape of a rectangular one

    Args:
        image (np.ndarray): The image to be transformed, can be a binary image as returned by :func:`~hmsm.utils.read_image` or just an image as read by skimage.imread
        offset (Optional[int], optional): The offset (in degrees, counterclockwise) of the beginning of the disc. Defaults to 0.
        binarize (Optional[bool], optional): Weather the output image should be binarized.

    Returns:
        np.ndarray: Image of the transformed medium
    """
    # Apply preprocessing

    if binarize:
        image = hmsm.utils.binarize_image(image)

    image = hmsm.utils.crop_image_to_contents(image.copy())

    logging.info("Determening disc measurements and center")

    center_x, center_y, a, b, theta = fit_ellipse_to_circumference(image)

    # Create a mask for all pixels that are within the disc
    ellipse_mask = cv2.ellipse(
        np.zeros((image.shape[0], image.shape[1]), np.uint8),
        (center_x, center_y),
        (int(a), int(b)),
        theta,
        0,
        360,
        1,
        -1,
    )

    # Right now the disc background is filled while the holes are empty
    # This should be fine, but does significantly increase computational demand
    if binarize:
        image = np.invert(image)

    logging.info("Extracting location information for all pixels")

    # Get all points that are within the disc

    coords = np.argwhere(ellipse_mask == 1)

    values = image[ellipse_mask == 1]

    # Shift points around the center to make the center be (0,0)

    coords = coords - np.array([center_x, center_y])

    # Calculate position in degrees

    degrees = np.arctan2(coords[:, 1], coords[:, 0]) * 180 / np.pi
    degrees = ((np.round(degrees, decimals=1) + 180) * 10).astype(np.int16)

    # Apply offset if applicable

    if not offset == 0:
        logging.info("Applying offset")
        degrees = degrees - (offset * 10)
        degrees[degrees < 0] = degrees[degrees < 0] + 3600

    degrees = degrees.astype(np.uint16)

    # Calculate distances

    dists = np.round(
        scipy.spatial.distance.cdist(np.array([[0, 0]]), coords)[0], decimals=0
    ).astype(np.uint16)

    # Build 2d image

    logging.info("Applying calculated transformations and creating output image")

    dims = (
        (3601, dists.max() + 10)
        if values.ndim == 1
        else (3601, dists.max() + 10, values.shape[1])
    )
    image_rect = np.zeros(dims, np.uint8)
    image_rect[degrees, dists] = values

    logging.info("Interpolating missing pixels in output image")

    mask = np.full((image_rect.shape[0], image_rect.shape[1]), True, bool)
    mask[degrees, dists] = False

    interpolated_image = hmsm.utils.interpolate_missing_pixels(image_rect, mask)

    return interpolated_image


def fit_ellipse_to_circumference(image: np.ndarray) -> Tuple[int, int, int, int, float]:
    """Fit ellipse equation to circumference of disc

    This method will try to find the outer edge of the disc in the provided image, fit an ellipse through the points on that edge and return the parameters of the fitted ellipse.

    Args:
        image (np.ndarray): Image of a disc shaped medium, binarization and edge detection will be run automatically

    Returns:
        Tuple[int, int, int, int, float]: xc, yx, a, b, theta as calculated by skimage.measure.EllipseModel
    """
    if image.ndim == 3 or np.unique(image).size > 2:
        image = hmsm.utils.binarize_image(image)

    # Label the image to find the outer edge of the disc

    edges = hmsm.utils.morphological_edge_detection(image)

    labels = skimage.measure.label(edges, background=0, connectivity=2)

    # We can generally assume the outer edge to be the first label, though we might implement additional methods for messier images in the future

    edge = np.argwhere(labels == 1)

    # Fit an ellipse to the outer edge to determine the image center

    ell = skimage.measure.EllipseModel()
    ell.estimate(edge)
    center_x, center_y, a, b, theta = ell.params
    center_x = int(center_x)
    center_y = int(center_y)
    a = int(a)
    b = int(b)

    return (center_x, center_y, a, b, theta)
