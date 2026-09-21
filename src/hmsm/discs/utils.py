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


def _fit_ellipse(data: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fit an ellipse to a set of 2D points using a direct least-squares conic fit

    Reimplements the algorithm behind skimage.measure.EllipseModel (Halir & Flusser,
    "Numerically stable direct least squares fitting of ellipses"). skimage takes the
    real part of its result only at the very end, after dividing the (potentially
    complex-typed) angle by pi; whether np.linalg.eig returns a real or a complex
    dtype for this data depends on floating point noise from the BLAS backend, and
    numpy has never supported the modulo operator on complex arrays, so skimage's own
    implementation can raise a TypeError. We take the real part right after the
    eigendecomposition instead, which is equivalent whenever skimage would otherwise
    succeed.

    Args:
        data (np.ndarray): Nx2 array of (row, column) coordinates to fit

    Raises:
        ValueError: Will be raised if an ellipse could not be fit to the given points

    Returns:
        Tuple[float, float, float, float, float]: x0, y0, width, height, phi (semi-axes and rotation in radians)
    """
    if len(data) < 5:
        raise ValueError("Need at least 5 data points to estimate an ellipse")

    data = data.astype(np.float64, copy=False)
    origin = data.mean(axis=0)
    data = data - origin
    scale = data.std()

    if scale < np.finfo(np.float64).tiny:
        raise ValueError(
            "Standard deviation of data is too small to estimate ellipse with meaningful precision"
        )

    data = data / scale

    x = data[:, 0]
    y = data[:, 1]

    # Quadratic and linear parts of the design matrix
    D1 = np.vstack([x**2, x * y, y**2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T

    # Scatter matrix
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2

    # Constraint matrix
    C1 = np.array([[0.0, 0.0, 2.0], [0.0, -1.0, 0.0], [2.0, 0.0, 0.0]])

    try:
        M = np.linalg.inv(C1) @ (S1 - S2 @ np.linalg.inv(S3) @ S2.T)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix while estimating ellipse")

    eig_vals, eig_vecs = np.linalg.eig(M)
    eig_vecs = eig_vecs.real

    # Eigenvector must satisfy 4ac - b^2 > 0 to describe a valid ellipse
    cond = 4 * eig_vecs[0, :] * eig_vecs[2, :] - eig_vecs[1, :] ** 2
    a1 = eig_vecs[:, cond > 0]

    if a1.shape[1] != 1:
        raise ValueError("Eigenvector constraints not met while estimating ellipse")

    a, b, c = a1.ravel()
    a2 = -np.linalg.inv(S3) @ S2.T @ a1
    d, f, g = a2.ravel()

    b /= 2.0
    d /= 2.0
    f /= 2.0

    x0 = (c * d - b * f) / (b**2.0 - a * c)
    y0 = (a * f - b * d) / (b**2.0 - a * c)

    numerator = a * f**2 + c * d**2 + g * b**2 - 2 * b * d * f - a * c * g
    term = np.sqrt((a - c) ** 2 + 4 * b**2)
    denominator1 = (b**2 - a * c) * (term - (a + c))
    denominator2 = (b**2 - a * c) * (-term - (a + c))
    width = np.sqrt(2 * numerator / denominator1)
    height = np.sqrt(2 * numerator / denominator2)

    phi = 0.5 * np.arctan((2.0 * b) / (a - c))
    if a > c:
        phi += 0.5 * np.pi

    if width < height:
        width, height = height, width
        phi += np.pi / 2

    phi %= np.pi

    params = np.nan_to_num([x0, y0, width, height, phi])
    params[:4] *= scale
    params[:2] += origin

    return tuple(params)


def fit_ellipse_to_circumference(image: np.ndarray) -> Tuple[int, int, int, int, float]:
    """Fit ellipse equation to circumference of disc

    This method will try to find the outer edge of the disc in the provided image, fit an ellipse through the points on that edge and return the parameters of the fitted ellipse.

    Args:
        image (np.ndarray): Image of a disc shaped medium, binarization and edge detection will be run automatically

    Returns:
        Tuple[int, int, int, int, float]: xc, yx, a, b, theta as calculated by the direct least-squares ellipse fit
    """
    if image.ndim == 3 or np.unique(image).size > 2:
        image = hmsm.utils.binarize_image(image)

    # Label the image to find the outer edge of the disc

    edges = hmsm.utils.morphological_edge_detection(image)

    labels = skimage.measure.label(edges, background=0, connectivity=2)

    # We can generally assume the outer edge to be the first label, though we might implement additional methods for messier images in the future

    edge = np.argwhere(labels == 1)

    # Fit an ellipse to the outer edge to determine the image center

    center_x, center_y, a, b, theta = _fit_ellipse(edge)
    center_x = int(center_x)
    center_y = int(center_y)
    a = int(a)
    b = int(b)

    return (center_x, center_y, a, b, theta)
