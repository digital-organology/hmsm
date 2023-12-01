# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import scipy.signal
import skimage.color
import skimage.filters
import skimage.morphology


def v_channel(
    generator,
    image: np.ndarray,
    bg_color: str,
    threshold: float,
    upper_threshold: Optional[float] = None,
    roll_detection_threshold: Optional[str | float] = None,
) -> Dict[str, np.ndarray]:
    """Creates a mask for the holes on the role by using thresholding on the v channel of the image

    Args:
        image (np.ndarray): Input image
        bg_color (str): Color of the background in the provided image. Currently only "black" and "white" are supported arguments.
        threshold (float): Threshold to use for masking
        upper_threshold (Optional[float]): If provided, all values threshold < value < upper_threshold will be considered to be printed annotations. Defaults to None.
        roll_detection_threshold (Optional[str | float]): To be used for rolls which have a somewhat non uniform background. If provided the area of the roll will be detected using thresholding based binarization. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the generated mask as numpy array
    """
    # Since we only care about the v_channel we can speed the process up by a factor of about 5 by not calculating the h and s channels
    # This yields very slightly different results than skimage.color.rgb2hsv would give us
    # but since the differences are in the region of 10^-16 this should be fine

    if roll_detection_threshold is not None:
        left_edge, right_edge = _get_roll_edges(image, roll_detection_threshold)

    v_channel = (image / 255).max(axis=2)
    image = (
        v_channel < threshold if bg_color == "black" else (1 - v_channel) < threshold
    )
    footprint = skimage.morphology.diamond(3)
    image = skimage.morphology.binary_opening(image, footprint)
    image = skimage.morphology.binary_closing(image, footprint)

    if roll_detection_threshold is not None:
        for i in range(0, len(image)):
            image[i, 0 : (left_edge[i] - 1)] = True
            image[i, (right_edge[i]) :] = True
    else:
        left_edge, right_edge = _get_roll_edges(image)

    if upper_threshold is None:
        return {"holes": image, "edges": (left_edge, right_edge)}

    holes_dilated = skimage.morphology.binary_dilation(image, footprint)
    annotations = (np.invert(holes_dilated)) & (v_channel < 0.75)
    annotations = skimage.morphology.binary_opening(annotations, footprint)

    if roll_detection_threshold is not None:
        for i in range(0, len(annotations)):
            annotations[i, 0 : (left_edge[i] - 1)] = True
            annotations[i, (right_edge[i]) :] = True

    return {
        "holes": image,
        "annotations": annotations,
        "edges": (left_edge, right_edge),
    }


def _get_roll_edges(
    mask: np.ndarray, threshold: Optional[str | float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect roll edges on the given mask

    Args:
        mask (np.ndarray): Mask containing the roll. If threshold is set this is expected to be the original rgb role image.
        threshold (Optional[str  |  float], optional): If set will perform binarization with this threshold and use the resulting mask to detect the roll edges. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Positions of the left and right roll along the y axis of the roll scan.
    """
    if threshold is not None:
        mask = skimage.color.rgb2gray(mask)

        if threshold == "auto":
            threshold = skimage.filters.threshold_otsu(mask)

        mask = mask > threshold

    if mask[0, 0] == True:
        mask = np.invert(mask)

    if threshold is not None:
        mask = skimage.morphology.binary_closing(mask, skimage.morphology.diamond(5))

    rows, cols = np.where(mask)

    _, idx = np.unique(rows, return_index=True)

    left_edge = np.minimum.reduceat(cols, idx)
    left_edge = scipy.signal.savgol_filter(left_edge, 20, 3).round().astype(np.uint16)

    right_edge = np.maximum.reduceat(cols, idx)
    right_edge = scipy.signal.savgol_filter(right_edge, 20, 3).round().astype(np.uint16)

    return (left_edge, right_edge)
