# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import random
from typing import Dict, Optional

import cv2
import numpy as np
import skimage.color
import skimage.morphology


def v_channel(
    generator,
    image: np.ndarray,
    bg_color: str,
    threshold: float,
    upper_threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Creates a mask for the holes on the role by using thresholding on the v channel of the image

    Args:
        image (np.ndarray): Input image
        bg_color (str): Color of the background in the provided image. Currently only "black" and "white" are supported arguments.
        threshold (float): Threshold to use for masking

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the generated mask as numpy array
    """
    # Since we only care about the v_channel we can speed the process up by a factor of about 5 by not calculating the h and s channels
    # This yields very slightly different results than skimage.color.rgb2hsv would give us
    # but since they're in the region of 10^-16 this should be fine

    v_channel = (image / 255).max(axis=2)
    image = v_channel < threshold if bg_color == "black" else v_channel > threshold
    footprint = skimage.morphology.diamond(3)
    image = skimage.morphology.binary_opening(image, footprint)
    image = skimage.morphology.binary_closing(image, footprint)

    if upper_threshold is None:
        return {"holes": image}

    holes_dilated = skimage.morphology.binary_dilation(image, footprint)
    annotations = (np.invert(holes_dilated)) & (v_channel < 0.75)
    annotations = skimage.morphology.binary_opening(annotations, footprint)
    return {"holes": image, "annotations": annotations}


# def _estimate_annotation_pararameters(
#     generator: hmsm.rolls.masking.MaskGenerator,
#     image: np.ndarray,
#     bg_color: str,
#     threshold: float,
# ) -> None:
#     sample_indices = random.sample(
#         range(int(image.shape[0] * 0.2), int(image.shape[0] * 0.8)), 5
#     )
#     stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
#     footprint = skimage.morphology.diamond(3)

#     for idx in sample_indices:
#         chunk = image[idx : idx + 4000, :]
#         v_channel = (chunk / 255).max(axis=2)
#         mask_holes = (
#             v_channel < threshold if bg_color == "black" else v_channel > threshold
#         )
#         mask_holes = skimage.morphology.binary_dilation(mask_holes, footprint)

#         pixels = chunk[mask_holes == False].reshape((-1, 3))
#         pixels = np.float32(pixels)
#         _, labels, (centers) = cv2.kmeans(
#             pixels, 2, None, stop_criteria, 10, cv2.KMEANS_RANDOM_CENTERS
#         )

#     cluster_centers = []
#     pass


# def v_channel_with_rgb_segmentation(
#     generator: MaskGenerator,
#     image: np.ndarray,
#     bg_color: str,
#     threshold: float,
# ) -> Dict[str, np.ndarray]:
#     if not hasattr(generator, "cluster_centers"):
#         _estimate_annotation_pararameters(generator, image, bg_color, threshold)
#     pass
