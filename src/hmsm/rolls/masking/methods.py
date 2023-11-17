# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

from typing import Dict

import numpy as np
import skimage.color
import skimage.morphology


def v_channel(
    image: np.ndarray, bg_color: str, threshold: float
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
    return {"holes": image}
