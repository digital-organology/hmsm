# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import skimage.color
import skimage.morphology


def v_channel(image, bg_color, threshold):
    image = skimage.color.rgb2hsv(image)
    image = (
        image[:, :, 2] < threshold
        if threshold == "black"
        else image[:, :, 2] > threshold
    )
    footprint = skimage.morphology.diamond(3)
    image = skimage.morphology.binary_opening(image, footprint)
    image = skimage.morphology.binary_closing(image, footprint)
    image = np.invert(image)
    return {"holes": image}
