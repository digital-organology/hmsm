import logging
import os
import sys
import numpy as np
import cv2
from typing import Optional
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import img_as_ubyte

def read_as_binary_image(path: str, threshold: Optional[int] = None) -> np.ndarray:
    if not os.path.isfile(path):
        logging.error(f"The system could not find the specified file at path '{path}', aborting")
        sys.exit()

    img = imread(path)

    logging.info(f"Image read from '{path}'")

    img = rgb2gray(img)

    if threshold is None:
        threshold = threshold_otsu(img)
        logging.info(f"No threshold value was specified for binarization, used Otsu's method to estimate threshold to be {threshold * 255}")
    else:
        threshold = (threshold / 255)

    img_bin = img > threshold
    img_bin = np.invert(img_bin)

    return img_as_ubyte(img_bin)

def crop_image_to_contents(image: np.ndarray) -> np.ndarray:
    y_values, x_values = np.nonzero(image)

    y_min = y_values.min() - 20 if y_values.min() - 20 >= 0 else 0
    y_max = y_values.max() + 20 if y_values.max() + 20 <= image.shape[0] - 1 else image.shape[0] - 1

    x_min = x_values.min() - 20 if x_values.min() - 20 >= 0 else 0
    x_max = x_values.max() + 20 if x_values.max() + 20 <= image.shape[1] - 1 else image.shape[1] - 1

    cropped_image = image.copy()[y_min:y_max, x_min:x_max]
    return cropped_image

def morphological_edge_detection(image: np.ndarray, n_erosions: Optional[int] = 2) -> np.ndarray:
    kernel = np.ones((3,3), np.uint8)

    for n in range(n_erosions):
        image = cv2.erode(image, kernel)

    image_eroded = cv2.erode(image, kernel)

    edges = image - image_eroded

    return edges
