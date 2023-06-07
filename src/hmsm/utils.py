import logging
import os
import sys
import numpy as np
import pkg_resources
import cv2
from typing import Optional, Tuple
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import img_as_ubyte, img_as_float
from scipy import interpolate

def read_image(path: str, binarize: bool = False, threshold: Optional[int] = None) -> np.ndarray:
    try:
        img = imread(path)
    except FileNotFoundError:
        logging.error(f"The system could not find the specified file at path '{path}', could not read image file")
        raise

    img = imread(path)

    logging.info(f"Image read from '{path}'")

    return binarize_image(img, threshold) if binarize else img

def binarize_image(image: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        image = rgb2gray(image)
    
    if threshold is None:
        threshold = threshold_otsu(image)
    else:
        threshold = (threshold / 255)

    img_bin = image > threshold
    # img_bin = np.invert(img_bin)

    return img_as_ubyte(img_bin)

def crop_image_to_contents(image: np.ndarray) -> np.ndarray:
    output_image = image.copy()

    # Binarize the image to make sure we don't get stiffled by some weird gray values
    if image.ndim == 3 or np.unique(image).size > 2:
        image = binarize_image(image)

    # Check if the background is ones and if so invert the image for cropping

    max_value = image.max()

    if image[0,0] == max_value and image[0,-1] == max_value and image[-1, 0] == max_value and image[-1,-1] == max_value:
        img = np.invert(image.copy())
    else:
        img = image

    y_values, x_values = np.nonzero(img)

    y_min = y_values.min() - 20 if y_values.min() - 20 >= 0 else 0
    y_max = y_values.max() + 20 if y_values.max() + 20 <= img.shape[0] - 1 else img.shape[0] - 1

    x_min = x_values.min() - 20 if x_values.min() - 20 >= 0 else 0
    x_max = x_values.max() + 20 if x_values.max() + 20 <= img.shape[1] - 1 else img.shape[1] - 1

    output_image = output_image[y_min:y_max, x_min:x_max]
    return output_image

def morphological_edge_detection(image: np.ndarray, n_erosions: Optional[int] = 2) -> np.ndarray:
    kernel = np.ones((3,3), np.uint8)

    for n in range(n_erosions):
        image = cv2.erode(image, kernel)

    image_eroded = cv2.erode(image, kernel)

    edges = image - image_eroded

    return edges

def interpolate_missing_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    unknown_coords = np.argwhere(mask == True)

    interpolated_values = interpolate.griddata(
        np.argwhere(mask == False), image[mask == False], unknown_coords,
        method="nearest", fill_value= [0,0,0]
    )

    interpolated_image = image.copy()
    interpolated_image[unknown_coords[:,0], unknown_coords[:,1]] = interpolated_values
    return interpolated_image

def get_lut() -> np.ndarray:
    file = pkg_resources.resource_filename(__name__, "data/lut.npy")

    with open(file, "rb") as f:
        lut = np.load(f)

    return lut

def image_from_coords(coords: dict|list, shape: Tuple[int, int]) -> np.ndarray:
    if isinstance(coords, list):
        coords = dict(zip(range(1, len(coords) + 1), coords))
    

    bg = np.zeros(shape, np.uint8)

    for id, coords in coords:
        bg[coords[:,0], coords[:,1]] = id

    lut = get_lut()

    clr_img = cv2.LUT(cv2.merge((bg, bg, bg)), lut)
    return clr_img