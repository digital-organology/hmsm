# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import logging
import numpy as np
import pkg_resources
import cv2
import skimage.filters
import skimage.color
import skimage.io
import skimage
import scipy
from typing import Optional, Tuple

def read_image(path: str, binarize: bool = False, threshold: Optional[int] = None) -> np.ndarray:
    """Read an image from disk

    Wrapper around skimage.io.imread that will also perform binarization if desired

    Args:
        path (str): Filepath to read the image from
        binarize (bool, optional): Perform binarization on the input image. Defaults to False.
        threshold (Optional[int], optional): Threshold to use for binarization. If missing will use Otsu's method to estimate. Defaults to None.

    Returns:
        np.ndarray: Image as read from path, with binarization applied if requested
    """
    try:
        img = skimage.io.imread(path)
    except FileNotFoundError:
        logging.error(f"The system could not find the specified file at path '{path}', could not read image file")
        raise

    logging.info(f"Image read from '{path}'")

    return binarize_image(img, threshold) if binarize else img

def binarize_image(image: np.ndarray, threshold: Optional[int] = None) -> np.ndarray:
    """Binarize image

    This will convert the input image to grayscale if necessary and then perform binarization, either with the provided threshold value or with an estimated one.

    Args:
        image (np.ndarray): Input image, can be color (rgb) or grayscale.
        threshold (Optional[int], optional): Threshold to use for binarization. Will be estimated using Otsu's method if missing. Defaults to None.

    Returns:
        np.ndarray: Binarized input image
    """
    if image.ndim == 3 and image.shape[2] == 3:
        image = skimage.color.rgb2gray(image)
    
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(image)
    else:
        threshold = (threshold / 255)

    img_bin = image > threshold

    return skimage.img_as_ubyte(img_bin)

def crop_image_to_contents(image: np.ndarray) -> np.ndarray:
    """Crop image to contents

    This will binarize the input image if required and crop it to the area that actually has content.
    The method will test the pixels at the very edge of the image and if they are all filled after binarization,
    assume the image has a light background and invert it before cropping.

    Args:
        image (np.ndarray): Image to crop, can be rgb or grayscale

    Returns:
        np.ndarray: Cropped image
    """
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
    """Detect edges on the input image

    This will perform morphological edge detection on the input image.
    Edges will be calculated by performing an erosion on the input image and then calculating the difference between the eroded image and the original input.

    Args:
        image (np.ndarray): Input image, needs to be binarized
        n_erosions (Optional[int], optional): Number of erosions to perform. These are additional erosions performed before the erosion used for edge detection and can help with better edge seperation. Defaults to 2.

    Returns:
        np.ndarray: Image of the same shape as the input image with edges
    """
    kernel = np.ones((3,3), np.uint8)

    for n in range(n_erosions):
        image = cv2.erode(image, kernel)

    image_eroded = cv2.erode(image, kernel)

    edges = image - image_eroded

    return edges

def interpolate_missing_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate missing pixels in an image

    Args:
        image (np.ndarray): Input image with unknown pixel values, needs to be a color image with three channels
        mask (np.ndarray): Bool array of the same shape as image with True on positions where pixel values are unknown in Image and should be interpolated

    Returns:
        np.ndarray: Input image with interpolated pixels
    """
    unknown_coords = np.argwhere(mask == True)

    interpolated_values = scipy.interpolate.griddata(
        np.argwhere(mask == False), image[mask == False], unknown_coords,
        method="nearest", fill_value= [0,0,0]
    )

    interpolated_image = image.copy()
    interpolated_image[unknown_coords[:,0], unknown_coords[:,1]] = interpolated_values
    return interpolated_image

def get_lut() -> np.ndarray:
    """Load a lut included in the package

    Returns:
        np.ndarray: LUT
    """
    file = pkg_resources.resource_filename(__name__, "data/lut.npy")

    with open(file, "rb") as f:
        lut = np.load(f)

    return lut

def image_from_coords(coords: dict|list, shape: Optional[Tuple[int, int]] = None, labels: Optional[bool|list] = None) -> np.ndarray:
    """Make an image from coordinate lists

    Args:
        coords (dict | list): List or dictionary of coordinate lists
        shape (Optional[Tuple[int, int]], optional): Dimensions of the output image. If missing will be infered from the provided coordinate lists. Defaults to None.
        labels (Optional[bool | list], optional): Labels to plot on the image. If missing no labels will be printed, when a list is provided it will be used as labels, if set to true the dictionary keys will be used as labels (these might be autogenerated if coords is a list). Defaults to None.

    Raises:
        ValueError: Will be raised if labels is a list of not exactly the same length as coords


    Returns:
        np.ndarray: Color image with the plotted coordinate lists.
    """
    if isinstance(coords, list):
        coords = dict(zip(range(1, len(coords) + 1), coords))

    
    if shape is None:
        max_values = np.vstack([el.max(axis = 0) for el in coords.values()]).max(axis = 0)
        shape = (max_values[0] + 10, max_values[1] + 10)

    bg = np.zeros(shape, np.uint8)

    for id, coord_list in coords.items():
        bg[coord_list[:,0], coord_list[:,1]] = id

    lut = get_lut()

    clr_img = cv2.LUT(cv2.merge((bg, bg, bg)), lut)

    if not labels is None:
        if isinstance(labels, list):
            if len(labels) != len(coords):
                raise ValueError("There must be exactly as many labels as coordinate lists to plot")
        else:
            labels = list(coords.keys())
    
        # This has memory overhead and isn't optimal, but there isn't a really good ways to iterate over a dictionary and a list at the same time

        coords = list(coords.values())

        for idx in range(len(coords)):
            cv2.putText(clr_img,
                        str(labels[idx]),
                        tuple(np.mean(coords[idx], 0).astype(np.uint32)[[1,0]]),
                        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2)

    return clr_img


def to_coord_lists(edge_image: np.ndarray) -> dict:
    # Label connected components
    labels = skimage.measure.label(edge_image, background = 0, connectivity = 2)

    # Most fast ways to do this only work on 1d arrays, so we flatten out the array first and get the indices for each unique element then
    # after which we stich everything back together to get 2d indices

    labels_flat = labels.ravel()
    labels_flat_sorted = np.argsort(labels_flat)
    keys, indices_flattend = np.unique(labels_flat[labels_flat_sorted], return_index=True)
    labels_ndims = np.unravel_index(labels_flat_sorted, labels.shape)
    labels_ndims = np.c_[labels_ndims] if labels.ndim > 1 else labels_flat_sorted
    indices = np.split(labels_ndims, indices_flattend[1:])
    coords = dict(zip(keys, indices))
    
    # We can most likely get away with just deleting the first element (as it should always be 0, meaning the background)
    coords.pop(0, None)

    logger = logging.getLogger()

    if logger.isEnabledFor(logging.DEBUG):
        image = image_from_coords(indices, edge_image.shape, keys)
        cv2.imwrite(os.path.join("debug_data", "labels.jpg"), image)


    return coords