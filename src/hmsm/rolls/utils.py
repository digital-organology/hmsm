# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import skimage.filters
import random
import logging
import numpy as np
import skimage.io
import skimage
from typing import Optional, Tuple

def find_threshold(image: np.ndarray, method: Optional[str] = "threshold_mean", n_chunks: Optional[int] = 5, chunk_size: Optional[int] = 4000, limits: Optional[Tuple[float, float]] = (0.1,0.66)) -> float:
    """Find binarization threshold

    This method will find the binarization threshold on the given grayscale image by running the provided method for threshold determination on several chunks of the image.
    This approach is choosen over finding the threshold on the complete image to preserve memory resources.

    Args:
        image (str): Image to generate thresholds on. Must be a grayscale image in the form of a 2-Dimensional numpy array.
        method (Optional[str], optional): Method to use for threshold approximation, needs to be implemented in skimage.filters. Defaults to "threshold_mean".
        n_chunks (Optional[int], optional): Number of chunks to generate for threshold approximation. Defaults to 5.
        chunk_size (Optional[int], optional): Lenght of each chunk. Defaults to 4000.
        limits (Optional[Tuple[float, float]], optional): Relative vertical bounds within the input image to use generate sample chunks within. Defaults to (0.1,0.66).

    Returns:
        float: Approximated threshold
    """
    assert image.ndim == 2    
    chunk_start_idx = random.sample(range(round(image.shape[0] * limits[0]), round(image.shape[0] * limits[1])), n_chunks)

    try:
        threshold_method = getattr(skimage.filters, method)
    except AttributeError:
        logging.error("Invalid thresholding method supplied, needs to be a method implemented in skimage.filters")
        raise
    thresholds = [threshold_method(image[start_idx:start_idx+chunk_size]) for start_idx in chunk_start_idx]
    return sum(thresholds) / len(thresholds)

def read_as_binary_image(image_path: str, **kwargs) -> np.ndarray:
    """Reads a binarized version of an image

    Args:
        image_path (str): Path to the image to be read
        **kwargs: Additional arguments passed to the find_threshold method

    Raises:
        ValueError: Will be raised if the image could not be read from the specified path

    Returns:
        np.ndarray: Binarized version of the image at the provided path
    """    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read image from specified path")
    threshold = find_threshold(image, **kwargs)
    image = image > threshold
    return image

def create_masks(image_path: str, n_masks: Optional[int] = 2, **kwargs):

    # Create binarized image
    binary = read_as_binary_image(image_path, **kwargs)

    # Read color image
    pixels = skimage.io.imread(image_path)
    # Store shape of the image for later
    img_shape = pixels.shape[:2]

    # Filter out all pixels we are interested in
    pixels = pixels[binary == False].reshape((-1, 3))
    pixels = np.float32(pixels)

    # Cluster into two clusters based on color

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    _, labels, (centers) = cv2.kmeans(pixels, n_masks, None, stop_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    del pixels

    centers = np.uint8(centers)
    labels = labels.flatten()

    binary_mask = skimage.img_as_ubyte(binary)
    skimage.io.imsave(f"mask_{n_masks}.tif", binary_mask)
    del binary_mask

    masks = []

    for mask_id in range(0, n_masks):
        mask = np.zeros(img_shape, np.bool_)
        mask[binary == False] = labels != mask_id
        mask_ubyte = skimage.img_as_ubyte(mask)
        skimage.io.imsave(f"mask_{mask_id}.tif", mask_ubyte)
        del mask_ubyte
        masks.append(mask)

    masks.append(binary)

    return masks
    

