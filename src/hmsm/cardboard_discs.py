import cv2
import numpy as np
from skimage.measure import EllipseModel, label
from scipy.spatial import distance
from typing import Optional
from hmsm.utils import read_as_binary_image, crop_image_to_contents, morphological_edge_detection

# For debugging only
import matplotlib.pyplot as plt


def process_disc():
    pass

def transform_to_rectangle(input_image: np.ndarray, offset: Optional[int] = 0) -> np.ndarray:
    """Transforms an image of a circular music storage medium to the shape of a rectangular one

    Args:
        image (np.ndarray): The image to be transformed, is expected to be a binary image as returned by :func:`~hmsm.utils.read_as_binary_image`
        offset (Optional[int], optional): The offset (in degrees, counterclockwise) of the beginning of the disc. Defaults to 0.

    Returns:
        np.ndarray: Image of the transformed medium
    """
    # Read image and apply preprocessing
    
    image = crop_image_to_contents(input_image.copy())
    edges = morphological_edge_detection(image)

    # Label the image to find the outer edge of the disc

    labels = label(edges, background = 0, connectivity = 2)

    # We can generally assume the outer edge to be the first label, though we might implement additional methods for messier images in the future

    edge = np.argwhere(labels == 1)

    # Fit an ellipse to the outer edge to determine the image center
    # TODO: Refactor into helper function

    ell = EllipseModel()
    ell.estimate(edge)
    center_x, center_y, a, b, theta = ell.params
    center_x = int(center_x)
    center_y = int(center_y)
    a = int(a)
    b = int(b)

    # Create a mask for all pixels that are within the disc
    ellipse_mask = cv2.ellipse(np.zeros_like(image), (center_x, center_y), (int(a), int(b)), theta, 0, 360, 1, -1)

    image = np.invert(image)

    image[ellipse_mask == 0] = 0

    points_filled = np.argwhere(image == 255)

    # Shift points around the center to make the center be (0,0)

    points_filled = points_filled - np.array([center_x, center_y])

    # Calculate position in degrees

    degrees_filled = np.arctan2(points_filled[:,1], points_filled[:,0]) * 180 / np.pi
    degrees_filled = ((np.round(degrees_filled, decimals = 1) + 180) * 10).astype(np.int16)

    # Apply offset if applicable

    if not offset == 0:
        degrees_filled = degrees_filled - (offset * 10)
        degrees_filled[degrees_filled < 0] = degrees_filled[degrees_filled < 0] + 3600

    degrees_filled = degrees_filled.astype(np.uint16)

    # Calculate distances

    dists_filled = np.round(distance.cdist(np.array([[0, 0]]), points_filled)[0], decimals = 0).astype(np.uint16)

    # Convert into triplets

    triplets = np.column_stack((degrees_filled, dists_filled, np.full(dists_filled.shape, 1)))

    # Build 2d image

    image_rect = np.zeros((3601, dists_filled.max() + 10), np.uint8)
    image_rect[triplets[:,0], triplets[:,1]] = triplets[:,2]

    # Fill holes

    kernel = np.ones((3,3),np.uint8)
    image_rect = cv2.morphologyEx(image_rect, cv2.MORPH_CLOSE, kernel)

    return image_rect