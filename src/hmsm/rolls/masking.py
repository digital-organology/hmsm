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
import scipy.spatial.distance
from typing import Optional, Tuple, List

def estimate_mask_parameters(image: np.ndarray, method: Optional[str] = "threshold_mean", n_chunks: Optional[int] = 5, chunk_size: Optional[int] = 4000, limits: Optional[Tuple[float, float]] = (0.1,0.66), n_clusters: Optional[int] = 2) -> Tuple[float, List[np.ndarray]]:
    """Estimates parameters required for mask generation

    This method will estimate the binararization threshold as well as the cluster centers that are used to generate image masks by calculating
    approximate values on a number of slices of the original image.

    Args:
        image (np.ndarray): Image to estimate parameters on. Is expected to be a BGR color image.
        method (Optional[str], optional): Method to use for threshold approximation. Must be implemented in skimage.filters. Defaults to "threshold_mean".
        n_chunks (Optional[int], optional): Number of chunks to use for parameter estimation. Defaults to 5.
        chunk_size (Optional[int], optional): Size of the chunks to use for parameter estimation. Defaults to 4000.
        limits (Optional[Tuple[float, float]], optional): Relative limits of the regions inside the image to consider for slice selection. Set to 0,1 to use the entire image. Defaults to (0.1,0.66).
        n_clusters (Optional[int], optional): Number of cluster centers to generate. Defaults to 2.

    Returns:
        Tuple[float, List[np.ndarray]]: Tuple of (threshold, cluster_centers).
    """    
    
    assert image.ndim == 3

    chunk_start_idx = random.sample(range(round(image.shape[0] * limits[0]), round(image.shape[0] * limits[1])), n_chunks)
    
    try:
        threshold_method = getattr(skimage.filters, method)
    except AttributeError:
        logging.error("Invalid thresholding method supplied, needs to be a method implemented in skimage.filters")
        raise

    threshold = [threshold_method(cv2.cvtColor(image[start_idx:start_idx+chunk_size], cv2.COLOR_BGR2GRAY)) for start_idx in chunk_start_idx]
    threshold = sum(threshold) / len(threshold)

    cluster_centers = [_find_cluster_centers(image[start_idx:start_idx+chunk_size], threshold, n_clusters) for start_idx in chunk_start_idx]

    # There is alomost definitly a better (meaning a more readable) way to do this
    # This finds the centers that are the closest for each cluster and calculates their mean center
    # We need to do this, as kmeans has no guarantee to return the centers in a fixed order
    centers = [np.mean([x[scipy.spatial.distance.cdist(np.expand_dims(cluster_centers[0][i], axis = 0), x).argmin()] for x in cluster_centers[1:]], axis = 0) for i in range(n_clusters)]

    return (threshold, np.vstack(centers))

def _find_cluster_centers(image: np.ndarray, threshold: float, n_clusters: Optional[int] = 2) -> np.ndarray:
    """Helper method that runs kmeans clustering on the colorspace of the provided image

    Args:
        image (np.ndarray): Image to use for clustering
        threshold (float): Threshold to use for binarization. Only pixels binarized to 0/False will be considered for the clustering.
        n_clusters (Optional[int], optional): Number of clusters to generate. Defaults to 2.

    Returns:
        np.ndarray: Array of cluster centers.
    """    
    binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > threshold
    

    # Filter out all pixels we are interested in
    image = image[binary == False].reshape((-1, 3))
    image = np.float32(image)

    # Cluster into two clusters based on color

    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    _, labels, (centers) = cv2.kmeans(image, n_clusters, None, stop_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    return centers


def create_masks(image: np.ndarray, threshold: float, cluster_centers: List[np.ndarray]) -> List[np.ndarray]:
    """Generates binary masks for each cluster center provided

    Args:
        image (np.ndarray): Input image
        threshold (float): Threshold to use for binarization. Only pixels binarized to 0/False will be considered when creating masks.
        cluster_centers (List[np.ndarray]): Cluster centers to use for mask assignment.

    Returns:
        List[np.ndarray]: List containing a binary array for each mask.
    """    
    image_binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) > threshold

    classes = image[image_binary == False]

    classes = scipy.spatial.distance.cdist(classes, cluster_centers).argmin(axis = 1)

    masks = []

    for mask_id in range(len(cluster_centers)):
        mask = np.zeros(image.shape[:2], np.bool_)
        mask[image_binary == False] = classes != mask_id
        masks.append(mask)

    masks.append(image_binary)

    return masks

def create_chunk_masks(image_path: str, chunk_size: Optional[int] = 4000, n_clusters: Optional[int] = 2, n_chunks: Optional[int] = 5) -> None:
    """Create masks for the input image

    This method will generate masks based on color space segmentation for the given input image.
    To ease memory pressure this process is done chunkwise on the input image.

    Args:
        image_path (str): Path to the input image.
        chunk_size (Optional[int], optional): Vertical size of the chunks in which the image will be processed. Defaults to 4000.
        n_clusters (Optional[int], optional): Number of cluster for which to create masks. Defaults to 2.
        n_chunks (Optional[int], optional): Number of chunks to use for parameter estimation. Defaults to 5.
    """    

    logging.info(f"Reading input image from {image_path}...")
    image = skimage.io.imread(image_path)

    logging.info(f"Estimating required parameters using {n_chunks} slices of {chunk_size} pixels height each")
    threshold, centers = estimate_mask_parameters(image, chunk_size = chunk_size, n_clusters = n_clusters)

    logging.info(f"Parameters estimated, beginning mask creation")

    for start_idx in range(0, len(image), chunk_size):
        end_idx = start_idx + chunk_size if start_idx + chunk_size <= len(image) else len(image)

        logging.info(f"Processing chunk from {start_idx} to {end_idx}")

        masks = create_masks(image[start_idx:end_idx], threshold, centers)

        for mask_id in range(len(masks)):
            mask_ubyte = skimage.img_as_ubyte(masks[mask_id])
            filename = os.path.join("masks", f"mask_{start_idx}_{end_idx}_{mask_id}.tif")
            skimage.io.imsave(filename, mask_ubyte)