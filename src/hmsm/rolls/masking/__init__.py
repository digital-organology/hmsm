# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import logging
import random
from typing import List, Optional, Self, Tuple

import cv2
import numpy as np
import scipy.spatial.distance
import skimage
import skimage.filters
import skimage.io
import skimage.morphology

import hmsm.rolls.masking.methods


class MaskGenerator:
    chunk_size: int = None

    bg_color: str = None

    method: str = None

    image: np.ndarray = None

    parameters = None

    threshold: float = None

    centers: List[np.ndarray] = None

    _current_idx: int = None

    def __init__(
        self,
        image: np.ndarray,
        bg_color: str,
        method: str,
        chunk_size: Optional[int] = 4000,
        **kwargs,
    ) -> Self:
        """Creates a new object to generate masks from an image

        Args:
            image (np.ndarray): Image to generate masks for. Is expected to be a 3-Dimensional numpy array.
            bg_color (str): The color of the background in image. Must currently be either 'black' or 'white'.
            method (str): Method to use for mask generation.
            chunk_size (Optional[int], optional): Size of chunks to generate when iterated over. Defaults to 4000.
            **kwargs: Additional named arguments that will be passed to the mask generting method.

        Returns:
            Self: New MaskGenerator object
        """

        # TODO: Potentially validate the input parameters here

        self.image = image
        self.bg_color = bg_color
        self.method = method
        self.parameters = kwargs
        self.chunk_size = chunk_size
        self.current_idx = 0

    def get_masks(self, bounds: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """Create masks

        Method that will create masks with the parameters set for this objects instance.

        Args:
            bounds (Optional[Tuple[int, int]], optional): If set the masks will be generated for the chunk along the vertical bounds specified. If unspecified will generate masks for the entire image. Defaults to None.

        Returns:
            List[np.ndarray]: List of masks for the image. These will always be in the same order for one instance but the order might differ between instances (even for the same image)
        """
        chunk = self.image if bounds is None else self.image[bounds[0] : bounds[1]]

        try:
            generator_method = getattr(hmsm.rolls.masking.methods, self.method)
        except AttributeError:
            logging.error(
                "Invalid thresholding method supplied, needs to be a method implemented in hmsm.rolls.masking.methods"
            )
            raise

        try:
            mask = generator_method(chunk, self.bg_color, **self.parameters)
        except TypeError:
            logging.error(
                "Invalid parameters were supplied to the binarization/masking method. See the original exception for details."
            )
            raise

        return mask

    def __iter__(self) -> Self:
        """Gets the iterator for this object

        Raises:
            RuntimeError: Raised if required parameters are not set

        Returns:
            Self: The generator object
        """

        self._current_idx = 0
        return self

    def __next__(self) -> Tuple[Tuple[int, int], List[np.ndarray]]:
        """Gets the next chunk for the iterator

        Raises:
            StopIteration: Will be raised when the image has been completetly iterated over

        Returns:
            Tuple[Tuple[int, int], List[np.ndarray]]: The bounds of the chunk returned and a list of masks created for that chunk
        """
        if self._current_idx == len(self.image):
            raise StopIteration

        start_idx = self._current_idx
        end_idx = (
            self._current_idx + self.chunk_size
            if self._current_idx + self.chunk_size <= len(self.image)
            else len(self.image)
        )
        self._current_idx = end_idx

        return ((start_idx, end_idx), self.get_masks((start_idx, end_idx)))

    ##################################
    #                                #
    # Deprecated for the time beeing #
    #                                #
    ##################################

    # def estimate_mask_parameters(
    #     self,
    #     method: Optional[str] = "threshold_mean",
    #     n_chunks: Optional[int] = 5,
    #     chunk_size: Optional[int] = 4000,
    #     limits: Optional[Tuple[float, float]] = (0.1, 0.66),
    # ) -> Self:
    #     """Estimates parameters required for mask generation

    #     This method will estimate the binararization threshold as well as the cluster centers that are used to generate image masks by calculating
    #     approximate values on a number of slices of the original image.

    #     Args:
    #         method (Optional[str], optional): Method to use for threshold approximation. Must be implemented in skimage.filters. Defaults to "threshold_mean".
    #         n_chunks (Optional[int], optional): Number of chunks to use for parameter estimation. Defaults to 5.
    #         chunk_size (Optional[int], optional): Size of the chunks to use for parameter estimation. Defaults to 4000.
    #         limits (Optional[Tuple[float, float]], optional): Relative limits of the regions inside the image to consider for slice selection. Set to 0,1 to use the entire image. Defaults to (0.1,0.66).

    #     Returns:
    #         Tuple[float, List[np.ndarray]]: Tuple of (threshold, cluster_centers).
    #     """

    #     assert self.image.ndim == 3

    #     chunk_start_idx = random.sample(
    #         range(
    #             round(self.image.shape[0] * limits[0]),
    #             round(self.image.shape[0] * limits[1]),
    #         ),
    #         n_chunks,
    #     )

    #     try:
    #         threshold_method = getattr(skimage.filters, method)
    #     except AttributeError:
    #         logging.error(
    #             "Invalid thresholding method supplied, needs to be a method implemented in skimage.filters"
    #         )
    #         raise

    #     threshold = [
    #         threshold_method(
    #             cv2.cvtColor(
    #                 self.image[start_idx : start_idx + chunk_size], cv2.COLOR_BGR2GRAY
    #             )
    #         )
    #         for start_idx in chunk_start_idx
    #     ]
    #     self.threshold = sum(threshold) / len(threshold)

    #     # A single cluster means we will just binarize
    #     if self.n_clusters == 1:
    #         return self

    #     cluster_centers = [
    #         self._find_cluster_centers(self.image[start_idx : start_idx + chunk_size])
    #         for start_idx in chunk_start_idx
    #     ]

    #     # There is alomost definitly a better (meaning a more readable) way to do this
    #     # This finds the centers that are the closest for each cluster and calculates their mean center
    #     # We need to do this, as kmeans has no guarantee to return the centers in a fixed order
    #     self.centers = [
    #         np.mean(
    #             [
    #                 x[
    #                     scipy.spatial.distance.cdist(
    #                         np.expand_dims(cluster_centers[0][i], axis=0), x
    #                     ).argmin()
    #                 ]
    #                 for x in cluster_centers[1:]
    #             ],
    #             axis=0,
    #         )
    #         for i in range(self.n_clusters)
    #     ]

    #     return self

    # def _find_cluster_centers(self, chunk: np.ndarray) -> np.ndarray:
    #     """Helper method that runs kmeans clustering on the colorspace of the provided image

    #     Args:
    #         chunk (np.ndarray): Image to use for clustering.

    #     Returns:
    #         np.ndarray: Array of cluster centers.
    #     """
    #     binary = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY) > self.threshold

    #     # Filter out all pixels we are interested in
    #     chunk = chunk[binary == False].reshape((-1, 3))
    #     chunk = np.float32(chunk)

    #     # Cluster into two clusters based on color

    #     stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    #     _, labels, (centers) = cv2.kmeans(
    #         chunk, self.n_clusters, None, stop_criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    #     )

    #     centers = np.uint8(centers)
    #     return centers

    # def get_masks(self, bounds: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
    #         """Create masks

    #         Method that will create masks with the parameters set for this objects instance.

    #         Args:
    #             bounds (Optional[Tuple[int, int]], optional): If set the masks will be generated for the chunk along the vertical bounds specified. If unspecified will generate masks for the entire image. Defaults to None.

    #         Returns:
    #             List[np.ndarray]: List of masks for the image. These will always be in the same order for one instance but the order might differ between instances (even for the same image)
    #         """
    #         chunk = self.image if bounds is None else self.image[bounds[0] : bounds[1]]

    #         image_binary = cv2.cvtColor(chunk, cv2.COLOR_BGR2GRAY) > self.threshold

    #         if self.n_clusters == 1:
    #             return [np.invert(image_binary), image_binary]

    #         classes = chunk[image_binary == False]

    #         classes = scipy.spatial.distance.cdist(classes, self.centers).argmin(axis=1)

    #         masks = []

    #         for mask_id in range(len(self.centers)):
    #             mask = np.zeros(chunk.shape[:2], np.bool_)
    #             mask[image_binary == False] = classes == mask_id
    #             mask = skimage.morphology.binary_opening(
    #                 mask, skimage.morphology.diamond(3)
    #             )
    #             masks.append(mask)

    #         masks.append(image_binary)

    #         return masks
