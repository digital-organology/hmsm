# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import logging
import math
from typing import List, Optional, Self, Tuple

import numpy as np

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
        self._current_idx = 0

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

    def get_number_iterations(self) -> int:
        """Returns the number of remaining iterations

        Returns:
            int: Number of iterations remaining
        """
        return math.ceil((len(self.image) - self._current_idx) / self.chunk_size)
