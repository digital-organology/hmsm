# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import itertools
import logging
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.spatial.distance
import skimage
import skimage.color
import skimage.filters
import skimage.io
import skimage.measure

import hmsm.rolls
import hmsm.utils
from hmsm.rolls.masking import MaskGenerator


def create_chunk_masks(
    image_path: str,
    chunk_size: Optional[int] = 4000,
    n_clusters: Optional[int] = 2,
    n_chunks: Optional[int] = 5,
) -> None:
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

    generator = hmsm.rolls.masking.MaskGenerator(image, chunk_size, n_clusters)

    logging.info(
        f"Estimating required parameters using {n_chunks} slices of {chunk_size} pixels height each"
    )

    generator.estimate_mask_parameters(n_chunks=n_chunks)

    logging.info(f"Parameters estimated, beginning mask creation")

    for bounds, masks in iter(generator):
        (start_idx, end_idx) = bounds
        logging.info(f"Writing images for chunk from {start_idx} to {end_idx}")
        for mask_id in range(len(masks)):
            mask_ubyte = skimage.img_as_ubyte(masks[mask_id])
            filename = os.path.join(
                "masks", f"mask_{start_idx}_{end_idx}_{mask_id}.tif"
            )
            skimage.io.imsave(filename, mask_ubyte)


def estimate_hole_parameters(
    generator: MaskGenerator,
    mask_idx: int,
    n_chunks: Optional[int] = 5,
    chunk_size: Optional[int] = 4000,
    bounds: Optional[Tuple[float, float]] = (0.1, 0.66),
) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """Estimates parameters required for image to midi transformation of piano rolls

    Args:
        generator (MaskGenerator): MaskGenerator that has been configured for the piano roll to estimate parameters for
        mask_idx (int): Index of the mask containing the holes on the roll that are to be interpreted as note holes
        n_chunks (Optional[int], optional): Number of chunks to use for parameter estimation. Defaults to 5.
        chunk_size (Optional[int], optional): Verical size of each chunk to use for parameter estimation. Defaults to 4000.
        bounds (Optional[Tuple[float, float]], optional): Vertical area of the roll within which the test chunks will be generated. Defaults to (0.1, 0.66).

    Returns:
        Tuple[Tuple[int, int], Tuple[float, float]]: Width bounds and Density bounds for note detection
    """
    width_range = 0.1

    chunk_start_idx = random.sample(
        range(
            round(generator.image.shape[0] * bounds[0]),
            round(generator.image.shape[0] * bounds[1]),
        ),
        n_chunks,
    )

    width_bounds = list()
    density_bounds = list()

    for start_idx in chunk_start_idx:
        end_idx = (
            start_idx + chunk_size
            if start_idx + chunk_size <= len(generator.image)
            else len(generator.image)
        )

        mask = generator.get_masks((start_idx, end_idx))[mask_idx]

        labels = skimage.measure.label(mask, background=False, connectivity=2)
        coord_list = hmsm.utils.to_coord_lists(labels)

        logger = logging.getLogger()
        if logger.isEnabledFor(logging.DEBUG):
            clr_image = hmsm.utils.image_from_coords(coord_list, mask.shape)
            filename = os.path.join(
                "debug_data",
                f"parameter_estimation_detected_components_{start_idx}_{end_idx}.tif",
            )
            cv2.imwrite(filename, clr_image)

        widths = list()
        density = list()

        for component in coord_list.values():
            dims = component.max(axis=0) - component.min(axis=0)
            widths.append(dims[1])
            density.append(len(component) / (dims[0] * dims[1]))

        widths = np.array(widths)
        density = np.array(density)

        if logger.isEnabledFor(logging.DEBUG):
            width_filter = (
                np.abs(widths - np.median(widths)) < np.median(widths) * width_range
            )
            density_filter = np.abs(density - np.median(density)) < 2 * np.std(density)
            # TODO: Fix this. This is clearly not a well written expression.
            clr_image = hmsm.utils.image_from_coords(
                list(
                    itertools.compress(
                        list(coord_list.values()),
                        np.invert(width_filter & density_filter).tolist(),
                    )
                ),
                mask.shape,
            )
            filename = os.path.join(
                "debug_data",
                f"parameter_estimation_rejected_components_{start_idx}_{end_idx}.tif",
            )
            cv2.imwrite(filename, clr_image)

            clr_image = hmsm.utils.image_from_coords(
                list(
                    itertools.compress(
                        list(coord_list.values()),
                        (width_filter & density_filter).tolist(),
                    )
                ),
                mask.shape,
            )
            filename = os.path.join(
                "debug_data",
                f"parameter_estimation_accepted_components_{start_idx}_{end_idx}.tif",
            )
            cv2.imwrite(filename, clr_image)

        width_median = np.median(widths)
        width_bounds.append(
            np.array(
                (
                    width_median - (width_range * width_median),
                    width_median + (2 * width_range * width_median),
                )
            )
        )

        density_median = np.median(density)
        density_std = np.std(density)
        density_bounds.append(
            np.array((density_median - density_std, density_median + density_std))
        )

    density_bounds = np.vstack(density_bounds).mean(axis=0)
    width_bounds = np.vstack(width_bounds).mean(axis=0).round()

    return (tuple(width_bounds), tuple(density_bounds))


def get_initial_alignment_grid(
    roll_width_mm: int, track_measurements: List
) -> np.ndarray:
    """Gets the initial alignment grid from the provided track measurements and converts it into relative positions

    Args:
        roll_width_mm (int): Widht of the roll in mm
        track_measurements (List): List containing information about each track on the roll

    Returns:
        np.ndarray: Initial alignment grid with relative positions
    """
    alignment_grid = np.array([list(v.values()) for v in track_measurements])
    alignment_grid[:, 0:2] = alignment_grid[:, 0:2] / roll_width_mm
    return alignment_grid


def guess_background_color(image: np.ndarray, n_points: Optional[int] = 1000) -> str:
    sample_points_y = np.random.choice(
        image.shape[0] - 1,
        n_points,
        replace=False if n_points >= image.shape[0] else True,
    )

    sample_points_x = np.concatenate(
        [
            np.random.choice(10, int(n_points / 2), replace=True),
            np.random.choice(
                np.arange(image.shape[1] - 11, image.shape[1] - 1),
                int(n_points / 2),
                replace=True,
            ),
        ]
    )

    sample_points = image[sample_points_y, sample_points_x]

    sample_points = skimage.color.rgb2hsv(sample_points)

    mean_value = np.mean(sample_points, axis=0)[2]

    if 0.2 <= mean_value <= 0.8:
        logging.warning(
            f"Found inconclusive blackness value of {mean_value} when trying to determine the background color. This might result in problems during further processing."
        )

    return "black" if mean_value < 0.5 else "white"
