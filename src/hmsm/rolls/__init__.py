# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import functools
import logging
import os
from typing import Optional, Tuple

import numpy as np
import scipy.signal
import skimage.io
import skimage.morphology

import hmsm.rolls.masking
import hmsm.rolls.utils
import hmsm.utils


def process_roll(
    input_path: str,
    output_path: str,
    method: str,
    config: dict,
    n_clusters: Optional[int] = 2,
    chunk_size: Optional[int] = 4000,
    parameter_estimation_chunks: Optional[int] = 5,
) -> None:
    logging.info(f"Reading input image from {input_path}...")

    image = skimage.io.imread(input_path)

    generator = hmsm.rolls.masking.MaskGenerator(image, chunk_size, n_clusters)

    logging.info(
        f"Estimating required masking parameters using {parameter_estimation_chunks} slices of {chunk_size} pixels height each"
    )

    generator.estimate_mask_parameters(
        n_chunks=parameter_estimation_chunks, chunk_size=chunk_size
    )

    logging.info(
        f"Estimating required note detection parameters using {parameter_estimation_chunks} slices of {chunk_size} pixels height each"
    )

    mask_idx = min(enumerate(generator.centers), key=lambda x: np.mean(x[1]))[0]

    width_bounds, density_bounds = hmsm.rolls.utils.estimate_hole_parameters(
        generator,
        mask_idx=mask_idx,
        n_chunks=parameter_estimation_chunks,
        chunk_size=chunk_size,
    )

    logging.info(
        f"Estimated width bounds at ({width_bounds[0]}, {width_bounds[1]}), density bounds at ({density_bounds[0]}, {density_bounds[1]})"
    )

    logging.info("Beginning processign of note data")

    alignment_grid = hmsm.rolls.utils.get_initial_alignment_grid(
        config["roll_width_mm"], config["track_measurements"]
    )

    note_data = list()

    for bounds, masks in iter(generator):
        logging.info(f"Processing chunk from {bounds[0]} to {bounds[1]}")
        notes, alignment_grid = extract_note_data(
            masks[mask_idx], masks[-1], alignment_grid, width_bounds, density_bounds
        )

        if notes is None:
            continue

        notes[:, 0:2] = notes[:, 0:2] + bounds[0]
        note_data.append(notes)

    # Currently superfluous
    # note_data = list(filter(lambda x: x is not None, note_data))

    note_data = np.vstack(note_data)

    logger = logging.getLogger()
    if logger.isEnabledFor(logging.DEBUG):
        filename = os.path.join("debug_data", "note_data.csv")
        np.savetxt(filename, note_data, delimiter=",")
    pass


def extract_note_data(
    note_mask: np.ndarray,
    binary_mask: np.ndarray,
    alignment_grid: np.ndarray,
    width_bounds: Tuple[float, float],
    density_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    labels = skimage.measure.label(note_mask, background=False, connectivity=2)
    components = list(hmsm.utils.to_coord_lists(labels).values())

    components = list(
        filter(
            functools.partial(
                _filter_component,
                width_bounds=width_bounds,
                density_bounds=density_bounds,
            ),
            components,
        )
    )

    # If there are no notes in this chunk we skip it and continue
    if len(components) == 0:
        return (None, alignment_grid)

    left_edge, right_edge = _get_roll_edges(binary_mask)

    note_data = list()

    for component in components:
        # Calculate relative position on the roll (from the left roll edge)
        height, width = tuple(component.max(axis=0) - component.min(axis=0))

        y_position = int(round(component[:, 0].min() + (height / 2)))
        x_position = int(round(component[:, 1].min() + (width / 2)))
        roll_width = right_edge[y_position] - left_edge[y_position]
        left_dist = (component[:, 1].min() - left_edge[y_position]) / roll_width
        right_dist = (component[:, 1].max() - left_edge[y_position]) / roll_width

        # Find the closest track (based on the left track edge for now, might want to include the right edge in the future)
        track_idx = np.abs(alignment_grid[:, 0] - left_dist).argmin()

        # Calculate the misalignment of the note center from the track center
        relative_x_position = (x_position - left_edge[y_position]) / roll_width
        misalignment = relative_x_position - alignment_grid[track_idx, 0:2].mean()

        # If the misalignment is negative the note is to the left of the track
        # and we need to shift the track to the left and vice versa for a positive misalignment
        # We apply a correction to the alignment grid which we base on the difference in misalignments
        # however we only correct for a fraction of the actual misalignment to prevent overcorrecting
        correction = misalignment * 0.2
        alignment_grid[:, 0:2] = alignment_grid[:, 0:2] + correction

        # Calculate note data
        note_data.append(
            np.array(
                [
                    component[:, 0].min(),
                    component[:, 0].max(),
                    int(alignment_grid[track_idx, 2]),
                ]
            )
        )

    if len(note_data) == 0:
        return (None, alignment_grid)

    return (np.vstack(note_data), alignment_grid)


def _get_roll_edges(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = skimage.morphology.binary_closing(mask, skimage.morphology.diamond(5))

    rows, cols = np.where(mask)

    _, idx = np.unique(rows, return_index=True)

    left_edge = np.minimum.reduceat(cols, idx)
    left_edge = scipy.signal.savgol_filter(left_edge, 20, 3).round()

    right_edge = np.maximum.reduceat(cols, idx)
    right_edge = scipy.signal.savgol_filter(right_edge, 20, 3).round()

    return (left_edge, right_edge)


def _filter_component(
    component: np.ndarray,
    width_bounds: Tuple[float, float],
    density_bounds: Tuple[float, float],
) -> bool:
    dims = component.max(axis=0) - component.min(axis=0)

    if dims[1] < width_bounds[0] or dims[1] > width_bounds[1]:
        return False

    density = len(component) / (dims[0] * dims[1])

    if density < density_bounds[0] or density > density_bounds[1]:
        return False

    return True
