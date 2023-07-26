# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import functools
import logging
import os
from typing import Optional, Tuple

import numpy as np
import scipy.signal
import skimage.io
import skimage.morphology

import hmsm.midi
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
    """Main processing function for piano rolls

    This function will perform image to midi digitization of a piano roll.

    Args:
        input_path (str): Path to the input image of a piano roll
        output_path (str): Path to which to write the created midi file
        method (str): Method to use for digitization. This parameter is currently unused and can be set to any value.
        config (dict): Configuration data to use for midi creation. Needs to contain, amongst others, initial guesses on the position of the tracks.
        n_clusters (Optional[int], optional): Number of clusters when segmenting the colorspace of the roll image. Suggested to be 2 for rolls with written annotations. Defaults to 2.
        chunk_size (Optional[int], optional): Vertical size of chunks to segment the image to for processing. Defaults to 4000.
        parameter_estimation_chunks (Optional[int], optional): Number of chunks to generate for estimating required digitization parameters. Higher numbers should improve accuracy but will have a negative performance impact. Defaults to 5.
    """
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

    logging.info(f"Estimated width bounds are ({width_bounds[0]}, {width_bounds[1]})")

    logging.info(
        f"Estimated density bounds are ({density_bounds[0]}, {density_bounds[1]})"
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
        filename = os.path.join("debug_data", "note_data_raw.csv")
        np.savetxt(filename, note_data, delimiter=",")

    logging.info("Processing control information [NOT IMPLEMENTED YET]")

    note_data = process_control_tracks(note_data)

    logging.info("Merging notes")

    note_data = merge_notes(note_data)

    note_start = (note_data[:, 0]).tolist()
    note_duration = (note_data[:, 1] - note_data[:, 0]).tolist()
    midi_tone = note_data[:, 2].tolist()

    if logger.isEnabledFor(logging.DEBUG):
        filename = os.path.join("debug_data", "note_data_processed.csv")
        debug_array = np.hstack([note_start, note_duration, midi_tone])
        np.savetxt(filename, debug_array, delimiter=",")

    logging.info("Creating midi data")

    midi = hmsm.midi.create_midi(note_start, note_duration, midi_tone, scaling_factor=3)

    logging.info(f"Writing midi file to {output_path}")

    midi.save(output_path)

    logging.info("Done, bye")


def process_control_tracks(notes: np.ndarray) -> np.ndarray:
    """Process detected control tracks on the roll

    This part of the processing pipeline is yet to be implemented. The method will currently discard all control information.

    Args:
        notes (np.ndarray): Array containing detected notes

    Returns:
        np.ndarray: Currently: Input array except for notes on control tracks
    """
    # Does not currently do anything but filter out the control information
    return notes[notes[:, 2] != -1, :]


def merge_notes(notes: np.ndarray) -> np.ndarray:
    """Merge notes that sound as one

    On original playback devices holes that are closely grouped will sound as a single, longer note.
    This method emulates that behaviour by merging notes that are closer than a calculated distance threshold.

    Args:
        notes (np.ndarray): Array containing detected notes

    Returns:
        np.ndarray: Input array with close notes merged
    """
    offset = notes[:, 0].min()
    notes[:, 0:2] = notes[:, 0:2] - offset

    merge_threshold = round(np.mean(notes[:, 1] - notes[:, 0]) * 2)

    midi_notes = np.unique(notes[:, 2])

    # Should be superfluous as the processing will yield notes already ordered
    notes = notes[notes[:, 0].argsort()]

    notes_merged = list()

    for tone in midi_notes:
        tone_notes = notes[notes[:, 2] == tone, :]
        note_start_idx = (
            np.diff(tone_notes[:, 0], prepend=(merge_threshold * -1 - 1))
            > merge_threshold
        )
        note_end_idx = np.roll(note_start_idx, -1)

        tone_notes_merged = tone_notes[note_start_idx, :]
        tone_notes_merged[:, 1] = tone_notes[note_end_idx, 1]

        notes_merged.append(tone_notes_merged)

    notes_merged = np.vstack(notes_merged)

    return notes_merged


def extract_note_data(
    note_mask: np.ndarray,
    binary_mask: np.ndarray,
    alignment_grid: np.ndarray,
    width_bounds: Tuple[float, float],
    density_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts note information from masks

    Primary data extraction method for piano rolls that will detect notes on the provided masks and return them as tabular data.

    Args:
        note_mask (np.ndarray): Mask that contains the holes on a piano roll
        binary_mask (np.ndarray): Mask that is binarized to be True where the piano roll is and false outside of it, used for detecting the edges of the roll
        alignment_grid (np.ndarray): Array containing information about the location of all tracks on the given piano roll
        width_bounds (Tuple[float, float]): Bounds within which the width of a given component must be to be considered as a hole in the piano roll
        density_bounds (Tuple[float, float]): Bounds within which the density of a given component must be to be considered as a hole in the piano roll

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the extracted tabular data of notes on the roll and the alignment grid with calculated corrections applied
    """
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
    """Finds the edges of the piano roll from the given mask

    Args:
        mask (np.ndarray): Mask that is binarized to be True where the roll is and False everywhere else

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two one dimensional arrays that contain the position of the roll edge along the given mask
    """
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
    """Checks if the given compoennt fits within the given bounds

    Args:
        component (np.ndarray): Array of coordinates that belong to the component in question
        width_bounds (Tuple[float, float]): Bounds within which the widht of the given component must be
        density_bounds (Tuple[float, float]): Bounds within which the density of the given component must be

    Returns:
        bool: True if the component falls within the bounds provided, False otherwise
    """
    dims = component.max(axis=0) - component.min(axis=0)

    if dims[1] < width_bounds[0] or dims[1] > width_bounds[1]:
        return False

    density = len(component) / (dims[0] * dims[1])

    if density < density_bounds[0] or density > density_bounds[1]:
        return False

    return True
