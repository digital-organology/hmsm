# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import datetime
import functools
import logging
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.signal
import scipy.sparse.csgraph
import scipy.spatial.distance
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
    bg_color: Optional[str] = "guess",
    chunk_size: Optional[int] = 4000,
    skip_lines: Optional[int] = 0,
    parameter_estimation_chunks: Optional[int] = 5,
) -> None:
    """Main processing function for piano rolls

    This function will perform image to midi digitization of a piano roll.

    Args:
        input_path (str): Path to the input image of a piano roll
        output_path (str): Path to which to write the created midi file
        method (str): Method to use for digitization. This parameter is currently unused and can be set to any value.
        config (dict): Configuration data to use for midi creation. Needs to contain, amongst others, initial guesses on the position of the tracks.
        bg_color (Optional[str], optional): Color of the background in the supplied roll scan. Must currently be one of 'black' and 'white' or 'guess' in which case it will be attempted to infer the color from the supplied scan. Defaults to "guess".
        chunk_size (Optional[int], optional): Vertical size of chunks to segment the image to for processing. Defaults to 4000.
        parameter_estimation_chunks (Optional[int], optional): Number of chunks to generate for estimating required digitization parameters. Higher numbers should improve accuracy but will have a negative performance impact. Defaults to 5.
    """
    logging.info(f"Reading input image from {input_path}...")

    image = skimage.io.imread(input_path)

    # TODO: Check if skip lines < height of image

    if skip_lines > 0:
        image = image[skip_lines:, :]

    if not bg_color in ["black", "white", "guess"]:
        raise ValueError(
            f"Background color must be one of 'black', 'white' or 'guess', got '{bg_color}'"
        )

    if bg_color == "guess":
        bg_color = hmsm.rolls.utils.guess_background_color(image)

    generator = hmsm.rolls.masking.MaskGenerator(
        image,
        bg_color,
        config["binarization_method"],
        chunk_size,
        **config["binarization_options"],
    )

    logging.info("Beginning processing of note data")

    alignment_grid = hmsm.rolls.utils.get_initial_alignment_grid(
        config["roll_width_mm"], config["track_measurements"]
    )

    note_data = list()

    width_bounds = _calculate_hole_width_range(config["hole_width_mm"])

    for bounds, masks in iter(generator):
        logging.info(f"Processing chunk from {bounds[0]} to {bounds[1]}")

        left_edge, right_edge = _get_roll_edges(np.invert(masks["holes"]))

        notes, alignment_grid = extract_note_data(
            masks["holes"], (left_edge, right_edge), alignment_grid, width_bounds
        )

        if notes is not None:
            notes[:, 0:2] = notes[:, 0:2] + bounds[0]
            note_data.append(notes)

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
    return notes[notes[:, 2] >= 0, :]


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


def _calculate_hole_width_range(
    width_mm: float,
    tolerances: Optional[Tuple[float, float]] = (0.5, 1.25),
    resolution: Optional[int] = 300,
) -> Tuple[int, int]:
    # TODO: Make this work for rolls that have multiple types of holes
    bounds = (
        int(width_mm / 25.4 * resolution * tolerances[0]),
        int(width_mm / 25.4 * resolution * tolerances[1]),
    )
    return bounds if bounds[1] > bounds[0] else (bounds[1], bounds[0])


def extract_note_data(
    note_mask: np.ndarray,
    roll_edges: Tuple[np.ndarray, np.ndarray],
    alignment_grid: np.ndarray,
    width_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts note information from masks

    Primary data extraction method for piano rolls that will detect notes on the provided masks and return them as tabular data.

    Args:
        note_mask (np.ndarray): Mask that contains the holes on a piano roll
        roll_edges (Tuple[np.ndarray, np.ndarray]): Arrays containing the the edge of the roll along the entire subchunk.
        alignment_grid (np.ndarray): Array containing information about the location of all tracks on the given piano roll
        width_bounds (Tuple[float, float]): Bounds within which the width of a given component must be to be considered as a hole in the piano roll

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
            ),
            components,
        )
    )

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        mask_debug = skimage.util.img_as_ubyte(note_mask)

    # If there are no notes in this chunk we skip it and continue
    if len(components) == 0:
        return (None, alignment_grid)

    left_edge, right_edge = roll_edges

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

        # Currently not beeing used

        # # Calculate the misalignment of the note center from the track center
        # relative_x_position = (x_position - left_edge[y_position]) / roll_width
        # misalignment = relative_x_position - alignment_grid[track_idx, 0:2].mean()

        # # If the misalignment is negative the note is to the left of the track
        # # and we need to shift the track to the left and vice versa for a positive misalignment
        # # We apply a correction to the alignment grid which we base on the difference in misalignments
        # # however we only correct for a fraction of the actual misalignment to prevent overcorrecting
        # correction = misalignment * 0.2
        # alignment_grid[:, 0:2] = alignment_grid[:, 0:2] + correction

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

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            cv2.putText(
                mask_debug,
                str(int(alignment_grid[track_idx, 2])),
                (
                    int(np.mean(component, axis=0)[1]),
                    int(np.mean(component, axis=0)[0]),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                123,
                2,
            )

    if len(note_data) == 0:
        return (None, alignment_grid)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        skimage.io.imsave(
            f"notes_aligned_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
            mask_debug,
        )

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
    width_bounds: Optional[Tuple[float, float]] = None,
    density_bounds: Optional[Tuple[float, float]] = None,
    height_to_width_ratio: Optional[Tuple[float, float]] = None,
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

    if width_bounds is not None and (
        dims[1] < width_bounds[0] or dims[1] > width_bounds[1]
    ):
        return False

    if density_bounds is not None:
        density = len(component) / (dims[0] * dims[1])

        if density < density_bounds[0] or density > density_bounds[1]:
            return False

    if height_to_width_ratio is not None:
        ratio = dims[0] / dims[1]
        if ratio < height_to_width_ratio[0] or ratio > height_to_width_ratio[1]:
            return False

    return True
