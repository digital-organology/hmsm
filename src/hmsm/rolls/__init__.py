# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import datetime
import functools
import itertools
import logging
import math
import os
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.signal
import scipy.sparse.csgraph
import scipy.spatial.distance
import skimage.io
import skimage.morphology

try:
    import enlighten
except ImportError:
    _has_enlighten = False
else:
    _has_enlighten = True

import hmsm.midi
import hmsm.rolls.masking
import hmsm.rolls.utils
import hmsm.utils


def process_roll(
    input_path: str,
    output_path: str,
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
        config (dict): Configuration data to use for midi creation. Needs to contain, amongst others, initial guesses on the position of the tracks.
        bg_color (Optional[str], optional): Color of the background in the supplied roll scan. Must currently be one of 'black' and 'white' or 'guess' in which case it will be attempted to infer the color from the supplied scan. Defaults to "guess".
        chunk_size (Optional[int], optional): Vertical size of chunks to segment the image to for processing. Defaults to 4000.
        skip_lines (Optional[int], optional): Number of lines to skip from the beginning of the roll scan, useful for excluding the roll head from beeing processed. Defaults to 0.
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
        logging.info(
            f"Background color of the input image was not specified, attempting automatic detection"
        )
        bg_color = hmsm.rolls.utils.guess_background_color(image)
        logging.info(f"Background color was detected to be {bg_color}")

    generator = hmsm.rolls.masking.MaskGenerator(
        image,
        bg_color,
        config["binarization_method"],
        chunk_size,
        **config["binarization_options"],
    )

    logging.info(
        "Beginning processing of the roll scan and extraction of musical action signals"
    )

    alignment_grid = hmsm.rolls.utils.get_initial_alignment_grid(
        config["roll_width_mm"], config["track_measurements"]
    )

    note_data = list()

    width_bounds = _calculate_hole_width_range(config["hole_width_mm"])

    if _has_enlighten:
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=generator.get_number_iterations(),
            desc="Processing roll scan",
            unit="chunks",
        )

    dyn_line = []

    for bounds, masks in iter(generator):
        if not _has_enlighten:
            logging.info(f"Processing chunk from {bounds[0]} to {bounds[1]}")

        try:
            left_edge, right_edge = _get_roll_edges(masks["holes"])
            notes, alignment_grid = extract_note_data(
                masks["holes"],
                (left_edge, right_edge),
                alignment_grid,
                width_bounds,
            )
        except Exception as e:
            logging.warning(
                f"Encountere an exception when trying to process the chunk [{bounds[0]}, {bounds[1]}] will assume that this is because the roll is finished and stop processing. Exception encountered: {traceback.format_exc()}"
            )
            break

        if "annotations" in masks:
            dyn = _process_annotations(masks["annotations"], bounds[0])
            dyn_line.append(dyn)
            # skimage.io.imsave(
            #     f"annotations_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
            #     masks["annotations"],
            # )

        if notes is not None:
            notes[:, 0:2] = notes[:, 0:2] + bounds[0]
            note_data.append(notes)

        if _has_enlighten:
            progress_bar.update()

    if _has_enlighten:
        manager.stop()

    note_data = np.vstack(note_data)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        filename = os.path.join("debug_data", "note_data_raw.csv")
        np.savetxt(filename, note_data, delimiter=",")

    logging.info("Post-processing extracted data...")

    if dyn_line:
        dyn_line = _postprocess_dynamics_line(dyn_line)

    note_data = merge_notes(note_data, config["hole_width_mm"])

    note_start = (note_data[:, 0]).tolist()
    note_duration = (note_data[:, 1] - note_data[:, 0]).tolist()
    midi_tone = note_data[:, 2].tolist()

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        filename = os.path.join("debug_data", "note_data_processed.csv")
        debug_array = np.hstack([note_start, note_duration, midi_tone])
        np.savetxt(filename, debug_array, delimiter=",")

    logging.info("Finished post processing, generating midi data...")

    midi_generator = hmsm.midi.MidiGenerator(3)
    midi_generator.make_midi(
        note_start,
        note_duration,
        midi_tone,
        config["hole_width_mm"],
        dyn_line,
    )

    logging.info(f"Writing midi file to {output_path}")

    midi_generator.write_midi(output_path)

    logging.info("Done, bye")


def _postprocess_dynamics_line(line: list):
    line = list(itertools.chain(*line))
    line = np.vstack(line).astype(np.uint32)

    dists = scipy.spatial.distance.cdist(line, line)
    dists[dists == 0] = float("inf")

    closest_node = dists.min(axis=0).astype(np.uint16)

    NODE_DISTANCE_THRESHOLD = 2.5 * np.mean(closest_node)

    # keep = closest_node <= NODE_DISTANCE_THRESHOLD

    keep = (dists <= NODE_DISTANCE_THRESHOLD).sum(axis=0) > 2

    line = line[keep, :]

    line = line[line[:, 0].argsort(), :]

    # line_interpolated = np.arange(line[:, 0].min(), line[:, 0].max())

    # line_interpolated = np.column_stack(
    #     (line_interpolated, np.zeros_like(line_interpolated))
    # )

    # Prepare points on line data

    groups = line[:, 0].copy()
    line = np.delete(line, 0, axis=1)

    _id, _pos, g_count = np.unique(groups, return_index=True, return_counts=True)

    g_sum = np.add.reduceat(line[groups.argsort()], _pos, axis=0)
    g_mean = g_sum / g_count[:, None]
    line = np.column_stack((_id, g_mean)).astype(np.uint32)

    current_point = line[0, :]

    line_interpolated = []
    line_interpolated.append(current_point[1])

    for i in range(1, len(line)):
        next_point = line[i, :]

        if (next_point[0] - current_point[0]) > 1:
            m = (int(next_point[1]) - int(current_point[1])) / (
                int(next_point[0]) - int(current_point[0])
            )

            b = int(current_point[1]) - (m * int(current_point[0]))

            line_interpolated.append(
                np.arange(current_point[0] + 1, next_point[0]) * m + b
            )

        line_interpolated.append(next_point[1])
        current_point = next_point

    line_interpolated = (
        np.column_stack(
            (
                np.arange(line[:, 0].min(), (line[:, 0].max() + 1)),
                np.hstack(line_interpolated),
            )
        )
        .round()
        .astype(np.uint32)
    )

    line_interpolated[:, 1] = scipy.signal.savgol_filter(
        line_interpolated[:, 1], 500, 2
    ).round()

    return line_interpolated


def _process_annotations(mask: np.ndarray, offset: int) -> np.ndarray:
    # coords = np.argwhere(mask)
    # uq = np.unique(np.argwhere(mask)[:, 0], return_index=True)
    # annots = dict(zip(uq[0] + offset, np.split(coords[:, 1], uq[1][1:])))
    MIN_NUM_PIXELS = 200

    coords = hmsm.utils.to_coord_lists(mask)
    coords = list(coords.values())
    for val in coords:
        val[:, 0] = val[:, 0] + offset

    coords = [val for val in coords if len(val) >= MIN_NUM_PIXELS]

    coords = [np.mean(val, axis=0) for val in coords]

    return coords


def merge_notes(notes: np.ndarray, hole_size_mm: Optional[float]) -> np.ndarray:
    """Merge notes that sound as one

    On original playback devices holes that are closely grouped will sound as a single, longer note.
    This method emulates that behaviour by merging notes that are closer than a calculated distance threshold.

    Args:
        notes (np.ndarray): Array containing detected notes

    Returns:
        np.ndarray: Input array with close notes merged
    """
    CORRECTION_FACTOR = 1.75

    offset = notes[:, 0].min()
    notes[:, 0:2] = notes[:, 0:2] - offset

    merge_threshold = (
        CORRECTION_FACTOR * math.floor(hole_size_mm / 25.4 * 300)
        if hole_size_mm is not None
        else round(np.mean(notes[:, 1] - notes[:, 0]) * CORRECTION_FACTOR)
    )

    midi_notes = np.unique(notes[:, 2])

    # Should be superfluous as the processing will yield notes already ordered
    notes = notes[notes[:, 0].argsort()]

    notes_merged = list()

    for tone in midi_notes:
        tone_notes = notes[notes[:, 2] == tone, :]

        current_row = tone_notes[0, :]

        for i in range(1, len(tone_notes)):
            if (tone_notes[i, 0] - tone_notes[i - 1, 1]) < merge_threshold:
                current_row[1] = tone_notes[i, 1]
            else:
                notes_merged.append(current_row)
                current_row = tone_notes[i, :]

        notes_merged.append(current_row)

    # rows.append(current_row)

    #     note_start_idx = (
    #         np.diff(tone_notes[:, 0], prepend=(merge_threshold * -1 - 1))
    #         > merge_threshold
    #     )
    #     note_end_idx = np.roll(note_start_idx, -1)

    #     tone_notes_merged = tone_notes[note_start_idx, :]
    #     tone_notes_merged[:, 1] = tone_notes[note_end_idx, 1]

    #     notes_merged.append(tone_notes_merged)

    notes_merged = np.vstack(notes_merged)

    return notes_merged


def _calculate_hole_width_range(
    width_mm: float,
    tolerances: Optional[Tuple[float, float]] = (0.5, 1.25),
    resolution: Optional[int] = 300,
) -> Tuple[int, int]:
    """Calculate the accapatable size of components to be used as hole

    Args:
        width_mm (float): Physical width of the holes on the roll in mm.
        tolerances (Optional[Tuple[float, float]], optional): Tolerances to use when calculating the interval. Defaults to (0.5, 1.25).
        resolution (Optional[int], optional): Resolution of the input scan in dpi. Defaults to 300.

    Returns:
        Tuple[int, int]: Lower and upper bounds for component filtering, in pixels
    """
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
    if mask[0, 0] == True:
        mask = np.invert(mask)

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
    height_bounds: Optional[Tuple[float, float]] = None,
    height_to_width_ratio: Optional[Tuple[float, float]] = None,
) -> bool:
    """Checks if the given component fits within the given bounds

    Args:
        component (np.ndarray): Array of coordinates that belong to the component in question
        width_bounds (Optional[Tuple[float, float]], optional): Bounds within which the widht of the given component must be. Defaults to None.
        density_bounds (Optional[Tuple[float, float]], optional): Bounds within which the density of the given component must be. Defaults to None.
        height_bounds (Optional[Tuple[float, float]], optional): Bounds within which the height of the given component must be. Defaults to None.
        height_to_width_ratio (Optional[Tuple[float, float]], optional): Bounds within which the height to width ratio of the given component must be. Defaults to None.

    Returns:
        bool: True if the component falls within the bounds provided, False otherwise
    """
    dims = component.max(axis=0) - component.min(axis=0)

    if width_bounds is not None and (
        dims[1] < width_bounds[0] or dims[1] > width_bounds[1]
    ):
        return False

    if height_bounds is not None and (
        dims[0] < height_bounds[0] or dims[0] > height_bounds[1]
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
