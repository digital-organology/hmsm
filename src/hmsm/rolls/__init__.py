# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import datetime
import functools
import itertools
import logging
import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import scipy.signal
import scipy.sparse.csgraph
import scipy.spatial.distance
import skimage.io
import skimage.morphology
import skimage.util

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
    tempo: Optional[int] = 50,
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
        tempo (Optional[int], optional): Tempo of the roll in feet-per-minute times 10. Defaults to 50.
    """
    logging.info(f"Reading input image from {input_path}...")

    image = skimage.io.imread(input_path)

    if image.shape[2] == 4:
        logging.info("Image appears to have an alpha channel which will be dropped")
        image = image[:, :, 0:3]

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
    hole_width = (
        config["hole_width_mm"][0]
        if isinstance(config["hole_width_mm"], list)
        else config["hole_width_mm"]
    )

    if _has_enlighten:
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=generator.get_number_iterations(),
            desc="Processing roll scan",
            unit="chunks",
        )

    dyn_line = []
    pedal_markers = []

    good_chunk_found = False

    for bounds, masks in iter(generator):
        if not _has_enlighten:
            logging.info(f"Processing chunk from {bounds[0]} to {bounds[1]}")

        if masks is None:
            if good_chunk_found == False:
                logging.info(
                    f"Chunk {bounds[0]} to {bounds[0]} failed to process at the roll start and will be skipped"
                )
                continue
            else:
                logging.info(
                    f"Chunk {bounds[0]} to {bounds[0]} failed to process, this likely indicates the roll end."
                )
                break

        left_edge, right_edge = masks["edges"]

        if good_chunk_found == False and (
            (left_edge.max() - left_edge.min()) > 250
            or (right_edge.max() - right_edge.min()) > 250
        ):
            start_idx = _find_roll_start(left_edge, right_edge, image.shape[1])
            if start_idx is None:
                logging.info(
                    f"Chunk {bounds[0]} to {bounds[1]} appears to be faulty, this might be because it contains the roll head or because of defects with the roll scan, skipping"
                )
                continue
            masks = {
                k: v[start_idx:, :] if isinstance(v, np.ndarray) else v
                for k, v in masks.items()
            }

        notes, alignment_grid = extract_note_data(
            masks["holes"],
            (left_edge, right_edge),
            alignment_grid,
            width_bounds,
        )

        if "annotations" in masks:
            if "pedal_cutoff" in config:
                dyn, pedal = _process_annotations(
                    masks["annotations"],
                    bounds[0],
                    (left_edge, right_edge),
                    config["pedal_cutoff"],
                )
                pedal_markers.append(pedal)
            else:
                dyn = _process_annotations(masks["annotations"], bounds[0])
            dyn_line.append(dyn)

        if notes is not None:
            notes[:, 0:2] = notes[:, 0:2] + bounds[0]
            note_data.append(notes)
            good_chunk_found = True

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
        logging.info("Processing dynamics annotations")
        dyn_line = _postprocess_dynamics_line(dyn_line)
    else:
        dyn_line = None

    if pedal_markers:
        pedal_markers = _postprocess_pedal_markers(pedal_markers)
        note_data = (
            np.vstack([note_data, pedal_markers])
            if pedal_markers is not None
            else note_data
        )

    note_data = merge_notes(note_data, hole_width)

    note_start = (note_data[:, 0]).tolist()
    note_duration = (note_data[:, 1] - note_data[:, 0]).tolist()
    midi_tone = note_data[:, 2].tolist()

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        filename = os.path.join("debug_data", "note_data_processed.csv")
        debug_array = np.hstack([note_start, note_duration, midi_tone])
        np.savetxt(filename, debug_array, delimiter=",")

    logging.info("Finished post processing, generating midi data...")

    midi_generator = hmsm.midi.MidiGenerator(tempo)
    midi_generator.make_midi(
        note_start,
        note_duration,
        midi_tone,
        hole_width,
        dyn_line,
    )

    logging.info(f"Writing midi file to {output_path}")

    midi_generator.write_midi(output_path)

    logging.info("Done, bye")


def _postprocess_pedal_markers(markers: list) -> np.ndarray:
    """Postprocess printed pedal annotations

    This will take the data generated by the annotation processing method and
    transform it into data points that can be used to set pedal signals in the midi creationb process

    Args:
        markers (list): Pedal marks found in the annotations

    Returns:
        np.ndarray: Processed pedal control information
    """
    markers = list(itertools.chain(*markers))
    if len(markers) == 0:
        return None
    markers = np.vstack(markers)

    markers = markers[markers[:, 1] > 3000]

    markers = markers[markers[:, 0].argsort()]

    for i in range(1, len(markers)):
        if markers[i, 0] - markers[i - 1, 0] <= 100:
            markers[i - 1, 1] = markers[i - 1, 1] + markers[i, 1]
            markers[i - 1, 2] = max(markers[i - 1, 2], markers[i, 2])
            markers[i - 1, 3] = min(markers[i - 1, 3], markers[i, 3])
            markers[i, 1] = 0

    markers = markers[markers[:, 1] > 0]

    markers = np.c_[markers, markers[:, 2] - markers[:, 3]]
    markers[:, 4] = markers[:, 2] - markers[:, 3]

    if len(markers) < 10:
        logging.info(
            f"Found only {len(markers)} data points for pedal markers, this likely indicates there are no pedal annotations and the detected data points are noise, skipping."
        )
        return None

    cutoffs = np.mean(markers, axis=0)

    notes = list()
    note_start = None

    for i in range(0, len(markers)):
        if markers[i, 4] > cutoffs[4]:
            # If the previous marker was a pedal start we check if the next marker is a start marker
            # if it is we most likely got this one wrong and it is an end marker
            # If it isnt we most likely missed the previous end marker and overwrite
            if (i + 1) < len(markers) and (
                note_start is None or markers[i + 1, 4] < cutoffs[4]
            ):
                note_start = markers[i, 0]
            elif note_start is not None:
                notes.append(
                    np.array(
                        [
                            note_start,
                            markers[i, 0],
                            -3,
                        ]
                    )
                )
                note_start = None
            else:
                # We only get here if we're at the last entry and it's a start marker but we're missing the corresponding end mark
                # so we just skip it
                continue
        else:
            # Basically the same thing as before: If the previous marker was an end marker we check the next marker.
            # If it also is an end marker we assume we got this one wrong and set it as a start marker.
            # If it is a start marker we might have missed the corresponding start marker and discard this data point
            if note_start is not None:
                notes.append(
                    np.array(
                        [
                            note_start,
                            markers[i, 0],
                            -3,
                        ]
                    )
                )
                note_start = None
            elif (i + 1) < len(markers) and (markers[i + 1, 4] < cutoffs[4]):
                note_start = markers[i, 0]

    return np.vstack(notes)


def _find_roll_start(
    left_edge: np.ndarray,
    right_edge: np.ndarray,
    image_width: int,
) -> int | None:
    """Find the beginning of the roll

    Will attempt to find the point at which the roll begins to be straight from the detected
    edges of the roll. Note that this approach is imperfect, some rolls have labels even beyond the triangular roll head.

    Args:
        left_edge (np.ndarray): Array of the detected positions of the left roll edge
        right_edge (np.ndarray): Array of the detected positions of the right roll edge
        image_width (int): Width of the image beeing processed

    Returns:
        int | None: Index at which the roll beginns beeing straight or None if it can't be found
    """
    assert len(left_edge) == len(right_edge)

    SHIFT_THRESHOLD = 20

    max_idx = len(left_edge) - 1

    for i in reversed(range(0, len(left_edge), 100)):
        segment_left = left_edge[i:max_idx]
        segment_right = right_edge[i:max_idx]
        if (
            (segment_left.max() - segment_left.min()) > SHIFT_THRESHOLD
            or (segment_right.max() - segment_right.min()) > SHIFT_THRESHOLD
        ) or (segment_left.mean() < 5 and segment_right.mean() > (image_width * 0.99)):
            return i if max_idx != len(left_edge) - 1 else None
        max_idx = i

    return 0


def _postprocess_dynamics_line(line: list) -> np.ndarray:
    """Process printed dynamics annotations

    Will take the elements detected on the dynamics line and interpolate
    dynamics measures for every point between the first and last provided points

    Args:
        line (list): List containing coordinates of nodes detected as dynamics annotations

    Returns:
        np.ndarray: Interpolated dynamics values for the entire area covered by the provided data
    """
    line = list(itertools.chain(*line))
    if len(line) == 0:
        return None
    line = np.vstack(line).astype(np.uint32)

    if len(line) < 200:
        logging.info(
            f"Found only {len(line)} data points for the dynamics line, this most likely indicates the roll doesnt have any printed dynamics annotations and the data found is noise. Skipping."
        )
        return None

    dists = scipy.spatial.distance.cdist(line, line)
    dists[dists == 0] = float("inf")

    closest_node = dists.min(axis=0).astype(np.uint16)

    NODE_DISTANCE_THRESHOLD = 2.5 * np.mean(closest_node)

    # keep = closest_node <= NODE_DISTANCE_THRESHOLD

    keep = (dists <= NODE_DISTANCE_THRESHOLD).sum(axis=0) > 2

    line = line[keep, :]

    line = line[line[:, 0].argsort(), :]

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


def _process_annotations(
    mask: np.ndarray,
    offset: Optional[int] = 0,
    edges: Optional[tuple] = None,
    pedal_cutoff: Optional[float] = None,
) -> List | Tuple[List, List]:
    """Extract annotated information from mask

    Will process the mask and extract information for the printed dynamics line and (if present) printed pedal annotations

    Args:
        mask (np.ndarray): Binary mask of the printed annotations
        offset (int): Offset of the chunk that mask represents, will be used to adjust position information
        edges (Optional[tuple], optional): Tuple containing positions of the roll edges. Only required if pedal markers are present on the roll. Defaults to None.
        pedal_cutoff (Optional[float], optional): Relative cutoff point (from the left side of the roll) at which the pedal annotations stop and the dynamics line begins. For cutoff <= 0.5 it is assumed the pedal marks are on the left side of the roll, for values > 0.5 on the right side. Defaults to None.

    Returns:
        List | Tuple[List, List]: The extracted annotation data
    """
    MIN_NUM_PIXELS = 200 if pedal_cutoff is None else 250

    # Correct processing of the pedal is more sensitive to holes in the mask, so we run an additional dilation

    if pedal_cutoff is not None:
        mask = skimage.morphology.binary_dilation(mask, skimage.morphology.diamond(5))

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        skimage.io.imsave(
            f"annotations_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg",
            skimage.util.img_as_ubyte(mask),
        )

    coords = hmsm.utils.to_coord_lists(mask)
    coords = list(coords.values())
    for val in coords:
        val[:, 0] = val[:, 0] + offset

    coords = [
        val
        for val in coords
        if len(val) >= MIN_NUM_PIXELS
        and len(val) <= (mask.shape[0] + mask.shape[1] * 10)
    ]

    # If there are pedal annotations on the roll we check each data point for its location relative to the pedal cutoff limit

    if pedal_cutoff is not None:
        pedal = list()
        dyn_line = list()
        for val in coords:
            y, x = tuple(np.mean(val, axis=0).astype(int))
            if (
                pedal_cutoff <= 0.5
                and ((edges[1][y - offset] - edges[0][y - offset]) * pedal_cutoff)
                + edges[0][y - offset]
                > x
            ) or (
                pedal_cutoff > 0.5
                and ((edges[1][y - offset] - edges[0][y - offset]) * pedal_cutoff)
                + edges[0][y - offset]
                < x
            ):
                pedal.append(
                    np.array(
                        (
                            y,
                            len(val),
                            val.max(axis=0)[1] - edges[0][y - offset],
                            val.min(axis=0)[1] - edges[0][y - offset],
                        )
                    )
                )
            else:
                dyn_line.append(np.mean(val, axis=0))
    else:
        dyn_line = [np.mean(val, axis=0) for val in coords]

    if pedal_cutoff is not None:
        return dyn_line, pedal
    return dyn_line


def merge_notes(
    notes: np.ndarray,
    hole_size_mm: Optional[float] = None,
) -> np.ndarray:
    """Merge notes that sound as one

    On original playback devices holes that are closely grouped will sound as a single, longer note.
    This method emulates that behaviour by merging notes that are closer than a calculated distance threshold.

    Args:
        notes (np.ndarray): Array containing detected notes
        hole_size_mm (Optional[float]): Size of the holes on the roll, will be used to calculate the merging threshold. If missing this will be guessed from the data. Defaults to None.

    Returns:
        np.ndarray: Input array with close notes merged
    """
    # Holes with a gap < hole_size * CORRECTION FACTOR are assumed to sound as a single note
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

    notes_merged = np.vstack(notes_merged)

    return notes_merged


def _calculate_hole_width_range(
    width_mm: float | List[float],
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
    tolerances = (
        (tolerances[0], tolerances[1]) if tolerances[0] < tolerances[1] else tolerances
    )

    width_mm = list([width_mm]) if isinstance(width_mm, float) else width_mm

    bounds_min = min(
        [int(width / 25.4 * resolution * tolerances[0]) for width in width_mm]
    )
    bounds_max = max(
        [int(width / 25.4 * resolution * tolerances[1]) for width in width_mm]
    )

    return (bounds_min, bounds_max)


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
