# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import functools
import itertools
import logging
import os
from typing import List, Optional, Tuple

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
    dynamics_line = list()

    for bounds, masks in iter(generator):
        logging.info(f"Processing chunk from {bounds[0]} to {bounds[1]}")

        left_edge, right_edge = _get_roll_edges(masks[-1])

        notes, alignment_grid = extract_note_data(
            masks[mask_idx],
            (left_edge, right_edge),
            alignment_grid,
            width_bounds,
            density_bounds,
        )

        if notes is not None:
            notes[:, 0:2] = notes[:, 0:2] + bounds[0]
            note_data.append(notes)

        dynamics_line_nodes = process_annotations(
            masks[1 - mask_idx], (left_edge, right_edge), bounds
        )
        if dynamics_line_nodes is not None:
            dynamics_line.append(dynamics_line_nodes)
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

    logging.info("Processing dynamics line")

    dynamics_line = np.vstack(dynamics_line)

    dynamics_line = process_dynamics_line(dynamics_line)

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


def process_dynamics_line(nodes: List[np.ndarray]) -> np.ndarray:
    nodes = nodes.astype(np.uint64)
    line_coords = list()

    for i in range(len(nodes) - 1):
        y = np.arange(nodes[i, 0], nodes[i + 1, 0])

        m = (int(nodes[i, 1]) - int(nodes[i + 1, 1])) / (
            int(nodes[i, 0]) - int(nodes[i + 1, 0])
        )
        b = (
            (int(nodes[i, 0]) * int(nodes[i + 1, 1]))
            - (int(nodes[i + 1, 0]) * int(nodes[i, 1]))
        ) / (int(nodes[i, 0]) - int(nodes[i + 1, 0]))

        line_coords.append(np.column_stack((y, b + (y * m))))

    line_coords = np.vstack(line_coords)

    return line_coords


def process_annotations(
    annotation_mask: np.ndarray,
    roll_edges: Tuple[np.ndarray, np.ndarray],
    bounds: Tuple[int, int],
) -> np.ndarray:
    left_edge, right_edge = roll_edges
    labels = skimage.measure.label(annotation_mask, background=False, connectivity=2)
    components = list(hmsm.utils.to_coord_lists(labels).values())

    dynamics_line_nodes = _get_dynamics_line(components, bounds)

    return dynamics_line_nodes

    # dynamics_line = list(
    #     filter(
    #         functools.partial(
    #             _filter_component,
    #             width_bounds=(15, 45),
    #             height_to_width_ratio=(0.5, 2),
    #         ),
    #         components,
    #     )
    # )

    # node_centers = np.array([np.mean(comp, axis=0) for comp in dynamics_line])

    # dists = scipy.spatial.distance.cdist(node_centers, node_centers)
    # dists[dists == 0] = float("inf")

    # distance_threshold = 200

    # adj_matrix = dists < distance_threshold

    # n_components, labels = scipy.sparse.csgraph.connected_components(
    #     adj_matrix, False, return_labels=True
    # )

    # assignments = np.zeros(len(node_centers), np.uint8)

    # # Do this until all elements are assigned
    # while np.any(assignments == 0):
    #     cluster_id = np.max(assignments) + 1
    #     # Select the unassigned element that is at the top most point of the image as the start point
    #     current_element_idx = node_centers[assignments == 0, 0].argmin(axis=0)
    #     current_element = node_centers[current_element_idx]
    #     assignments[current_element_idx] = cluster_id

    #     # Find the next closest element to the starting element
    #     next_element = dists[assignments == 0, current_element_idx].argmin()

    #     # TODO: Infere the threshold from data
    #     while (
    #         dists[assignments == 0, current_element_idx][next_element]
    #         < distance_threshold
    #     ):
    #         if (
    #             current_element[0] - node_centers[assignments == 0][next_element][0]
    #             > 10
    #         ):
    #             # Elements should generally be located below the current one, we only tolerate a very small upwards tilt
    #             # There should be a better way to do this...
    #             values = dists[assignments == 0, current_element_idx]
    #             values[next_element] = float("inf")
    #             dists[assignments == 0, current_element_idx] = values
    #             # Go to next closest element
    #             next_element = dists[assignments == 0, current_element_idx].argmin()
    #             continue

    #         current_element_idx = next_element
    #         current_element = node_centers[assignments == 0][current_element_idx]

    #         new_assignments = assignments[assignments == 0]
    #         new_assignments[current_element_idx] = cluster_id
    #         assignments[assignments == 0] = new_assignments

    #         next_element = dists[assignments == 0, current_element_idx].argmin()

    # min_distances = np.min(dists, axis=0)
    # median = np.median(min_distances)
    # dists = np.delete(dists, min_distances > median * 1.5, axis=0)
    # dists = np.delete(dists, min_distances > median * 1.5, axis=1)

    # dynamics_line = list(
    #     itertools.compress(dynamics_line, (min_distances <= median * 1.5).tolist())
    # )

    # clr_img = hmsm.utils.image_from_coords(dynamics_line, annotation_mask.shape)
    # filename = os.path.join("masks", f"dynamics_line_{bounds[0]}_{bounds[1]}.tif")

    # skimage.io.imsave(filename, clr_img)
    # cluster_distances = dists[labels == cluster_id, :][:, labels == cluster_id]

    # dists = dists[min_distances < median * 1.5, min_distances < median * 1.5]
    # pass


def _get_dynamics_line(components: np.ndarray, bounds: Tuple[int, int]) -> np.ndarray:
    dynamics_line = list(
        filter(
            functools.partial(
                _filter_component,
                width_bounds=(15, 45),
                height_to_width_ratio=(0.5, 2),
            ),
            components,
        )
    )

    node_centers = np.array([np.mean(comp, axis=0) for comp in dynamics_line])

    dists = scipy.spatial.distance.cdist(node_centers, node_centers)
    dists[dists == 0] = float("inf")

    distance_threshold = 200

    adj_matrix = dists < distance_threshold

    n_components, labels = scipy.sparse.csgraph.connected_components(
        adj_matrix, False, return_labels=True
    )

    component_sizes = np.bincount(labels)

    min_components = (bounds[1] - bounds[0]) / 400

    if np.max(component_sizes) < min_components or n_components > 10:
        return None

    cluster_id = np.argmax(component_sizes)
    cluster_distances = dists[labels == cluster_id, :][:, labels == cluster_id].copy()
    cluster_distances[cluster_distances == float("inf")] = 0
    cluster_nodes = node_centers[labels == cluster_id]

    first_node = cluster_nodes[:, 0].argmin()
    last_node = cluster_nodes[:, 0].argmax()

    path_distances, path_predecessors = scipy.sparse.csgraph.shortest_path(
        cluster_distances.astype(np.uint64), directed=False, return_predecessors=True
    )

    path = list([last_node])
    current_node = last_node

    while path_predecessors[first_node, current_node] != -9999:
        path.append(path_predecessors[first_node, current_node])
        current_node = path_predecessors[first_node, current_node]

    path = path[::-1]

    cluster_nodes = cluster_nodes[path]

    cluster_nodes[:, 0] = cluster_nodes[:, 0] + bounds[0]

    return cluster_nodes


def extract_note_data(
    note_mask: np.ndarray,
    roll_edges: Tuple[np.ndarray, np.ndarray],
    alignment_grid: np.ndarray,
    width_bounds: Tuple[float, float],
    density_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts note information from masks

    Primary data extraction method for piano rolls that will detect notes on the provided masks and return them as tabular data.

    Args:
        note_mask (np.ndarray): Mask that contains the holes on a piano roll
        roll_edges (Tuple[np.ndarray, np.ndarray]): Arrays containing the the edge of the roll along the entire subchunk.
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
