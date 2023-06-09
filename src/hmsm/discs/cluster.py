# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import hmsm.utils
import hmsm.discs.utils
import hmsm.midi
import logging
import cv2
import sklearn.cluster
from scipy.spatial import distance
from typing import Optional, Tuple

def process_disc(input_image: np.ndarray, output_path: str, config: dict, offset: Optional[int] = 0, verbose: Optional[bool] = False) -> None:
    """Performs clustering based digitization of an input image

    This method will take the provided input image of a cardboard disc and attempt to extract all required information to create a midi file from it.

    Args:
        input_image (np.ndarray): Image of a cardboard disc
        output_path (str): File path to write the output midi file to
        config (dict): Dictionary containing configuration information for the digitization process, e.g. the mapping from disc tracks to midi numbers
        offset (Optional[int], optional): Offset of the disc start, in degrees counted counterclockwise. Defaults to 0.
        verbose (Optional[bool], optional): Should verbose output be enabled. Defaults to False.
    """
    # Binarize input image

    logging.info("Preprocessing image")

    if "binarization_threshold" in config:
        logging.debug(f"Using provided value of {config['binarization_threshold']} as binarziation threshold")
        bin_image = hmsm.utils.binarize_image(input_image, config['binarization_threshold'])
    else:
        logging.debug("No binarization threshold provided, will use Otsu's method to estimate optimal threshold")
        bin_image = hmsm.utils.binarize_image(input_image)

    # Crop to contents

    bin_image = hmsm.utils.crop_image_to_contents(bin_image)

    # Detect edges

    if "n_erosions" in config:
        logging.debug(f"Using provided value of {config['n_erosions']} for edge detection")
        edges = hmsm.utils.morphological_edge_detection(bin_image, config["n_erosions"])
    else:
        logging.debug("Using default value for number of erosions")
        edges = hmsm.utils.morphological_edge_detection(bin_image)

    # Fit ellipse and find center

    center_x, center_y, a, b, theta = hmsm.discs.utils.fit_ellipse_to_circumference(bin_image)

    # Remove everything on the inner disc

    ellipse_mask = cv2.ellipse(np.zeros((edges.shape[0], edges.shape[1]), np.uint8), (center_x, center_y), (int(a * config["radius_inner"]), int(b * config["radius_inner"])), theta, 0, 360, 1, -1)

    edges[ellipse_mask == 1] = 0

    # Convert to coordinates

    coords = hmsm.discs.utils.to_coord_lists(edges)

    # Get the outer edge of the disc for distance calculations
    # We could also get the edge from the edge image that is the outer edge,
    # however this could lead to wrong calculations if the disc has physical damages

    outer_edge = cv2.ellipse(np.zeros((edges.shape[0], edges.shape[1]), np.uint8), (center_x, center_y), (int(a), int(b)), theta, 0, 360, 1, 1)
    outer_edge = np.argwhere(outer_edge == 1)

    # We can now drop the edge of the outer edge, it should always be the one with the most pixels
    
    del coords[max(coords, key = lambda k: coords[k].shape[0])]

    track_mapping = assign_tracks(coords, (center_y, center_x), outer_edge, len(config["track_mapping"]), config["first_track"])
    
    # Calculate the start and end positions of every hole relative to the circumference of the disc

    timing_data = calculate_note_timings(coords, (center_y, center_x), offset)
    
    # Combine with track information to create midi

    midi_notes = [config["track_mapping"][str(track)] for track in track_mapping.values()]

    midi_file = hmsm.midi.create_midi(timing_data[:,1].tolist(), timing_data[:,3].tolist(), midi_notes)

    midi_file.save(output_path)

    # plot_data = zip(track_mapping.values(), coords.values())


    # print("Hi from the clustering digitization method")
    # pass

def calculate_note_timings(coords: dict, center: Tuple[int, int], offset: Optional[int] = 0) -> np.ndarray:
    """Calculate timing information

    Given a dictionary of edges (in the form of coordinate lists) this method will calculate the timing information (beginning, end and duration) for the associated edges.

    Args:
        coords (dict): Dictionary with edge ids as keys and coordinate lists as values
        center (Tuple[int, int]): Position of the center of the disc, in the form (y, x)
        offset (Optional[int], optional): Offset of the disc beginning, in degrees counterclockwise. Defaults to 0.

    Returns:
        np.ndarray: An array containing four columns: The note id, note beginning, note end and note duration, all except the id in degrees
    """
    start_positions = list()
    end_positions = list()
    (center_y, center_x) = center

    for edge in coords.values():
        # Shift coordinates by the center of the disc to make them centered around (0,0) and calculate the rotation in degrees
        degrees = np.arctan2(edge[:,1] - center_y, edge[:,0] - center_x) * 180 / np.pi
        start_positions.append(degrees.min())
        end_positions.append(degrees.max())

    timing_data = np.empty((len(coords), 4), np.float64)
    timing_data[:,0] = list(coords.keys())
    timing_data[:,1] = start_positions
    timing_data[:,2] = end_positions
    timing_data[:,3] = timing_data[:,2] - timing_data[:,1]
    timing_data[:,[1,2]] = timing_data[:,[1,2]] + 180

    if offset > 0:
        timing_data[:,[1,2]] = timing_data[:,[1,2]] - offset
        timing_data[:,[1,2]][timing_data[:,[1,2]] < 0] = timing_data[:,[1,2]][timing_data[:,[1,2]] < 0] + 360

    return timing_data

def assign_tracks(coords: dict, center: Tuple[int, int], outer_edge: np.ndarray, n_tracks: int, first_track: float) -> dict:
    """Find the track assignments for the given edges

    This will find the track each edge belongs to by performing a cluster analysis on the calculated distances between the edge and the center and outer edge of the disc.

    Args:
        coords (dict): Dic
        center (Tuple[int, int]): Position of the center of the disc, in the form (y, x)
        outer_edge (np.ndarray): Coordinate list of the outer edge of the disc
        n_tracks (int): Theoretical number of tracks on the disc, regardless of whether they are used on this particular medium
        first_track (float): Relative distance of the first track between disc center and outer edge

    Returns:
        dict: A dictionary containing track assignments for all edge ids, keys are edge ids as in coords, values are assigned track numbers
    """

    logger = logging.getLogger()

    # List to store the distance from edge center to disc center relative to the disc radius
    # Using relative distances allows us to compensate for some warping and also to
    # use the same paramters for clustering regardless of the actual image size
    distances = list()
    
    for edge in coords.values():
        # This is not super accurate and one might consider using the closest or farthest point of each edge relative to the center
        # however practical testing showed that there is no practical difference in clustering quality
        edge_center = np.mean(edge, 0).astype(np.uint16)
        edge_center = np.expand_dims(edge_center, axis = 0)
        dist_inner = distance.cdist(edge_center, [center]).min()
        dist_outer = distance.cdist(edge_center, outer_edge).min()
        distances.append(dist_inner / (dist_outer + dist_inner))

    # Use MeanShift clustering to group edges together
    # MeanShift has the advantage of not needing a priori knowledge of the number of clusters
    # It has also proven to be more robust compared to algorithms for natural break optimization/1-D Clustering for our use case
    # As it is meant to cluster 2-D Data, we just add a dummy second dimension which is 0 always

    cluster_data = np.array(distances)
    cluster_data = cluster_data * 1000
    cluster_data = np.column_stack((cluster_data, np.zeros(len(cluster_data))))
    cluster_data = cluster_data.astype(np.uint32)
    # Using a bandwidth of 8 has shown to work generally well in empirical testing
    # It should do so across multiple disc sizes as we scale for the disc radius
    mean_shift = sklearn.cluster.MeanShift(bandwidth = 8, bin_seeding = True)
    mean_shift.fit(cluster_data)
    labels = mean_shift.labels_

    # Sort classes and create mapping
    assignments = np.column_stack((labels, np.array(distances)))
    ids = np.unique(assignments[:,0])
    cluster_means = [np.mean(assignments[assignments[:,0] == i, 1]) for i in ids]
    cluster_means = np.column_stack((ids, cluster_means))
    cluster_means = cluster_means[cluster_means[:,1].argsort()]

    # At this point we might have ended up with more clusters than there are tracks on the disc
    # So we merge them back together until we have at max as many clusters as tracks

    while cluster_means.shape[0] > n_tracks:
        dists = np.diff(cluster_means[:,1])
        merge_to_idx = np.argmin(dists)
        merge_to = cluster_means[merge_to_idx,0]
        merge_from = cluster_means[merge_to_idx + 1,0]

        assignments[assignments[:,0] == merge_from,0] = merge_to
        assignments[assignments[:,0] > merge_to,0] = assignments[assignments[:,0] > merge_to,0] - 1

        labels[labels == merge_from] = merge_to
        labels[labels > merge_to] = labels[labels > merge_to] - 1

        ids = np.unique(assignments[:,0])
        cluster_means = [np.mean(assignments[assignments[:,0] == i, 1]) for i in ids]

        cluster_means = np.column_stack((ids, cluster_means))
        cluster_means = cluster_means[cluster_means[:,1].argsort()]

    # Change the labels so they are sorted from the inside of the disc to the outside

    copy = np.zeros_like(labels)
    for cluster in np.unique(labels):
        copy[labels == cluster] = np.argwhere(cluster_means[:,0] == cluster)[0][0]

    labels = copy + 1

    assignments = np.column_stack((list(coords.keys()), labels))

    # If we end up with less tracks than are on the format we want to
    # find the places where unpopulated tracks are located and shift everything accordingly

    # Calculate average track distance

    track_distances = np.column_stack((np.arange(1, cluster_means.shape[0] + 1), cluster_means[:,1]))

    track_distances = np.insert(track_distances, 0, np.array((0, first_track)), 0)
    
    track_distances = np.column_stack((track_distances[:,0], np.append(0, np.diff(track_distances[:,1], axis = 0))))

    median_track_width = np.median(track_distances, axis = 0)[1]

    # Shift tracks until we have the correct number of tracks

    while np.max(track_distances, axis = 0)[0] < n_tracks:
        max_row_idx = np.argmax(track_distances, axis = 0)[1]
        max_row = track_distances[max_row_idx]

        if round((max_row[1] - median_track_width) / median_track_width) < 1:
            break
        
        track_distances[max_row_idx][1] = max_row[1] - median_track_width

        # Increment all rows after this one up

        for i in range(max_row_idx, len(track_distances)):
            track_distances[i][0] = track_distances[i][0] + 1

    # Alter assignments

    copy = assignments.copy()

    for track in np.unique(assignments[:,1]):
        copy[:,1][assignments[:,1] == track] = track_distances[int(track), 0]

    assignments = dict(zip(copy[:,0], copy[:,1]))

    if logger.isEnabledFor(logging.DEBUG):
        plot_data = dict(zip(assignments.values(), coords.values()))
        debug_image = hmsm.utils.image_from_coords(plot_data, plot_labels = True)

    return assignments

