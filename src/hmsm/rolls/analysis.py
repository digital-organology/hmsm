# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import functools
import json
import logging
import math
import pathlib
from typing import List, Optional, Self, Tuple

import cv2
import numpy as np
import skimage.filters
import skimage.io
import skimage.morphology
import skimage.util
import sklearn.cluster

try:
    import pandas as pd
except ImportError:
    _has_pandas = False
else:
    _has_pandas = True

try:
    import enlighten
except ImportError:
    _has_enlighten = False
else:
    _has_enlighten = True

import hmsm.rolls
import hmsm.rolls.masking
import hmsm.rolls.masking.methods
import hmsm.rolls.utils


def analyze_roll(
    image_path: str,
    output_path: str,
    roll_physical_width: float,
    skip_lines: Optional[int] = 0,
    hole_width: Optional[float] = 1.5,
    threshold: Optional[float] = 0.15,
    bandwidth: Optional[float] = 2,
    chunk_size: Optional[int] = 4000,
) -> None:
    """Detect track positions on a roll scan

    This method will try to detect the track positions on the given roll scan image and create a configuration profile stub for processing rolls of this type as well as various debug images.

    Args:
        image_path (str): Path to the input image
        output_path (str): Path to write the generated configuration stub to
        roll_physical_width (float): Physical width of the roll in the scan
        skip_lines (Optional[int], optional): Number of lines to skip from the beginning of the roll scan, useful for excluding the roll head from beeing processed. Defaults to 0.
        hole_width (Optional[float], optional): Width of the holes on the roll scan in mm. Defaults to 1.5.
        threshold (Optional[float], optional): Theshold to be used for binarization of the roll scan, must be between 0 and 1. Defaults to 0.15.
        bandwidth (Optional[float], optional): Bandwidth to use for the underlying clustering alogorithm when grouping notes to tracks. Defaults to 2.
        chunk_size (Optional[int], optional): Size of the chunks to use when processing the roll. Defaults to 4000.

    Raises:
        ImportError: Will be raised if pandas is not available
    """
    if not _has_pandas:
        raise ImportError("This function requires pandas to be installed")

    logging.info("Reading input image from provided path")

    try:
        image = skimage.io.imread(image_path)
    except FileNotFoundError:
        logging.error("Failed to read image from provided path")
        raise

    image = image[skip_lines:, :]

    image_path = f"{output_path}_images/"
    pathlib.Path(image_path).mkdir(exist_ok=True)

    logging.info(
        f"Output images will be written under '{image_path}', existing files under that path may be overwritten"
    )

    logging.info("Beginning first pass over the roll scan to determine track positions")

    notes = []

    if _has_enlighten:
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=math.ceil(image.shape[0] / chunk_size),
            desc="Detecting track positions",
            unit="chunks",
        )

    bg_color = hmsm.rolls.utils.guess_background_color(image)
    logging.info(f"Determined background color of the provided image to be {bg_color}")

    width_bounds = hmsm.rolls._calculate_hole_width_range(hole_width)

    for start in range(0, image.shape[0], chunk_size):
        end = (
            start + chunk_size
            if start + chunk_size < image.shape[0]
            else image.shape[0]
        )

        if not _has_enlighten:
            logging.info(f"Processing chunk from {start} to {end}")

        mask = hmsm.rolls.masking.methods.v_channel(
            image[start:end, :, :], bg_color, threshold
        )
        mask = mask["holes"]

        left_edge, right_edge = hmsm.rolls._get_roll_edges(np.invert(mask))

        labels = skimage.measure.label(mask, background=False, connectivity=2)
        components = list(hmsm.utils.to_coord_lists(labels).values())

        components = list(
            filter(
                functools.partial(
                    hmsm.rolls._filter_component,
                    width_bounds=width_bounds,
                    height_bounds=(width_bounds[0], 1000),
                ),
                components,
            )
        )

        for component in components:
            # Calculate relative position on the roll (from the left roll edge)
            height, width = tuple(component.max(axis=0) - component.min(axis=0))

            y_position = int(round(component[:, 0].min() + (height / 2)))
            x_position = int(round(component[:, 1].min() + (width / 2)))
            roll_width = right_edge[y_position] - left_edge[y_position]
            left_dist = (component[:, 1].min() - left_edge[y_position]) / roll_width
            right_dist = (component[:, 1].max() - left_edge[y_position]) / roll_width

            note_height = y_position + start

            note_data = dict(
                {
                    "left_dist": left_dist,
                    "right_dist": right_dist,
                    "height": note_height,
                }
            )

            notes.append(note_data)

        if _has_enlighten:
            progress_bar.update()

    if _has_enlighten:
        manager.stop()

    logging.info("Finished first pass over the roll scan, calculating track positions")

    df = pd.DataFrame.from_records(notes)

    left_dists = df.left_dist.to_numpy() * 1000
    left_dists = np.column_stack((left_dists, np.zeros(len(left_dists))))

    # Potentially adapt bandwith dynamically to get the desired number of tracks

    mean_shift = sklearn.cluster.MeanShift(bandwidth=bandwidth)
    mean_shift.fit(
        np.column_stack(((df.left_dist.to_numpy() * 1000), np.zeros(len(df))))
    )
    labels = mean_shift.labels_

    logging.info(f"Found {max(labels) + 1} tracks on the provided roll scan")

    df["track"] = labels

    config = dict(
        {
            "media_type": "roll",
            "method": "roll",
            "roll_width_mm": roll_physical_width,
            "hole_width_mm": hole_width,
            "binarization_method": "v_channel",
            "binarization_options": {"threshold": threshold},
        }
    )

    tm = (
        df.groupby("track")
        .mean()
        .sort_values("left_dist")
        .reset_index()
        .apply(
            lambda row: dict(
                {
                    "left": (row.left_dist * roll_physical_width),
                    "right": (row.right_dist * roll_physical_width),
                    "tone": row.name,
                }
            ),
            axis=1,
        )
        .to_list()
    )

    config["track_measurements"] = tm

    logging.info("Beginning second pass over the roll to assign tracks")

    notes = []

    alignment_grid = hmsm.rolls.utils.get_initial_alignment_grid(
        config["roll_width_mm"], config["track_measurements"]
    )

    if _has_enlighten:
        manager = enlighten.get_manager()
        progress_bar = manager.counter(
            total=math.ceil(image.shape[0] / chunk_size),
            desc="Assigning tracks",
            unit="chunks",
        )

    for start in range(0, image.shape[0], chunk_size):
        end = (
            start + chunk_size
            if start + chunk_size < image.shape[0]
            else image.shape[0]
        )

        if not _has_enlighten:
            logging.info(f"Processing chunk from {start} to {end}")

        chunk = image[start:end, :, :]

        mask = hmsm.rolls.masking.methods.v_channel(chunk, bg_color, threshold)
        mask = mask["holes"]

        left_edge, right_edge = hmsm.rolls._get_roll_edges(np.invert(mask))

        labels = skimage.measure.label(mask, background=False, connectivity=2)
        components = list(hmsm.utils.to_coord_lists(labels).values())

        components = list(
            filter(
                functools.partial(
                    hmsm.rolls._filter_component,
                    width_bounds=width_bounds,
                    height_bounds=(width_bounds[0], 1000),
                ),
                components,
            )
        )

        for component in components:
            # Calculate relative position on the roll (from the left roll edge)
            height, width = tuple(component.max(axis=0) - component.min(axis=0))

            y_position = int(round(component[:, 0].min() + (height / 2)))
            x_position = int(round(component[:, 1].min() + (width / 2)))
            roll_width = right_edge[y_position] - left_edge[y_position]
            left_dist = (component[:, 1].min() - left_edge[y_position]) / roll_width
            right_dist = (component[:, 1].max() - left_edge[y_position]) / roll_width

            track_idx = np.abs(alignment_grid[:, 0] - left_dist).argmin()

            notes.append(
                np.array(
                    [
                        component[:, 0].min() + start,
                        component[:, 0].max() + start,
                        int(alignment_grid[track_idx, 2]),
                    ]
                )
            )

            chunk[component[:, 0], component[:, 1]] = [255, 0, 0]

            cv2.putText(
                chunk,
                str(int(alignment_grid[track_idx, 2])),
                (
                    int(np.mean(component, axis=0)[1]),
                    int(np.mean(component, axis=0)[0]),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        skimage.io.imsave(os.path.join(image_path, f"{start}_{end}.jpg"), chunk)

        if _has_enlighten:
            progress_bar.update()

    if _has_enlighten:
        manager.stop()

    logging.info(f"Writing configuration data to '{output_path}'")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    return
