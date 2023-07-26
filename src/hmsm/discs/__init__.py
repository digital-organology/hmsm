# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import logging
from typing import Optional

import hmsm.discs.cluster
import hmsm.utils


def process_disc(
    input_path: str,
    output_path: str,
    method: str,
    config: dict,
    offset: Optional[int] = 0,
) -> None:
    """Perform image based midi generation on a disc shaped medium

    This is a wrapper method that will read the input image and then dispatch the appropriate processing method.

    Args:
        input_path (str): File path to the input image
        output_path (str): File path to the output midi file
        method (str): Method to use for digitization, currently only 'cluster' is implemented
        config (dict): Dictionary containing required configuration parameters
    """
    logging.info(f"Reading input image from {input_path}")

    input = hmsm.utils.read_image(input_path)

    logging.info("Input image read successfully")

    if method == "cluster":
        hmsm.discs.cluster.process_disc(
            input, output_path, config=config, offset=offset
        )
