# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import skimage.filters
import random
import logging
import numpy as np
import skimage.io
import skimage
import scipy.spatial.distance
import hmsm.rolls.masking
from typing import Optional, Tuple, List

def create_chunk_masks(image_path: str, chunk_size: Optional[int] = 4000, n_clusters: Optional[int] = 2, n_chunks: Optional[int] = 5) -> None:
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

    logging.info(f"Estimating required parameters using {n_chunks} slices of {chunk_size} pixels height each")

    generator.estimate_mask_parameters()

    logging.info(f"Parameters estimated, beginning mask creation")

    for bounds, masks in iter(generator):
        (start_idx, end_idx) = bounds
        logging.info(f"Writing images for chunk from {start_idx} to {end_idx}")
        for mask_id in range(len(masks)):
            mask_ubyte = skimage.img_as_ubyte(masks[mask_id])
            filename = os.path.join("masks", f"mask_{start_idx}_{end_idx}_{mask_id}.tif")
            skimage.io.imsave(filename, mask_ubyte)
