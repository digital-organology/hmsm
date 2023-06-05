import cv2
import numpy as np
import logging
from skimage.measure import EllipseModel, label
from scipy.spatial import distance
from typing import Optional, Tuple
from hmsm.utils import read_image
import hmsm.discs.cluster


# For debugging only
import matplotlib.pyplot as plt

def process_disc(input_path: str, output_path: str, method: str, config: dict, verbose: Optional[bool] = False) -> None:
    logging.info(f"Reading input image from {input_path}")

    input = read_image(input_path)

    logging.info("Input image read successfully")

    if method == "cluster":
        hmsm.discs.cluster.process_disc(input, output_path, config, verbose)

