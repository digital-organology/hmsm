import numpy as np
import hmsm.utils
import hmsm.discs.utils
import logging
import cv2
from typing import Optional

def process_disc(input_image: np.ndarray, output_path: str, config: dict, verbose: Optional[bool] = False):
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

    print("Hi from the clustering digitization method")
    pass


