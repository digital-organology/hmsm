import logging
import os
import sys
import argparse
import cv2
from hmsm import utils, cardboard_discs
from skimage.io import imread, imsave

def disc2roll(argv = sys.argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help = "Input image file")
    parser.add_argument("output", help = "Filename to write output image to")
    parser.add_argument("-t", "--threshold", dest = "threshold", default = None, const = None, nargs= "?", help = "Threshold value to use for binarization, must be in range [0,255] if provided")
    parser.add_argument("-b", "--binarize", dest = "binarize", action = "store_true", help = "Binarize the image")
    parser.add_argument("-o", "--offset", dest = "offset", default = 0, const = 0, nargs= "?", type = int, help = "Offset of the disc starting position, in degrees. Must be in range [0,360] if provided")
    parser.add_argument("-d", "--debug", action = "store_true", help = "Enable debug output. Note that this will also output various messages from other used python packages.")
    parser.set_defaults(debug = False, offset = 0, threshold = None, binarize = False)                                                   
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level = logging.DEBUG if args.debug else logging.INFO,
        format = "%(asctime)s [%(levelname)s]: %(message)s"
    )

    try:
        image = imread(args.input)
    except FileNotFoundError:
        logging.error(f"The system could not find the specified file at path '{args.input}', could not read image file")
        sys.exit()
    logging.info("Beginning transformation process")
    output = cardboard_discs.transform_to_rectangle(image, args.offset, args.binarize)
    logging.info("Transformation processed")
    imsave(args.output, output)
    logging.info(f"Output written to {args.output}")
