# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import logging
import sys
import argparse
import traceback
import hmsm.discs.utils
import hmsm.discs
import hmsm.config
import skimage.io
import pathlib

def disc2roll(argv = sys.argv):
    """CLI entrypoint for disc to roll transformation

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
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
        image = skimage.io.imread(args.input)
    except FileNotFoundError:
        logging.error(f"The system could not find the specified file at path '{args.input}', could not read image file")
        sys.exit()
    logging.info("Beginning transformation process")
    output = hmsm.discs.utils.transform_to_rectangle(image, args.offset, args.binarize)
    logging.info("Transformation processed")
    skimage.io.imsave(args.output, output)
    logging.info(f"Output written to {args.output}")

def disc2midi(argv = sys.argv):
    """CLI entrypoint for disc to midi transformation

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help = "Input image file")
    parser.add_argument("output", help = "Filename to write output midi to")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("-c", "--config", dest = "config", required = True, help = "Configuration to use for digitization. Must be either the name of a provided profile, path to a json file containing the required information or a json string with configuration data.")
    required_named.add_argument("-m", "--method", dest = "method", required=True, help = "Method to use for digitization. Currently only 'cluster' is supported.")
    parser.add_argument("-o", "--offset", dest = "offset", default = 0, const = 0, nargs= "?", type = int, help = "Offset of the disc starting position, in degrees. Must be in range [0,360] if provided")
    parser.add_argument("-d", "--debug", action = "store_true", help = "Enable debug output. Note that this will also output various messages from other used python packages and add siginificant calculation overhead for creating debug information.")
    parser.set_defaults(debug = False, offset = 0)                                                   
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level = logging.DEBUG if args.debug else logging.INFO,
        format = "%(asctime)s [%(levelname)s]: %(message)s"
    )

    if args.debug:
        pathlib.Path("debug_data").mkdir(exist_ok = True)

    try:
        config = hmsm.config.get_config(args.config, args.method)
    except Exception:
        logging.error("Failed to read configuration, the following exception occured:")
        traceback.print_exc()
        sys.exit(1)

    hmsm.discs.process_disc(args.input, args.output, args.method, config, args.offset)