# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import argparse
import datetime
import logging
import pathlib
import sys
import traceback

import skimage.io

import hmsm.config
import hmsm.discs

# import hmsm.rolls.utils
import hmsm.discs.generator
import hmsm.discs.utils
import hmsm.rolls.analysis
import hmsm.rolls.utils


def disc2roll(argv=sys.argv):
    """CLI entrypoint for disc to roll transformation

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Filename to write output image to")
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=None,
        const=None,
        nargs="?",
        help="Threshold value to use for binarization, must be in range [0,255] if provided",
    )
    parser.add_argument(
        "-b",
        "--binarize",
        dest="binarize",
        action="store_true",
        help="Binarize the image",
    )
    parser.add_argument(
        "-o",
        "--offset",
        dest="offset",
        default=0,
        const=0,
        nargs="?",
        type=int,
        help="Offset of the disc starting position, in degrees. Must be in range [0,360] if provided",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output. Note that this will also output various messages from other used python packages.",
    )
    parser.set_defaults(debug=False, offset=0, threshold=None, binarize=False)
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    try:
        image = skimage.io.imread(args.input)
    except FileNotFoundError:
        logging.error(
            f"The system could not find the specified file at path '{args.input}', could not read image file"
        )
        sys.exit()
    logging.info("Beginning transformation process")
    output = hmsm.discs.utils.transform_to_rectangle(image, args.offset, args.binarize)
    logging.info("Transformation processed")
    skimage.io.imsave(args.output, output)
    logging.info(f"Output written to {args.output}")


def disc2midi(argv=sys.argv):
    """CLI entrypoint for disc to midi transformation

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Filename to write output midi to")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "-c",
        "--config",
        dest="config",
        required=True,
        help="Configuration to use for digitization. Must be either the name of a provided profile, path to a json file containing the required information or a json string with configuration data.",
    )
    required_named.add_argument(
        "-m",
        "--method",
        dest="method",
        required=True,
        help="Method to use for digitization. Currently only 'cluster' is supported.",
    )
    parser.add_argument(
        "-o",
        "--offset",
        dest="offset",
        default=0,
        const=0,
        nargs="?",
        type=int,
        help="Offset of the disc starting position, in degrees. Must be in range [0,360] if provided",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output. Note that this will also output various messages from other used python packages and add siginificant calculation overhead for creating debug information.",
    )
    parser.set_defaults(debug=False, offset=0)
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    if args.debug:
        pathlib.Path("debug_data").mkdir(exist_ok=True)

    try:
        config = hmsm.config.get_config(args.config, args.method)
    except Exception:
        logging.error("Failed to read configuration, the following exception occured:")
        traceback.print_exc()
        sys.exit(1)

    hmsm.discs.process_disc(args.input, args.output, args.method, config, args.offset)


def roll2masks(argv=sys.argv):
    """CLI entrypoint for mask creation on piano rolls

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image file")
    parser.add_argument(
        "-s",
        "--chunk_size",
        dest="chunk_size",
        default=4000,
        const=4000,
        nargs="?",
        type=int,
        help="Size of the image chunks to use for processing",
    )
    parser.add_argument(
        "-n",
        "--n_clusters",
        dest="n_clusters",
        default=2,
        const=2,
        nargs="?",
        type=int,
        help="Number of clusters to consider when creating masks. Note that there will always be n+1 clusters as one cluster is implicitly created during binarization.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output. Note that this will also output various messages from other used python packages and add siginificant calculation overhead for creating debug information.",
    )
    parser.set_defaults(debug=False, n_clusters=2, chunk_size=4000)
    args = parser.parse_args(argv[1:])

    pathlib.Path("masks").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    hmsm.rolls.utils.create_chunk_masks(args.input, args.chunk_size, args.n_clusters)


def midi2disc(argv=sys.argv):
    """CLI entrypoint for disc image creation from a midi file

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input midi file")
    parser.add_argument("output", help="Output image file")
    parser.add_argument(
        "-t",
        "--type",
        dest="type",
        default="ariston_24",
        const="ariston_24",
        nargs="?",
        type=str,
        help="Type of disc to create",
    )
    parser.add_argument(
        "-s",
        "--size",
        dest="size",
        default=4000,
        const=4000,
        nargs="?",
        type=int,
        help="Diameter of the disc to create (in pixels)",
    )
    parser.add_argument(
        "-l",
        "--logo-file",
        dest="logo_file",
        default=None,
        const=None,
        nargs="?",
        type=str,
        help="Logo file to apply to the disc",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name",
        default=None,
        const=None,
        nargs="?",
        type=str,
        help="Name to use for the disc title.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug output."
    )
    parser.set_defaults(
        type="ariston_24", size=4000, logo_file=None, name=None, debug=False
    )
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    hmsm.discs.generator.generate_disc(
        args.input, args.output, args.size, args.type, args.name, args.logo_file
    )


def roll2midi(argv=sys.argv):
    """CLI entrypoint for piano roll to midi transformation

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Filename to write output midi to")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "-c",
        "--config",
        dest="config",
        required=True,
        help="Configuration to use for digitization. Must be either the name of a provided profile, path to a json file containing the required information or a json string with configuration data.",
    )
    parser.add_argument(
        "-s",
        "--chunk_size",
        dest="chunk_size",
        default=4000,
        const=4000,
        nargs="?",
        type=int,
        help="Size of the image chunks to use for processing",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output. Note that this will also output various messages from other utilized third party python packages that use PEP 282 logging (notably scikit image) and add siginificant calculation overhead for creating debug information.",
    )
    parser.add_argument(
        "-l",
        "--skip-lines",
        dest="skip_lines",
        default=0,
        const=0,
        nargs="?",
        type=int,
        help="Optional number of lines to skip before starting to process the file.",
    )
    parser.add_argument(
        "-t",
        "--tempo",
        dest="tempo",
        default=50,
        const=50,
        nargs="?",
        type=int,
        help="Tempo of the roll. Unit is feet-per-minute (fpm) * 10, as is annotated on (most) rolls.",
    )
    parser.add_argument(
        "-b",
        "--background",
        dest="bg_color",
        default="guess",
        const="guess",
        nargs="?",
        type=str,
        help="Color of the background in the provided roll scan. Currently only black or white backgrounds are supported. If ommited this will be estimated from a sample of pixels.",
    )
    parser.set_defaults(debug=False, bg_color="guess")
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    print(
        "This program is licensed to you under the terms of the GNU General Public License v3.0 or later and comes with ABSOLUTELY NO WARRANTY.\n\n"
    )

    if args.debug:
        debug_path = f"debug_data"
        pathlib.Path(debug_path).mkdir(exist_ok=True)
        pathlib.Path(f"{debug_path}/masks").mkdir(exist_ok=True)
    try:
        config = hmsm.config.get_config(args.config, "roll")
    except Exception:
        logging.error("Failed to read configuration, the following exception occured:")
        traceback.print_exc()
        sys.exit(1)

    hmsm.rolls.process_roll(
        args.input,
        args.output,
        config,
        args.bg_color,
        args.chunk_size,
        args.skip_lines,
        args.tempo,
    )


def roll2config(argv=sys.argv):
    """CLI entrypoint for initial analysis and estimation of configuration parameters

    Args:
        argv (list, optional): Command line arguments. Defaults to sys.argv.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input image file")
    parser.add_argument("output", help="Output json file")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "-w",
        "--width",
        dest="width",
        default=-1,
        const=-1,
        nargs="?",
        type=float,
        help="Width of the physical roll in mm",
    )
    parser.add_argument(
        "-l",
        "--skip_lines",
        dest="line_skip",
        default=0,
        const=0,
        nargs="?",
        type=float,
        help="Number of lines to skip from the beginning of the roll",
    )
    parser.add_argument(
        "-s",
        "--hole_size",
        dest="hole_width",
        default=1.5,
        const=1.5,
        nargs="?",
        type=float,
        help="Width of the holes on the roll. Currently only roles with uniform holes are supported.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=0.15,
        const=0.15,
        nargs="?",
        type=float,
        help="Threshold to use when binarizing, must be between 0 and 1",
    )
    parser.add_argument(
        "-b",
        "--bandwidth",
        dest="bandwidth",
        default=2,
        const=2,
        nargs="?",
        type=float,
        help="Bandwith to use for the underlying alogrithm when finding tracks. Higher values will result in more tracks beeing grouped together while lower values will separate more tracks. You will likely not have to touch this.",
    )
    parser.add_argument(
        "-c",
        "--chunk_size",
        dest="chunk_size",
        default=4000,
        const=4000,
        nargs="?",
        type=float,
        help="(Vertical) size of chunks to use for processing the piano roll.",
    )
    args = parser.parse_args(argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
    )

    hmsm.rolls.analysis.analyze_roll(
        args.input,
        args.output,
        args.width,
        args.line_skip,
        args.hole_width,
        args.threshold,
        args.bandwidth,
        args.chunk_size,
    )
