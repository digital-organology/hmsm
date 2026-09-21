# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Command line entry points.

Everything here parses arguments, sets up logging and hands off to a pipeline
package. Processing logic belongs in those packages, not in this module.

The commands are reachable two ways: as subcommands of ``hmsm``, and under
their own names (``roll2midi``, ``disc2midi`` and so on) for the sake of
existing habits and scripts.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Callable, List, Optional, Sequence

from hmsm import __version__

logger = logging.getLogger("hmsm")

_LICENSE_NOTICE = (
    "This program is licensed to you under the terms of the GNU General "
    "Public License v3.0 or later and comes with ABSOLUTELY NO WARRANTY."
)

_DEBUG_DIR = "debug_data"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ``hmsm`` command.

    Args:
        argv: Argument vector including the program name. Defaults to ``sys.argv``.

    Returns:
        A process exit code.
    """
    argv = list(sys.argv if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv[1:])

    if not getattr(args, "handler", None):
        parser.print_help()
        return 1

    _setup_logging(args)

    try:
        args.handler(args)
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        logger.debug("Traceback follows", exc_info=True)
        return 1

    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hmsm",
        description="Image based digitization of historical musical storage media.",
        epilog=_LICENSE_NOTICE,
    )
    parser.add_argument("--version", action="version", version=f"hmsm {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    for name, build in _COMMANDS.items():
        build(subparsers.add_parser(name, help=_HELP[name], description=_HELP[name]))

    return parser


def _common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the options every command shares."""
    group = parser.add_argument_group("output")
    group.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help=(
            f"Log debug output and write diagnostic artefacts to ./{_DEBUG_DIR}. "
            "Also surfaces debug messages from third party libraries and adds "
            "significant processing overhead."
        ),
    )
    group.add_argument(
        "-q", "--quiet", action="store_true", help="Only report warnings and errors."
    )
    return parser


def _roll_source_args(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by the commands that read a roll scan.

    Note:
        ``--band-height`` deliberately has no short form. The previous CLI
        spelled it ``-s`` on ``roll2midi`` and used ``-s`` for the hole size on
        ``roll2config``; giving it a short form again would let an old command
        line keep working while quietly meaning something else.
    """
    parser.add_argument("input", help="Roll scan to read.")
    parser.add_argument(
        "--band-height",
        type=int,
        default=4000,
        metavar="ROWS",
        help=(
            "Number of scan rows to process at a time. Affects memory use, "
            "not the result. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-l",
        "--skip-rows",
        type=int,
        default=0,
        metavar="ROWS",
        help=(
            "Rows to skip from the top of the scan, to step past a roll head "
            "the automatic detection cannot handle. Default: %(default)s."
        ),
    )


def _roll2midi_parser(parser: argparse.ArgumentParser) -> None:
    _roll_source_args(parser)
    parser.add_argument("output", help="MIDI file to write.")
    required = parser.add_argument_group("required named arguments")
    required.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="PROFILE",
        help=(
            "Format profile: the name of a bundled preset, a path to a json "
            "file, or a json string. Run 'hmsm profiles' for the presets."
        ),
    )
    parser.add_argument(
        "-t",
        "--tempo",
        type=int,
        default=50,
        help=(
            "Roll tempo in feet per minute times ten, as annotated on most "
            "rolls. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-b",
        "--background",
        default="guess",
        choices=("guess", "black", "white"),
        help="Background colour of the scan. Default: %(default)s.",
    )
    _common(parser)
    parser.set_defaults(handler=_run_roll2midi)


def _roll2config_parser(parser: argparse.ArgumentParser) -> None:
    _roll_source_args(parser)
    parser.add_argument("output", help="Profile stub to write.")
    required = parser.add_argument_group("required named arguments")
    required.add_argument(
        "-w",
        "--width",
        type=float,
        required=True,
        metavar="MM",
        help="Physical width of the roll, in mm.",
    )
    parser.add_argument(
        "-s",
        "--hole-size",
        type=float,
        default=1.5,
        metavar="MM",
        help=(
            "Width of the holes on the roll, in mm. Only rolls with uniform "
            "holes are supported here. Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "How far a pixel has to be from the paper towards the scanner "
            "background to count as a hole, between 0 and 1. The default puts "
            "the boundary half way, which is where the edge of a hole is. "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "-b",
        "--bandwidth",
        type=float,
        default=2.0,
        help=(
            "Clustering bandwidth, in thousandths of the roll width. Raise it "
            "to group more positions into one track, lower it to separate "
            "more. You will rarely need to touch this. Default: %(default)s."
        ),
    )
    _common(parser)
    parser.set_defaults(handler=_run_roll2config)


def _disc2midi_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Disc scan to read.")
    parser.add_argument("output", help="MIDI file to write.")
    required = parser.add_argument_group("required named arguments")
    required.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="PROFILE",
        help=(
            "Format profile: the name of a bundled preset, a path to a json "
            "file, or a json string."
        ),
    )
    required.add_argument(
        "-m",
        "--method",
        required=True,
        choices=("cluster",),
        help="Digitization method to use.",
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=int,
        default=0,
        metavar="DEGREES",
        help="Rotational offset of the disc's starting position. Default: %(default)s.",
    )
    _common(parser)
    parser.set_defaults(handler=_run_disc2midi)


def _disc2roll_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="Disc scan to read.")
    parser.add_argument("output", help="Unrolled image to write.")
    parser.add_argument(
        "-o",
        "--offset",
        type=int,
        default=0,
        metavar="DEGREES",
        help="Rotational offset of the disc's starting position. Default: %(default)s.",
    )
    parser.add_argument(
        "-b", "--binarize", action="store_true", help="Binarize the output image."
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=None,
        metavar="VALUE",
        help=(
            "Threshold to binarize with, in [0, 255]. Estimated with Otsu's "
            "method if omitted."
        ),
    )
    _common(parser)
    parser.set_defaults(handler=_run_disc2roll)


def _midi2disc_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="MIDI file to render.")
    parser.add_argument("output", help="Disc image to write.")
    parser.add_argument(
        "-t",
        "--type",
        default="ariston_24",
        help="Disc format to render. Default: %(default)s.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=4000,
        metavar="PIXELS",
        help="Diameter of the disc to create. Default: %(default)s.",
    )
    parser.add_argument(
        "-l", "--logo-file", default=None, help="Logo to place on the disc."
    )
    parser.add_argument("-n", "--name", default=None, help="Title to put on the disc.")
    _common(parser)
    parser.set_defaults(handler=_run_midi2disc)


def _profiles_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-m",
        "--media",
        default=None,
        choices=("roll", "cluster"),
        help="Only list profiles for this kind of medium.",
    )
    _common(parser)
    parser.set_defaults(handler=_run_profiles)


_COMMANDS: dict[str, Callable[[argparse.ArgumentParser], None]] = {
    "roll2midi": _roll2midi_parser,
    "roll2config": _roll2config_parser,
    "disc2midi": _disc2midi_parser,
    "disc2roll": _disc2roll_parser,
    "midi2disc": _midi2disc_parser,
    "profiles": _profiles_parser,
}

_HELP = {
    "roll2midi": "Digitize a piano roll scan into a MIDI file.",
    "roll2config": "Detect the tracks on a roll scan and write a profile stub.",
    "disc2midi": "Digitize a disc scan into a MIDI file.",
    "disc2roll": "Unroll a disc scan into a roll shaped image.",
    "midi2disc": "Render a MIDI file as a disc image.",
    "profiles": "List the bundled format profiles.",
}


# --------------------------------------------------------------------------
# Command implementations
# --------------------------------------------------------------------------


def _run_roll2midi(args: argparse.Namespace) -> None:
    import hmsm.rolls
    from hmsm.profiles import load_profile

    print(_LICENSE_NOTICE, end="\n\n")

    transcription = hmsm.rolls.roll_to_midi(
        input_path=args.input,
        output_path=args.output,
        profile=load_profile(args.config, "roll"),
        background=args.background,
        band_height=args.band_height,
        skip_rows=args.skip_rows,
        tempo=args.tempo,
        debug_dir=_DEBUG_DIR if args.debug else None,
    )
    logger.info("Wrote %d notes to %s", len(transcription.notes), args.output)


def _run_roll2config(args: argparse.Namespace) -> None:
    from hmsm.rolls.analysis import analyze_roll

    profile = analyze_roll(
        image_path=args.input,
        output_path=args.output,
        roll_width_mm=args.width,
        skip_rows=args.skip_rows,
        hole_width_mm=args.hole_size,
        threshold=args.threshold,
        bandwidth=args.bandwidth,
        band_height=args.band_height,
    )
    logger.info(
        "Wrote a stub with %d tracks to %s. Assign MIDI notes to the tracks "
        "before using it; see docs/CONFIG.md.",
        len(profile.tracks),
        args.output,
    )


def _run_disc2midi(args: argparse.Namespace) -> None:
    import hmsm.discs
    from hmsm.profiles import load_profile

    if args.debug:
        pathlib.Path(_DEBUG_DIR).mkdir(exist_ok=True)

    hmsm.discs.process_disc(
        args.input,
        args.output,
        args.method,
        load_profile(args.config, args.method),
        args.offset,
    )


def _run_disc2roll(args: argparse.Namespace) -> None:
    import skimage.io

    import hmsm.discs.utils
    from hmsm.io import open_source

    with open_source(args.input) as source:
        image = source.read_rows(0, source.height)

    logger.info("Unrolling disc")
    output = hmsm.discs.utils.transform_to_rectangle(
        image, args.offset, args.binarize, args.threshold
    )
    skimage.io.imsave(args.output, output)
    logger.info("Wrote %s", args.output)


def _run_midi2disc(args: argparse.Namespace) -> None:
    import hmsm.discs.generator

    hmsm.discs.generator.generate_disc(
        args.input, args.output, args.size, args.type, args.name, args.logo_file
    )


def _run_profiles(args: argparse.Namespace) -> None:
    from hmsm.profiles import available_presets

    presets = available_presets(args.media)
    if not presets:
        print("No bundled profiles found.")
        return

    width = max(len(name) for name in presets)
    print(f"{'PROFILE'.ljust(width)}  MEDIUM")
    for name, media in sorted(presets.items()):
        print(f"{name.ljust(width)}  {media}")


# --------------------------------------------------------------------------
# Logging and legacy entry points
# --------------------------------------------------------------------------


def _setup_logging(args: argparse.Namespace) -> None:
    if getattr(args, "debug", False):
        level = logging.DEBUG
    elif getattr(args, "quiet", False):
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s]: %(message)s")


def _legacy(command: str) -> Callable[..., int]:
    """Build an entry point that dispatches straight to one subcommand."""

    def entry(argv: Optional[Sequence[str]] = None) -> int:
        argv = list(sys.argv if argv is None else argv)
        return main([argv[0], command, *argv[1:]])

    entry.__name__ = command
    entry.__doc__ = f"Entry point for '{command}'. Equivalent to 'hmsm {command}'."
    return entry


roll2midi = _legacy("roll2midi")
roll2config = _legacy("roll2config")
disc2midi = _legacy("disc2midi")
disc2roll = _legacy("disc2roll")
midi2disc = _legacy("midi2disc")


if __name__ == "__main__":
    raise SystemExit(main())
