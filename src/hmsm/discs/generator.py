# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import logging
import math
import os

import cv2
import mido
import numpy as np

try:
    import vnoise
except ImportError:
    _has_vnoise = False
else:
    _has_vnoise = True

try:
    import cairosvg
except ImportError:
    _has_cairo = False
else:
    _has_cairo = True

from typing import List, Optional, Tuple


def generate_disc(
    midi_path: str,
    image_path: str,
    size: int,
    type: str,
    name: Optional[str],
    logo_file: Optional[str],
) -> None:
    """Generate an image from a midi file

    Method to convert a midi file into an image of a corresponding cardboard disc

    Args:
        midi_path (str): Path to the midi file to be converted into an image
        image_path (str): Path to write the generated image to
        size (int): Diameter (in pixels) of the disc to generate. The generated image will have some minor padding added to this value.
        type (str): Type of disc to generate. Currently only ariston_24 is supported.
        name (Optional[str]): Title of the disc that will be written on the disc
        logo_file (Optional[str]): Path to an svg file to draw on the disc. This required cairosvg to be installed

    Raises:
        ValueError: Will be raised if an unsupported type of disc is supplied
    """
    if type == "ariston_24":
        disc_image, center = _generate_ariston_24_base_disc(size, name, logo_file)
    else:
        raise ValueError("Disc type {type} is not (currently) supported by this method")

    logging.info("Getting configuration data")

    config = _get_config(type)

    logging.info("Reading and processing midi file")

    note_data = _midi_to_absolute_timings(
        midi_path, list(config["track_mapping"].keys())
    )

    logging.info("Drawing holes on disc")

    disc_image = _draw_notes(disc_image, note_data, config, size / 2, center)

    logging.info(f"Writing disc image to {image_path}")

    canvas_rgb = cv2.cvtColor(disc_image, cv2.COLOR_BGRA2RGBA)
    cv2.imwrite(image_path, canvas_rgb)

    logging.info("Done")


def _generate_ariston_24_base_disc(
    size: int, title: Optional[str], logo_file: Optional[str]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Generate base image of an ariston brand cardboard disc with 24 tracks

    This will also draw some decorations and (if provided) the title of the disc as well as a logo file on the disc.

    Args:
        size (int): Diameter (in pixels) of the disc to generate
        title (Optional[str]): Title of the disc to draw onto the base image
        logo_file (Optional[str]): Path to an svg file to draw onto the disc

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: Base image of the disc and the coordinates of the disc center point
    """

    # Create base canvas with disc

    logging.info("Creating canvas and drawing disc background")

    canvas = np.zeros((size + 100, size + 100, 3), np.uint8)

    center_x = center_y = int((size / 2) + 50)

    radius = int(size / 2)

    canvas = cv2.circle(canvas, (center_x, center_y), radius, (128, 88, 53), -1)

    # Get a mask of the disc
    disc_mask = cv2.inRange(canvas, np.array([1, 1, 1]), np.array([255, 255, 255]))

    logging.info("Applying perlin noise to the disc background")
    # Apply noise to make the disc look more natural
    canvas = _apply_noise(canvas, size)

    printing_color = (40, 40, 40)

    logging.info("Drawing decorations on disc")
    # Draw full decorative lines

    lines_to_draw = list(
        [(0.96641, 0.00528), (0.95584, 0.00226), (0.38415, 0.00603), (0.37471, 0.00679)]
    )

    for relative_center_radius, relative_width in lines_to_draw:
        canvas = cv2.circle(
            canvas,
            (center_x, center_y),
            round(relative_center_radius * radius),
            printing_color,
            math.ceil(relative_width * radius),
            cv2.LINE_4,
        )

    # Draw repreating partial lines

    partial_lines_to_draw = list(
        [(0.36905, 0.00452, 4.5, 2.13), (0.36288, 0.00981, 4.5, 3.39)]
    )

    for (
        relative_center_radius,
        relative_width,
        repeat_every,
        draw_angle,
    ) in partial_lines_to_draw:
        radius_min = round((relative_center_radius - (relative_width / 2)) * radius)
        radius_max = round((relative_center_radius + (relative_width / 2)) * radius)
        for start_angle in np.arange(0, 360, repeat_every):
            rr, cc = _get_partial_circle_coords(
                radius_min,
                radius_max,
                (center_x, center_y),
                start_angle,
                draw_angle,
                canvas.shape[0:2],
            )
            canvas[rr, cc] = printing_color

    # Add beginning line

    beginning_min_radius = 0.40830 * radius
    beginning_max_radius = 0.92849 * radius
    text = "_The beginning. Le commencement. Der Anfang._"
    font_scale = _fit_text_in_width(text, beginning_max_radius - beginning_min_radius)

    canvas = cv2.putText(
        canvas,
        text,
        (round(radius - beginning_max_radius + 50), (radius + 50)),
        cv2.FONT_HERSHEY_TRIPLEX,
        font_scale,
        printing_color,
        1,
    )

    if not title is None:
        logging.info("Drawing disc title")
        title_radius = round(0.311 * radius)
        canvas = _put_title(
            canvas, title, printing_color, title_radius, (center_x, center_y)
        )

    if not logo_file is None:
        logging.info("Drawing logo file")
        canvas = _put_logos(
            canvas, logo_file, title_radius, (center_x, center_y), printing_color
        )

    logging.info("Adding alpha channel to the image")

    # Add alpha channel to image

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)

    # Make hole in the middle

    canvas = cv2.circle(canvas, (center_y, center_x), 0, (0, 0, 0, 0), 20)
    canvas[disc_mask == 0] = (0, 0, 0, 0)

    return (canvas, (center_x, center_y))


def _draw_notes(
    canvas: np.ndarray,
    note_data: np.ndarray,
    config: dict,
    radius: int,
    center: Tuple[int, int],
) -> np.ndarray:
    """Draw holes for the given notes on the given disc image

    Args:
        canvas (np.ndarray): Base image to draw onto
        note_data (np.ndarray): Data on the notes to draw. Must be a three column numpy array with midi tone, start_time and end_time as columns. Timings must be absolute, in clockwise degrees and be scaled to [0,360].
        config (dict): Configuration data containing information on midi note to track mappings and the width of the holes and padding between them
        radius (int): Radius of the disc that will be drawn on in pixels.
        center (Tuple[int, int]): Center point of the disc that will be drawn on

    Returns:
        np.ndarray: Input image with holes drawn onto
    """
    # Calculate the width of each note and each padding between them

    printable_area = (config["note_end"] - config["note_start"]) * radius

    n_notes = len(config["track_mapping"])

    n_padding = n_notes - 2

    total_padding_width = n_padding * config["padding_to_note_ratio"]

    note_width = printable_area / (n_notes + total_padding_width)

    padding_width = config["padding_to_note_ratio"] * note_width

    # Shift the start of the disc to 180 degrees

    note_data[:, 1:3] = note_data[:, 1:3] + 180
    note_data[:, 1:3][note_data[:, 1:3] > 360] = (
        note_data[:, 1:3][note_data[:, 1:3] > 360] - 360
    )

    # Convert start_angle to width
    note_data[:, 1] = note_data[:, 2] - note_data[:, 1]

    # Get the start angle which is the inverse of the end angle as the discs are played counter clockwise
    note_data[:, 2] = 360 - note_data[:, 2]

    # Change midi tones to track index
    note_data[:, 0] = np.array(
        [config["track_mapping"][val] for val in note_data[:, 0]]
    )

    if np.any(note_data[note_data[:, 1] < 0]):
        broken_notes = note_data[note_data[:, 1] < 0]
        broken_notes[:, 1] = broken_notes[:, 1] + 360
        note_data[note_data[:, 1] < 0] = broken_notes

    # Draw these annoying things
    for note in note_data:
        min_radius = round(
            (config["note_start"] * radius)
            + ((note[0] - 1) * note_width)
            + ((note[0] - 1) * padding_width)
        )
        max_radius = round(min_radius + note_width)
        rr, cc = _get_partial_circle_coords(
            min_radius, max_radius, center, note[2], note[1], canvas.shape[:2]
        )
        canvas[rr, cc] = (0, 0, 0, 0)

    return canvas


def _midi_to_absolute_timings(
    midi_file: str, allowed_notes: Optional[List[int]]
) -> np.ndarray:
    """Reads a midi file and converts the timings to absolute units

    Will also handle conversion and scaling of the timing data.

    Args:
        midi_file (str): Path to a midi file

    Raises:
        ValueError: Will be raised if the midi file cannot be found at the specified location

    Returns:
        np.ndarray: Array containing midi tones, start- and endangles (in degrees, scaled to [0,360])
    """
    if not os.path.isfile(midi_file):
        raise ValueError("Midi file does not exist")

    midi_file = mido.MidiFile(midi_file)

    message_types = list()
    message_notes = list()
    message_times = list()
    skipped_notes = 0

    for msg in midi_file.tracks[0]:
        if isinstance(msg, mido.midifiles.meta.MetaMessage):
            continue

        if not (msg.type == "note_on" or msg.type == "note_off"):
            continue

        if not msg.note in allowed_notes:
            skipped_notes += 1
            continue

        message_types.append(msg.type)
        message_notes.append(msg.note)
        message_times.append(msg.time)

    if not skipped_notes == 0:
        logging.warning(
            f"Skipped {skipped_notes} notes because they contained tones not contained in the format"
        )

    message_types = np.array(message_types)
    message_notes = np.array(message_notes)
    message_times = np.array(message_times)
    # Convert to absolute times
    message_times = np.cumsum(message_times).astype(np.float64)
    # Rescale message times to degrees with 2.5 degrees buffer on each side
    message_times = (
        355.0 * (message_times - message_times.min()) / np.ptp(message_times) + 2.5
    )

    hole_data = list()

    for note in np.unique(message_notes):
        start_times = message_times[message_notes == note][0::2]
        end_times = message_times[message_notes == note][1::2]
        hole_data.append(
            np.column_stack((np.full(start_times.shape, note), start_times, end_times))
        )

    hole_data = np.vstack(hole_data)

    return hole_data


def _get_config(type: str) -> dict:
    """Get configuration data required to draw holes

    Args:
        type (str): Type of the disc for which the configuration data will be returned

    Returns:
        dict: Configuration data for the given disc type
    """
    if type == "ariston_24":
        return dict(
            {
                "track_mapping": {
                    45: 1,
                    47: 2,
                    50: 3,
                    52: 4,
                    57: 5,
                    59: 6,
                    61: 7,
                    62: 8,
                    64: 9,
                    66: 10,
                    68: 11,
                    69: 12,
                    71: 13,
                    73: 14,
                    74: 15,
                    75: 16,
                    76: 17,
                    78: 18,
                    79: 19,
                    80: 20,
                    81: 21,
                    83: 22,
                    85: 23,
                    86: 24,
                },
                "note_start": 0.40,
                "note_end": 0.96377,
                "padding_to_note_ratio": 0.14285,
            }
        )


def _put_logos(
    canvas: np.ndarray,
    logos_file: str,
    radius: int,
    center: Tuple[int, int],
    color: Tuple[int, int, int],
    offset: Optional[int] = 100,
) -> np.ndarray:
    """Draw logo file

    Args:
        canvas (np.ndarray): Image to draw the logo file on
        logos_file (str): Path to an svg file containing the given logos
        radius (int): Radius of the disc on the given image
        center (Tuple[int, int]): Center point of the disc
        color (Tuple[int,int,int]): Color to use when drawing the bounding rectangle around the logos
        offset (Optional[int], optional): Offset from the center point when drawing the logos. Defaults to 100.

    Raises:
        ImportError: Will be raised if cairosvg is not available
        ValueError: Will be raised if the svg file cannot be read from the given location

    Returns:
        np.ndarray: Input image with logos drawn on
    """
    if not _has_cairo:
        raise ImportError("Placing logos required the 'cairosvg' package")

    if not os.path.isfile(logos_file):
        raise ValueError("Logo file does not exist")

    bottom_right_x = round(radius * math.sin(math.radians(45)) + center[0])
    bottom_right_y = round(radius * math.cos(math.radians(45)) + center[1])

    top_left_x = center[0] + offset
    top_left_y = round(center[1] + (radius * math.cos(math.radians(135))))

    width = bottom_right_y - top_left_y
    height = bottom_right_x - top_left_x

    png_buffer = cairosvg.svg2png(
        url=logos_file, output_width=width, output_height=height
    )
    byte_array = np.frombuffer(png_buffer, np.uint8)
    logo_mat = cv2.imdecode(byte_array, cv2.IMREAD_UNCHANGED)
    r_channel = logo_mat[:, :, 2].copy()
    logo_mat[:, :, 2] = logo_mat[:, :, 0]
    logo_mat[:, :, 0] = r_channel

    canvas = cv2.rectangle(
        canvas, (top_left_y, top_left_x), (bottom_right_y, bottom_right_x), color, 2
    )

    alpha = logo_mat[:, :, 3].astype(float) / 255
    alpha = cv2.merge((alpha, alpha, alpha))

    fg = logo_mat[:, :, 0:3].astype(float)

    fg = cv2.multiply(alpha, fg)

    bg = canvas[top_left_x:bottom_right_x, top_left_y:bottom_right_y, :].astype(float)
    bg = cv2.multiply(1 - alpha, bg)

    blended = cv2.add(fg, bg).astype(np.uint8)

    canvas[top_left_x:bottom_right_x, top_left_y:bottom_right_y, :] = blended

    return canvas


def _put_title(
    canvas: np.ndarray,
    title: str,
    color: Tuple[int, int, int],
    radius: int,
    center: Tuple[int, int],
    offset: Optional[int] = 100,
) -> np.ndarray:
    """Draw title on disc

    Args:
        canvas (np.ndarray): Disc image to draw the title on
        title (str): Title to draw on the disc. Must currently be a single line of text.
        color (Tuple[int, int, int]): Color to draw the title in
        radius (int): Radius of the disc in the input image
        center (Tuple[int, int]): Center point of the disc in the input image
        offset (Optional[int], optional): Offset from the center point when drawing the title. Defaults to 100.

    Returns:
        np.ndarray: Input image with title text drawn onto
    """
    top_left_x = round(radius * math.sin(math.radians(225)) + center[0])
    top_left_y = round(radius * math.cos(math.radians(225)) + center[1])

    bottom_right_x = center[0] - offset
    bottom_right_y = round(radius * math.cos(math.radians(315)) + center[1])

    width = bottom_right_y - top_left_y
    height = bottom_right_x - top_left_x

    if "<br>" in title:
        title = title.split("<br>")

    font_scale = _fit_text_in_rectangle(title, width, height)

    # Get text dimensions to center

    if isinstance(title, list):
        heights = list()
        widths = list()

        for title_line in title:
            text_width, text_height = cv2.getTextSize(
                title_line, cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=1
            )[0]
            heights.append(text_height)
            widths.append(text_width)

        vertical_padding = (height - sum(heights)) / (len(title) + 1)
        vertical_offset = top_left_x + vertical_padding

        canvas = cv2.rectangle(
            canvas, (top_left_y, top_left_x), (bottom_right_y, bottom_right_x), color, 2
        )

        for title_line in title:
            text_width, text_height = cv2.getTextSize(
                title_line, cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=1
            )[0]
            line_x = round(vertical_offset + text_height)
            vertical_offset = line_x + vertical_padding
            line_y = round(((width - text_width) / 2) + top_left_y)
            canvas = cv2.putText(
                canvas,
                title_line,
                (line_y, line_x),
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                color,
                3,
            )
    else:
        text_width, text_height = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=1
        )[0]

        title_x = round(bottom_right_x - ((height - text_height) / 2))
        title_y = round(((width - text_width) / 2) + top_left_y)

        canvas = cv2.rectangle(
            canvas, (top_left_y, top_left_x), (bottom_right_y, bottom_right_x), color, 2
        )

        canvas = cv2.putText(
            canvas,
            title,
            (title_y, title_x),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale,
            color,
            3,
        )

    return canvas


def _get_partial_circle_coords(
    min_radius: int,
    max_radius: int,
    center: Tuple[int, int],
    start_angle: float,
    width: float,
    canvas_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get coordinates of a partial circle within the specified radii

    Args:
        min_radius (int): Radius to start drawing the circle
        max_radius (int): Radius to end drawing the circle
        center (Tuple[int, int]): Center point to draw the circle around
        start_angle (float): Angle to begin the circle
        width (float): Angle for which to draw the circle to
        canvas_shape (Tuple[int, int]): Size of the canvas to draw onto

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y coordinates of all points on the filles partial circle
    """
    # This is definitly a stupid and bad approach that is also quite expensive but at this point i spent a good 2 hours on this so i don't really care anymore

    background = np.zeros(canvas_shape, dtype=np.uint8)

    for radius in np.arange(min_radius, max_radius):
        background = cv2.ellipse(
            background,
            center,
            (radius, radius),
            0,
            start_angle,
            start_angle + width,
            1,
            2,
        )

    # background = skimage.morphology.binary_closing(background)

    return np.nonzero(background)


def _apply_noise(canvas: np.ndarray, disc_mask: np.ndarray) -> np.ndarray:
    """Apply perlin noise to input image

    Args:
        canvas (np.ndarray): Input image to which noise will be applied
        disc_mask (np.ndarray): Only pixels which are 1 on this mask will have noise applied to

    Returns:
        np.ndarray: Input image with perlin noise applied
    """
    if not _has_vnoise:
        raise ImportError(
            "This function requires the optional dependency vnoise to be installed"
        )

    noise = vnoise.Noise()

    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV)

    noise_maps = list(
        [
            noise.noise2(
                np.linspace(0, 1, canvas.shape[0]),
                np.linspace(0, 1, canvas.shape[0]),
                grid_mode=True,
                octaves=10,
            ),
            noise.noise2(
                np.linspace(1, 0, canvas.shape[0]),
                np.linspace(1, 0, canvas.shape[0]),
                grid_mode=True,
                octaves=10,
            ),
            noise.noise2(
                np.linspace(0, 1, canvas.shape[0]),
                np.linspace(1, 0, canvas.shape[0]),
                grid_mode=True,
                octaves=10,
            ),
        ]
    )

    # Right now we don't apply noise to the H channel, though we might change that later
    for channel, noise_map in enumerate(noise_maps[0:2], 1):
        noise_map[disc_mask == 0] = 0
        canvas[:, :, channel] = canvas[:, :, channel] + (noise_map * 100)

    canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2RGB)

    return canvas


def _fit_text_in_width(text: str, width: int) -> float:
    """Find the biggest font scale that fit the given text within the given width

    Args:
        text (str): Text to fit
        width (int): Width to fit the text within

    Returns:
        float: Biggest font_scale that will fit given text in the given width
    """
    for font_scale in np.flip(np.arange(0, (width / 250), 0.1)):
        text_width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_TRIPLEX, fontScale=font_scale, thickness=1
        )[0][0]
        if text_width < width:
            return font_scale

    raise ValueError("Could not fit text to scale, is the printable area large enough?")


def _fit_text_in_rectangle(text: str | List, width: int, height: int) -> float:
    """Find the biggest font scale that fits the given text within the given rectangle

    Args:
        text (str|List): Text to fit
        width (int): Width to fit the text within
        height (int): Height to fit the text within

    Returns:
        float: Biggest font_scale that will fit given text in the given rectangle
    """
    for font_scale in np.flip(np.arange(0.1, (width / 100), 0.1)):
        if isinstance(text, list):
            heights = list()
            widths = list()
            for text_line in text:
                text_size = cv2.getTextSize(
                    text_line,
                    cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=font_scale,
                    thickness=1,
                )[0]
                heights.append(text_size[1])
                widths.append(text_size[0])

            if not all(text_width < width for text_width in widths):
                continue

            if not (sum(heights) + (len(heights) * 10)) < height:
                continue

            return font_scale
        else:
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=1
            )[0]
            if text_size[0] < width and text_size[1] < height:
                return font_scale

    raise ValueError("Could not fit text to scale, is the printable area large enough?")
