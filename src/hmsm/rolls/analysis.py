# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Deriving a profile stub from a roll scan.

Given a scan of an unfamiliar format, find the tracks it uses and write out a
profile skeleton for it.  Only tracks that are actually punched on the scan in
hand can be found, so a roll that exercises every track, such as a test roll
or a scale, gives the best result; otherwise run this over several scans and
combine what comes out.

The tones in the stub are placeholders. Assigning real MIDI notes to the
detected tracks is the part that needs a human; see ``docs/CONFIG.md``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import sklearn.cluster

from hmsm.io import ImageSource, open_source
from hmsm.profiles import HOLE_DENSITY_BOUNDS, HOLE_WIDTH_TOLERANCE, RollProfile
from hmsm.rolls.binarization import HOLE_GROW, HOLE_SEED, segment
from hmsm.rolls.digitizer import DEFAULT_BAND_HEIGHT
from hmsm.rolls.holes import filter_components, find_components
from hmsm.rolls.paper import PaperModel, sample_scan
from hmsm.units import DEFAULT_DPI, mm_to_px

logger = logging.getLogger(__name__)

#: Track positions are clustered in units of a thousandth of the roll width,
#: which is what makes the bandwidth argument a comprehensible number.
_POSITION_SCALE = 1000


@dataclass(frozen=True)
class DetectedTracks:
    """Track positions found on a scan.

    Attributes:
        left: Left side of each track, as a fraction of the roll width.
        right: Right side of each track, as a fraction of the roll width.
        hole_count: How many holes were seen on each track.
    """

    left: np.ndarray
    right: np.ndarray
    hole_count: np.ndarray

    def __len__(self) -> int:
        return len(self.left)

    def to_profile(
        self,
        roll_width_mm: float,
        hole_width_mm: float,
        threshold: float = HOLE_GROW,
    ) -> RollProfile:
        """Build a profile stub from the detected tracks.

        Tones are numbered from zero and have to be replaced with the actual
        MIDI notes of the format before the profile is usable, and any printed
        markings the format carries have to be declared as ink layers; see
        ``docs/CONFIG.md``.
        """
        options = {} if threshold == HOLE_GROW else {"hole_grow": threshold}
        return RollProfile.from_dict(
            {
                "media_type": "roll",
                "method": "roll",
                "roll_width_mm": roll_width_mm,
                "hole_width_mm": hole_width_mm,
                "binarization_method": "paper_relative",
                "binarization_options": options,
                "track_measurements": [
                    {
                        "left": float(self.left[i] * roll_width_mm),
                        "right": float(self.right[i] * roll_width_mm),
                        "tone": i,
                    }
                    for i in range(len(self))
                ],
            }
        )


def detect_tracks(
    source: ImageSource,
    hole_width_mm: float = 1.5,
    threshold: float = HOLE_GROW,
    bandwidth: float = 2.0,
    band_height: int = DEFAULT_BAND_HEIGHT,
    skip_rows: int = 0,
    background: Optional[str] = None,
    dpi: float = DEFAULT_DPI,
) -> DetectedTracks:
    """Find the track positions used on a roll scan.

    Every hole on the scan is located and its position relative to the roll
    edges recorded; the positions then cluster into tracks.

    Args:
        source: The scan to analyse.
        hole_width_mm: Nominal width of the holes on the roll.
        threshold: Transmission at which a pixel counts as a hole: zero is
            clean paper and one is the scanner background showing through.
            The default puts the boundary half way, which is where the edge
            of a hole actually is.
        bandwidth: Clustering bandwidth, in thousandths of the roll width.
            Raise it to group more positions into one track, lower it to
            separate more.
        band_height: Number of scan rows to process at a time.
        skip_rows: Rows to skip from the top of the scan.
        background: Scan background colour, detected from the scan if omitted.
        dpi: Resolution of the scan.

    Raises:
        ValueError: If no holes could be found on the scan.

    Returns:
        The detected tracks, ordered left to right.
    """
    paper = PaperModel.estimate(sample_scan(source), dpi=dpi, background=background)
    logger.info("Treating the scan as having a %s background", paper.background_name)

    low, high = sorted(HOLE_WIDTH_TOLERANCE)
    nominal = mm_to_px(hole_width_mm, dpi)
    width_bounds = (int(nominal * low), int(nominal * high))

    positions = _hole_positions(
        source, paper, threshold, width_bounds, band_height, skip_rows
    )

    if len(positions) == 0:
        raise ValueError(
            "No holes were found on this scan. Try a different binarization "
            "threshold or check the declared hole width."
        )

    logger.info("Found %d holes; clustering them into tracks", len(positions))

    # Cluster on the left side of each hole only. The right side follows from
    # it, and using one coordinate keeps the bandwidth easy to reason about.
    mean_shift = sklearn.cluster.MeanShift(bandwidth=bandwidth)
    labels = mean_shift.fit_predict(
        np.column_stack((positions[:, 0] * _POSITION_SCALE, np.zeros(len(positions))))
    )

    tracks = _average_per_cluster(positions, labels)
    logger.info("Detected %d tracks on the provided scan", len(tracks))
    return tracks


def analyze_roll(
    image_path: str,
    output_path: str,
    roll_width_mm: float,
    skip_rows: int = 0,
    hole_width_mm: float = 1.5,
    threshold: float = HOLE_GROW,
    bandwidth: float = 2.0,
    band_height: int = DEFAULT_BAND_HEIGHT,
) -> RollProfile:
    """Analyse a roll scan and write a profile stub for its format.

    Args:
        image_path: Path to the roll scan.
        output_path: Path to write the profile stub to.
        roll_width_mm: Physical width of the roll.
        skip_rows: Rows to skip from the top of the scan.
        hole_width_mm: Nominal width of the holes on the roll.
        threshold: Transmission at which a pixel counts as a hole, in ``[0, 1]``.
        bandwidth: Clustering bandwidth, in thousandths of the roll width.
        band_height: Number of scan rows to process at a time.

    Returns:
        The profile stub that was written.
    """
    with open_source(image_path) as source:
        tracks = detect_tracks(
            source,
            hole_width_mm=hole_width_mm,
            threshold=threshold,
            bandwidth=bandwidth,
            band_height=band_height,
            skip_rows=skip_rows,
            dpi=source.dpi or DEFAULT_DPI,
        )

    profile = tracks.to_profile(roll_width_mm, hole_width_mm, threshold)

    logger.info("Writing profile stub for %d tracks to '%s'", len(tracks), output_path)
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(_profile_to_dict(profile), handle, ensure_ascii=False, indent=4)

    return profile


def _hole_positions(
    source: ImageSource,
    paper: PaperModel,
    threshold: float,
    width_bounds: Tuple[int, int],
    band_height: int,
    skip_rows: int,
) -> np.ndarray:
    """Collect the relative left and right position of every hole on a scan."""
    found: List[np.ndarray] = []

    for (start, stop), band in source.bands(band_height, start=skip_rows):
        try:
            masks = segment(
                "paper_relative",
                band,
                paper,
                hole_seed=max(threshold, HOLE_SEED),
                hole_grow=threshold,
            )
        except Exception as exc:
            logger.info(
                "Skipping band %d-%d, which failed to segment: %s", start, stop, exc
            )
            continue

        components = filter_components(
            find_components(masks.holes),
            width_bounds=width_bounds,
            height_bounds=(width_bounds[0], band_height / 2),
            density_bounds=HOLE_DENSITY_BOUNDS,
        )
        if len(components) == 0:
            continue

        rows = np.clip(components.center_row, 0, len(masks.edges) - 1)
        roll_width = np.maximum(masks.edges.width[rows], 1)
        left = masks.edges.left[rows]
        found.append(
            np.column_stack(
                (
                    (components.left - left) / roll_width,
                    (components.right - left) / roll_width,
                )
            )
        )

    return np.vstack(found) if found else np.empty((0, 2))


def _average_per_cluster(positions: np.ndarray, labels: np.ndarray) -> DetectedTracks:
    """Average the hole positions within each cluster into one track."""
    order = labels.argsort(kind="stable")
    positions, labels = positions[order], labels[order]

    _, first, counts = np.unique(labels, return_index=True, return_counts=True)
    means = np.add.reduceat(positions, first, axis=0) / counts[:, None]

    by_position = means[:, 0].argsort()
    return DetectedTracks(
        left=means[by_position, 0],
        right=means[by_position, 1],
        hole_count=counts[by_position],
    )


def _profile_to_dict(profile: RollProfile) -> dict:
    """Serialize a profile back to the JSON shape documented in CONFIG.md."""
    data = {
        "media_type": "roll",
        "method": "roll",
        "roll_width_mm": profile.roll_width_mm,
        "binarization_method": profile.binarization_method,
        "binarization_options": dict(profile.binarization_options),
        "track_measurements": [
            {"left": t.left_mm, "right": t.right_mm, "tone": t.tone}
            for t in profile.tracks
        ],
    }
    if profile.hole_width_mm:
        widths = profile.hole_width_mm
        data["hole_width_mm"] = widths[0] if len(widths) == 1 else list(widths)
    if profile.hole_length_mm is not None:
        data["hole_length_mm"] = list(profile.hole_length_mm)
    if profile.ink_layers:
        data["ink_layers"] = [
            {"name": layer.name, "role": layer.role, "region": list(layer.region)}
            for layer in profile.ink_layers
        ]
    return data
