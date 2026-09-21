# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

"""Format profiles: what a physical medium looks like, in millimetres.

A profile describes a piano roll or disc format rather than a particular scan,
so everything in it is a physical measurement.  Converting those to pixels is
the job of the pipeline, which knows the resolution of the scan in hand.

Profiles are loaded from one of three places, in this order: a path to a
``.json`` file, a raw JSON string, or the name of one of the presets bundled
in ``hmsm/data/config.json``.  See ``docs/CONFIG.md`` for the field reference.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from hmsm.units import DEFAULT_DPI, mm_to_px

logger = logging.getLogger(__name__)

#: How far a connected component's width may deviate from the nominal hole
#: width and still be accepted as a hole.
HOLE_WIDTH_TOLERANCE = (0.5, 1.25)


class ProfileError(ValueError):
    """Raised when a profile is missing, malformed or unsuitable for the task."""


@dataclass(frozen=True)
class Track:
    """One track on a roll, measured from the left edge of the roll.

    Attributes:
        left_mm: Distance from the left roll edge to the left side of the track's holes.
        right_mm: Distance from the left roll edge to the right side of the track's holes.
        tone: MIDI note the track sounds, or a negative control code. See ``docs/FORMATS.md``.
    """

    left_mm: float
    right_mm: float
    tone: int

    @property
    def is_control(self) -> bool:
        """Whether this track carries control information rather than a pitch."""
        return self.tone < 0


@dataclass(frozen=True)
class RollProfile:
    """Everything the roll pipeline needs to know about a roll format.

    Attributes:
        roll_width_mm: Physical width of the roll.
        tracks: The tracks present on the format.
        binarization_method: Name of the function in :mod:`hmsm.rolls.binarization` to segment with.
        binarization_options: Keyword arguments passed to that function.
        hole_width_mm: Nominal hole widths, most frequent first.
        pedal_cutoff: Where printed pedal markers stop and the dynamics line begins,
            as a fraction of the roll width, or None if the format has no printed
            pedal annotations.
        name: Name of the preset this was loaded from, for logging.
    """

    roll_width_mm: float
    tracks: Tuple[Track, ...]
    binarization_method: str = "v_channel"
    binarization_options: Mapping[str, Any] = field(default_factory=dict)
    hole_width_mm: Tuple[float, ...] = ()
    pedal_cutoff: Optional[float] = None
    name: Optional[str] = None

    @property
    def has_printed_annotations(self) -> bool:
        """Whether the format carries printed annotations to extract."""
        return self.binarization_options.get("upper_threshold") is not None

    @property
    def primary_hole_width_mm(self) -> Optional[float]:
        """The most frequent hole width on the format, used for note merging."""
        return self.hole_width_mm[0] if self.hole_width_mm else None

    def alignment_grid(self) -> np.ndarray:
        """Track positions as fractions of the roll width.

        Returns:
            An ``(n, 3)`` array of ``[left, right, tone]``, sorted by position,
            where left and right are fractions of the roll width in ``[0, 1]``.
        """
        grid = np.array(
            [[t.left_mm, t.right_mm, t.tone] for t in self.tracks], dtype=np.float64
        )
        grid[:, 0:2] /= self.roll_width_mm
        return grid[grid[:, 0].argsort()]

    def hole_width_bounds_px(
        self,
        dpi: float = DEFAULT_DPI,
        tolerance: Tuple[float, float] = HOLE_WIDTH_TOLERANCE,
    ) -> Tuple[int, int]:
        """Width range, in pixels, a component must fall in to count as a hole.

        Args:
            dpi: Resolution of the scan being processed.
            tolerance: Lower and upper multipliers applied to the nominal widths.

        Returns:
            Inclusive ``(minimum, maximum)`` component width in pixels.
        """
        if not self.hole_width_mm:
            raise ProfileError("Profile does not declare 'hole_width_mm'")
        low, high = sorted(tolerance)
        widths = [mm_to_px(w, dpi) for w in self.hole_width_mm]
        return (int(min(widths) * low), int(max(widths) * high))

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], name: Optional[str] = None
    ) -> "RollProfile":
        """Build a profile from its JSON representation, validating as we go."""
        missing = {"roll_width_mm", "track_measurements", "binarization_method"} - set(
            data
        )
        if missing:
            raise ProfileError(
                f"Roll profile is missing required field(s): {', '.join(sorted(missing))}"
            )

        tracks = tuple(
            Track(float(m["left"]), float(m["right"]), int(m["tone"]))
            for m in data["track_measurements"]
        )
        if not tracks:
            raise ProfileError("Roll profile declares no tracks")

        hole_width = data.get("hole_width_mm", ())
        if isinstance(hole_width, (int, float)):
            hole_width = (float(hole_width),)
        else:
            hole_width = tuple(float(w) for w in hole_width)

        pedal_cutoff = data.get("pedal_cutoff")
        if pedal_cutoff is not None and not 0.0 < float(pedal_cutoff) < 1.0:
            raise ProfileError(
                f"'pedal_cutoff' must be a fraction of the roll width, got {pedal_cutoff}"
            )

        return cls(
            roll_width_mm=float(data["roll_width_mm"]),
            tracks=tracks,
            binarization_method=str(data["binarization_method"]),
            binarization_options=dict(data.get("binarization_options", {})),
            hole_width_mm=hole_width,
            pedal_cutoff=None if pedal_cutoff is None else float(pedal_cutoff),
            name=name or data.get("name"),
        )


@dataclass(frozen=True)
class DiscProfile:
    """Geometry of a disc format, for the clustering based disc pipeline.

    Attributes:
        radius_inner: Radius of the inner label, as a fraction of the disc radius.
        first_track: Radial position of the first track, as a fraction of the disc radius.
        track_mapping: Track number to MIDI note.
        binarization_threshold: Fixed threshold to binarize with. Otsu's method
            is used where this is not given.
        n_erosions: Erosions to apply before edge detection, to separate edges
            that would otherwise touch.
        name: Name of the preset this was loaded from, for logging.
    """

    radius_inner: float
    first_track: float
    track_mapping: Mapping[str, int]
    binarization_threshold: Optional[float] = None
    n_erosions: Optional[int] = None
    name: Optional[str] = None

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], name: Optional[str] = None
    ) -> "DiscProfile":
        missing = {"radius_inner", "first_track", "track_mapping"} - set(data)
        if missing:
            raise ProfileError(
                f"Disc profile is missing required field(s): {', '.join(sorted(missing))}"
            )
        return cls(
            radius_inner=float(data["radius_inner"]),
            first_track=float(data["first_track"]),
            track_mapping=dict(data["track_mapping"]),
            binarization_threshold=data.get("binarization_threshold"),
            n_erosions=data.get("n_erosions"),
            name=name or data.get("name"),
        )

    def as_dict(self) -> Dict[str, Any]:
        """The profile in the plain dict form the disc pipeline still expects.

        The disc pipeline has not been ported to profile objects yet; this is
        the seam between the two.
        """
        data: Dict[str, Any] = {
            "radius_inner": self.radius_inner,
            "first_track": self.first_track,
            "track_mapping": dict(self.track_mapping),
        }
        if self.binarization_threshold is not None:
            data["binarization_threshold"] = self.binarization_threshold
        if self.n_erosions is not None:
            data["n_erosions"] = self.n_erosions
        return data


def load_profile(spec: str, media: str = "roll") -> RollProfile | DiscProfile:
    """Resolve a profile specification into a profile object.

    Args:
        spec: A path to a ``.json`` file, a JSON string, or the name of a bundled preset.
        media: Which kind of profile is wanted; ``"roll"`` or ``"cluster"``.

    Raises:
        ProfileError: If the profile cannot be found, parsed, or is for a different medium.

    Returns:
        The parsed and validated profile.
    """
    data, name = _resolve(spec, media)

    if media == "roll":
        return RollProfile.from_dict(data, name)
    if media == "cluster":
        return DiscProfile.from_dict(data, name)
    raise ProfileError(f"Unknown media type '{media}'")


def available_presets(media: Optional[str] = None) -> Dict[str, str]:
    """List the bundled presets.

    Args:
        media: Restrict the listing to presets for this medium, if given.

    Returns:
        Preset name to the medium it applies to.
    """
    presets = _bundled_presets()
    return {
        name: data.get("method", "")
        for name, data in presets.items()
        if media is None or data.get("method") == media
    }


def _resolve(spec: str, media: str) -> Tuple[Mapping[str, Any], Optional[str]]:
    """Find the raw profile data behind a specification."""
    if re.search(r"\.json$", spec):
        logger.info("Reading profile from file '%s'", spec)
        try:
            with open(spec, "r", encoding="UTF-8") as handle:
                return json.load(handle), spec
        except FileNotFoundError as exc:
            raise ProfileError(f"Could not open profile file '{spec}'") from exc
        except json.JSONDecodeError as exc:
            raise ProfileError(
                f"Profile file '{spec}' is not valid JSON: {exc}"
            ) from exc

    if re.search(r"^\{.*\}$", spec.strip(), re.DOTALL):
        logger.info("Reading profile from JSON string")
        try:
            return json.loads(spec), None
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Profile string is not valid JSON: {exc}") from exc

    presets = _bundled_presets()
    if spec not in presets:
        candidates = ", ".join(sorted(available_presets(media)))
        raise ProfileError(
            f"No bundled profile named '{spec}'. Available for '{media}': {candidates}"
        )

    data = presets[spec]
    if data.get("method") != media:
        raise ProfileError(
            f"Profile '{spec}' is for '{data.get('method')}' media, "
            f"but a '{media}' profile is required here"
        )
    logger.info("Using bundled profile '%s'", spec)
    return data, spec


def _bundled_presets() -> Mapping[str, Mapping[str, Any]]:
    """Load the presets shipped with the package."""
    resource = importlib.resources.files("hmsm").joinpath("data/config.json")
    with resource.open("r", encoding="UTF-8") as handle:
        return json.load(handle)
