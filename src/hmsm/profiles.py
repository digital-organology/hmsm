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

#: How much of its bounding box a component must fill to be a punched hole.
#: A hole is convex, so it fills most of its box; a tear, a fold or a smear
#: of dirt that happens to be the right width does not.
HOLE_DENSITY_BOUNDS = (0.45, 1.0)

#: The roles :class:`InkLayer` understands. A layer with any other role is
#: segmented like the rest but nothing downstream reads it, which is how a
#: format's markings can be looked at before it is decided what they mean.
INK_ROLES = ("dynamics", "pedal")


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
class InkLayer:
    """One strip of a roll that carries printed markings.

    Printed markings are laid out by lane: a Hupfeld Phonola has its pedal
    words down the far bass edge and its dynamics line just inside them, an
    Aeolian Themodist Metrostyle has a red tempo line and a grey dynamics line
    in two more lanes again. Declaring the lanes rather than a single cutoff
    is what lets a format say how many kinds of marking it carries and where
    each one lives.

    Attributes:
        name: Identifies the layer's mask, and names it in debug output.
        role: What the pipeline should make of the layer. ``"dynamics"``
            traces a continuous line and turns it into velocities,
            ``"pedal"`` pairs markers into sustain spans. Any other role is
            extracted but not interpreted, which is the intended way to look
            at a marking whose meaning is not settled: the Metrostyle line and
            the Phonola's accentuation marks are both waiting on that.
        region: Where the layer sits across the roll, as ``(left, right)``
            fractions of the roll width measured from the left paper edge.
    """

    name: str
    role: str
    region: Tuple[float, float]

    @property
    def is_interpreted(self) -> bool:
        """Whether the pipeline knows what to do with this layer."""
        return self.role in INK_ROLES

    def as_option(self) -> Dict[str, Any]:
        """The layer in the plain form the binarization methods take."""
        return {"name": self.name, "region": list(self.region)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InkLayer":
        missing = {"name", "role", "region"} - set(data)
        if missing:
            raise ProfileError(
                f"Ink layer is missing required field(s): {', '.join(sorted(missing))}"
            )

        region = tuple(float(x) for x in data["region"])
        if len(region) != 2:
            raise ProfileError(
                f"Ink layer '{data['name']}' needs a region of exactly two "
                f"fractions, got {list(region)}"
            )
        if not 0.0 <= region[0] < region[1] <= 1.0:
            raise ProfileError(
                f"Ink layer '{data['name']}' has region {list(region)}, which is "
                "not an increasing pair of fractions of the roll width"
            )

        return cls(name=str(data["name"]), role=str(data["role"]), region=region)


@dataclass(frozen=True)
class RollProfile:
    """Everything the roll pipeline needs to know about a roll format.

    Attributes:
        roll_width_mm: Physical width of the roll.
        tracks: The tracks present on the format.
        binarization_method: Name of the function in :mod:`hmsm.rolls.binarization` to segment with.
        binarization_options: Keyword arguments passed to that function.
        hole_width_mm: Nominal hole widths, most frequent first.
        hole_length_mm: Shortest and longest a punched hole may be along the
            roll, or None to accept any length. The upper bound is what keeps
            a fold or a torn edge running down the roll out of the note table.
        ink_layers: The lanes of printed marking the format carries, in the
            order they appear across the roll.
        name: Name of the preset this was loaded from, for logging.
    """

    roll_width_mm: float
    tracks: Tuple[Track, ...]
    binarization_method: str = "paper_relative"
    binarization_options: Mapping[str, Any] = field(default_factory=dict)
    hole_width_mm: Tuple[float, ...] = ()
    hole_length_mm: Optional[Tuple[float, float]] = None
    ink_layers: Tuple[InkLayer, ...] = ()
    name: Optional[str] = None

    @property
    def has_printed_annotations(self) -> bool:
        """Whether the format carries printed annotations to extract."""
        return bool(self.ink_layers)

    def layer(self, role: str) -> Optional[InkLayer]:
        """The first layer declared with ``role``, if the format has one."""
        return next((layer for layer in self.ink_layers if layer.role == role), None)

    @property
    def primary_hole_width_mm(self) -> Optional[float]:
        """The most frequent hole width on the format, used for note merging."""
        return self.hole_width_mm[0] if self.hole_width_mm else None

    def segmentation_options(self) -> Dict[str, Any]:
        """The options to hand :func:`hmsm.rolls.binarization.segment`.

        The declared ink layers travel with them, so a segmentation method
        returns its ink already split by lane.
        """
        options = dict(self.binarization_options)
        if self.ink_layers:
            options["ink_layers"] = [layer.as_option() for layer in self.ink_layers]
        return options

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

    def hole_length_bounds_px(
        self, dpi: float = DEFAULT_DPI
    ) -> Optional[Tuple[int, int]]:
        """Length range, in pixels, a component must fall in to count as a hole.

        Args:
            dpi: Resolution of the scan being processed.

        Returns:
            Inclusive ``(minimum, maximum)`` component height in pixels, or
            None where the format does not constrain hole length.
        """
        if self.hole_length_mm is None:
            return None
        low, high = self.hole_length_mm
        return (int(mm_to_px(low, dpi)), int(mm_to_px(high, dpi)))

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

        hole_length = data.get("hole_length_mm")
        if hole_length is not None:
            hole_length = tuple(float(x) for x in hole_length)
            if len(hole_length) != 2 or not 0 <= hole_length[0] < hole_length[1]:
                raise ProfileError(
                    f"'hole_length_mm' must be an increasing pair of lengths, "
                    f"got {list(hole_length)}"
                )

        layers = tuple(
            InkLayer.from_dict(entry) for entry in data.get("ink_layers", ())
        )
        _check_layers_do_not_overlap(layers)

        unknown = {layer.role for layer in layers} - set(INK_ROLES)
        if unknown:
            logger.info(
                "Ink layer role(s) %s are not interpreted by this pipeline; "
                "those layers will be segmented but not read",
                ", ".join(sorted(unknown)),
            )

        return cls(
            roll_width_mm=float(data["roll_width_mm"]),
            tracks=tracks,
            binarization_method=str(data["binarization_method"]),
            binarization_options=dict(data.get("binarization_options", {})),
            hole_width_mm=hole_width,
            hole_length_mm=hole_length,
            ink_layers=layers,
            name=name or data.get("name"),
        )


def _check_layers_do_not_overlap(layers: Sequence[InkLayer]) -> None:
    """Reject layers that claim the same strip of roll.

    A pixel belongs to one marking, so overlapping lanes would have the same
    ink turn up as two different annotations.
    """
    ordered = sorted(layers, key=lambda layer: layer.region)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.region[0] < earlier.region[1]:
            raise ProfileError(
                f"Ink layers '{earlier.name}' and '{later.name}' overlap: "
                f"{list(earlier.region)} and {list(later.region)}"
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
