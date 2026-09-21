# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import json

import numpy as np
import pytest

from hmsm.profiles import (
    DiscProfile,
    ProfileError,
    RollProfile,
    available_presets,
    load_profile,
)

MINIMAL = {
    "roll_width_mm": 100.0,
    "binarization_method": "v_channel",
    "hole_width_mm": 2.0,
    "track_measurements": [
        {"left": 50.0, "right": 52.0, "tone": 60},
        {"left": 10.0, "right": 12.0, "tone": 55},
    ],
}


def test_every_bundled_preset_loads():
    for name, media in available_presets().items():
        assert load_profile(name, media).name == name


def test_preset_for_the_wrong_medium_is_refused():
    with pytest.raises(ProfileError, match="roll"):
        load_profile("phonola", "cluster")


def test_unknown_preset_lists_the_alternatives():
    with pytest.raises(ProfileError, match="animatic"):
        load_profile("no-such-profile", "roll")


def test_profile_can_come_from_a_json_string():
    profile = load_profile(json.dumps(MINIMAL), "roll")
    assert profile.roll_width_mm == 100.0
    assert len(profile.tracks) == 2


def test_profile_can_come_from_a_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(MINIMAL))
    assert load_profile(str(path), "roll").roll_width_mm == 100.0


def test_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(ProfileError, match="Could not open"):
        load_profile(str(tmp_path / "absent.json"), "roll")


def test_malformed_json_is_reported_clearly():
    with pytest.raises(ProfileError, match="not valid JSON"):
        load_profile('{"roll_width_mm": }', "roll")


def test_missing_required_field_is_named():
    del (data := dict(MINIMAL))["roll_width_mm"]
    with pytest.raises(ProfileError, match="roll_width_mm"):
        RollProfile.from_dict(data)


def test_pedal_cutoff_must_be_a_fraction():
    with pytest.raises(ProfileError, match="fraction"):
        RollProfile.from_dict({**MINIMAL, "pedal_cutoff": 42})


def test_alignment_grid_is_relative_and_sorted():
    grid = RollProfile.from_dict(MINIMAL).alignment_grid()
    assert grid.shape == (2, 3)
    np.testing.assert_allclose(grid[:, 0], [0.10, 0.50])
    np.testing.assert_allclose(grid[:, 2], [55, 60])


def test_hole_width_bounds_span_every_declared_width():
    profile = RollProfile.from_dict({**MINIMAL, "hole_width_mm": [2.7, 1.7]})
    low, high = profile.hole_width_bounds_px(300)
    assert low < high
    # The narrowest hole sets the floor and the widest sets the ceiling.
    assert (
        low
        == RollProfile.from_dict(
            {**MINIMAL, "hole_width_mm": 1.7}
        ).hole_width_bounds_px(300)[0]
    )
    assert (
        high
        == RollProfile.from_dict(
            {**MINIMAL, "hole_width_mm": 2.7}
        ).hole_width_bounds_px(300)[1]
    )


def test_single_hole_width_is_accepted_as_a_scalar():
    assert RollProfile.from_dict(MINIMAL).primary_hole_width_mm == 2.0


def test_annotations_are_detected_from_the_binarization_options():
    assert not RollProfile.from_dict(MINIMAL).has_printed_annotations
    with_upper = RollProfile.from_dict(
        {**MINIMAL, "binarization_options": {"threshold": 0.1, "upper_threshold": 0.6}}
    )
    assert with_upper.has_printed_annotations


def test_disc_profile_round_trips_through_its_legacy_dict():
    profile = load_profile("ariston", "cluster")
    assert isinstance(profile, DiscProfile)
    data = profile.as_dict()
    assert data["radius_inner"] == profile.radius_inner
    assert data["track_mapping"] == dict(profile.track_mapping)
