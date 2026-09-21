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


def test_ink_layer_regions_must_be_increasing_fractions():
    for region in ([0.5, 0.2], [0.0, 1.5], [-0.1, 0.5], [0.3]):
        with pytest.raises(ProfileError, match="region|two fractions"):
            RollProfile.from_dict({**MINIMAL, "ink_layers": [_layer(region=region)]})


def test_ink_layers_may_not_overlap():
    with pytest.raises(ProfileError, match="overlap"):
        RollProfile.from_dict(
            {
                **MINIMAL,
                "ink_layers": [
                    _layer(name="pedal", region=[0.0, 0.3]),
                    _layer(name="dynamics", region=[0.2, 1.0]),
                ],
            }
        )


def test_touching_ink_layers_are_fine():
    profile = RollProfile.from_dict(
        {
            **MINIMAL,
            "ink_layers": [
                _layer(name="pedal", role="pedal", region=[0.0, 0.1]),
                _layer(name="dynamics", region=[0.1, 1.0]),
            ],
        }
    )
    assert [layer.name for layer in profile.ink_layers] == ["pedal", "dynamics"]
    assert profile.layer("pedal").region == (0.0, 0.1)
    assert profile.layer("tempo") is None


def test_an_ink_layer_missing_a_field_is_named():
    with pytest.raises(ProfileError, match="role"):
        RollProfile.from_dict(
            {**MINIMAL, "ink_layers": [{"name": "x", "region": [0.0, 0.5]}]}
        )


def test_an_uninterpreted_role_is_kept_but_flagged():
    profile = RollProfile.from_dict(
        {**MINIMAL, "ink_layers": [_layer(name="metrostyle", role="tempo")]}
    )
    layer = profile.layer("tempo")
    assert layer is not None and not layer.is_interpreted
    # It still reaches segmentation, so its mask can be looked at.
    assert profile.segmentation_options()["ink_layers"] == [
        {"name": "metrostyle", "region": [0.0, 1.0]}
    ]


def test_hole_length_must_be_an_increasing_pair():
    with pytest.raises(ProfileError, match="hole_length_mm"):
        RollProfile.from_dict({**MINIMAL, "hole_length_mm": [40.0, 4.0]})


def test_hole_length_bounds_convert_to_pixels():
    assert RollProfile.from_dict(MINIMAL).hole_length_bounds_px(300) is None
    low, high = RollProfile.from_dict(
        {**MINIMAL, "hole_length_mm": [1.0, 25.4]}
    ).hole_length_bounds_px(300)
    assert (low, high) == (11, 300)


def _layer(name="dynamics", role="dynamics", region=(0.0, 1.0)):
    return {"name": name, "role": role, "region": list(region)}


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


def test_annotations_are_detected_from_the_declared_ink_layers():
    assert not RollProfile.from_dict(MINIMAL).has_printed_annotations
    assert "ink_layers" not in RollProfile.from_dict(MINIMAL).segmentation_options()
    with_ink = RollProfile.from_dict({**MINIMAL, "ink_layers": [_layer()]})
    assert with_ink.has_printed_annotations


def test_disc_profile_round_trips_through_its_legacy_dict():
    profile = load_profile("ariston", "cluster")
    assert isinstance(profile, DiscProfile)
    data = profile.as_dict()
    assert data["radius_inner"] == profile.radius_inner
    assert data["track_mapping"] == dict(profile.track_mapping)
