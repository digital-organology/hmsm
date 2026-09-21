# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import mido
import numpy as np
import pytest

from hmsm.midi import TICKS_PER_BEAT, VELOCITY_BASE, MidiGenerator
from hmsm.midi.controls import ControlCode


def render(starts, lengths, tones, **kwargs):
    generator = MidiGenerator(kwargs.pop("fpm", 60))
    return generator, generator.make_midi(starts, lengths, tones, **kwargs)


def messages(midi_file, kind=None):
    return [m for m in midi_file.tracks[0] if kind is None or m.type == kind]


def test_tempo_converts_rows_to_seconds():
    # 60 means six feet per minute; at 300 dpi that is 21600 rows a minute.
    assert MidiGenerator(60).seconds_per_row == pytest.approx(60 / 21600)
    assert MidiGenerator(120).seconds_per_row == pytest.approx(
        MidiGenerator(60).seconds_per_row / 2
    )


def test_tempo_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        MidiGenerator(0)


def test_writing_before_rendering_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="make_midi"):
        MidiGenerator(60).write(str(tmp_path / "x.mid"))


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal length"):
        MidiGenerator(60).make_midi([0], [1, 2], [60])


def test_each_note_becomes_an_on_and_an_off():
    _, midi = render([0, 1000], [500, 500], [60, 72])
    assert len(messages(midi, "note_on")) == 2
    assert len(messages(midi, "note_off")) == 2


def test_notes_carry_the_base_velocity_without_a_dynamics_line():
    _, midi = render([0], [500], [60])
    assert messages(midi, "note_on")[0].velocity == VELOCITY_BASE


def test_timing_does_not_drift_over_a_long_roll():
    # Notes exactly one row apart; the last one must land where it belongs
    # rather than accumulating a rounding error per event.
    generator = MidiGenerator(60)
    starts = list(range(0, 100_000, 7))
    midi = generator.make_midi(starts, [1] * len(starts), [60] * len(starts))

    tick = 0
    first_on = None
    for message in midi.tracks[0]:
        tick += message.time
        if message.type == "note_on":
            first_on = tick if first_on is None else first_on
            last_on = tick

    expected = round(
        mido.second2tick(
            starts[-1] * generator.seconds_per_row, TICKS_PER_BEAT, mido.bpm2tempo(120)
        )
    )
    assert abs(last_on - expected) <= 1


def test_pedal_spans_become_sustain_control_changes():
    pedal = int(ControlCode.PEDAL)
    _, midi = render([0, 100, 200], [50, 50, 50], [60, pedal, 61], hole_size_mm=2.7)
    sustain = [m for m in messages(midi, "control_change") if m.control == 64]
    # An initial state, then the span itself.
    assert len(sustain) >= 3
    assert sustain[0].value == 0


def test_no_sustain_messages_without_a_pedal_track():
    _, midi = render([0], [50], [60])
    assert messages(midi, "control_change") == []


def test_dynamics_line_drives_velocity():
    dynamics = np.column_stack((np.arange(0, 1000), np.linspace(0, 100, 1000)))
    _, midi = render([10, 900], [50, 50], [60, 60], dynamics_line=dynamics)
    quiet, loud = messages(midi, "note_on")
    assert quiet.velocity < loud.velocity


def test_notes_outside_the_dynamics_line_clamp_to_its_ends():
    dynamics = np.column_stack((np.arange(500, 600), np.full(100, 50)))
    _, midi = render([0, 10_000], [50, 50], [60, 60], dynamics_line=dynamics)
    assert {m.velocity for m in messages(midi, "note_on")} == {40}


def test_a_flat_dynamics_line_does_not_divide_by_zero():
    dynamics = np.column_stack((np.arange(0, 100), np.full(100, 7)))
    _, midi = render([10], [50], [60], dynamics_line=dynamics)
    assert messages(midi, "note_on")[0].velocity >= 1


def test_boost_tracks_raise_the_velocity_of_notes_underneath_them():
    boost = int(ControlCode.BASS_BOOST_ON)
    _, midi = render([0, 100, 5000], [1000, 50, 50], [boost, 60, 60])
    inside, outside = messages(midi, "note_on")
    assert inside.velocity > outside.velocity


def test_boost_applies_only_to_its_half_of_the_keyboard():
    bass = int(ControlCode.BASS_BOOST_ON)
    _, midi = render([0, 100, 100], [1000, 50, 50], [bass, 60, 80])
    boosted, unboosted = messages(midi, "note_on")
    assert boosted.note == 60 and unboosted.note == 80
    assert boosted.velocity > unboosted.velocity


def test_velocities_stay_in_range():
    dynamics = np.column_stack((np.arange(0, 100), np.linspace(0, 1000, 100)))
    boost = int(ControlCode.BASS_BOOST_ON)
    _, midi = render([0, 50], [200, 10], [boost, 60], dynamics_line=dynamics)
    for message in messages(midi, "note_on"):
        assert 1 <= message.velocity <= 127


def test_unknown_control_codes_are_ignored_with_a_warning(caplog):
    _, midi = render([0, 100], [50, 50], [-99, 60])
    assert len(messages(midi, "note_on")) == 1
    assert "-99" in caplog.text


def test_releases_precede_attacks_at_the_same_instant():
    # Two notes meeting exactly: the release must come first so the second
    # note is not immediately silenced by the first note's note_off.
    _, midi = render([0, 500], [500, 500], [60, 60])
    kinds = [m.type for m in messages(midi) if m.type in ("note_on", "note_off")]
    assert kinds == ["note_on", "note_off", "note_on", "note_off"]


def test_written_file_can_be_read_back(tmp_path):
    generator, _ = render([0, 1000], [500, 500], [60, 72])
    path = str(tmp_path / "out.mid")
    generator.write(path)
    assert len(mido.MidiFile(path).tracks) == 1
