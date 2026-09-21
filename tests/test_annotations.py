# Copyright (c) 2023 David Fuhry, Museum of Musical Instruments, Leipzig University

import numpy as np
import pytest

from hmsm.midi.controls import ControlCode
from hmsm.profiles import InkLayer
from hmsm.rolls.annotations import (
    AnnotationCollector,
    Fragments,
    pedal_spans,
    trace_line,
)
from hmsm.rolls.binarization import InkMask
from hmsm.rolls.edges import RollEdges

DPI = 300


def fragments(rows, columns, widths=6.0, areas=400.0):
    rows = np.asarray(rows, dtype=np.float64)
    columns = np.asarray(columns, dtype=np.float64)
    widths = np.broadcast_to(np.asarray(widths, dtype=np.float64), rows.shape)
    return Fragments(
        row=rows,
        column=columns,
        left=columns - widths / 2,
        right=columns + widths / 2,
        area=np.broadcast_to(np.asarray(areas, dtype=np.float64), rows.shape).copy(),
    )


def dotted_line(count=120, pitch=60, start=500.0, slope=0.4):
    """A dotted line running down the roll and slowly out across it."""
    rows = np.arange(count) * pitch
    return rows, start + rows * slope / pitch


def test_a_dotted_line_is_traced_through_every_row_it_spans():
    rows, columns = dotted_line()
    line = trace_line(fragments(rows, columns), DPI)

    assert line is not None
    assert line[0, 0] == rows[0] and line[-1, 0] == rows[-1]
    assert len(line) == rows[-1] - rows[0] + 1
    # The traced column follows the marks it was built from.
    at = {int(r): int(c) for r, c in line}
    for row, column in zip(rows[5:-5], columns[5:-5]):
        assert abs(at[int(row)] - column) <= 3


def test_printing_off_the_line_does_not_pull_the_line_off_course():
    # Accent marks and a maker's watermark share the lane with the dynamics
    # line on a Hupfeld roll, and sit hundreds of pixels away from it.
    rows, columns = dotted_line()
    strays = np.arange(10, 110, 7)
    all_rows = np.concatenate((rows, rows[strays] + 5))
    all_columns = np.concatenate((columns, columns[strays] + 900))

    order = all_rows.argsort(kind="stable")
    line = trace_line(fragments(all_rows[order], all_columns[order]), DPI)

    assert line is not None
    at = {int(r): int(c) for r, c in line}
    for row, column in zip(rows[5:-5], columns[5:-5]):
        assert abs(at[int(row)] - column) <= 20


def test_a_line_that_steps_sharply_across_the_roll_is_followed():
    # The Hupfeld line runs in straight segments and does change direction
    # abruptly; it must not be smoothed into a diagonal or cut in two.
    rows = np.arange(120) * 60
    columns = np.where(rows < 3600, 500.0, 2200.0)
    line = trace_line(fragments(rows, columns), DPI)

    assert line is not None
    assert line[0, 1] < 700 and line[-1, 1] > 2000


def test_scattered_specks_are_not_a_line():
    rng = np.random.default_rng(0)
    rows = np.sort(rng.integers(0, 60_000, 200).astype(float))
    columns = rng.integers(0, 3000, 200).astype(float)
    assert trace_line(fragments(rows, columns), DPI) is None


def test_too_little_ink_is_not_a_line():
    rows, columns = dotted_line(count=10)
    assert trace_line(fragments(rows, columns), DPI) is None


def test_a_short_run_of_marks_on_a_long_roll_is_not_a_line():
    # Fifty accent marks in one corner of a roll trace nicely among
    # themselves, but they do not span the roll and are not its dynamics line.
    rows, columns = dotted_line(count=60, pitch=60)
    rows = np.concatenate((rows, [200_000.0]))
    columns = np.concatenate((columns, [500.0]))
    assert trace_line(fragments(rows, columns), DPI) is None


def pedal_markers(count=12, pitch=2000, wide=250.0, narrow=70.0):
    """Alternating "Ped." and flower markers down the bass edge."""
    rows = np.arange(count) * pitch + 1000.0
    widths = np.where(np.arange(count) % 2 == 0, wide, narrow)
    return fragments(rows, np.full(count, 120.0), widths, areas=4000.0)


def test_pedal_markers_pair_into_spans():
    spans = pedal_spans(pedal_markers(), DPI)

    assert spans is not None
    assert len(spans) == 6
    assert (spans[:, 2] == int(ControlCode.PEDAL)).all()
    # Each span opens on a wide marker and closes on the next narrow one.
    np.testing.assert_array_equal(spans[0], [1000, 3000, int(ControlCode.PEDAL)])
    assert (spans[:, 1] > spans[:, 0]).all()


def test_a_marker_broken_into_pieces_is_still_one_marker():
    # "Ped." scans as the word and its full stop, a few millimetres apart.
    whole = pedal_markers()
    pieces = Fragments(
        row=np.concatenate((whole.row, whole.row[::2] + 40)),
        column=np.concatenate((whole.column, whole.column[::2] + 140)),
        left=np.concatenate((whole.left, whole.left[::2] + 260)),
        right=np.concatenate((whole.right, whole.right[::2] + 280)),
        area=np.concatenate((whole.area, np.full(len(whole.row[::2]), 400.0))),
    )
    order = pieces.row.argsort(kind="stable")
    pieces = Fragments(
        *(getattr(pieces, f)[order] for f in ("row", "column", "left", "right", "area"))
    )

    assert len(pedal_spans(pieces, DPI)) == len(pedal_spans(whole, DPI))


def test_specks_are_not_pedal_markers():
    small = pedal_markers()
    small = Fragments(
        small.row, small.column, small.left, small.right, np.full(len(small.row), 50.0)
    )
    assert pedal_spans(small, DPI) is None


def test_too_few_markers_are_not_a_pedal_sequence():
    assert pedal_spans(pedal_markers(count=4), DPI) is None


def test_marker_widths_split_even_when_one_kind_dominates():
    # A long run of "Ped." with only a few flowers between them: the mean
    # width sits above the narrow markers and would call them all wide.
    widths = np.array([250.0] * 18 + [70.0, 250.0, 70.0, 250.0])
    rows = np.arange(len(widths)) * 2000.0 + 1000.0
    marks = fragments(rows, np.full(len(widths), 120.0), widths, areas=4000.0)

    spans = pedal_spans(marks, DPI)
    assert spans is not None and len(spans) >= 2


def collector_with(layers, masks, edges=None, offset=0):
    edges = edges or RollEdges(
        np.full(masks[0].shape[0], 10, np.int32),
        np.full(masks[0].shape[0], 490, np.int32),
    )
    collector = AnnotationCollector(layers, dpi=DPI)
    collector.add_band(
        {layer.name: InkMask(mask, offset) for layer, mask in zip(layers, masks)},
        edges,
        row_offset=0,
    )
    return collector


def test_fragment_positions_are_measured_from_the_left_paper_edge():
    layer = InkLayer(name="dynamics", role="dynamics", region=(0.0, 1.0))
    mask = np.zeros((200, 200), np.uint8)
    mask[50:70, 100:120] = 1  # in the lane's own coordinates

    collector = collector_with([layer], [mask], offset=300)
    found = collector.fragments("dynamics")

    assert found is not None and len(found) == 1
    # Lane column 110, lane starts at scan column 300, paper edge at 10.
    assert abs(found.column[0] - (300 + 110 - 10)) < 1


def test_specks_and_whole_labels_are_both_ignored():
    layer = InkLayer(name="dynamics", role="dynamics", region=(0.0, 1.0))
    mask = np.zeros((600, 600), np.uint8)
    mask[10:12, 10:12] = 1  # a speck of paper grain
    mask[100:500, 100:500] = 1  # the roll's paper label
    mask[50:70, 550:570] = 1  # a genuine dot

    found = collector_with([layer], [mask]).fragments("dynamics")
    assert found is not None and len(found) == 1


def test_a_layer_with_an_unread_role_is_collected_but_not_interpreted(caplog):
    layers = [InkLayer(name="metrostyle", role="tempo", region=(0.0, 1.0))]
    mask = np.zeros((200, 200), np.uint8)
    mask[50:70, 100:120] = 1

    collector = collector_with(layers, [mask])
    assert collector.dynamics_line() is None
    assert collector.pedal_events() is None
    assert len(collector.fragments("metrostyle")) == 1

    with caplog.at_level("INFO"):
        collector.report_uninterpreted()
    assert "metrostyle" in caplog.text and "tempo" in caplog.text


def test_an_empty_lane_is_not_an_error():
    layers = [InkLayer(name="dynamics", role="dynamics", region=(0.0, 0.1))]
    collector = AnnotationCollector(layers, dpi=DPI)
    edges = RollEdges(np.zeros(100, np.int32), np.full(100, 99, np.int32))
    collector.add_band({"dynamics": InkMask(np.zeros((100, 0), np.uint8), 0)}, edges)
    assert collector.fragments("dynamics") is None
    assert collector.dynamics_line() is None


@pytest.mark.parametrize("dpi", [300, 600])
def test_thresholds_follow_the_scan_resolution(dpi):
    # The same roll scanned at twice the resolution gives the same line.
    rows, columns = dotted_line()
    at_300 = trace_line(fragments(rows, columns), 300)
    at_600 = trace_line(fragments(rows * 2, columns * 2), 600)
    assert at_300 is not None and at_600 is not None
    assert abs(len(at_600) - 2 * len(at_300)) <= 2
