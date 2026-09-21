# Architecture

This document describes how the package is put together and where to hook into
it. For the format profile file format see [CONFIG.md](CONFIG.md); for which
physical formats are supported see [FORMATS.md](FORMATS.md).

## Layout

```
hmsm/
  units.py        millimetres <-> pixels
  profiles.py     what a physical format looks like; loading and validation
  io/             reading scans band by band
  rolls/          the piano roll pipeline
    paper.py          the colours a scan is made of, and channels against them
    binarization.py   band -> hole mask, ink masks, paper edges
    edges.py          tracking where the paper sits, per row
    holes.py          connected components -> note table
    annotations.py    dynamics line and pedal markers
    notes.py          the note table and merging holes into notes
    digitizer.py      the orchestrator
    analysis.py       roll2config: detecting tracks on an unknown format
  discs/          the disc pipeline
  midi/           note tables -> MIDI
  cli.py          argument parsing only
```

Dependencies run one way: `rolls` and `discs` depend on `io`, `profiles`,
`units` and `midi`; nothing depends on `cli`.

## The note table

Between hole detection and MIDI output the music is an `(n, 3)` integer array
of `[start_row, end_row, tone]`, in scan rows.

**A negative `tone` is a control code, not a pitch.** The codes are
`hmsm.midi.controls.ControlCode` and are listed in [FORMATS.md](FORMATS.md).
Any code that touches a note table has to preserve that convention.

Row positions are absolute scan rows until `hmsm.rolls.notes.rebase` moves the
first note to row zero at the end of the pipeline. The dynamics line is in the
same coordinate system and is shifted along with it, which is what keeps
velocities lined up with the notes they belong to. Its *column*, by contrast,
is measured from the left paper edge rather than from the edge of the scan, so
a roll that drifts sideways does not leak that drift into velocity.

## Reading scans

Roll scans reach two gigabytes decoded, but the pipeline only ever looks at a
few thousand rows at a time. `hmsm.io.open_source` returns an `ImageSource`,
which exposes a scan as something you read horizontal bands out of:

```python
with open_source("scan.tif") as source:
    for (start, stop), pixels in source.bands(4000):
        ...
```

For strip and tile based TIFFs this is `TiffSource`, which decodes only the
segments a band touches, on a thread pool. Anything else is decoded once into
an `ArraySource`, which has the same interface. Memory use is therefore set by
the band size, not by the length of the scan: the two full-length scans in
`assets/`, around 190 000 rows each, take under twenty seconds and about
1.2 GB whatever their length.

`context_bands` is the same thing with a few rows of surrounding scan included
either side. Morphology and smoothing give different answers at the edge of a
band than in the middle of an uninterrupted scan, so the pipeline works on
padded bands and discards the padding. That is what makes results independent
of where the band boundaries fall.

## Measuring a scan against its paper

Nothing on a roll has an absolute brightness worth thresholding. The paper is
beige, pink, red or green depending on the maker, has aged for a century, and
may have been scanned over a black bed or a white one. What is constant is how
each thing on the roll *differs from the paper around it*, and that is what
`hmsm.rolls.paper` measures.

`PaperModel.estimate` reads two reference colours off a sample of the scan:
the **paper**, which is what the bulk of the scan is made of, and the
**background**, which is what a hole shows through. Against those it gives
three channels:

| channel | zero at | one at | used for |
| ------- | ------- | ------ | -------- |
| `transmission` | paper | background | finding holes |
| `shade` | paper | black | finding ink |
| `chroma` | anything on the paper-background line | — | telling ink from holes |

`transmission` is position along the line from paper to background, so a hole
reads as one whether the background is black or white and whatever colour the
paper is. `shade` is how much darker than the paper a pixel is, which is what
printing does regardless of what the background does. `chroma` is the distance
*off* that line: a hole can only ever show a mixture of paper and background,
so it stays on the line however dark it is, while heavily printed ink does not.
That last one is what separates a Hupfeld dynamics line, which is printed dark
enough to pass for a hole, from an actual hole.

`localize_channels` then re-zeroes `transmission` and `shade` against a block
median of the band, taking out the scanner's uneven lighting and whatever a
century of storage did to the paper locally. It is a separate call because it
has to be told where the roll is first: a block straddling the paper edge would
otherwise be levelled against the scanner bed.

Adding support for a new paper colour is therefore not a code change and not
even usually a profile change — the reference colours come off the scan.

## The roll pipeline

`RollDigitizer.run` drives four stages over each band:

1. **Segmentation** (`binarization.segment`) splits the band into a hole mask,
   one ink mask per lane of printing the format declares, and the position of
   the paper edges. The default method, `paper_relative`, works in the
   paper-relative channels above; `v_channel`, which thresholds absolute
   brightness, is kept as a cheaper baseline.
2. **Edge tracking** (`edges.detect_edges`) records where the left and right
   edge of the paper sits in every row. Hole positions are expressed relative
   to those edges, which compensates for the roll drifting sideways or curving
   over the length of a scan.
3. **Hole extraction** (`holes.extract_notes`) labels the connected components
   of the hole mask, rejects anything whose width, length or fill is
   implausible for a hole, and assigns each survivor to the track its middle
   falls on. A component that falls on no track is rejected rather than snapped
   to the nearest one.
4. **Annotations** (`annotations.AnnotationCollector`) accumulates fragments of
   printing per lane, which are only assembled once the whole scan has been
   read, because both a line and a sequence of pedal markers need the whole.

Afterwards `notes.merge_notes` joins holes that sound as one note and
`MidiGenerator` renders the result.

A band that fails to segment carries meaning: before the first holes are seen
it is the roll head and is skipped, afterwards it is the end of the paper and
processing stops.

### Ink lanes

Printed markings are laid out by lane: the Hupfeld Phonola has its pedal words
down the far bass edge and its dynamics line inside them, an Aeolian Themodist
Metrostyle has a red tempo line in a lane of its own besides. A profile
declares those lanes as `ink_layers`, each with a `role`, and segmentation
returns one mask per lane rather than one mask for all printing.

A layer whose `role` is neither `dynamics` nor `pedal` is segmented and
counted but not interpreted. That is deliberate: it is how a marking can be
looked at before it has been decided what it means, and the Metrostyle line
and the Phonola's accentuation marks are both waiting on that.

### Finding the dynamics line

The dynamics line shares its lane with accent marks, printer's ornament and
the maker's watermark, so which ink belongs to it cannot be decided one mark
at a time. `annotations.trace_line` decides it for the sequence as a whole:
every mark may follow any earlier mark within reach along the roll, at a cost
of however far sideways that step is, and every mark taken in is worth a fixed
credit against that cost. The best-scoring chain is the line. It picks up the
dots of a dotted line because they are cheap to reach, follows the line
wherever it goes including straight across the roll, and leaves an accent mark
a thousand pixels off it alone because no credit covers the detour there and
back.

Positions along the line are recorded relative to the left paper edge, so a
roll that wanders sideways down a fifty foot scan does not read as a slow
crescendo.

### Adding a segmentation method

Register a function under a name and reference that name from a profile's
`binarization_method`. Nothing else has to change:

```python
from hmsm.rolls.binarization import BandMasks, binarizer
from hmsm.rolls.edges import detect_edges

@binarizer("my_method")
def my_method(band, paper, my_option=1.0):
    holes = ...          # uint8 mask
    return BandMasks(holes=holes, edges=detect_edges(holes))
```

The method is handed the band and the scan's `PaperModel`; a profile's
`binarization_options` are passed through as keyword arguments, along with
`ink_layers` where the profile declares any.

## Physical measurements

Profiles are in millimetres so that a profile describes a format rather than a
particular scan. `hmsm.units` converts to pixels, and the resolution comes from
the scan's own metadata where it declares one, falling back to 300 dpi.

## Known weaknesses

- **Roll start and end detection.** `digitizer.find_roll_start` only looks at
  how far the paper edges move within a hundred row window, so a roll head with
  a long shallow taper, or a label extending past the head, reads as straight
  roll. This is what puts stray notes at the beginning of a transcription.
- **Coloured annotations are not yet told apart.** `PaperModel.chroma` gives
  the two signed channels that would separate a red Metrostyle line from a
  grey dynamics line printed on the same roll, but nothing reads them: lanes
  are currently separated by position across the roll only. Two markings in
  the same lane in different colours would be traced as one.
- **The roll head is a different paper.** A head is often printed on stock of
  another colour entirely, and the reference colours come from the roll as a
  whole. The local level correction absorbs a good deal of that, but the head
  is not segmented as reliably as the body.
- **The disc pipeline** has not been ported onto the streaming and profile
  machinery; `DiscProfile.as_dict` is the seam between the two. It also still
  uses `hmsm.utils.to_coord_lists`, which materialises the coordinates of every
  set pixel and is where its memory use comes from.
- **`hmsm.midi.create_midi`** is the legacy rendering path, still used by the
  disc pipeline. New MIDI features belong in `MidiGenerator`.
