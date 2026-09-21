# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`hmsm` (HMSM-Tools) digitizes **H**istorical **M**usical **S**torage **M**edia — piano rolls and cardboard discs — from scans into MIDI, and can render MIDI back into disc images. It is the digitization component of the BMBF-funded DISKOS project at the Research Center Digital Organology, Leipzig University. GPL-3.0-or-later.

## Development

```bash
pip install -e ".[dev]"                    # src layout; entry points come from pyproject.toml
pip install -e ".[midi2disc,progressbar]"  # optional extras (cairosvg+vnoise, enlighten)
pytest                                     # ~120 tests, a couple of seconds
```

Python >= 3.11 (the code uses `typing.Self`-era syntax and `from __future__ import annotations`); developed against 3.13. Formatting is **black** (configured in `.vscode/settings.json`, format-on-save plus import organization).

There is no CI. Beyond the test suite, verification is done by running the CLI against real scans. `assets/` holds two disc images and two full-length roll scans; `examples/` holds head/playback/end excerpts cut from those rolls, plus low-resolution examples of rolls on differently coloured paper and with an extra annotation line:

```bash
hmsm roll2midi -c phonola -t 60 examples/phonola_roll_playback.tif out.mid
hmsm disc2midi -c ariston -m cluster assets/5070081_22.JPG out.mid
hmsm disc2roll --offset 92 assets/5070081_11.JPG roll.JPG
hmsm profiles                              # list bundled format profiles
```

Tests that need one of those scans skip themselves when it is absent, so the suite runs on a clean checkout.

Pass `-d`/`--debug` for verbose logging plus note tables written under `./debug_data`. The digitizer creates that directory itself, so calling library code directly with a `debug_dir` works.

## Architecture

`docs/ARCHITECTURE.md` is the reference; this is the short version.

Dependencies run one way: `rolls` and `discs` depend on `io`, `profiles`, `units` and `midi`. Nothing depends on `cli`.

### Entrypoints (`src/hmsm/cli.py`)

One `hmsm` command with subcommands (`roll2midi`, `roll2config`, `disc2midi`, `disc2roll`, `midi2disc`, `profiles`), plus a thin alias per subcommand under its old console-script name. `cli.py` only parses arguments, sets up logging and dispatches — keep processing logic out of it.

`--band-height` deliberately has no short form: the old CLI used `-s` for it on `roll2midi` and for the hole size on `roll2config`, so a short form would let stale command lines run while meaning something else.

### Profiles (`src/hmsm/profiles.py`)

`load_profile(spec, media)` resolves three ways, in order: a path ending in `.json`, a raw JSON string, or a preset name from the bundled `src/hmsm/data/config.json`. Returns a frozen `RollProfile` or `DiscProfile`; malformed input raises `ProfileError`. Profiles are in **millimetres** — `hmsm.units` converts to pixels using the scan's declared resolution, defaulting to 300 dpi.

A profile describes a format, never a scan: paper colour and scanner background are read off the scan itself and are not configurable. Printed markings are declared as `ink_layers`, a lane across the roll plus a `role` (`dynamics`, `pedal`, or anything else, which is segmented but not interpreted).

`docs/CONFIG.md` documents every field of a roll profile and `docs/FORMATS.md` tracks which physical formats each preset covers — update both when adding or changing a preset.

### Reading scans (`src/hmsm/io/`)

`open_source(path)` returns an `ImageSource`. For strip and tile based TIFFs that is `TiffSource`, which decodes only the segments a requested row range touches, on a thread pool; anything else is decoded once into an `ArraySource` with the same interface. Memory is set by the band size, not the length of the scan.

`source.context_bands(height, margin)` yields bands padded with surrounding rows that the caller discards afterwards. That padding is what makes results independent of where band boundaries fall; without it, morphology and edge smoothing behave differently at a band edge. Don't remove it.

A full-length scan (~190 000 rows) takes under 20 s and about 1.2 GB. `DEFAULT_BAND_HEIGHT` is deliberately left at 4000: halving it does cut memory to ~800 MB but costs 30% more time, and `_HEAD_TRAVEL_THRESHOLD` is an absolute edge travel per band, so changing the band height moves where roll-start detection thinks the head ends.

### Measuring against the paper (`src/hmsm/rolls/paper.py`)

Nothing on a roll has an absolute brightness worth thresholding, so `PaperModel` reads two reference colours off the scan — the **paper** (what the scan is mostly made of) and the **background** (what shows through a hole) — and expresses everything against them:

- `transmission`: position along the paper→background line. A hole is one, whatever colour either end is. This is how black and white background scans, and paper of any colour, go down one code path.
- `shade`: how much darker than the paper a pixel is. Printing darkens paper whatever the background does, so this and not transmission is what ink is found on.
- `chroma`: distance *off* that line. A hole can only show a mixture of paper and background, so it stays on the line however dark it is; the Hupfeld dynamics line is printed dark enough to pass for a hole on transmission alone and is separated by this. Nothing reads the two signed channels yet — they are the hook for telling a red Metrostyle line from a grey dynamics line.

`localize_channels` re-zeroes transmission and shade against a block median, taking out uneven scanner lighting and locally aged paper. It is a separate call from `channels()` because it has to be told where the roll is first — flatten outside the paper onto zero, or a block straddling the edge is levelled against the scanner bed.

### Roll pipeline (`src/hmsm/rolls/`)

`RollDigitizer.run()` drives, per band: `binarization.segment` (hole mask, one ink mask per declared lane, paper edges) → `edges.detect_edges` (per-row paper position) → `holes.extract_notes` (components → note table) → `annotations.AnnotationCollector` (accumulates fragments per lane). Then `notes.merge_notes`, `notes.rebase`, and `MidiGenerator`.

Binarization methods are registered with the `@binarizer("name")` decorator and selected by a profile's `binarization_method`; they take the band and the scan's `PaperModel`, and `binarization_options` are splatted in as kwargs. `paper_relative` is the default; `v_channel` (absolute thresholds) is kept as a cheaper baseline. Adding one means writing a function returning `BandMasks` — nothing else changes.

Holes are found with a two-level (hysteresis) threshold on transmission: seed at 0.75, grow out to 0.5, which is where the edge of a blurred hole actually is. A region is then rejected if its average colour sits off the paper–background line, which is what keeps heavily printed ink out of the note table. Track assignment is by the hole's **middle** against the track's middle, and a component falling outside every track's cell is dropped rather than snapped to the nearest one.

The dynamics line is reconstructed by `annotations.trace_line`, which scores chains of marks: a fixed credit per mark taken in, less the sideways distance travelled between them. The best chain is the line. That is what lets it follow the line straight across the roll while ignoring accent marks and watermarks in the same lane — decided for the sequence as a whole, not one mark at a time. Positions are recorded relative to the left paper edge so roll drift does not read as a crescendo.

A band that fails to segment carries meaning: before the first holes are seen it is the roll head and is skipped; afterwards it is the end of the paper and processing stops.

**Performance-critical choices, do not undo:**
- Morphology goes through `cv2.morphologyEx` on uint8, not `skimage.morphology`, which dispatches boolean input to scipy's *grayscale* filter path and is ~150-245x slower for identical output.
- Component analysis goes through `cv2.connectedComponentsWithStats`, which returns the bounding boxes and areas the pipeline actually needs. `hmsm.utils.to_coord_lists` materialises every set pixel's coordinates (~1 GB per 4000-row band) and is only still used by the disc pipeline.
- The value channel is a per-pixel max over uint8 channels, not a float conversion.

### The note table

An `(n, 3)` int64 array of `[start_row, end_row, tone]` in scan rows. **Negative `tone` values are control codes, not pitches** (`hmsm.midi.controls.ControlCode`; table in `docs/FORMATS.md`). Any code touching note arrays must keep that convention.

Rows are absolute scan rows until `notes.rebase` moves the first note to zero at the end; it shifts the dynamics line by the same amount, which is what keeps velocities aligned with their notes. The dynamics line's *column* is measured from the left paper edge, not from the edge of the scan, so roll drift does not leak into velocity.

### Disc pipeline (`src/hmsm/discs/`)

Not yet ported onto the streaming and profile machinery — `DiscProfile.as_dict()` is the seam. `discs/cluster.py` binarizes, crops, does morphological edge detection, fits an ellipse to the circumference, masks out the inner label, and assigns holes to tracks by radial distance. Note the comment in `cluster.py` about coordinate ordering: centres are `(y, x)` everywhere except the scipy distance call. `discs/generator.py` (`midi2disc`) renders discs from MIDI; its geometry lives in a hardcoded `_get_config(type)`, *not* in `data/config.json`.

### MIDI (`src/hmsm/midi/`)

`MidiGenerator` understands tempo scaling from feet-per-minute, velocity from the dynamics line and the boost tracks, and pedal spans. Event times are rounded in absolute terms and differenced afterwards, so rounding error cannot accumulate over the length of a roll. `create_midi()` at the bottom of `midi/__init__.py` is the legacy path still used by the disc pipeline, marked for removal. New MIDI features belong in `MidiGenerator`.

### Cross-cutting conventions

- `hmsm/io/source.py` sets `OPENCV_IO_MAX_IMAGE_PIXELS` **before** importing cv2/skimage, as do `discs/cluster.py`, `discs/utils.py` and `utils.py`. That ordering is load-bearing for large scans — do not let an import sorter move those lines above the `os.environ` assignment.
- Optional dependencies (`enlighten`, `cairosvg`, `vnoise`) are imported in try/except blocks setting a `_has_*` flag; follow that pattern rather than adding hard dependencies.
- Progress and status go through the stdlib `logging` module, one logger per module (`logging.getLogger(__name__)`); `cli.py` configures it. Expensive debug work is guarded.

## Known weaknesses

All are documented in `docs/ARCHITECTURE.md`:

- **Roll start/end detection.** `digitizer.find_roll_start` only measures how far the paper edges move within a 100-row window, so a shallow-tapering head, or a label extending past the head, reads as straight roll. This still puts some stray notes at the beginning of a transcription, though far fewer than before: on `examples/animatic_t_roll_head.tif` the count in the head region went from 38 to 1 once the label's printing stopped reading as holes. `tests/test_digitizer.py::test_roll_start_misses_a_head_that_tapers_too_gently` pins the current behaviour.
- **Coloured annotations are not told apart.** Lanes are separated by position across the roll only. `PaperModel.chroma` gives the signed channels that would separate a red Metrostyle line from a grey dynamics line in the *same* lane, but nothing reads them yet. Accentuation marks on the Phonola are likewise segmented (if declared) but not interpreted — there is no control code for them.
- **The roll head is a different paper.** Heads are often printed on stock of another colour; the reference colours come from the roll as a whole, and the local level correction only partly absorbs that.
