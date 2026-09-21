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

`docs/CONFIG.md` documents every field of a roll profile and `docs/FORMATS.md` tracks which physical formats each preset covers — update both when adding or changing a preset.

### Reading scans (`src/hmsm/io/`)

`open_source(path)` returns an `ImageSource`. For strip and tile based TIFFs that is `TiffSource`, which decodes only the segments a requested row range touches, on a thread pool; anything else is decoded once into an `ArraySource` with the same interface. Memory is set by the band size, not the length of the scan — a 2 GB scan processes in a few hundred MB.

`source.context_bands(height, margin)` yields bands padded with surrounding rows that the caller discards afterwards. That padding is what makes results independent of where band boundaries fall; without it, morphology and edge smoothing behave differently at a band edge. Don't remove it.

### Roll pipeline (`src/hmsm/rolls/`)

`RollDigitizer.run()` drives, per band: `binarization.segment` (hole mask, annotation mask, paper edges) → `edges.detect_edges` (per-row paper position) → `holes.extract_notes` (components → note table) → `annotations.AnnotationCollector` (accumulates fragments). Then `notes.merge_notes`, `notes.rebase`, and `MidiGenerator`.

Binarization methods are registered with the `@binarizer("name")` decorator and selected by a profile's `binarization_method`; `binarization_options` are splatted in as kwargs. Adding one means writing a function returning `BandMasks` — nothing else changes.

A band that fails to segment carries meaning: before the first holes are seen it is the roll head and is skipped; afterwards it is the end of the paper and processing stops.

**Performance-critical choices, do not undo:**
- Morphology goes through `cv2.morphologyEx` on uint8, not `skimage.morphology`, which dispatches boolean input to scipy's *grayscale* filter path and is ~150-245x slower for identical output.
- Component analysis goes through `cv2.connectedComponentsWithStats`, which returns the bounding boxes and areas the pipeline actually needs. `hmsm.utils.to_coord_lists` materialises every set pixel's coordinates (~1 GB per 4000-row band) and is only still used by the disc pipeline.
- The value channel is a per-pixel max over uint8 channels, not a float conversion.

### The note table

An `(n, 3)` int64 array of `[start_row, end_row, tone]` in scan rows. **Negative `tone` values are control codes, not pitches** (`hmsm.midi.controls.ControlCode`; table in `docs/FORMATS.md`). Any code touching note arrays must keep that convention.

Rows are absolute scan rows until `notes.rebase` moves the first note to zero at the end; it shifts the dynamics line by the same amount, which is what keeps velocities aligned with their notes.

### Disc pipeline (`src/hmsm/discs/`)

Not yet ported onto the streaming and profile machinery — `DiscProfile.as_dict()` is the seam. `discs/cluster.py` binarizes, crops, does morphological edge detection, fits an ellipse to the circumference, masks out the inner label, and assigns holes to tracks by radial distance. Note the comment in `cluster.py` about coordinate ordering: centres are `(y, x)` everywhere except the scipy distance call. `discs/generator.py` (`midi2disc`) renders discs from MIDI; its geometry lives in a hardcoded `_get_config(type)`, *not* in `data/config.json`.

### MIDI (`src/hmsm/midi/`)

`MidiGenerator` understands tempo scaling from feet-per-minute, velocity from the dynamics line and the boost tracks, and pedal spans. Event times are rounded in absolute terms and differenced afterwards, so rounding error cannot accumulate over the length of a roll. `create_midi()` at the bottom of `midi/__init__.py` is the legacy path still used by the disc pipeline, marked for removal. New MIDI features belong in `MidiGenerator`.

### Cross-cutting conventions

- `hmsm/io/source.py` sets `OPENCV_IO_MAX_IMAGE_PIXELS` **before** importing cv2/skimage, as do `discs/cluster.py`, `discs/utils.py` and `utils.py`. That ordering is load-bearing for large scans — do not let an import sorter move those lines above the `os.environ` assignment.
- Optional dependencies (`enlighten`, `cairosvg`, `vnoise`) are imported in try/except blocks setting a `_has_*` flag; follow that pattern rather than adding hard dependencies.
- Progress and status go through the stdlib `logging` module, one logger per module (`logging.getLogger(__name__)`); `cli.py` configures it. Expensive debug work is guarded.

## Known weaknesses

Both are scheduled for rework and documented in `docs/ARCHITECTURE.md`:

- **Roll start/end detection.** `digitizer.find_roll_start` only measures how far the paper edges move within a 100-row window, so a shallow-tapering head, or a label extending past the head, reads as straight roll. This is what puts stray notes at the beginning of a transcription. `tests/test_digitizer.py::test_roll_start_misses_a_head_that_tapers_too_gently` pins the current behaviour.
- **Annotation extraction.** The dynamics line and pedal markers are separated from holes by brightness alone, so annotations in a colour close to the paper are missed.
