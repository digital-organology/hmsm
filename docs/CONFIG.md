# Configuration for Roll Formats

This file provides an overview over the configuration format used to create a profile for piano roll to midi transformation.

A profile describes a *physical format*, not a particular scan: every measurement in it is in millimetres and is converted to pixels using the resolution the scan declares (falling back to 300 dpi).
The colours of the paper and of the scanner background are **not** part of a profile, because they are properties of the scan; they are read off the scan itself.

Configuration information is `json` based and a profile looks as follows:

```
{
    "media_type": "roll",
    "method": "roll",
    "roll_width_mm": 286.0,
    "binarization_method": "paper_relative",
    "binarization_options": {},
    "hole_width_mm": 2.7,
    "ink_layers": [
        {"name": "pedal", "role": "pedal", "region": [0.0, 0.095]},
        {"name": "dynamics", "role": "dynamics", "region": [0.095, 1.0]}
    ],
    "track_measurements": [
        {
            "left": 2.594324501953756,
            "right": 4.876675703902138,
            "tone": 29
        },
        [...]
    ]
}
```

Let's go over this piece by piece.

```
"media_type": "roll",
"method": "roll",
```

These fields contain meta information for the application to know which methods to dispatch.
For piano roll processing currently only these exact values are supported.

```
"roll_width_mm": 286.0,
```

This field contains the physical width of the roll in mm.
The unit itself is not actually important, but needs to be consistent with the measurements of the tracks later on.

## Segmentation

```
"binarization_method": "paper_relative",
"binarization_options": {},
```

This block sets information the application needs to segment the roll scan into meaningful elements.
The `binarization_method` field lets the application know which method to use segmenting the scan.
Two methods are available.

### `paper_relative` (the default)

This method measures every pixel against two reference colours read off the scan itself: the **paper**, which is whatever the scan is mostly made of, and the **background**, which is what shows through a punched hole.
From those it derives *transmission* (zero on clean paper, one where the background shows through), *shade* (how much darker than the paper a pixel is) and *chroma* (how far off the line between the two colours it lies).

Because every threshold is expressed against the scan's own paper, the defaults hold for beige, pink, red and green paper, for black and for white scanner backgrounds, and for paper that has darkened unevenly with age.
**A profile normally needs no `binarization_options` at all.**
Where one is needed the following are accepted:

| Option | Default | Meaning |
| ------ | ------- | ------- |
| `hole_grow` | `0.5` | Transmission at which a pixel is on the edge of a hole. Half way between paper and background is where the edge of a blurred hole actually lies, so this is rarely worth changing. |
| `hole_seed` | `0.75` | Transmission a region has to reach somewhere before it is a hole at all. Raise it if paper speckle is coming through, lower it if faintly scanned holes are being missed. |
| `ink_shade` | `0.10` | Shade above which a pixel counts as printing. Clean paper stays within about three percent of zero, so this is several times the noise. Lower it for very faint printing, raise it if paper grain is being collected as ink. |
| `hole_neutrality` | `0.015` | How far off the paper-background line a dark region may sit and still be a hole rather than heavily printed ink. Only relevant for formats whose printing is nearly as dark as a hole. |

### `v_channel`

The original method, which thresholds absolute pixel brightness.
It is cheaper and perfectly adequate on a clean, high-contrast scan of beige paper, but its thresholds have to be retuned for every paper colour, and on a dark red Welte roll a setting that works for beige marks the entire roll as printing.
It takes:

| Option | Meaning |
| ------ | ------- |
| `threshold` | Darkness below which a pixel counts as a hole, in `[0, 1]`. For black backgrounds a pixel with `value < threshold` is a hole, for white backgrounds `(1 - value) < threshold`. |
| `upper_threshold` | Darkness below which a pixel counts as printing. Pixels with `threshold < value < upper_threshold` are treated as printed annotation. Omit for formats with no printing. |
| `roll_detection_threshold` | Find the paper edges by thresholding a grayscale version of the scan rather than by taking the extent of the hole mask, for use where the background is noisy. `"auto"` picks the threshold with Otsu's method. |

## Holes

```
"hole_width_mm": 2.7,
```

is used for filtering the detected holes on the roll and removing artifacts.
If the roll contains holes of multiple sizes (like the Hupfeld 73 Phonola Solodant for example) you can also set this parameter to a list; the size of the most frequent hole type should be first, as that is the one used to decide when two holes sound as a single note.

```
"hole_length_mm": [1.5, 40.0],
```

is optional and bounds how long a punched hole may be along the roll.
The upper bound is what keeps a fold or a torn edge running down the paper out of the note table.
Leave it out to accept a hole of any length; a roll that holds long sustained notes punched as one slot needs either a generous bound or none.

## Printed annotations

```
"ink_layers": [
    {"name": "pedal", "role": "pedal", "region": [0.0, 0.095]},
    {"name": "dynamics", "role": "dynamics", "region": [0.095, 1.0]}
]
```

Printed markings are laid out by lane, and `ink_layers` declares them.
Each entry has three fields:

- `name` identifies the layer's mask and names it in debug output.
- `region` is where the lane sits across the roll, as a pair of fractions of the roll width measured from the left paper edge. Lanes may touch but must not overlap. Positions are taken relative to the paper edges row by row, so a roll that drifts sideways keeps its lanes lined up with its printing.
- `role` says what to make of the layer:

| Role | Handling |
| ---- | -------- |
| `dynamics` | The marks are traced into a continuous line and its distance from the paper edge becomes note velocity. |
| `pedal` | The marks are paired into sustain spans: a wide marker (Hupfeld's "Ped.") opens one and a narrow marker (a printer's flower) closes it. |
| anything else | The lane is segmented and its marks counted, but nothing reads them. |

That last row is the intended way to look at a marking before it has been decided what it means.
Declaring a layer as, say, `"role": "tempo"` will report at startup that the role is not interpreted, and log how many marks it found, without affecting the transcription.

Note that the Hupfeld dynamics line is **not** confined to a narrow lane: it sweeps right across the note tracks, so its region is most of the roll.
Only the pedal words sit in a lane of their own.
Sharing a lane with the note holes is fine — holes are cut out of the ink mask before the line is traced — and so is sharing it with accent marks and a watermark, which the trace steps around.

A profile with no `ink_layers` is a format with no printing, and nothing is extracted.

## Track measurements

```
"track_measurements": [
    {
        "left": 2.594324501953756,
        "right": 4.876675703902138,
        "tone": 29
    },
    [...]
]
```

These contain information on the tracks that are on the format.
Each entry has 3 values: `left` and `right` are the distance (in mm, or the same unit as `roll_width_mm`) from the left edge of the roll to the left and right side of the holes on this track.
`tone` is the midi note that this track is assigned to.
Negative numbers are used for control tracks; for a list of supported values see [FORMATS.md](FORMATS.md).
The order of the values does not matter.

A hole is assigned to the track its *middle* falls on, so a hole scanned wider or narrower than nominal still lands on its own track.
A component that falls outside every track — more than half a track pitch beyond the outermost one — is discarded rather than snapped onto the nearest track, so dirt in the margin does not become a note.
The flip side is that a track the profile does not declare is silently dropped; run with `-d` and watch for "Dropped N component(s) that sit on no track" if a format is producing fewer notes than it should.

## Getting a head start

If you want to get a headstart in creating these measurements (or are just lazy), you can use the `roll2config` utility we provide.
An example call could look like this:

```
hmsm roll2config -w 286 -s 2.7 roll_scan.tif config_stub.json
```

This tells the program that the roll has a physical width of 286 millimetres and holes of 2.7 millimetres width.
The detected tracks will be written in the target json file, along with `binarization_method` and `hole_width_mm`; tones, control codes and any `ink_layers` the format carries have to be filled in by hand.

This can obviously only detect tracks that are actually used on the scan provided, so you might have to run it on multiple scans and combine the results to get a complete track listing.
