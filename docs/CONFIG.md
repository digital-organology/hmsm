# Configuration for Roll Formats

This file provides an overview over the configuration format used to create a profile for piano roll to midi transformation.

Configuration information is `json` based and a profile looks as follows:

```
{
    "media_type": "roll",
    "method": "roll",
    "roll_width_mm": 286.0,
    "binarization_method": "v_channel",
    "binarization_options": {
        "threshold": 0.10,
        "upper_threshold": 0.75,
        "roll_detection_threshold": "auto"
    },
    "pedal_cutoff": 0.085,
    "hole_width_mm": 2.7,
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

```
"binarization_method": "v_channel",
"binarization_options": {
    "threshold": 0.10,
    "upper_threshold": 0.75,
    "roll_detection_threshold": "auto"
},
```

This block sets sets information the application needs to segment the roll scan into meaningful elements.
The `binarization_method` field let's the application know which method to use segmenting the scan.
Currently only `v_channel` is available, a method that works on computing the v_channel (aka the "blackness" level) for each pixel, but other methods could be implemented e.g. by using distance meaasures in the RGB colorspace.

The `threshold` sets the limit for which values will be considered belonging to the roll and which values will be considered belonging to the background.
This depends on the background color, for black backgrounds every value with `value < threshold` will be considered background, for white backgrounds it will be `(1 - value) < threshold`.

If `upper_threshold` is set all pixels with values `threshold < value < upper_threshold` will be considered to belong to printed annotations on the roll and be processed as such.

Additionally, if the background is somewhat messy (we have experienced some levels of white noise on our usualy black background scans) setting `roll_detection_threshold` will tell the application to detect the area of the roll by performing binary thresholding on a grayscale version of the roll with the provided threshold.
If set to `auto` otsus method will be used to compute the theshold.

```
"pedal_cutoff": 0.085,
```

Setting `pedal_cutoff` will tell the processing software that the printed annotations contain pedal markers (assumed to be on the left side of the roll), the value indicates the (relative) area on the left side of the roll where they are printed.

```
"hole_width_mm": 2.7,
```

is used for filtering the detected holes on the roll and removing artifacts.
This needs to be in millimeters as it will be used to calculate absolute measurements and assumes the resolution of the input scan image to be 300dpi.
If the roll contains holes of multiple sizes (like the Hupfeld 73 Phonola Solodant for example) you can also set this parameter to a list, the size of the most frequent hole type should be first.

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

The final parameter are the `track_measurements`.
These contain information on the tracks that are on the format.
Each entry has 3 values: `left` and `right` set the distance (in mm or the same unit as `roll_width_mm`) are the distance from measured from the left and right side of the holes on this track to the left edge of the role.
`tone` is the midi note that this track is assigned to.
Negative numbers are used for control tracks, for a list of supported values see [FORMATS.md](FORMATS.md).
The order of the values does not matter.

If you want to get a headstart in creating these measurements (or are just lazy), you can use the `roll2config` utility we provide.
An example call could look like this:

```
roll2config -w 286 -s 2.7 -t 0.1 roll_scan.tif config_stub.json
```

This tells the program that the roll has a physical width of 286 millimeters, holes of 2.7 millimeters width and to use a threshold of 0.1 for segmentation.
The detected tracks will be written in the target json file.

This can obviously only detect tracks that are acutally used on the scan provided, so you might have to run it on multiple scans and combine the results to get a complete track listing.