# Roll Format Support Status

This document provides an overview over the piano roll formats supported for midi generation out-of-the-box.

Additional information on the formats (in german) can be found [here](https://musixplora.de/mxp/2002522).

## Support for specific Formats

Due to historical and geographical reasons our collection contains a large amount of Hupfeld Piano Rolls.
We provide specific configuration profiles for some of these formats with the application.

| Provided Preset Name | Applicable Formats | Support for Midi Creation | Support for Control Information | Notes |
| -------------------- | ------------------ | ------------------------- | ------------------------------- | ----- |
| `animatic` | Hupfeld 88 Animatic, Hupfeld 88 Animatic T, Hupfeld 88 Animatic Clavitist | ✅ | Complete Support for Animatic rolls, Support for Animatic T and Animatic Clavitist limited to the functionality level of Animatic | A very small number (<0.1%) of false positives may currently occure, where parts of the printed dynamics line are misinterpreted as holes in the roll |
| `phonola` | Hupfeld 73 Phonola | ✅ | ✅ | Printed information (dynamics line, pedal annotations) is automatically extracted and processed but might be subject to small inaccuracies |
| `phonola_solodant` | Hupfeld 73 Phonola Solodant | ✅ | ✅ | See `phonola` |
| `clavitist` | Hupfeld Clavitist, Hupfeld Clavitist Spezialrolle, Hupfeld Clavitist Universal | ✅ | Pedal control supported, other control tracks currently unspported | |

## Additional Formats (Partially) Supported

We provide generic profiles that potentially only extract a subset of the control information present on any given roll format.
Support for three general types of rolls come bundled with the application, one for 73 tone Hupfeld rolls and two that support 65 tone and 88 tone rolls that conform with the Buffalo 1908 Convention specifications and should provide atleast a basic level of support for virtually all roll formats released after that.

| Provided Preset Name | Tested Formats | Support for Midi Creation | Support for Control Information | Notes |
| -------------------- | ------------------ | ------------------------- | ------------------------------- | ----- |
| `generic_73`, `generic_73_no_dynamics`, `generic_73_fsharp`, `generic_73_fsharp_no_dynamics` | Hupfeld 73 Phonoliszt, Hupfeld 73 Concert Phonoliszt, potentially other Hupfeld Formats | ✅ | Printed annotations (dynamics, pedal) are fully supported, only limited support for dynamics coded in the roll | Support for generic Hupfeld formats with 73 Tones. Some of these formats encode f-sharp (MIDI 30) on Track 39 while other use this track for control information. Use `_fsharp` for the former. |
| `generic_65`, `generic_65_no_dynamics` | Aeolian Piano 65, Aeolian Pianola 65, Aeolian Universal, Imperial, K.N.R., Kohler & Campbell, Phil AG Ducanola, The Autopiano Company  | ✅ | Printed dynamics line only | Profile that supports 65 tone rolls that conform to the Buffalo 1908 Convention. The 2 (optional) control tracks present on some rolls are ignored if present. Printed dynamics annotation is automatically extracted and applied. Dynamics processing will not work for formats with multiple printed annotation lines (double dynamics, dynamics and tempo). |
| `generic_88` | Blohut Rolla Artis, Concordia, Hupfeld Tri-Phonola, S.M. 88 S.M., Wöhle & Co. Diamant| ✅ | Printed dynamics lines are supported if present, encoded pedal and dynamics boost (for bass and discant) work, all other information is discarded. | Supports rolls according to the 88 tone format specified by the Buffalo 1908 convention.|


## Support Status for Control Information

This table has the control information currenlty supported for automatic extraction and in midi creation.

| Numerical ID | Description | Support Status |
| ------------ | ----------- | -------------- |
| `-3` | Pedal information on Hupfeld type rolls. The pedal is set to on at the beginning of the roll and toggled off while this track is active. | ✅ |
| `-10`, `-11`, `-20`, `-21` | Dynamics information on some Hupfeld type rolls. Notes beeing played while these tracks are active will have increased velocity. Always used in pairs and separate for bass (`-10`, `-11`) and discant (`-20`, `-21`). | ✅ |
| `N/A` | Printed dynamics information | Generally fully supported, however anomalies might occure where annotations are in very similar color to the scan background |
| `N/A` | Printed pedal information | ✅ |

## Adding Support for Additional Formats

The application is highly configurable through configuration profiles.
If you plan to convert a format not covered by the provided profiles, we provide the `roll2config` tool to help you create a configuration stub that will give you a head start in adding support for a new roll type.
The tool will analyse a given roll scan and detect the location of tracks present on the roll.
You can then assign midi tones and (optionally) control codes to the detected tracks to get a full profile.
As the tool can only detect tracks that are actually beeing used on the given roll scan, using a roll that has all tracks populated (like a scale or a tester) is generally beneficial.
Run `roll2config --help` for more information and see [CONFIG.md](CONFIG.md) for more information on the configuration files used by the application.