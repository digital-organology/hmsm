# Roll Format Support Status

The following table provides an overview over the formats for which we ship support bundled with the application.
Addition formats can be used by creating a configuration profile for them, this may be assisted by the `roll2config` tool included in the application.
Note that the provided presets may provide support for roll formats not listed here when they are identical in physical dimensions and information encoding.

Additional information on the formats can be found (in german) [here](https://musixplora.de/mxp/2002522).

| Provided Preset Name | Applicable Formats | Support for Midi Creation | Support for Control Information | Notes |
| -------------------- | ------------------ | ------------------------- | ------------------------------- | ----- |
| `animatic` | Hupfeld 88 Animatic, Hupfeld 88 Animatic T | ✅ | Complete Support for Animatic rolls, Support for Animatic T rolls beeing tested | A very small number (<0.1%) of false positives may currently occure, where parts of the printed dynamics line are misinterpreted as holes in the roll |
| `phonola` | Hupfeld 73 Phonola | ✅ | ✅ | Printed information (dynamics line, pedal annotations) is automatically extracted and processed but might be subject to small inaccuracies |
| `phonola_solodant` | Hupfeld 73 Phonola Solodant | ✅ | ✅ | See `phonola` |
| `clavitist` | Hupfeld Clavitist, Hupfeld Clavitist Spezialrolle, Hupfeld Clavitist Universal, Hupfeld 73 Phonoliszt, Hupfeld 73 Concert Phonoliszt | ✅ | Pedal control supported, other control tracks currently unspported | |
| `animatic_clavitist` | Hupfeld 88 Animatic Clavitist | ✅ | Pedal and dynamic control supported, other control tracks currently unspported | |
| `philag` | Phil AG 88 Philag | ❓ | ❓ | Format is identical to the Hupfeld 88 Animatic, however the use of black ink used for annotations renders our current approach (which is based on colorspace segmentation) problematic for scans with a black background (like ours are). A high error rate is to be expected.|

## Support Status for Control Information

This table has the control information currenlty supported for automatic extraction and in midi creation.

| Numerical ID | Description | Applicable Formats | Support Status |
| ------------ | ----------- | ------------------ | -------------- |
| `-3` | Pedal information on Hupfeld type rolls. The pedal is set to on at the beginning of the roll and toggled off while this track is active. | `clavitist`, `animatic_clavitist`, `animatic`, `philag` | ✅ |
| `-10`, `-11`, `-20`, `-21` | Dynamics information on some Hupfeld type rolls. Notes beeing played while these tracks are active will have increased velocity. Always used in pairs and separate for bass (`-10`, `-11`) and discant (`-20`, `-21`). | `animatic_clavitist`, `animatic`, `philag_88` | ✅ |
| `N/A` | Printed dynamics information | `animatic`, `phonola`, `phonola_solodant`, `philag` | Generally fully supported, however anomalies might occure where annotations are in very similar color to the scan background |
| `N/A` | Printed pedal information | `phonola`, `phonola_solodant` | ✅ |