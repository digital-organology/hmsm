# HMSM-Tools

This repository contains tool for the digitization and analysis of **H**istorical **M**usical **S**torage **M**edia.
This python package is the software implementation for the digitization part of the [BMBF](https://www.bmbf.de/bmbf/de/home/home_node.html) funded research project [DISKOS](https://organology.uni-leipzig.de/index.php/forschung/diskos) at the [Research Center Digital Organology](https://organology.uni-leipzig.de/) at Leipzig University.

Development is currently underway with support for additional formats/components beeing added.

## Installation

In the future we will publish this package to PyPI, in the meantime the development version can by installed directly from Github:

```{bash}
pip install git+https://github.com/digital-organology/hmsm.git
```

For additional installation information see [INSTALL.md](docs/INSTALL.md).

## Usage

### Piano Roll Digitization

We support digitization for a number of formats of piano rolls out of the box, for an overview see [FORMATS.md](docs/FORMATS.md).
Support for additional formats in our collection is currently beeing added along with extending the support for control information present on the formats already supported.
The application also ships with the `roll2config` tool which can be used to get a headstart in the creation of a processing profile for new formats.

To process a roll, use the provided `roll2midi` utility, like so:

```{bash}
roll2midi -l 5000 -c clavitist hupfeld_clavitist_roll.tif out.mid 
```

This will:

* `-l 5000` skip the first 5000 lines of the image provided, which can be useful in ignoring the head of the roll (if present) which may or may not introduce erroneously detected note information if not skipped
* `-c clavitist` use the `clavitist` profile bundled with the application. You may also pass the path to a json file containing a custom configuration or even a raw json string here.
* `hupfeld_clavitist_roll.tif` read the roll scan from this file
' `out.mid` write the generated midi file here

For more inforamtion on the command line interface for roll digitization pass the `-h` or `--help` parameter:

```
roll2midi --help
```

### Cardboard Disc Digitization

We currently support Image-to-Midi transformation for Ariston brand cardboard discs with 24 tracks. We expect our process to work for all types of discs that enode information in the same general way.

During development we use photographs of discs like the following:

![Ariston Brand Disc](assets/5070081_22.JPG)

These images are overexposed and have the start position of the disc aligned to 0 Degrees.
While we only did minimal testing up until now, we expect our approach to also work for normal pictures, as long as there is sufficient contrast between the disc and the image background.
Currently this may require addition preprocessing.
Be sure to pass the rotation of the start position of the disc using the `--offset` parameter.

To digitize the example image included in this repository call the included command line utiltiy, like so:

```
disc2midi -c ariston -m cluster assets/5070081_22.JPG out.mid
```

We also include a utility to transform circular media into a rectangular representation (similar to piano rolls).
This can be useful if you prefer to use an existing digitization solution for piano rolls for the image to midi transformation.
It should generally work with all circular media types as long as there is sufficient contrast between background and medium.
To transform the included color photography of the same disc as above use:

```
disc2roll --offset 92 assets/5070081_11.JPG roll.JPG
```

### Cardboard Disc Generation

Included is an additional utitily that will create the image of a cardboard disc from a provided midi file. All notes that are in the midi file and not supported by the disc format will be ignored.

```
midi2disc -n "My Awesome Disc" midi_input.mid disc_output.jpg
```

## License

We provide this software under the GNU-GPL (Version 3-or-later, at your discretion).

The photographs included in this repository (located unter `assets/`) are taken from our research platform [MusiXplora](https://www.musixplora.de/) and are generally provided under a CC BY-SA 4.0 License unless otherwise specified.
