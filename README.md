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
If you miss support for a format of interest for you, we provide tooling to help you create configuration information for that format.
For additional information on this, see [CONFIG.md](docs/CONFIG.md).

To process a roll, use the provided `roll2midi` utility, like so:

```{bash}
roll2midi -c clavitist -t 60 hupfeld_animatic_roll.tif out.mid 
```

This will:

* `-c animatic` use the `animatic` profile bundled with the application. You may also pass the path to a json file containing a custom configuration or even a raw json string here.
* `-t 60` set the roll speed to 60. The unit is feet-per-minute times 10, which is the unit annotated on (most) rolls. This means the roll will effectively be processed as if it were played back 6 feet per minute.
* `hupfeld_animatic_roll.tif` read the roll scan from this file
* `out.mid` write the generated midi file here

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

To digitize the example image included in this repository call the included command line utility, like so:

```{bash}
disc2midi -c ariston -m cluster assets/5070081_22.JPG out.mid
```

### Midi to Disc Transformation

Included in this package is functionality to create images of cardboard discs from midi files.
We currently include a profile for the Ariston 24 type of disc, though it should be relatively trivial to add support other types of discs.
You can test this with any midi file of your choosing, any notes that are not contained in the given format will be dropped automatically.

```{bash}
midi2disc -t ariston_24 -s 4000 -n "Title can have<br>multiple lines" input.mid output.png
```

Though this is not a core feature of our application we used midis generated from original discs to verify that the results are close to the original media.

### Disc to Roll Image Transformer

We also include a utility to transform circular media into a rectangular representation (similar to piano rolls).
This can be useful if you prefer to use an existing digitization solution for piano rolls for the image to midi transformation process.
It should generally work with all circular media types as long as there is sufficient contrast between background and medium for the algorithm to detect the edge of the medium correctly.
To transform the included color photography of the same disc as used above run:

```{bash}
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
