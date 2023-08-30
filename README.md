# HMSM-Tools

This repository contains tool for the digitization and analysis of **H**istorical **M**usic **S**torage **M**edia.
This python package is the software implementation for the [BMBF](https://www.bmbf.de/bmbf/de/home/home_node.html) funded research project [DISKOS](https://organology.uni-leipzig.de/index.php/forschung/diskos) at the [Research Center Digital Organology](https://organology.uni-leipzig.de/) at Leipzig University.

We are currently in the process of porting existing functionality to this package as well as implementing further functionality for the digitization and analysis of additional formats of historical storage media.

## Installation

In the future we will publish this package to PyPI, in the meantime the development version can by installed directly from Github:

```{bash}
pip install git+https://github.com/digital-organology/hmsm.git
```

For additional installation information see [INSTALL.md](docs/INSTALL.md).

## Usage

### Disc to Midi Transformation

We currently support Image-to-Midi transformation for Ariston brand cardboard discs with 24 tracks. We are in the process of supporting additional types of media and expect to eventually roll out support for most if not all types of [discs](https://www.musixplora.de/mxp/2003518) as well as [piano rolls](https://www.musixplora.de/mxp/2002522) in the collection of the Museum of Musical Instruments at Leipzig University.

During development we use photographs of discs like the following:

![Ariston Brand Disc](assets/5070081_22.JPG)

These images are overexposed and have the start position of the disc aligned to 0 Degrees.
While we only did minimal testing with other types of photographs up until now, we generally expect our approach to also work for normal pictures, as long as there is sufficient contrast between the disc and the image background.
Be sure to pass the rotation of the start position of the disc using the `--offset` parameter (rotation is measured in the running direction of the disc, meaning counter-clockwise).

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

## License

We provide this software under the GNU-GPL (Version 3-or-later, at your discretion).

The photographs included in this repository (located unter `assets/`) are taken from our research platform [MusiXplora](https://www.musixplora.de/) and are generally provided under a CC BY-SA 4.0 License.
