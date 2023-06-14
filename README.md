# HMSM-Tools

This repository contains tool for the digitization and analysis of **H**istorical **M**usical **S**torage **M**edia.
This python package is the software implementation for the [BMBF](https://www.bmbf.de/bmbf/de/home/home_node.html) funded research project [DISKOS](https://organology.uni-leipzig.de/index.php/forschung/diskos) at the [Research Center Digital Organology](https://organology.uni-leipzig.de/) at Leipzig University.

We are currently in the process of porting existing functionality to this package as well as implementing further functionality for the digitization and analysis of additional formats of historical storage media.

## Installation

In the future we will publish this package to PyPI, in the meantime the development version can by installed directly from Github:

```{bash}
pip install git+https://github.com/digital-organology/hmsm.git
```

For additional installation information see [INSTALL.md](docs/INSTALL.md).

## Usage

We currently support Image-to-Midi transformation for Ariston brand cardboard discs with 24 tracks. We are in the process of supporting additional types of media and expect to eventuelly roll out support for most if not all types of [discs](https://www.musixplora.de/mxp/2003518) as well as [piano rolls](https://www.musixplora.de/mxp/2002522) in the collection of the Museum of Musical Instruments at Leipzig University.

During development we use photographs of discs like the following:

![Ariston Brand Disc](assets/5070081_22.JPG)

These images are overexposed and have the start position of the disc aligned to 0 Degrees.
While we only did minimal testing up until now, we expect our approach to also work for normal pictures, as long as there is sufficient contrast between the disc and the image background.
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

## License

We provide this software under the GNU-GPL (Version 3-or-later, at your discretion).

The photographs included in this repository (located unter `assets/`) are taken from our research platform [MusiXplora](https://www.musixplora.de/) and are generally provided under a CC BY-SA 4.0 License.
